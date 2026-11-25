/*
 * xr_openxr.c - immersive VR mode on an OpenXR runtime (Meta Quest).
 *
 * The game is a flat 4:3 picture, so VR mode is a cinema: one quad layer, a
 * 3 m wide screen 2.2 m in front of where the player's head started, in an
 * otherwise empty space. The Vulkan backend (nv2a_pb_vulkan.c) makes its
 * device through the runtime (XR_KHR_vulkan_enable2) and copies each
 * presented frame into an image (nv2a_vk_frame); a thread of this file's
 * own runs the runtime's frame loop at the headset's rate and blits the
 * latest frame into the layer's swapchain, so the guest keeps its own 60 Hz.
 *
 * The motion controllers become player 1's Duke pad (xbox_InputSetExtPad):
 *   left / right thumbstick  left / right stick      triggers   LT / RT
 *   A B X Y                  A B X Y                 grips      white / black
 *   menu (left)              START                   left stick click  BACK
 * Burnout 3's accelerate and brake are the triggers, boost A, X looks
 * behind, Y cycles the race camera, as on a pad.
 *
 * The right stick click is the camera button, not the game's: a short tap
 * cycles Burnout's camera (a Y tap), a long press (XR_LONG_NS) switches
 * between external mode (the screen fixed in the room, where the
 * headset's recenter puts it) and in-world mode (for now a wide screen
 * locked to the head; per-eye stereo from the car is to come).
 */
#if defined(XBR_HAVE_OPENXR) && defined(__ANDROID__)

#define XR_USE_PLATFORM_ANDROID
#define XR_USE_GRAPHICS_API_VULKAN
#include <jni.h>
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <SDL3/SDL.h>

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "xr_openxr.h"
#include "nv2a_vk_xr.h"
#include "host_pad.h"
#include "xinput_xbox.h"

#define LOGP "[xr] "

static XrInstance s_xi;
static XrSystemId s_sys;
static XrSession  s_sess;
static XrSpace    s_space;
static XrSwapchain s_sc;
static uint32_t   s_sc_w = 1280, s_sc_h = 960, s_sc_n;
static VkImage    s_sc_img[8];
static VkDevice   s_dev;
static VkQueue    s_queue;
static VkCommandPool   s_pool;
static VkCommandBuffer s_cb;
static VkFence    s_fence;
static int        s_running, s_quit;
static pthread_t  s_thread;

static PFN_xrGetVulkanGraphicsRequirements2KHR p_req;
static PFN_xrCreateVulkanInstanceKHR  p_inst;
static PFN_xrGetVulkanGraphicsDevice2KHR p_pdev;
static PFN_xrCreateVulkanDeviceKHR    p_dev;

static int xr_ok(XrResult r, const char *what)
{
    if (XR_SUCCEEDED(r))
        return 1;
    fprintf(stderr, LOGP "%s failed: %d\n", what, (int)r);
    return 0;
}

/* ---- the controllers --------------------------------------------------------- */

static XrActionSet s_aset;
enum { A_STICK_L, A_STICK_R, A_TRIG_L, A_TRIG_R, A_GRIP_L, A_GRIP_R,
       A_A, A_B, A_X, A_Y, A_MENU, A_CLICK_L, A_CLICK_R, A_N };
static XrAction s_act[A_N];
static XrPath   s_hand[2];
static pthread_mutex_t s_padlock = PTHREAD_MUTEX_INITIALIZER;
static XBOX_INPUT_STATE s_pad;
#define XR_LONG_NS 600000000ll    /* a long press of the camera button */
#define XR_TAP_NS  150000000ll    /* the Y tap it sends */
static int64_t s_cam_down, s_y_until;   /* press start, end of the Y tap */
static int     s_cam_held_long;
static int     s_inworld;             /* 0 external, 1 in-world */
static XrSpace s_view;                /* head-locked space, in-world mode */

static int64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}
static int s_pad_live;

static int ext_pad(XBOX_INPUT_STATE *st)
{
    int live;
    pthread_mutex_lock(&s_padlock);
    live = s_pad_live;
    if (live)
        *st = s_pad;
    pthread_mutex_unlock(&s_padlock);
    return live;
}

static XrAction make_action(const char *name, XrActionType type, int both)
{
    XrActionCreateInfo ci = { XR_TYPE_ACTION_CREATE_INFO };
    XrAction a = XR_NULL_HANDLE;
    snprintf(ci.actionName, sizeof ci.actionName, "%s", name);
    snprintf(ci.localizedActionName, sizeof ci.localizedActionName, "%s", name);
    ci.actionType = type;
    ci.countSubactionPaths = both ? 2 : 0;
    ci.subactionPaths = both ? s_hand : NULL;
    xr_ok(xrCreateAction(s_aset, &ci, &a), name);
    return a;
}

static XrPath path(const char *s)
{
    XrPath p = XR_NULL_PATH;
    xrStringToPath(s_xi, s, &p);
    return p;
}

static int actions_init(void)
{
    XrActionSetCreateInfo si = { XR_TYPE_ACTION_SET_CREATE_INFO };
    static const struct { int a; const char *p; } bind[] = {
        { A_STICK_L, "/user/hand/left/input/thumbstick" },
        { A_STICK_R, "/user/hand/right/input/thumbstick" },
        { A_TRIG_L,  "/user/hand/left/input/trigger/value" },
        { A_TRIG_R,  "/user/hand/right/input/trigger/value" },
        { A_GRIP_L,  "/user/hand/left/input/squeeze/value" },
        { A_GRIP_R,  "/user/hand/right/input/squeeze/value" },
        { A_A,       "/user/hand/right/input/a/click" },
        { A_B,       "/user/hand/right/input/b/click" },
        { A_X,       "/user/hand/left/input/x/click" },
        { A_Y,       "/user/hand/left/input/y/click" },
        { A_MENU,    "/user/hand/left/input/menu/click" },
        { A_CLICK_L, "/user/hand/left/input/thumbstick/click" },
        { A_CLICK_R, "/user/hand/right/input/thumbstick/click" },
    };
    XrActionSuggestedBinding sb[A_N];
    XrInteractionProfileSuggestedBinding ps = { XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
    XrSessionActionSetsAttachInfo at = { XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO };
    unsigned i;

    s_hand[0] = path("/user/hand/left");
    s_hand[1] = path("/user/hand/right");
    snprintf(si.actionSetName, sizeof si.actionSetName, "pad");
    snprintf(si.localizedActionSetName, sizeof si.localizedActionSetName, "Gamepad");
    if (!xr_ok(xrCreateActionSet(s_xi, &si, &s_aset), "xrCreateActionSet"))
        return 0;
    s_act[A_STICK_L] = make_action("stick_l", XR_ACTION_TYPE_VECTOR2F_INPUT, 0);
    s_act[A_STICK_R] = make_action("stick_r", XR_ACTION_TYPE_VECTOR2F_INPUT, 0);
    s_act[A_TRIG_L]  = make_action("trigger_l", XR_ACTION_TYPE_FLOAT_INPUT, 0);
    s_act[A_TRIG_R]  = make_action("trigger_r", XR_ACTION_TYPE_FLOAT_INPUT, 0);
    s_act[A_GRIP_L]  = make_action("grip_l", XR_ACTION_TYPE_FLOAT_INPUT, 0);
    s_act[A_GRIP_R]  = make_action("grip_r", XR_ACTION_TYPE_FLOAT_INPUT, 0);
    s_act[A_A]       = make_action("button_a", XR_ACTION_TYPE_BOOLEAN_INPUT, 0);
    s_act[A_B]       = make_action("button_b", XR_ACTION_TYPE_BOOLEAN_INPUT, 0);
    s_act[A_X]       = make_action("button_x", XR_ACTION_TYPE_BOOLEAN_INPUT, 0);
    s_act[A_Y]       = make_action("button_y", XR_ACTION_TYPE_BOOLEAN_INPUT, 0);
    s_act[A_MENU]    = make_action("start", XR_ACTION_TYPE_BOOLEAN_INPUT, 0);
    s_act[A_CLICK_L] = make_action("click_l", XR_ACTION_TYPE_BOOLEAN_INPUT, 0);
    s_act[A_CLICK_R] = make_action("click_r", XR_ACTION_TYPE_BOOLEAN_INPUT, 0);
    for (i = 0; i < A_N; i++) {
        sb[i].action = s_act[bind[i].a];
        sb[i].binding = path(bind[i].p);
    }
    ps.interactionProfile = path("/interaction_profiles/oculus/touch_controller");
    ps.countSuggestedBindings = A_N;
    ps.suggestedBindings = sb;
    if (!xr_ok(xrSuggestInteractionProfileBindings(s_xi, &ps), "touch bindings"))
        return 0;
    at.countActionSets = 1;
    at.actionSets = &s_aset;
    return xr_ok(xrAttachSessionActionSets(s_sess, &at), "xrAttachSessionActionSets");
}

static float get_f(int a)
{
    XrActionStateGetInfo gi = { XR_TYPE_ACTION_STATE_GET_INFO };
    XrActionStateFloat st = { XR_TYPE_ACTION_STATE_FLOAT };
    gi.action = s_act[a];
    return XR_SUCCEEDED(xrGetActionStateFloat(s_sess, &gi, &st)) && st.isActive
           ? st.currentState : 0.0f;
}

static int get_b(int a)
{
    XrActionStateGetInfo gi = { XR_TYPE_ACTION_STATE_GET_INFO };
    XrActionStateBoolean st = { XR_TYPE_ACTION_STATE_BOOLEAN };
    gi.action = s_act[a];
    return XR_SUCCEEDED(xrGetActionStateBoolean(s_sess, &gi, &st)) && st.isActive
           && st.currentState;
}

static XrVector2f get_v(int a)
{
    XrActionStateGetInfo gi = { XR_TYPE_ACTION_STATE_GET_INFO };
    XrActionStateVector2f st = { XR_TYPE_ACTION_STATE_VECTOR2F };
    XrVector2f z = { 0, 0 };
    gi.action = s_act[a];
    return XR_SUCCEEDED(xrGetActionStateVector2f(s_sess, &gi, &st)) && st.isActive
           ? st.currentState : z;
}

static SHORT axis(float v)
{
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    return (SHORT)(v * 32767.0f);
}

static BYTE analog(float v)
{
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return (BYTE)(v * 255.0f + 0.5f);
}

static void actions_sync(void)
{
    XrActiveActionSet as = { s_aset, XR_NULL_PATH };
    XrActionsSyncInfo si = { XR_TYPE_ACTIONS_SYNC_INFO };
    XBOX_INPUT_STATE p;
    XrVector2f l, r;

    si.countActiveActionSets = 1;
    si.activeActionSets = &as;
    if (!XR_SUCCEEDED(xrSyncActions(s_sess, &si)))
        return;
    memset(&p, 0, sizeof p);
    l = get_v(A_STICK_L);
    r = get_v(A_STICK_R);
    p.Gamepad.sThumbLX = axis(l.x);
    p.Gamepad.sThumbLY = axis(l.y);
    p.Gamepad.sThumbRX = axis(r.x);
    p.Gamepad.sThumbRY = axis(r.y);
    p.Gamepad.bAnalogButtons[XBOX_BUTTON_A] = get_b(A_A) ? 255 : 0;
    p.Gamepad.bAnalogButtons[XBOX_BUTTON_B] = get_b(A_B) ? 255 : 0;
    p.Gamepad.bAnalogButtons[XBOX_BUTTON_X] = get_b(A_X) ? 255 : 0;
    p.Gamepad.bAnalogButtons[XBOX_BUTTON_Y] = get_b(A_Y) ? 255 : 0;
    p.Gamepad.bAnalogButtons[XBOX_BUTTON_WHITE] = analog(get_f(A_GRIP_L));
    p.Gamepad.bAnalogButtons[XBOX_BUTTON_BLACK] = analog(get_f(A_GRIP_R));
    p.Gamepad.bAnalogButtons[XBOX_BUTTON_LTRIGGER] = analog(get_f(A_TRIG_L));
    p.Gamepad.bAnalogButtons[XBOX_BUTTON_RTRIGGER] = analog(get_f(A_TRIG_R));
    if (get_b(A_MENU))
        p.Gamepad.wButtons |= XBOX_GAMEPAD_START;
    if (get_b(A_CLICK_L))
        p.Gamepad.wButtons |= XBOX_GAMEPAD_BACK;
    /* The camera button: a tap is the game's Y (its camera), a hold
     * switches mode once per press. */
    {
        int64_t t = now_ns();
        if (get_b(A_CLICK_R)) {
            if (!s_cam_down)
                s_cam_down = t;
            else if (!s_cam_held_long && t - s_cam_down >= XR_LONG_NS) {
                s_cam_held_long = 1;
                s_inworld = !s_inworld;
                fprintf(stderr, LOGP "%s mode\n", s_inworld ? "in-world" : "external");
            }
        } else if (s_cam_down) {
            if (!s_cam_held_long)
                s_y_until = t + XR_TAP_NS;
            s_cam_down = 0;
            s_cam_held_long = 0;
        }
        if (t < s_y_until)
            p.Gamepad.bAnalogButtons[XBOX_BUTTON_Y] = 255;
    }
    /* The left stick's direction as the d-pad too: menus move on either. */
    if (l.y > 0.7f)  p.Gamepad.wButtons |= XBOX_GAMEPAD_DPAD_UP;
    if (l.y < -0.7f) p.Gamepad.wButtons |= XBOX_GAMEPAD_DPAD_DOWN;
    if (l.x < -0.7f) p.Gamepad.wButtons |= XBOX_GAMEPAD_DPAD_LEFT;
    if (l.x > 0.7f)  p.Gamepad.wButtons |= XBOX_GAMEPAD_DPAD_RIGHT;
    pthread_mutex_lock(&s_padlock);
    p.dwPacketNumber = s_pad.dwPacketNumber
                       + (memcmp(&p.Gamepad, &s_pad.Gamepad, sizeof p.Gamepad) != 0);
    s_pad = p;
    s_pad_live = 1;
    pthread_mutex_unlock(&s_padlock);
}

/* ---- Vulkan through the runtime (nv2a_vk_xr.h) ------------------------------------ */

static VkResult vk_instance(const VkInstanceCreateInfo *ci, VkInstance *out)
{
    XrGraphicsRequirementsVulkan2KHR req = { XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR };
    XrVulkanInstanceCreateInfoKHR xi = { XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR };
    VkInstanceCreateInfo c = *ci;
    VkApplicationInfo app = *ci->pApplicationInfo;
    VkResult vr = VK_ERROR_INITIALIZATION_FAILED;

    p_req(s_xi, s_sys, &req);   /* required before the instance */
    if (app.apiVersion < VK_API_VERSION_1_1)
        app.apiVersion = VK_API_VERSION_1_1;
    c.pApplicationInfo = &app;
    xi.systemId = s_sys;
    xi.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    xi.vulkanCreateInfo = &c;
    if (!xr_ok(p_inst(s_xi, &xi, out, &vr), "xrCreateVulkanInstanceKHR"))
        return VK_ERROR_INITIALIZATION_FAILED;
    return vr;
}

static VkPhysicalDevice vk_pdev(VkInstance inst)
{
    XrVulkanGraphicsDeviceGetInfoKHR gi = { XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR };
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    gi.systemId = s_sys;
    gi.vulkanInstance = inst;
    xr_ok(p_pdev(s_xi, &gi, &pd), "xrGetVulkanGraphicsDevice2KHR");
    return pd;
}

static VkResult vk_device(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, VkDevice *out)
{
    XrVulkanDeviceCreateInfoKHR di = { XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR };
    VkResult vr = VK_ERROR_INITIALIZATION_FAILED;
    di.systemId = s_sys;
    di.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
    di.vulkanPhysicalDevice = pd;
    di.vulkanCreateInfo = ci;
    if (!xr_ok(p_dev(s_xi, &di, out, &vr), "xrCreateVulkanDeviceKHR"))
        return VK_ERROR_INITIALIZATION_FAILED;
    return vr;
}

static void *frame_thread(void *arg);

static void vk_ready(VkInstance inst, VkPhysicalDevice pd, VkDevice dev, uint32_t qfam,
                     VkQueue queue)
{
    XrGraphicsBindingVulkan2KHR gb = { XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR };
    XrSessionCreateInfo si = { XR_TYPE_SESSION_CREATE_INFO };
    XrReferenceSpaceCreateInfo rs = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    XrSwapchainCreateInfo sc = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
    VkCommandPoolCreateInfo cp = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    VkCommandBufferAllocateInfo ca = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    XrSwapchainImageVulkan2KHR imgs[8];
    int64_t fmts[64];
    uint32_t nf = 0, i;

    s_dev = dev;
    s_queue = queue;
    gb.instance = inst;
    gb.physicalDevice = pd;
    gb.device = dev;
    gb.queueFamilyIndex = qfam;
    gb.queueIndex = 0;
    si.next = &gb;
    si.systemId = s_sys;
    if (!xr_ok(xrCreateSession(s_xi, &si, &s_sess), "xrCreateSession"))
        return;
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    rs.poseInReferenceSpace.orientation.w = 1.0f;
    if (!xr_ok(xrCreateReferenceSpace(s_sess, &rs, &s_space), "xrCreateReferenceSpace"))
        return;
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    xr_ok(xrCreateReferenceSpace(s_sess, &rs, &s_view), "view space");
    actions_init();
    /* The swapchain's format: BGRA8 UNORM, the game's own bytes, when the
     * runtime offers it, else RGBA8 (the blit swizzles). Not sRGB: the
     * frame's values are already display values. */
    xrEnumerateSwapchainFormats(s_sess, 64, &nf, fmts);
    sc.format = 0;
    for (i = 0; i < nf && !sc.format; i++)
        if (fmts[i] == VK_FORMAT_B8G8R8A8_UNORM)
            sc.format = fmts[i];
    for (i = 0; i < nf && !sc.format; i++)
        if (fmts[i] == VK_FORMAT_R8G8B8A8_UNORM)
            sc.format = fmts[i];
    if (!sc.format && nf)
        sc.format = fmts[0];
    sc.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    sc.sampleCount = 1;
    sc.width = s_sc_w;
    sc.height = s_sc_h;
    sc.faceCount = 1;
    sc.arraySize = 1;
    sc.mipCount = 1;
    if (!xr_ok(xrCreateSwapchain(s_sess, &sc, &s_sc), "xrCreateSwapchain"))
        return;
    for (i = 0; i < 8; i++) {
        imgs[i].type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR;
        imgs[i].next = NULL;
    }
    xrEnumerateSwapchainImages(s_sc, 8, &s_sc_n, (XrSwapchainImageBaseHeader *)imgs);
    for (i = 0; i < s_sc_n; i++)
        s_sc_img[i] = imgs[i].image;
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = qfam;
    vkCreateCommandPool(dev, &cp, NULL, &s_pool);
    ca.commandPool = s_pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    vkAllocateCommandBuffers(dev, &ca, &s_cb);
    vkCreateFence(dev, &fi, NULL, &s_fence);
    fprintf(stderr, LOGP "session up: %ux%u swapchain (format %lld), %u images\n", s_sc_w,
            s_sc_h, (long long)sc.format, s_sc_n);
    pthread_create(&s_thread, NULL, frame_thread, NULL);
}

static const struct nv2a_vk_xr s_vk = { vk_instance, vk_pdev, vk_device, vk_ready };

/* ---- the frame loop -------------------------------------------------------------- */

static void barrier(VkImage img, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(s_cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

/* The latest game frame into swapchain image i (black before the first). */
static void draw(uint32_t i)
{
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkImage src;
    uint32_t w, h;
    uint64_t gen;
    int have;

    nv2a_vk_frame_lock();
    have = nv2a_vk_frame(&src, &w, &h, &gen);
    vkResetFences(s_dev, 1, &s_fence);
    vkResetCommandBuffer(s_cb, 0);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s_cb, &bi);
    barrier(s_sc_img[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    if (have) {
        VkImageBlit r;
        memset(&r, 0, sizeof r);
        r.srcSubresource.aspectMask = r.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        r.srcSubresource.layerCount = r.dstSubresource.layerCount = 1;
        r.srcOffsets[1].x = (int32_t)w;
        r.srcOffsets[1].y = (int32_t)h;
        r.srcOffsets[1].z = 1;
        r.dstOffsets[1].x = (int32_t)s_sc_w;
        r.dstOffsets[1].y = (int32_t)s_sc_h;
        r.dstOffsets[1].z = 1;
        vkCmdBlitImage(s_cb, src, VK_IMAGE_LAYOUT_GENERAL, s_sc_img[i],
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r, VK_FILTER_LINEAR);
    } else {
        VkClearColorValue black = { { 0, 0, 0, 1 } };
        VkImageSubresourceRange rr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(s_cb, s_sc_img[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black,
                             1, &rr);
    }
    barrier(s_sc_img[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkEndCommandBuffer(s_cb);
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s_cb;
    nv2a_vk_queue_lock();
    vkQueueSubmit(s_queue, 1, &si, s_fence);
    nv2a_vk_queue_unlock();
    vkWaitForFences(s_dev, 1, &s_fence, VK_TRUE, UINT64_MAX);
    nv2a_vk_frame_unlock();
}

static void handle_events(void)
{
    XrEventDataBuffer ev = { XR_TYPE_EVENT_DATA_BUFFER };
    while (xrPollEvent(s_xi, &ev) == XR_SUCCESS) {
        if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            const XrEventDataSessionStateChanged *e = (const XrEventDataSessionStateChanged *)&ev;
            if (e->state == XR_SESSION_STATE_READY) {
                XrSessionBeginInfo bi = { XR_TYPE_SESSION_BEGIN_INFO };
                bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                if (xr_ok(xrBeginSession(s_sess, &bi), "xrBeginSession"))
                    s_running = 1;
            } else if (e->state == XR_SESSION_STATE_STOPPING) {
                xrEndSession(s_sess);
                s_running = 0;
            } else if (e->state == XR_SESSION_STATE_EXITING
                       || e->state == XR_SESSION_STATE_LOSS_PENDING) {
                s_running = 0;
                s_quit = 1;
            }
        } else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
            s_quit = 1;
        }
        ev.type = XR_TYPE_EVENT_DATA_BUFFER;
        ev.next = NULL;
    }
}

static void *frame_thread(void *arg)
{
    (void)arg;
    while (!s_quit) {
        XrFrameWaitInfo wi = { XR_TYPE_FRAME_WAIT_INFO };
        XrFrameState fs = { XR_TYPE_FRAME_STATE };
        XrFrameBeginInfo bi = { XR_TYPE_FRAME_BEGIN_INFO };
        XrFrameEndInfo ei = { XR_TYPE_FRAME_END_INFO };
        XrCompositionLayerQuad quad = { XR_TYPE_COMPOSITION_LAYER_QUAD };
        const XrCompositionLayerBaseHeader *layers[1];

        handle_events();
        if (!s_running) {
            struct timespec ts = { 0, 50 * 1000000 };
            nanosleep(&ts, NULL);
            continue;
        }
        if (!XR_SUCCEEDED(xrWaitFrame(s_sess, &wi, &fs)))
            continue;
        nv2a_vk_queue_lock();
        xrBeginFrame(s_sess, &bi);
        nv2a_vk_queue_unlock();
        actions_sync();
        ei.displayTime = fs.predictedDisplayTime;
        ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        if (fs.shouldRender) {
            XrSwapchainImageAcquireInfo ai = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
            XrSwapchainImageWaitInfo wi2 = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
            XrSwapchainImageReleaseInfo ri = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
            uint32_t idx = 0;
            nv2a_vk_queue_lock();
            xrAcquireSwapchainImage(s_sc, &ai, &idx);
            nv2a_vk_queue_unlock();
            wi2.timeout = XR_INFINITE_DURATION;
            xrWaitSwapchainImage(s_sc, &wi2);
            draw(idx);
            nv2a_vk_queue_lock();
            xrReleaseSwapchainImage(s_sc, &ri);
            nv2a_vk_queue_unlock();
            /* External: a 3 m by 2.25 m screen (4:3), 2.2 m ahead of where
             * the head was at the last recenter. In-world (interim): locked
             * to the head and wide, about 100 degrees across. */
            quad.layerFlags = 0;
            quad.space = s_inworld && s_view ? s_view : s_space;
            quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            quad.subImage.swapchain = s_sc;
            quad.subImage.imageRect.extent.width = (int32_t)s_sc_w;
            quad.subImage.imageRect.extent.height = (int32_t)s_sc_h;
            quad.pose.orientation.w = 1.0f;
            quad.pose.position.z = -2.2f;
            quad.size.width = s_inworld ? 2.4f : 3.0f;
            quad.size.height = s_inworld ? 1.8f : 2.25f;
            if (s_inworld)
                quad.pose.position.z = -1.0f;
            layers[0] = (const XrCompositionLayerBaseHeader *)&quad;
            ei.layerCount = 1;
            ei.layers = layers;
        }
        nv2a_vk_queue_lock();
        xrEndFrame(s_sess, &ei);
        nv2a_vk_queue_unlock();
    }
    pthread_mutex_lock(&s_padlock);
    s_pad_live = 0;
    pthread_mutex_unlock(&s_padlock);
    return NULL;
}

/* ---- start ------------------------------------------------------------------------ */

int xbox_XrInit(void)
{
    PFN_xrInitializeLoaderKHR init_loader = NULL;
    XrLoaderInitInfoAndroidKHR li = { XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR };
    XrInstanceCreateInfoAndroidKHR ai = { XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR };
    XrInstanceCreateInfo ci = { XR_TYPE_INSTANCE_CREATE_INFO };
    XrSystemGetInfo sg = { XR_TYPE_SYSTEM_GET_INFO };
    static const char *ext[] = { XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME,
                                 XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME };
    const char *off = getenv("RECOMP_XR");
    JNIEnv *env = (JNIEnv *)SDL_GetAndroidJNIEnv();
    JavaVM *vm = NULL;
    jobject act;

    if ((off && off[0] == '0') || !env)
        return 0;
    (*env)->GetJavaVM(env, &vm);
    act = (jobject)SDL_GetAndroidActivity();
    if (!vm || !act)
        return 0;
    act = (*env)->NewGlobalRef(env, act);
    xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                          (PFN_xrVoidFunction *)&init_loader);
    if (!init_loader)
        return 0;
    li.applicationVM = vm;
    li.applicationContext = act;
    if (!xr_ok(init_loader((const XrLoaderInitInfoBaseHeaderKHR *)&li), "loader"))
        return 0;
    ai.applicationVM = vm;
    ai.applicationActivity = act;
    ci.next = &ai;
    snprintf(ci.applicationInfo.applicationName, sizeof ci.applicationInfo.applicationName,
             "xboxrecomp");
    ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    ci.enabledExtensionCount = 2;
    ci.enabledExtensionNames = ext;
    if (xrCreateInstance(&ci, &s_xi) != XR_SUCCESS) {
        fprintf(stderr, LOGP "no OpenXR runtime: a flat window\n");
        return 0;
    }
    sg.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    if (!xr_ok(xrGetSystem(s_xi, &sg, &s_sys), "xrGetSystem")) {
        xrDestroyInstance(s_xi);
        s_xi = XR_NULL_HANDLE;
        return 0;
    }
    xrGetInstanceProcAddr(s_xi, "xrGetVulkanGraphicsRequirements2KHR",
                          (PFN_xrVoidFunction *)&p_req);
    xrGetInstanceProcAddr(s_xi, "xrCreateVulkanInstanceKHR", (PFN_xrVoidFunction *)&p_inst);
    xrGetInstanceProcAddr(s_xi, "xrGetVulkanGraphicsDevice2KHR", (PFN_xrVoidFunction *)&p_pdev);
    xrGetInstanceProcAddr(s_xi, "xrCreateVulkanDeviceKHR", (PFN_xrVoidFunction *)&p_dev);
    if (!p_req || !p_inst || !p_pdev || !p_dev) {
        fprintf(stderr, LOGP "runtime has no XR_KHR_vulkan_enable2\n");
        return 0;
    }
    nv2a_pb_vulkan_set_xr(&s_vk);
    xbox_InputSetExtPad(ext_pad);
    fprintf(stderr, LOGP "OpenXR up: immersive mode, motion controllers as player 1\n");
    return 1;
}

#else
#include "xr_openxr.h"
int xbox_XrInit(void) { return 0; }
#endif

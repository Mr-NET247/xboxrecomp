/**
 * A Vulkan backend for the pushbuffer executor (struct nv2a_pb_backend), the
 * sibling of nv2a_pb_metal.m for every host with a Vulkan driver: Android,
 * Meta Quest, Linux, and macOS through MoltenVK for development.
 *
 * It answers the executor's hooks as the Metal backend does, function for
 * function, and keeps its rules: one colour target per guest surface (grown
 * to its largest clip), depth buffers per zeta surface, the shared texture
 * decode (nv2a_backend_common.h), shaders from the parsed vertex programs and
 * register combiners (d3d8_glsl.h, compiled to SPIR-V at run time), guest
 * memory written back only on demand, and visibility tests through the
 * shared tracker. Where Vulkan differs:
 *
 *  - Render passes are dynamic rendering (Vulkan 1.3, or VK_KHR_dynamic_
 *    rendering). Render targets and depth buffers live in the GENERAL layout,
 *    and every pass boundary and transfer gets a full barrier, standing in for
 *    the hazard tracking Metal does itself.
 *  - Per-draw data (vertices, indices, constants) goes into host-visible rings
 *    and is bound with push descriptors (VK_KHR_push_descriptor).
 *  - Texture and seed uploads are recorded into an upload command buffer that
 *    is submitted just before the frame's, so a new texture is complete
 *    before any draw that samples it, as replaceRegion is in Metal.
 *  - The viewport has a negative height, so clip space is Metal's (y up) and
 *    the same shaders and winding apply.
 *  - Frames reach the window by readback (xbox_FramebufferWindowPresentPixels,
 *    the SDL renderer window), as Metal's metal_present=readback.
 *
 * Opt-in: RECOMP_PB_BACKEND=vulkan (nv2a_pb_vulkan_register_from_env). The
 * Metal backend's debug keys (metal_no_rtt, metal_point, metal_occ, ...)
 * steer this one too. Everything runs on the NV2A ack thread.
 */
#if defined(XBR_HAVE_VULKAN)

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

#include "recomp_env.h"
#include "nv2a_pb_state.h"
#include "nv2a_2d.h"
#include "nv2a_backend_common.h"
#include "nv2a_stats.h"
#include "nv2a_tex_pack.h"
#include "nv2a_tex_dump.h"
#include "d3d8_glsl.h"
#include "nv2a_basic_glsl.h"
#include "nv2a_vk_xr.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <pthread.h>

ptrdiff_t xbox_GetMemoryOffset(void);
extern void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch);
extern void xbox_FramebufferWindowPresent(uint32_t fb_va, uint32_t pitch);
extern int  xbox_FramebufferWindowRunning(void);
extern void xbox_FramebufferWindowPresentPixels(const void *px, uint32_t w, uint32_t h,
                                                uint32_t pitch, uint32_t bpp);

#define LOGP "[vulkan] "
#define VK_MAX_DIM 8192u   /* every Vulkan device's 2D limit is at least this */

#define VKC(call, what) do { VkResult _r = (call); if (_r != VK_SUCCESS) { \
    fprintf(stderr, LOGP "%s failed: %d\n", what, (int)_r); } } while (0)

/* ---- shader ABI: as nv2a_pb_metal.m's ---------------------------------------- */

_Static_assert(sizeof(struct nv2a_vp_consts) == 80, "nv2a_vp_consts is not 5 x 16 bytes");
_Static_assert(sizeof(((struct nv2a_pb_vsh *)0)->c) == NV2A_VS_MAX_CONSTANTS * 16,
               "vertex constants are not 192 float4");
_Static_assert(sizeof(struct nv2a_ps_consts) == 560, "nv2a_ps_consts size changed");

typedef struct {
    struct nv2a_ps_consts b;
    float tex_bias[4];
} VkPsConsts;
_Static_assert(offsetof(VkPsConsts, tex_bias) == 560 && sizeof(VkPsConsts) == 576,
               "VkPsConsts.tex_bias moved");

#define VTX_F4 7
static const int s_vtx_attr[VTX_F4] = {0, 3, 4, 9, 10, 11, 12};

/* ---- device ------------------------------------------------------------------- */

static VkInstance        s_inst;
static VkPhysicalDevice  s_pdev;
static VkDevice          s_dev;
static VkQueue           s_queue;
static uint32_t          s_qfam;
static VkPhysicalDeviceProperties s_props;
static VkPhysicalDeviceMemoryProperties s_mem;
static VkFormat          s_dfmt;          /* depth/stencil format */
static int               s_failed;
static int               s_bc, s_aniso, s_precise_occ;
static VkPipelineLayout  s_playout;
static VkDescriptorSetLayout s_dsl;
static VkPipelineCache   s_pcache;
static VkQueryPool       s_qpool;
static PFN_vkCmdPushDescriptorSetKHR s_push_desc;   /* NULL: sets from slot pools */
static PFN_vkCmdBeginRenderingKHR    s_begin_rendering;
static PFN_vkCmdEndRenderingKHR      s_end_rendering;
static int                           s_legacy;   /* no dynamic rendering: VkRenderPass */

/* ---- a VR runtime's share (nv2a_vk_xr.h) ---------------------------------------- */
static const struct nv2a_vk_xr *s_xr;
/* The presenter's Vulkan window and its surface (Android). */
SDL_Window *xbox_FramebufferWindowVulkan(void);
static SDL_Window  *s_win;
static VkSurfaceKHR s_surf;
/* The stereo pass without a headset (RECOMP_XR_STEREO set): a test of it on
 * a desktop, its pairs in the host dumps (metal_host_dump). */
static int xr_on(void)
{
    static int test = -1;
    if (test < 0)
        test = recomp_env(RENV_XR_STEREO) != NULL;
    return s_xr || test;
}
static pthread_mutex_t s_qlock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t s_flock = PTHREAD_MUTEX_INITIALIZER;
void nv2a_pb_vulkan_set_xr(const struct nv2a_vk_xr *xr) { s_xr = xr; }
void nv2a_vk_queue_lock(void)   { pthread_mutex_lock(&s_qlock); }
void nv2a_vk_queue_unlock(void) { pthread_mutex_unlock(&s_qlock); }
void nv2a_vk_frame_lock(void)   { pthread_mutex_lock(&s_flock); }
void nv2a_vk_frame_unlock(void) { pthread_mutex_unlock(&s_flock); }
static shaderc_compiler_t            s_cc;
static shaderc_compile_options_t     s_copt;
static VkShaderModule    s_vs_pass, s_fs_basic;

#define VIS_SLOTS 8192

static uint64_t s_presents, s_clears, s_draws, s_draws_skipped, s_draws_prog,
                s_draws_culled, s_draws_tex, s_tex_uploads, s_tex_rehash,
                s_tex_changed, s_flushes, s_writebacks, s_draws_stretch,
                s_zclears, s_draws_z, s_rtt_binds, s_self_copies, s_stale_drops,
                s_retired, s_evict_lost, s_blits_gpu, s_blits_guest, s_grows,
                s_partial_clears, s_decode_syncs, s_pitch_alias_binds,
                s_compile_errors, s_vs_compiled, s_fs_compiled, s_report_commits;
static struct nv2a_pair_set s_pitch_alias_seen;

/* ---- memory ------------------------------------------------------------------- */

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
    uint32_t i;
    for (i = 0; i < s_mem.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (s_mem.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    uint8_t       *map;
    VkDeviceSize   size, pos;
} Ring;

static int ring_make(Ring *r, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    VkMemoryRequirements mr;
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(s_dev, &bi, NULL, &r->buf) != VK_SUCCESS)
        return 0;
    vkGetBufferMemoryRequirements(s_dev, r->buf, &mr);
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                     | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX
            || vkAllocateMemory(s_dev, &ai, NULL, &r->mem) != VK_SUCCESS)
        return 0;
    vkBindBufferMemory(s_dev, r->buf, r->mem, 0);
    if (vkMapMemory(s_dev, r->mem, 0, VK_WHOLE_SIZE, 0, (void **)&r->map) != VK_SUCCESS)
        return 0;
    r->size = size;
    r->pos = 0;
    return 1;
}

static void ring_free(Ring *r)
{
    if (r->buf)
        vkDestroyBuffer(s_dev, r->buf, NULL);
    if (r->mem)
        vkFreeMemory(s_dev, r->mem, NULL);
    memset(r, 0, sizeof *r);
}

/* An image and its views. view is 2D (or cube, or 3D); layout is what the
 * last recorded command left it in. */
typedef struct {
    VkImage        img;
    VkDeviceMemory mem;
    VkImageView    view;
    VkFormat       fmt;
    uint32_t       w, h, levels, layers;
    VkImageViewType vtype;
    VkImageAspectFlags aspect;
} Img;

/* ---- command buffers ----------------------------------------------------------
 *
 * A slot is the frame's command buffer and its upload buffer, a fence and the
 * serial it was recorded under. flush(0) submits the open slot; flush(1) also
 * waits for every slot in flight, after which the rings are free. A resource
 * dropped while a slot may still read it waits in the graveyard for its
 * serial to complete. */
#define SLOTS 8
#define DSETS 8192   /* sets per slot without push descriptors */
typedef struct {
    VkCommandBuffer cmd, up;
    VkFence         fence;
    VkDescriptorPool dpool;   /* without push descriptors: this slot's sets */
    uint64_t        serial;
    int             busy;     /* submitted, fence not yet seen */
    int             up_used;
} Slot;
static VkCommandPool s_pool;
/* The next submit waits on s_sub_wait (a swapchain acquire) at the
 * transfer and colour stages and signals s_sub_signal (its present). */
static VkSemaphore s_sub_wait, s_sub_signal;
static Slot     s_slot[SLOTS];
static Slot    *s_cur;          /* recording, or NULL */
static uint64_t s_serial = 1;   /* the next slot's serial */
static uint64_t s_done;         /* every serial <= this has completed */
static uint64_t s_last;         /* the last submitted serial */

typedef struct { Img img; VkFramebuffer fb; uint64_t serial; } Grave;
static Grave   *s_grave;
static size_t   s_ngrave, s_grave_cap;

static void img_destroy_now(Img *im)
{
    if (im->view) vkDestroyImageView(s_dev, im->view, NULL);
    if (im->img)  vkDestroyImage(s_dev, im->img, NULL);
    if (im->mem)  vkFreeMemory(s_dev, im->mem, NULL);
    memset(im, 0, sizeof *im);
}

static void fb_forget(VkImageView v);

/* Free im once nothing in flight can read it. */
static void img_release(Img *im)
{
    if (!im->img)
        return;
    fb_forget(im->view);
    if (s_ngrave == s_grave_cap) {
        size_t n = s_grave_cap ? s_grave_cap * 2 : 64;
        Grave *g = (Grave *)realloc(s_grave, n * sizeof *g);
        if (!g) {
            vkDeviceWaitIdle(s_dev);
            img_destroy_now(im);
            return;
        }
        s_grave = g;
        s_grave_cap = n;
    }
    memset(&s_grave[s_ngrave], 0, sizeof s_grave[s_ngrave]);
    s_grave[s_ngrave].img = *im;
    s_grave[s_ngrave].serial = s_cur ? s_cur->serial : s_last;
    s_ngrave++;
    memset(im, 0, sizeof *im);
}

/* A fence wait that says so when it runs long: 2 s at a time, each
 * overrun logged with where and which slot, and a lost device reported
 * (a GPU fault ends every wait with VK_ERROR_DEVICE_LOST). */
static void fence_wait(VkFence f, const char *where, uint64_t serial)
{
    unsigned waited = 0;
    for (;;) {
        VkResult r = vkWaitForFences(s_dev, 1, &f, VK_TRUE, 2000000000ull);
        if (r == VK_SUCCESS)
            return;
        if (r != VK_TIMEOUT) {
            fprintf(stderr, LOGP "fence wait at %s (serial %llu): error %d%s\n", where,
                    (unsigned long long)serial, (int)r,
                    r == VK_ERROR_DEVICE_LOST ? " (device lost)" : "");
            return;
        }
        waited += 2;
        fprintf(stderr, LOGP "fence wait at %s (serial %llu): %u s and counting\n", where,
                (unsigned long long)serial, waited);
        fflush(stderr);
    }
}

static void slots_poll(void)
{
    size_t i, k;
    for (i = 0; i < SLOTS; i++) {
        Slot *s = &s_slot[i];
        if (s->busy && vkGetFenceStatus(s_dev, s->fence) == VK_SUCCESS) {
            s->busy = 0;
            if (s->serial > s_done) {
                /* done in order: everything before it is done too */
                uint64_t lo = UINT64_MAX;
                size_t j;
                for (j = 0; j < SLOTS; j++)
                    if (s_slot[j].busy && s_slot[j].serial < lo)
                        lo = s_slot[j].serial;
                s_done = lo == UINT64_MAX ? s_last : lo - 1;
            }
        }
    }
    for (i = k = 0; i < s_ngrave; i++) {
        if (s_grave[i].serial <= s_done) {
            img_destroy_now(&s_grave[i].img);
            if (s_grave[i].fb)
                vkDestroyFramebuffer(s_dev, s_grave[i].fb, NULL);
        }
        else
            s_grave[k++] = s_grave[i];
    }
    s_ngrave = k;
}

static int serial_done(uint64_t serial)
{
    if (serial <= s_done)
        return 1;
    slots_poll();
    return serial <= s_done;
}

static void wait_all(void)
{
    size_t i;
    for (i = 0; i < SLOTS; i++)
        if (s_slot[i].busy)
            fence_wait(s_slot[i].fence, "wait_all", s_slot[i].serial);
    slots_poll();
    s_done = s_last;
    slots_poll();
}

static Ring s_vb, s_ib, s_ub, s_stage_ring;   /* vertices, indices, constants, uploads */
static VkDeviceSize s_ssbo_align, s_ubo_align;

/* The rendering pass in the open slot. */
static int s_in_pass;
struct RenderTarget;
struct DepthTarget;
static struct RenderTarget *s_enc_rt;
static struct DepthTarget  *s_enc_dt;
static uint64_t s_enc_serial;
static void occ_pass_begin(void);
static void occ_pass_end(void);
static void occ_after_wait(void);
static struct OccQuery *s_occ_q;
static int s_occ_slot = -1;

/* The c[] memo: the ring offset that holds c[] at generation gen, valid
 * until the ring is reset. */
static struct { VkDeviceSize off; uint32_t gen; int ok; } s_cmemo;

static void ensure_cmd(void)
{
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    Slot *s = NULL;
    size_t i;
    if (s_cur)
        return;
    for (;;) {
        for (i = 0; i < SLOTS && !s; i++)
            if (!s_slot[i].busy)
                s = &s_slot[i];
        if (s)
            break;
        /* all in flight: wait for the oldest */
        {
            Slot *old = &s_slot[0];
            for (i = 1; i < SLOTS; i++)
                if (s_slot[i].serial < old->serial)
                    old = &s_slot[i];
            fence_wait(old->fence, "ensure_cmd", old->serial);
            slots_poll();
        }
    }
    vkResetFences(s_dev, 1, &s->fence);
    vkResetCommandBuffer(s->cmd, 0);
    vkResetCommandBuffer(s->up, 0);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s->cmd, &bi);
    s->serial = s_serial++;
    s->up_used = 0;
    if (s->dpool)
        vkResetDescriptorPool(s_dev, s->dpool, 0);
    s_cur = s;
}

/* The upload buffer of the open slot, begun on first use. */
static VkCommandBuffer up_cmd(void)
{
    ensure_cmd();
    if (!s_cur->up_used) {
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(s_cur->up, &bi);
        s_cur->up_used = 1;
    }
    return s_cur->up;
}

static void barrier_all(VkCommandBuffer cb)
{
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void img_layout(VkCommandBuffer cb, Img *im, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im->img;
    b.subresourceRange.aspectMask = im->aspect;
    b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    b.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

static void end_encoder(void)
{
    if (s_in_pass) {
        occ_pass_end();
        if (s_legacy)
            vkCmdEndRenderPass(s_cur->cmd);
        else
            s_end_rendering(s_cur->cmd);
        s_in_pass = 0;
    }
    s_enc_rt = NULL;
    s_enc_dt = NULL;
}

/* End the open pass and submit the slot; with wait, block until every slot
 * is done, after which the rings are free. */
static void flush(int wait)
{
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkCommandBuffer cbs[2];
    uint32_t n = 0;

    end_encoder();
    if (s_cur) {
        if (s_cur->up_used) {
            barrier_all(s_cur->up);
            vkEndCommandBuffer(s_cur->up);
            cbs[n++] = s_cur->up;
        }
        vkEndCommandBuffer(s_cur->cmd);
        cbs[n++] = s_cur->cmd;
        si.commandBufferCount = n;
        si.pCommandBuffers = cbs;
        {
            static const VkPipelineStageFlags ws = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            if (s_sub_wait) {
                si.waitSemaphoreCount = 1;
                si.pWaitSemaphores = &s_sub_wait;
                si.pWaitDstStageMask = &ws;
            }
            if (s_sub_signal) {
                si.signalSemaphoreCount = 1;
                si.pSignalSemaphores = &s_sub_signal;
            }
        }
        nv2a_vk_queue_lock();
        {
            VkResult sr = vkQueueSubmit(s_queue, 1, &si, s_cur->fence);
            if (sr != VK_SUCCESS)
                fprintf(stderr, LOGP "vkQueueSubmit failed: %d%s\n", (int)sr,
                        sr == VK_ERROR_DEVICE_LOST ? " (device lost)" : "");
        }
        nv2a_vk_queue_unlock();
        s_cur->busy = 1;
        s_last = s_cur->serial;
        s_cur = NULL;
        s_flushes++;
    } else if (wait) {
        s_flushes++;
    }
    if (wait) {
        wait_all();
        s_vb.pos = s_ib.pos = s_ub.pos = s_stage_ring.pos = 0;
        s_cmemo.ok = 0;
        occ_after_wait();
    }
}

/* `bytes` from a ring, aligned; on overflow the work so far is flushed (and
 * waited for) and the ring starts over. NULL if it can never fit. */
static void *ring_alloc(Ring *r, VkDeviceSize bytes, VkDeviceSize align, VkDeviceSize *off)
{
    VkDeviceSize at;
    if (align < 16)
        align = 16;
    at = (r->pos + align - 1) & ~(align - 1);
    if (bytes > r->size)
        return NULL;
    if (at + bytes > r->size) {
        flush(1);   /* resets every ring */
        at = 0;
    }
    r->pos = at + bytes;
    *off = at;
    return r->map + at;
}

/* flush(1) that keeps the rings' positions: inside a draw, whose vertices
 * may already be placed (a texture upload or a write-back mid-draw), the
 * waited-for slots are done and the data is still where it was. */
static void flush_keep(void)
{
    VkDeviceSize vb = s_vb.pos, ib = s_ib.pos, ub = s_ub.pos;
    flush(1);
    s_vb.pos = vb;
    s_ib.pos = ib;
    s_ub.pos = ub;
}

/* ---- shaders ------------------------------------------------------------------ */

static VkShaderModule compile_glsl(const char *src, int frag, const char *what)
{
    shaderc_compilation_result_t r;
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkShaderModule m = VK_NULL_HANDLE;

    r = shaderc_compile_into_spv(s_cc, src, strlen(src),
                                 frag ? shaderc_fragment_shader : shaderc_vertex_shader,
                                 what, "main", s_copt);
    if (shaderc_result_get_compilation_status(r) != shaderc_compilation_status_success) {
        s_compile_errors++;
        if (s_compile_errors <= 4)
            fprintf(stderr, LOGP "\x1b[1;31m%s: GLSL compile failed: %s\x1b[0m\n%s\n", what,
                    shaderc_result_get_error_message(r), src);
        shaderc_result_release(r);
        return VK_NULL_HANDLE;
    }
    ci.codeSize = shaderc_result_get_length(r);
    ci.pCode = (const uint32_t *)shaderc_result_get_bytes(r);
    if (vkCreateShaderModule(s_dev, &ci, NULL, &m) != VK_SUCCESS)
        m = VK_NULL_HANDLE;
    shaderc_result_release(r);
    return m;
}

/* ---- init ----------------------------------------------------------------------- */

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name)
{
    uint32_t i;
    for (i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name))
            return 1;
    return 0;
}

static int init_device(void)
{
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    const char *iext[16];
    uint32_t niext = 0, n = 0, i;
    VkExtensionProperties *ie = NULL, *de = NULL;
    VkPhysicalDevice pd[16];
    VkPhysicalDeviceFeatures have, want;
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dr = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    const char *dext[10];
    uint32_t ndext = 0, nde = 0, nq = 0;
    float prio = 1.0f;
    VkQueueFamilyProperties qf[16];
    int core13;

    vkEnumerateInstanceExtensionProperties(NULL, &n, NULL);
    ie = (VkExtensionProperties *)calloc(n ? n : 1, sizeof *ie);
    vkEnumerateInstanceExtensionProperties(NULL, &n, ie);
    if (has_ext(ie, n, "VK_KHR_portability_enumeration")) {
        iext[niext++] = "VK_KHR_portability_enumeration";
        ici.flags |= 0x00000001; /* VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR */
    }
    if (has_ext(ie, n, "VK_KHR_get_physical_device_properties2"))
        iext[niext++] = "VK_KHR_get_physical_device_properties2";
    /* The presenter's Vulkan window (Android): SDL's surface extensions. */
    s_win = xbox_FramebufferWindowVulkan();
    if (s_win) {
        Uint32 ne = 0, k;
        const char *const *se = SDL_Vulkan_GetInstanceExtensions(&ne);
        for (k = 0; se && k < ne && niext < 16; k++)
            iext[niext++] = se[k];
    }
    free(ie);
    app.pApplicationName = "xboxrecomp";
    /* 1.3: a driver hides its 1.3 core entry points (vkCmdBeginRendering)
     * from an instance that asked for less; one that has only 1.1 still
     * takes the request. */
    app.apiVersion = VK_API_VERSION_1_3;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = niext;
    ici.ppEnabledExtensionNames = iext;
    if ((s_xr ? s_xr->create_instance(&ici, &s_inst) : vkCreateInstance(&ici, NULL, &s_inst))
            != VK_SUCCESS) {
        fprintf(stderr, LOGP "no Vulkan instance\n");
        return 0;
    }
    if (s_win && !SDL_Vulkan_CreateSurface(s_win, s_inst, NULL, &s_surf)) {
        fprintf(stderr, LOGP "no window surface (%s): present by readback\n", SDL_GetError());
        s_surf = VK_NULL_HANDLE;
    }
    n = 16;
    if (vkEnumeratePhysicalDevices(s_inst, &n, pd) < 0 || !n) {
        fprintf(stderr, LOGP "no Vulkan device\n");
        return 0;
    }
    /* A discrete GPU first, else the first device; the VR runtime's own
     * choice when there is one. */
    s_pdev = s_xr ? s_xr->physical_device(s_inst) : pd[0];
    for (i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd[i], &p);
        if (!s_xr && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            s_pdev = pd[i];
            break;
        }
    }
    vkGetPhysicalDeviceProperties(s_pdev, &s_props);
    vkGetPhysicalDeviceMemoryProperties(s_pdev, &s_mem);
    core13 = s_props.apiVersion >= VK_API_VERSION_1_3;
    nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(s_pdev, &nq, qf);
    for (i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            break;
    if (i == nq) {
        fprintf(stderr, LOGP "no graphics queue\n");
        return 0;
    }
    s_qfam = i;

    vkEnumerateDeviceExtensionProperties(s_pdev, NULL, &nde, NULL);
    de = (VkExtensionProperties *)calloc(nde ? nde : 1, sizeof *de);
    vkEnumerateDeviceExtensionProperties(s_pdev, NULL, &nde, de);
    /* Push descriptors when the device has them; else each draw's set
     * comes from its slot's pool (the Android emulator, some phones). */
    if (has_ext(de, nde, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
        dext[ndext++] = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
    if (!core13 && !has_ext(de, nde, "VK_KHR_dynamic_rendering")) {
        /* Render pass objects instead (Vulkan 1.1 drivers, the emulator). */
        s_legacy = 1;
    } else if (!core13) {
        dext[ndext++] = "VK_KHR_dynamic_rendering";
        if (has_ext(de, nde, "VK_KHR_depth_stencil_resolve"))
            dext[ndext++] = "VK_KHR_depth_stencil_resolve";
        if (has_ext(de, nde, "VK_KHR_create_renderpass2"))
            dext[ndext++] = "VK_KHR_create_renderpass2";
    }
    if (s_surf) {
        VkBool32 ok = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(s_pdev, s_qfam, s_surf, &ok);
        if (ok && has_ext(de, nde, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            dext[ndext++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        } else {
            vkDestroySurfaceKHR(s_inst, s_surf, NULL);
            s_surf = VK_NULL_HANDLE;
        }
    }
    if (has_ext(de, nde, "VK_KHR_portability_subset"))
        dext[ndext++] = "VK_KHR_portability_subset";
    free(de);

    vkGetPhysicalDeviceFeatures(s_pdev, &have);
    memset(&want, 0, sizeof want);
    want.samplerAnisotropy = have.samplerAnisotropy;
    want.textureCompressionBC = have.textureCompressionBC;
    want.occlusionQueryPrecise = have.occlusionQueryPrecise;
    want.largePoints = have.largePoints;
    want.fragmentStoresAndAtomics = have.fragmentStoresAndAtomics;
    s_aniso = have.samplerAnisotropy;
    s_bc = have.textureCompressionBC;
    s_precise_occ = have.occlusionQueryPrecise;
    dr.dynamicRendering = VK_TRUE;
    qci.queueFamilyIndex = s_qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    dci.pNext = s_legacy ? NULL : &dr;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = ndext;
    dci.ppEnabledExtensionNames = dext;
    dci.pEnabledFeatures = &want;
    if ((s_xr ? s_xr->create_device(s_pdev, &dci, &s_dev)
              : vkCreateDevice(s_pdev, &dci, NULL, &s_dev)) != VK_SUCCESS) {
        fprintf(stderr, LOGP "%s: vkCreateDevice failed\n", s_props.deviceName);
        return 0;
    }
    vkGetDeviceQueue(s_dev, s_qfam, 0, &s_queue);
    s_push_desc = NULL;
    for (i = 0; i < ndext; i++)
        if (!strcmp(dext[i], VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
            s_push_desc = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(
                s_dev, "vkCmdPushDescriptorSetKHR");
    if (!s_legacy) {
        s_begin_rendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(
            s_dev, core13 ? "vkCmdBeginRendering" : "vkCmdBeginRenderingKHR");
        s_end_rendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(
            s_dev, core13 ? "vkCmdEndRendering" : "vkCmdEndRenderingKHR");
    }
    if (!s_legacy && (!s_begin_rendering || !s_end_rendering)) {
        /* Still hidden: render pass objects instead, which every device has. */
        fprintf(stderr, LOGP "no vkCmdBeginRendering from the driver: render pass objects\n");
        s_legacy = 1;
    }
    /* Depth32Float_Stencil8 as Metal and D3D11 keep it; else D24S8. */
    {
        VkFormat c[2] = { VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT };
        s_dfmt = VK_FORMAT_UNDEFINED;
        for (i = 0; i < 2 && !s_dfmt; i++) {
            VkFormatProperties fp;
            vkGetPhysicalDeviceFormatProperties(s_pdev, c[i], &fp);
            if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                s_dfmt = c[i];
        }
        if (!s_dfmt) {
            fprintf(stderr, LOGP "no depth/stencil format\n");
            return 0;
        }
    }
    s_ssbo_align = s_props.limits.minStorageBufferOffsetAlignment;
    s_ubo_align = s_props.limits.minUniformBufferOffsetAlignment;
    return 1;
}

static int init_objects(void)
{
    VkDescriptorSetLayoutBinding b[10];
    VkDescriptorSetLayoutCreateInfo dl = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPipelineLayoutCreateInfo pl = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkPipelineCacheCreateInfo pc = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    VkCommandPoolCreateInfo cp = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    VkCommandBufferAllocateInfo ca = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkQueryPoolCreateInfo qi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    static const VkDescriptorType types[10] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER };
    static const uint32_t nums[10] = { 0, 1, 2, 3, 4, 5, 8, 9, 10, 11 };
    int i;

    memset(b, 0, sizeof b);
    for (i = 0; i < 10; i++) {
        b[i].binding = nums[i];
        b[i].descriptorType = types[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    dl.flags = s_push_desc ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0;
    dl.bindingCount = 10;
    dl.pBindings = b;
    if (vkCreateDescriptorSetLayout(s_dev, &dl, NULL, &s_dsl) != VK_SUCCESS)
        return 0;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &s_dsl;
    if (vkCreatePipelineLayout(s_dev, &pl, NULL, &s_playout) != VK_SUCCESS)
        return 0;
    vkCreatePipelineCache(s_dev, &pc, NULL, &s_pcache);
    cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp.queueFamilyIndex = s_qfam;
    if (vkCreateCommandPool(s_dev, &cp, NULL, &s_pool) != VK_SUCCESS)
        return 0;
    ca.commandPool = s_pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (i = 0; i < SLOTS; i++) {
        if (vkAllocateCommandBuffers(s_dev, &ca, &s_slot[i].cmd) != VK_SUCCESS
                || vkAllocateCommandBuffers(s_dev, &ca, &s_slot[i].up) != VK_SUCCESS
                || vkCreateFence(s_dev, &fi, NULL, &s_slot[i].fence) != VK_SUCCESS)
            return 0;
        if (!s_push_desc) {
            VkDescriptorPoolSize ps[3] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2u * DSETS },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4u * DSETS },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4u * DSETS } };
            VkDescriptorPoolCreateInfo dp = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            dp.maxSets = DSETS;
            dp.poolSizeCount = 3;
            dp.pPoolSizes = ps;
            if (vkCreateDescriptorPool(s_dev, &dp, NULL, &s_slot[i].dpool) != VK_SUCCESS)
                return 0;
        }
    }
    qi.queryType = VK_QUERY_TYPE_OCCLUSION;
    qi.queryCount = VIS_SLOTS;
    if (vkCreateQueryPool(s_dev, &qi, NULL, &s_qpool) != VK_SUCCESS)
        return 0;
    /* As Metal's rings: vertices bounded by the 16-bit index range, indices
     * by the walker's cap (three 32-bit entries per index at most). */
    if (!ring_make(&s_vb, 16u << 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            || !ring_make(&s_ib, (VkDeviceSize)NV_MAX_INDICES * 3u * 4u,
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
            || !ring_make(&s_ub, 8u << 20, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)
            || !ring_make(&s_stage_ring, 64u << 20, VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        return 0;
    s_cc = shaderc_compiler_initialize();
    s_copt = shaderc_compile_options_initialize();
    if (!s_cc || !s_copt)
        return 0;
    shaderc_compile_options_set_target_env(s_copt, shaderc_target_env_vulkan,
                                           shaderc_env_version_vulkan_1_1);
    /* Exact arithmetic, as Metal's fast math off: no relaxed precision. */
    shaderc_compile_options_set_optimization_level(s_copt, shaderc_optimization_level_zero);
    s_vs_pass = compile_glsl(NV2A_GLSL_VS_PASS, 0, "vs_pass");
    s_fs_basic = compile_glsl(NV2A_GLSL_FS_BASIC, 1, "fs_basic");
    return s_vs_pass && s_fs_basic;
}

static int init(void)
{
    static int tried;
    if (tried)
        return !s_failed;
    tried = 1;
    s_failed = 1;
    if (!init_device() || !init_objects()) {
        fprintf(stderr, LOGP "initialisation failed: staying on the CPU rasteriser\n");
        return 0;
    }
    fprintf(stderr, LOGP "window surface: %s\n", s_surf ? "yes, swapchain present"
                                                      : "no, present by readback");
    fprintf(stderr, LOGP "device up: %s (Vulkan %u.%u.%u), depth %s, BC %d, aniso %d,"
            " precise occlusion %d, push descriptors %d, %s; present by readback, write-back"
            " on demand\n",
            s_props.deviceName, VK_VERSION_MAJOR(s_props.apiVersion),
            VK_VERSION_MINOR(s_props.apiVersion), VK_VERSION_PATCH(s_props.apiVersion),
            s_dfmt == VK_FORMAT_D32_SFLOAT_S8_UINT ? "D32S8" : "D24S8", s_bc, s_aniso,
            s_precise_occ, s_push_desc != NULL,
            s_legacy ? "render pass objects" : "dynamic rendering");
    s_failed = 0;
    if (s_xr)
        s_xr->device_ready(s_inst, s_pdev, s_dev, s_qfam, s_queue);
    return 1;
}

/* ---- images ---------------------------------------------------------------------- */

static int img_make(Img *im, VkImageType type, VkImageViewType vtype, VkFormat fmt,
                    uint32_t w, uint32_t h, uint32_t d, uint32_t levels, uint32_t layers,
                    VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    VkMemoryRequirements mr;
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };

    memset(im, 0, sizeof *im);
    ci.flags = vtype == VK_IMAGE_VIEW_TYPE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    ci.imageType = type;
    ci.format = fmt;
    ci.extent.width = w;
    ci.extent.height = h;
    ci.extent.depth = d;
    ci.mipLevels = levels;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(s_dev, &ci, NULL, &im->img) != VK_SUCCESS)
        return 0;
    vkGetImageMemoryRequirements(s_dev, im->img, &mr);
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX)
        ai.memoryTypeIndex = mem_type(mr.memoryTypeBits, 0);
    if (vkAllocateMemory(s_dev, &ai, NULL, &im->mem) != VK_SUCCESS) {
        vkDestroyImage(s_dev, im->img, NULL);
        im->img = VK_NULL_HANDLE;
        return 0;
    }
    vkBindImageMemory(s_dev, im->img, im->mem, 0);
    vi.image = im->img;
    vi.viewType = vtype;
    vi.format = fmt;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.levelCount = levels;
    vi.subresourceRange.layerCount = layers;
    if (vkCreateImageView(s_dev, &vi, NULL, &im->view) != VK_SUCCESS) {
        img_destroy_now(im);
        return 0;
    }
    im->fmt = fmt;
    im->w = w;
    im->h = h;
    im->levels = levels;
    im->layers = layers;
    im->vtype = vtype;
    im->aspect = aspect;
    return 1;
}

/* A BGRA8 colour target of hw x hh, in GENERAL (its layout for life). */
static int rt_image_new(Img *im, uint32_t hw, uint32_t hh)
{
    if (!img_make(im, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_B8G8R8A8_UNORM,
                  hw, hh, 1, 1, 1,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                  | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return 0;
    img_layout(up_cmd(), im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    return 1;
}

/* Copy `bytes` rows of src into the image at (0,0), w x h of level `level`,
 * layer `layer`, through the staging ring, in the upload buffer. row_px is
 * the source's row length in texels (or blocks x 4 for BC). */
static int img_upload(Img *im, uint32_t level, uint32_t layer, uint32_t w, uint32_t h,
                      const void *src, VkDeviceSize bytes, uint32_t row_px,
                      VkImageLayout layout)
{
    VkDeviceSize off;
    VkBufferImageCopy r;
    void *p;
    if (bytes > s_stage_ring.size)
        return 0;
    if (s_stage_ring.pos + bytes + 16 > s_stage_ring.size)
        flush_keep();
    p = ring_alloc(&s_stage_ring, bytes, 16, &off);
    if (!p)
        return 0;
    memcpy(p, src, (size_t)bytes);
    memset(&r, 0, sizeof r);
    r.bufferOffset = off;
    r.bufferRowLength = row_px;
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.imageSubresource.mipLevel = level;
    r.imageSubresource.baseArrayLayer = layer;
    r.imageSubresource.layerCount = 1;
    r.imageExtent.width = w;
    r.imageExtent.height = h;
    r.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(up_cmd(), s_stage_ring.buf, im->img, layout, 1, &r);
    return 1;
}

/* A host-visible buffer images are read back into (write-back, present). */
static Ring s_rb;

static const uint8_t *img_readback(Img *im, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    VkDeviceSize need = (VkDeviceSize)w * h * 4u;
    VkBufferImageCopy r;
    if (!s_rb.buf || s_rb.size < need) {
        if (s_rb.buf) {
            flush(1);
            ring_free(&s_rb);
        }
        if (!ring_make(&s_rb, need, VK_BUFFER_USAGE_TRANSFER_DST_BIT))
            return NULL;
    }
    end_encoder();
    ensure_cmd();
    barrier_all(s_cur->cmd);
    memset(&r, 0, sizeof r);
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.imageSubresource.layerCount = 1;
    r.imageOffset.x = (int32_t)x;
    r.imageOffset.y = (int32_t)y;
    r.imageExtent.width = w;
    r.imageExtent.height = h;
    r.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(s_cur->cmd, im->img, VK_IMAGE_LAYOUT_GENERAL, s_rb.buf, 1, &r);
    barrier_all(s_cur->cmd);
    /* the readback must not reset the rings: a draw may have placed its
     * vertices already (rt_sync runs inside on_draw) */
    flush_keep();
    return s_rb.map;
}

/* ---- render pass objects (no dynamic rendering) --------------------------------
 *
 * One pass per (depth attachment, colour load, depth load, stencil load),
 * every attachment in GENERAL before and after, as dynamic rendering keeps
 * them; pipelines and framebuffers are made against the all-load pass of
 * their depth case, which every pass of that case is compatible with. */
static VkRenderPass s_rp[2][8];

static VkRenderPass rp_get(int depth, int cclear, int zclear, int sclear)
{
    unsigned k = (unsigned)(cclear | zclear << 1 | sclear << 2);
    VkAttachmentDescription a[2];
    VkAttachmentReference cr = { 0, VK_IMAGE_LAYOUT_GENERAL }, dr = { 1, VK_IMAGE_LAYOUT_GENERAL };
    VkSubpassDescription sp;
    VkRenderPassCreateInfo ci = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    if (!depth)
        k &= 1u;
    if (s_rp[depth][k])
        return s_rp[depth][k];
    memset(a, 0, sizeof a);
    a[0].format = VK_FORMAT_B8G8R8A8_UNORM;
    a[0].samples = VK_SAMPLE_COUNT_1_BIT;
    a[0].loadOp = cclear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    a[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    a[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    a[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    a[0].initialLayout = a[0].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    a[1].format = s_dfmt;
    a[1].samples = VK_SAMPLE_COUNT_1_BIT;
    a[1].loadOp = zclear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    a[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    a[1].stencilLoadOp = sclear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    a[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    a[1].initialLayout = a[1].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    memset(&sp, 0, sizeof sp);
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments = &cr;
    sp.pDepthStencilAttachment = depth ? &dr : NULL;
    ci.attachmentCount = depth ? 2 : 1;
    ci.pAttachments = a;
    ci.subpassCount = 1;
    ci.pSubpasses = &sp;
    if (vkCreateRenderPass(s_dev, &ci, NULL, &s_rp[depth][k]) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return s_rp[depth][k];
}

#define FB_MAX 64
static struct { VkImageView c, d; uint32_t w, h; VkFramebuffer fb; } s_fb[FB_MAX];
static int s_fb_n;

static VkFramebuffer fb_get(VkImageView c, VkImageView d, uint32_t w, uint32_t h)
{
    VkFramebufferCreateInfo ci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    VkImageView v[2] = { c, d };
    VkFramebuffer fb;
    int i;
    for (i = 0; i < s_fb_n; i++)
        if (s_fb[i].c == c && s_fb[i].d == d && s_fb[i].w == w && s_fb[i].h == h)
            return s_fb[i].fb;
    ci.renderPass = rp_get(d != VK_NULL_HANDLE, 0, 0, 0);
    ci.attachmentCount = d ? 2 : 1;
    ci.pAttachments = v;
    ci.width = w;
    ci.height = h;
    ci.layers = 1;
    if (vkCreateFramebuffer(s_dev, &ci, NULL, &fb) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    if (s_fb_n == FB_MAX) {
        /* full: the oldest goes, once nothing in flight uses it */
        fb_forget(s_fb[0].c);
        if (s_fb_n == FB_MAX)
            return fb;   /* (cannot happen: fb_forget removed slot 0) */
    }
    s_fb[s_fb_n].c = c;
    s_fb[s_fb_n].d = d;
    s_fb[s_fb_n].w = w;
    s_fb[s_fb_n].h = h;
    s_fb[s_fb_n].fb = fb;
    s_fb_n++;
    return fb;
}

/* Every framebuffer over view v goes to the graveyard. */
static void fb_forget(VkImageView v)
{
    int i, k;
    for (i = k = 0; i < s_fb_n; i++) {
        if (s_fb[i].c == v || s_fb[i].d == v) {
            if (s_ngrave == s_grave_cap) {
                size_t n = s_grave_cap ? s_grave_cap * 2 : 64;
                Grave *g = (Grave *)realloc(s_grave, n * sizeof *g);
                if (!g) {
                    vkDeviceWaitIdle(s_dev);
                    vkDestroyFramebuffer(s_dev, s_fb[i].fb, NULL);
                    continue;
                }
                s_grave = g;
                s_grave_cap = n;
            }
            memset(&s_grave[s_ngrave], 0, sizeof s_grave[s_ngrave]);
            s_grave[s_ngrave].fb = s_fb[i].fb;
            s_grave[s_ngrave].serial = s_cur ? s_cur->serial : s_last;
            s_ngrave++;
        } else {
            s_fb[k++] = s_fb[i];
        }
    }
    s_fb_n = k;
}

/* ---- render targets: one per guest surface (nv2a_pb_metal.m's rules) ------------ */

#define DT_MAX 4
typedef struct DepthTarget {
    uint32_t zeta, w, h, hw, hh, pad;
    unsigned n;
    Img      tex;
    VkImageView dview;   /* depth aspect only, for the stereo pass (s_xr) */
    uint64_t last_use;
    int      fresh;
} DepthTarget;

#define RT_MAX 16
typedef struct RenderTarget {
    uint32_t     offset, addr, w, h, hw, hh, pad;
    unsigned     n;
    Img          tex;
    uint64_t     last_use;
    DepthTarget *dt;
    uint32_t     pitch, fmt;
    int          dirty, presented;
    struct nv2a_rt_own own;
    uint32_t     bar_flip[2];
    int          copy;
    uint32_t     copy_flip;
} RenderTarget;

static RenderTarget s_rt[RT_MAX];
static DepthTarget  s_dt[DT_MAX];
static uint64_t     s_tick;

static void rt_sync(RenderTarget *rt, uint32_t pitch);
static void rt_free(RenderTarget *rt);

static int alias_check_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = recomp_env(RENV_RT_ALIAS_CHECK);
        on = !(e && e[0] == '0');
    }
    return on;
}

static void rt_own_reset(RenderTarget *rt)
{
    if (!alias_check_on() || !rt->pitch)
        return;
    nv2a_rt_own_reset(&rt->own, (const uint8_t *)xbox_GetMemoryOffset(), rt->addr,
                      rt->pitch, rt->h, (uint32_t)s_presents);
}

static int rt_drop_if_stale(RenderTarget *rt, const char *why)
{
    static int said;
    if (!alias_check_on() || !rt->tex.img || !rt->pitch)
        return 0;
    if (!nv2a_rt_own_stale(&rt->own, (const uint8_t *)xbox_GetMemoryOffset(), rt->addr,
                           rt->pitch, rt->h, (uint32_t)s_presents))
        return 0;
    s_stale_drops++;
    if (said < 20 || recomp_env(RENV_METAL_TRACE)) {
        said++;
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target dropped (%s): the title"
                " rewrote its %u bytes; last drawn at flip %u, now %llu\n",
                rt->offset, rt->w, rt->h, why, rt->pitch * rt->h, rt->own.draw_flip,
                (unsigned long long)s_presents);
    }
    rt->dirty = 0;
    rt_free(rt);
    return 1;
}

static void rt_free(RenderTarget *rt)
{
    if (rt->dirty && rt->presented && rt_drop_if_stale(rt, "evict"))
        return;
    if (rt->dirty && rt->presented)
        rt_sync(rt, rt->pitch);
    if (s_enc_rt == rt)
        end_encoder();
    img_release(&rt->tex);
    memset(rt, 0, sizeof *rt);
}

static DepthTarget *depth_target(uint32_t w, uint32_t h)
{
    uint32_t zeta = nv2a_pb_vsh_state()->zeta_offset;
    DepthTarget *e = NULL, *lru = &s_dt[0];
    int i;

    for (i = 0; i < DT_MAX; i++) {
        DepthTarget *d = &s_dt[i];
        if (d->tex.img && d->zeta == zeta && d->w == w && d->h == h) {
            d->last_use = s_tick;
            return d;
        }
        if (!d->tex.img && !e)
            e = d;
        if (d->last_use < lru->last_use)
            lru = d;
    }
    if (!e) {
        e = lru;
        if (s_enc_dt == e)
            end_encoder();
        for (i = 0; i < RT_MAX; i++)
            if (s_rt[i].dt == e)
                s_rt[i].dt = NULL;
        if (e->dview) {
            vkDeviceWaitIdle(s_dev);
            vkDestroyImageView(s_dev, e->dview, NULL);
        }
        img_release(&e->tex);
        memset(e, 0, sizeof *e);
    }
    e->pad = nv2a_host_pad(w, h);
    e->n = nv2a_host_size(w + 2u * e->pad, h, nv2a_host_render_scale(), VK_MAX_DIM,
                          &e->hw, &e->hh);
    if (!img_make(&e->tex, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, s_dfmt, e->hw, e->hh, 1,
                  1, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                  | (xr_on() ? VK_IMAGE_USAGE_SAMPLED_BIT : 0),
                  VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
        return NULL;
    img_layout(up_cmd(), &e->tex, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    if (xr_on()) {
        VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vi.image = e->tex.img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = s_dfmt;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        if (vkCreateImageView(s_dev, &vi, NULL, &e->dview) != VK_SUCCESS)
            e->dview = VK_NULL_HANDLE;
    }
    e->zeta = zeta;
    e->w = w;
    e->h = h;
    e->last_use = s_tick;
    e->fresh = 1;
    fprintf(stderr, LOGP "zeta 0x%08X: %ux%u depth buffer (%ux%u host)\n", zeta, w, h,
            e->hw, e->hh);
    return e;
}

/* tex from the guest's bytes at addr when the surface is 32-bit, as the CPU
 * path draws over whatever memory holds; 0 when it is not. */
static int rt_seed(Img *tex, uint32_t addr, uint32_t pitch, uint32_t w, uint32_t h,
                   uint32_t pad, uint32_t hw, uint32_t hh)
{
    const uint8_t *src = (const uint8_t *)xbox_GetMemoryOffset() + addr;

    if (pitch < w * 4u || (pitch & 3u))
        return 0;
    if (hw == w)
        return img_upload(tex, 0, 0, w, h, src, (VkDeviceSize)pitch * (h - 1) + w * 4u,
                          pitch / 4u, VK_IMAGE_LAYOUT_GENERAL);
    {
        unsigned n = hw / (w + 2u * pad);
        uint8_t *up = (uint8_t *)calloc((size_t)hw * hh, 4u);
        int ok = 0;
        if (up) {
            nv2a_upscale_nearest32(src, pitch, w, h, n, up + (size_t)pad * n * 4u, hw * 4u);
            ok = img_upload(tex, 0, 0, hw, hh, up, (VkDeviceSize)hw * hh * 4u, hw,
                            VK_IMAGE_LAYOUT_GENERAL);
            free(up);
        }
        return ok;
    }
}

static void clear_image(Img *im, const float c[4])
{
    VkClearColorValue cv;
    VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    memcpy(cv.float32, c, sizeof cv.float32);
    end_encoder();
    ensure_cmd();
    barrier_all(s_cur->cmd);
    vkCmdClearColorImage(s_cur->cmd, im->img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &r);
    barrier_all(s_cur->cmd);
}

static void copy_image(Img *src, int32_t sx, int32_t sy, Img *dst, int32_t dx, int32_t dy,
                       uint32_t w, uint32_t h)
{
    VkImageCopy c;
    end_encoder();
    ensure_cmd();
    barrier_all(s_cur->cmd);
    memset(&c, 0, sizeof c);
    c.srcSubresource.aspectMask = c.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    c.srcSubresource.layerCount = c.dstSubresource.layerCount = 1;
    c.srcOffset.x = sx;
    c.srcOffset.y = sy;
    c.dstOffset.x = dx;
    c.dstOffset.y = dy;
    c.extent.width = w;
    c.extent.height = h;
    c.extent.depth = 1;
    vkCmdCopyImage(s_cur->cmd, src->img, VK_IMAGE_LAYOUT_GENERAL, dst->img,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    barrier_all(s_cur->cmd);
}

static int rt_grow(RenderTarget *rt, uint32_t w, uint32_t h)
{
    static int said;
    Img tex;
    uint32_t hw, hh, pad;
    unsigned n;
    int keep;

    pad = nv2a_host_pad(w, h);
    n = nv2a_host_size(w + 2u * pad, h, nv2a_host_render_scale(), VK_MAX_DIM, &hw, &hh);
    keep = rt->pad == pad && rt->hw == (rt->w + 2u * pad) * n && rt->hh == rt->h * n;
    if (!keep)
        rt_sync(rt, rt->pitch);
    end_encoder();
    if (!rt_image_new(&tex, hw, hh))
        return 0;
    if (!rt_seed(&tex, rt->addr, rt->pitch, w, h, pad, hw, hh)) {
        static const float black[4] = { 0, 0, 0, 0 };
        clear_image(&tex, black);
    }
    if (keep)
        copy_image(&rt->tex, 0, 0, &tex, 0, 0, rt->hw, rt->hh);
    if (said++ < 20)
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target grown to %ux%u\n",
                rt->offset, rt->w, rt->h, w, h);
    img_release(&rt->tex);
    rt->tex = tex;
    rt->w = w;
    rt->h = h;
    rt->hw = hw;
    rt->hh = hh;
    rt->pad = pad;
    rt->n = n;
    rt->dt = NULL;
    rt_own_reset(rt);
    s_grows++;
    return 1;
}

static RenderTarget *surface(void)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    uint32_t w = g->clip_x + g->clip_w, h = g->clip_y + g->clip_h;
    RenderTarget *free_rt = NULL, *lru = &s_rt[0];
    uint32_t hw, hh, pad;
    unsigned n;
    int i;

    if (!g->color_offset || !w || !h || w > 4096 || h > 4096)
        return NULL;
    s_tick++;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex.img && rt->offset == g->color_offset && rt->w == w && rt->h == h) {
            if (rt_drop_if_stale(rt, "reuse"))
                break;
            rt->last_use = s_tick;
            rt->fmt = g->format;
            rt->copy = 0;
            return rt;
        }
    }
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex.img || rt->offset != g->color_offset || rt->pitch != g->pitch)
            continue;
        if (rt_drop_if_stale(rt, "reuse"))
            break;
        if ((rt->w >= w && rt->h >= h)
                || rt_grow(rt, rt->w > w ? rt->w : w, rt->h > h ? rt->h : h)) {
            rt->last_use = s_tick;
            rt->fmt = g->format;
            rt->copy = 0;
            return rt;
        }
        break;
    }
    if (alias_check_on()) {
        uint32_t addr = nv2a_pb_dma_resolve(g->color_offset);
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            if (!rt->tex.img || !nv2a_rt_overlap(rt->addr, rt->pitch * rt->h, addr, g->pitch * h))
                continue;
            if (rt_drop_if_stale(rt, "overlap"))
                continue;
            rt_sync(rt, rt->pitch);
            if (s_retired++ < 20)
                fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target retired:"
                        " 0x%08X %ux%u overlaps it\n", rt->offset, rt->w, rt->h,
                        g->color_offset, w, h);
            rt_free(rt);
        }
    }
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex.img && !free_rt)
            free_rt = rt;
        if (rt->last_use < lru->last_use)
            lru = rt;
    }
    if (!free_rt) {
        free_rt = lru;
        if (!rt_drop_if_stale(free_rt, "evict")) {
            rt_sync(free_rt, free_rt->pitch);
            if (free_rt->dirty && s_evict_lost++ < 20)
                fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target evicted for"
                        " 0x%08X with no write-back (pitch %u), its pixels lost\n",
                        free_rt->offset, free_rt->w, free_rt->h, g->color_offset,
                        free_rt->pitch);
            rt_free(free_rt);
        }
    }
    pad = nv2a_host_pad(w, h);
    n = nv2a_host_size(w + 2u * pad, h, nv2a_host_render_scale(), VK_MAX_DIM, &hw, &hh);
    if (!rt_image_new(&free_rt->tex, hw, hh))
        return NULL;
    free_rt->offset = g->color_offset;
    free_rt->addr = nv2a_pb_dma_resolve(g->color_offset);
    free_rt->w = w;
    free_rt->h = h;
    free_rt->hw = hw;
    free_rt->hh = hh;
    free_rt->pad = pad;
    free_rt->n = n;
    free_rt->pitch = g->pitch;
    free_rt->fmt = g->format;
    free_rt->last_use = s_tick;
    rt_own_reset(free_rt);
    if (!rt_seed(&free_rt->tex, free_rt->addr, g->pitch, w, h, pad, hw, hh)) {
        static const float black[4] = { 0, 0, 0, 0 };
        clear_image(&free_rt->tex, black);
    }
    fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target (%ux%u host)\n",
            g->color_offset, w, h, hw, hh);
    return free_rt;
}

/* Open a rendering pass on rt: with `clear`, colour cleared to c, else
 * loaded; the depth buffer as begin_pass in nv2a_pb_metal.m. */
static void begin_pass(RenderTarget *rt, int clear, const double c[4],
                       unsigned zmask, float z, uint32_t st)
{
    VkRenderingAttachmentInfoKHR ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR };
    VkRenderingAttachmentInfoKHR da = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR };
    VkRenderingAttachmentInfoKHR sa = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR };
    VkRenderingInfoKHR ri = { VK_STRUCTURE_TYPE_RENDERING_INFO_KHR };
    DepthTarget *dt = rt->dt;

    end_encoder();
    ensure_cmd();
    barrier_all(s_cur->cmd);
    rt->dirty = 1;
    nv2a_rt_own_drawn(&rt->own, (uint32_t)s_presents);
    ca.imageView = rt->tex.view;
    ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (clear) {
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca.clearValue.color.float32[0] = (float)c[0];
        ca.clearValue.color.float32[1] = (float)c[1];
        ca.clearValue.color.float32[2] = (float)c[2];
        ca.clearValue.color.float32[3] = (float)c[3];
    } else {
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    }
    ri.renderArea.extent.width = rt->hw;
    ri.renderArea.extent.height = rt->hh;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    if (dt) {
        int zc = (zmask & 1u) || dt->fresh, sc = (zmask & 2u) || dt->fresh;
        da.imageView = sa.imageView = dt->tex.view;
        da.imageLayout = sa.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        da.storeOp = sa.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        da.loadOp = zc ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        sa.loadOp = sc ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        da.clearValue.depthStencil.depth = (zmask & 1u) ? z : 1.0f;
        sa.clearValue.depthStencil.stencil = (zmask & 2u) ? (st & 0xFFu) : 0u;
        ri.pDepthAttachment = &da;
        ri.pStencilAttachment = &sa;
        dt->fresh = 0;
    }
    if (s_legacy) {
        VkRenderPassBeginInfo bi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        VkClearValue cv[2];
        int zc = dt && da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
        int sc = dt && sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
        cv[0] = ca.clearValue;
        cv[1].depthStencil.depth = da.clearValue.depthStencil.depth;
        cv[1].depthStencil.stencil = sa.clearValue.depthStencil.stencil;
        bi.renderPass = rp_get(dt != NULL, clear, zc, sc);
        bi.framebuffer = fb_get(rt->tex.view, dt ? dt->tex.view : VK_NULL_HANDLE, rt->hw, rt->hh);
        bi.renderArea = ri.renderArea;
        bi.clearValueCount = dt ? 2 : 1;
        bi.pClearValues = cv;
        vkCmdBeginRenderPass(s_cur->cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
    } else {
        s_begin_rendering(s_cur->cmd, &ri);
    }
    s_in_pass = 1;
    s_enc_serial++;
    s_enc_rt = rt;
    s_enc_dt = dt;
    occ_pass_begin();
}

static void begin_encoder(RenderTarget *rt, int clear, const double c[4])
{
    begin_pass(rt, clear, c, 0, 1.0f, 0);
}

static void vk_on_clear(uint32_t param)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    RenderTarget *rt;
    double c[4];
    uint32_t argb, b[4], hb[4];
    VkClearAttachment ca;
    VkClearRect cr;

    if (!(param & 0xF0) || !init())
        return;
    rt = surface();
    if (!rt || !nv2a_clear_box(g, rt->w, rt->h, b))
        return;
    argb = nv2a_clear_argb(g);
    c[0] = (double)((argb >> 16) & 0xFF) / 255.0;
    c[1] = (double)((argb >> 8) & 0xFF) / 255.0;
    c[2] = (double)(argb & 0xFF) / 255.0;
    c[3] = (double)(argb >> 24) / 255.0;
    s_clears++;
    if (b[0] == 0 && b[1] == 0 && b[2] == rt->w && b[3] == rt->h) {
        begin_encoder(rt, 1, c);
        return;
    }
    if (!s_in_pass || s_enc_rt != rt)
        begin_encoder(rt, 0, NULL);
    /* A part of the target: vkCmdClearAttachments over the box, as a 3D
     * draw's clip maps it (an edge at 0 or w reaches the pads). Not a draw,
     * so no visibility test counts it. */
    nv2a_aspect_box(b, rt->w, rt->h, rt->pad, rt->n, NV2A_PLACE_CENTRE, 0, 0, hb);
    memset(&ca, 0, sizeof ca);
    ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ca.clearValue.color.float32[0] = (float)c[0];
    ca.clearValue.color.float32[1] = (float)c[1];
    ca.clearValue.color.float32[2] = (float)c[2];
    ca.clearValue.color.float32[3] = (float)c[3];
    memset(&cr, 0, sizeof cr);
    cr.rect.offset.x = (int32_t)hb[0];
    cr.rect.offset.y = (int32_t)hb[1];
    cr.rect.extent.width = hb[2] - hb[0];
    cr.rect.extent.height = hb[3] - hb[1];
    cr.layerCount = 1;
    if (cr.rect.extent.width && cr.rect.extent.height)
        vkCmdClearAttachments(s_cur->cmd, 1, &ca, 1, &cr);
    s_partial_clears++;
}

static void vk_on_zclear(uint32_t param)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    RenderTarget *rt;
    DepthTarget *dt;

    if (!(param & 3u) || !init())
        return;
    rt = surface();
    if (!rt)
        return;
    dt = depth_target(rt->w, rt->h);
    if (!dt)
        return;
    rt->dt = dt;
    begin_pass(rt, 0, NULL, param & 3u, nv2a_zclear_depth(g->format, v->zclear),
               v->zclear & 0xFFu);
    if (param & 1u)
        s_zclears++;
}

/* ---- vertices ------------------------------------------------------------------- */

static int fetch(const struct nv2a_pb_draw *d, const VertexAttr *a, uint32_t index,
                 float out[4])
{
    const uint8_t *p;
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    if (d->inline_data) {
        size_t at = (size_t)a->offset + (size_t)index * a->stride;
        if (at + 4 > d->inline_bytes)
            return 0;
        p = (const uint8_t *)d->inline_buf + at;
    } else {
        if (!a->offset)
            return 0;
        p = (const uint8_t *)xbox_GetMemoryOffset() + a->offset + (size_t)index * a->stride;
    }
    return nv2a_vtx_decode(a->type, a->size, p, out);
}

/* ---- pipelines ---------------------------------------------------------------
 *
 * Metal keeps cull, winding and depth/stencil out of its pipelines; Vulkan
 * (without the extended dynamic state) keeps them in. So the key is
 * nv2a_pb_metal.m's (shaders, blend, colour mask, depth attachment) plus the
 * topology class, the cull and winding, and the depth/stencil key. */

static VkBlendOp blend_op(uint32_t eq)
{
    switch (eq) {
    case 0x800A: return VK_BLEND_OP_SUBTRACT;
    case 0x800B:
    case 0xF005: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 0x8007: return VK_BLEND_OP_MIN;
    case 0x8008: return VK_BLEND_OP_MAX;
    default:     return VK_BLEND_OP_ADD;
    }
}

static VkBlendFactor blend_factor(uint32_t gl, int alpha)
{
    static const VkBlendFactor map[NV2A_BF_UNKNOWN + 1] = {
        VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
        VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
        VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
        VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
        VK_BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
        VK_BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
        VK_BLEND_FACTOR_ONE,
    };
    uint32_t bf = nv2a_blend_from_gl(gl);
    return map[alpha ? nv2a_blend_for_alpha(bf) : bf];
}

typedef struct {
    uint64_t vs, fs;
    uint64_t dss;          /* depth_key() */
    uint32_t key[9];
} PsoKey;
#define PSO_MAX 1024
static struct { PsoKey k; VkPipeline p; } s_pso[PSO_MAX];
static int s_pso_n, s_pso_last;

/* The depth/stencil key: nv2a_pb_metal.m's depth_state key. */
static uint64_t depth_key(int depth, int stencil)
{
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    uint32_t f = nv2a_cmp_from_gl(v->depth_func);
    uint64_t key = 0;
    if (depth)
        key |= 1u | f << 1 | (v->depth_mask ? 1u : 0u) << 4;
    if (stencil)
        key |= (uint64_t)(1u | (v->stencil_func & 7u) << 1
                          | nv2a_stencil_op_from_gl(v->stencil_op[0]) << 4
                          | nv2a_stencil_op_from_gl(v->stencil_op[1]) << 8
                          | nv2a_stencil_op_from_gl(v->stencil_op[2]) << 12
                          | (v->stencil_rmask & 0xFFu) << 16
                          | (v->stencil_wmask & 0xFFu) << 24) << 8;
    return key;
}

/* topo: 0 triangles, 1 lines, 2 points; cull 0 none, 1 back, 2 front. */
static VkPipeline pipeline(VkShaderModule vs, VkShaderModule fs, int topo, int cull,
                           int ccw, uint64_t dss)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    PsoKey pk;
    uint32_t *key = pk.key;
    VkPipeline p = VK_NULL_HANDLE;
    int i;

    memset(&pk, 0, sizeof pk);
    pk.vs = (uint64_t)(uintptr_t)vs;
    pk.fs = (uint64_t)(uintptr_t)fs;
    pk.dss = dss;
    key[0] = g->blend_enable ? 1 : 0;
    key[1] = key[0] ? g->blend_sfactor : 0;
    key[2] = key[0] ? g->blend_dfactor : 0;
    key[3] = g->color_keep;
    key[4] = key[0] ? (uint32_t)blend_op(g->blend_equation) : 0;
    key[5] = s_enc_dt ? 1 : 0;
    key[6] = (uint32_t)topo;
    key[7] = (uint32_t)cull;
    key[8] = (uint32_t)ccw;
    if (s_pso_last < s_pso_n && !memcmp(&s_pso[s_pso_last].k, &pk, sizeof pk))
        return s_pso[s_pso_last].p;
    for (i = 0; i < s_pso_n; i++)
        if (!memcmp(&s_pso[i].k, &pk, sizeof pk)) {
            s_pso_last = i;
            return s_pso[i].p;
        }
    {
        VkPipelineShaderStageCreateInfo st[2] = {
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
        VkPipelineVertexInputStateCreateInfo vi = {
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        VkPipelineInputAssemblyStateCreateInfo ia = {
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        VkPipelineViewportStateCreateInfo vp = {
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        VkPipelineRasterizationStateCreateInfo rs = {
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        VkPipelineMultisampleStateCreateInfo ms = {
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        VkPipelineDepthStencilStateCreateInfo ds = {
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        VkPipelineColorBlendAttachmentState cb;
        VkPipelineColorBlendStateCreateInfo cbs = {
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        static const VkDynamicState dyn[5] = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS,
            VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_REFERENCE };
        VkPipelineDynamicStateCreateInfo dy = {
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        VkFormat cfmt = VK_FORMAT_B8G8R8A8_UNORM;
        VkPipelineRenderingCreateInfoKHR rci = {
            VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
        VkGraphicsPipelineCreateInfo pci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        int dz = (int)(dss & 1u), sz = (int)(dss >> 8 & 1u);

        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = vs;
        st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = fs;
        st[1].pName = "main";
        ia.topology = topo == 2 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                    : topo == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                    :             VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = cull == 1 ? VK_CULL_MODE_BACK_BIT
                    : cull == 2 ? VK_CULL_MODE_FRONT_BIT : VK_CULL_MODE_NONE;
        rs.frontFace = ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
        rs.depthBiasEnable = VK_TRUE;
        rs.lineWidth = 1.0f;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        ds.depthTestEnable = dz ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = dz && (dss >> 4 & 1u) ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = dz ? (VkCompareOp)(dss >> 1 & 7u) : VK_COMPARE_OP_ALWAYS;
        ds.stencilTestEnable = sz ? VK_TRUE : VK_FALSE;
        if (sz) {
            uint64_t s = dss >> 8;
            ds.front.compareOp = (VkCompareOp)(s >> 1 & 7u);
            ds.front.failOp = (VkStencilOp)(s >> 4 & 0xFu);
            ds.front.depthFailOp = (VkStencilOp)(s >> 8 & 0xFu);
            ds.front.passOp = (VkStencilOp)(s >> 12 & 0xFu);
            ds.front.compareMask = (uint32_t)(s >> 16 & 0xFFu);
            ds.front.writeMask = (uint32_t)(s >> 24 & 0xFFu);
            ds.back = ds.front;
        }
        memset(&cb, 0, sizeof cb);
        cb.blendEnable = key[0] ? VK_TRUE : VK_FALSE;
        if (key[0]) {
            cb.srcColorBlendFactor = blend_factor(key[1], 0);
            cb.dstColorBlendFactor = blend_factor(key[2], 0);
            cb.srcAlphaBlendFactor = blend_factor(key[1], 1);
            cb.dstAlphaBlendFactor = blend_factor(key[2], 1);
            cb.colorBlendOp = cb.alphaBlendOp = (VkBlendOp)key[4];
        }
        cb.colorWriteMask = ((key[3] & 0x00FF0000u) ? 0 : VK_COLOR_COMPONENT_R_BIT)
                          | ((key[3] & 0x0000FF00u) ? 0 : VK_COLOR_COMPONENT_G_BIT)
                          | ((key[3] & 0x000000FFu) ? 0 : VK_COLOR_COMPONENT_B_BIT)
                          | ((key[3] & 0xFF000000u) ? 0 : VK_COLOR_COMPONENT_A_BIT);
        cbs.attachmentCount = 1;
        cbs.pAttachments = &cb;
        dy.dynamicStateCount = 5;
        dy.pDynamicStates = dyn;
        rci.colorAttachmentCount = 1;
        rci.pColorAttachmentFormats = &cfmt;
        if (key[5]) {
            rci.depthAttachmentFormat = s_dfmt;
            rci.stencilAttachmentFormat = s_dfmt;
        }
        if (s_legacy)
            pci.renderPass = rp_get(key[5] != 0, 0, 0, 0);
        else
            pci.pNext = &rci;
        pci.stageCount = 2;
        pci.pStages = st;
        pci.pVertexInputState = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vp;
        pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms;
        pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &cbs;
        pci.pDynamicState = &dy;
        pci.layout = s_playout;
        if (vkCreateGraphicsPipelines(s_dev, s_pcache, 1, &pci, NULL, &p) != VK_SUCCESS) {
            static int said;
            if (!said++)
                fprintf(stderr, LOGP "pipeline creation failed\n");
            return VK_NULL_HANDLE;
        }
    }
    if (s_pso_n < PSO_MAX) {
        i = s_pso_n++;
    } else {
        static unsigned next;
        i = (int)(next++ % PSO_MAX);
        /* an encoded draw may still use it: wait for the GPU first */
        flush_keep();
        vkDestroyPipeline(s_dev, s_pso[i].p, NULL);
    }
    s_pso[i].k = pk;
    s_pso[i].p = p;
    s_pso_last = i;
    return p;
}

/* ---- textures (nv2a_pb_metal.m's cache, the same keys and invalidation) ------ */

typedef struct nv2a_stage Stage;
static Stage s_stage[4];

#define TEX_MAX_TEXELS (2048u * 2048u)
#define TEX_CACHE 512
typedef struct {
    uint32_t offset, color, width, height, pitch, palette, levels;
    int      cube;
    uint32_t bytes, gen, used, pack_gen, pack_scale, pack_levels;
    uint64_t hash;
    int      live;
    Img      tex;
} TexEntry;

static TexEntry s_tc[TEX_CACHE];
static uint32_t s_tc_clock;

static uint32_t tex_cache_n(void)
{
    static long n = -1;
    if (n < 0) {
        n = recomp_env_int(RENV_TEX_CACHE, 256);
        if (n < 1) n = 1;
        if (n > TEX_CACHE) n = TEX_CACHE;
    }
    return (uint32_t)n;
}

static void tex_cache_stale(uint32_t addr, uint32_t bytes)
{
    uint32_t n = tex_cache_n(), lo = addr & 0x07FFFFFFu, hi = lo + bytes, i;
    for (i = 0; i < n; i++) {
        TexEntry *c = &s_tc[i];
        uint32_t clo = c->offset & 0x07FFFFFFu;
        if (c->live && clo < hi && lo < clo + c->bytes)
            c->gen = nv2a_pb_gpu_state()->flips - 1u;
    }
}

static uint32_t *s_texels;
static size_t    s_texels_cap;
static Img       s_black, s_magenta, s_black_cube, s_black_3d;

static int solid_make(Img *im, VkImageType type, VkImageViewType vt, uint32_t layers,
                      uint32_t argb)
{
    uint32_t l;
    if (!img_make(im, type, vt, VK_FORMAT_B8G8R8A8_UNORM, 1, 1, 1, 1, layers,
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return 0;
    img_layout(up_cmd(), im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    for (l = 0; l < layers; l++)
        img_upload(im, 0, l, 1, 1, &argb, 4, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    img_layout(up_cmd(), im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return 1;
}

static int no_mips(void)
{
    static int off = -1;
    if (off < 0)
        off = recomp_env(RENV_METAL_NO_MIPS) != NULL;
    return off;
}

static int memo_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = recomp_env(RENV_METAL_MEMO);
        on = !(e && *e == '0');
    }
    return on;
}

static void build_stages(void)
{
    extern uint32_t xbox_ContiguousAllocatedBytes(void);
    const uint32_t *regs = nv2a_pb_tex_regs();
    const uint8_t  *sets = nv2a_pb_tex_regs_set();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    static struct { uint32_t gen, prog, contig; int set, ok; } memo;
    uint32_t gen = nv2a_pb_tex_regs_gen(), contig = xbox_ContiguousAllocatedBytes();
    int n;

    if (memo_on() && memo.ok && memo.gen == gen && memo.prog == v->shader_prog
        && memo.set == v->shader_set && memo.contig == contig)
        return;
    memo.gen = gen;
    memo.prog = v->shader_prog;
    memo.set = v->shader_set;
    memo.contig = contig;
    memo.ok = 1;
    for (n = 0; n < 4; n++) {
        Stage *t = &s_stage[n];
        nv2a_stage_decode(n, &regs[(0x40u * n) / 4], &sets[(0x40u * n) / 4],
                          v->shader_set, v->shader_prog, nv2a_pb_dma_resolve,
                          NV2A_STAGE_P8_PALETTE | NV2A_STAGE_PITCH_LINEAR
                          | (no_mips() ? 0u : NV2A_STAGE_MIPS), t);
        if (t->mode == 5)
            t->mode = 0;
    }
}

static void stages_cube_rt(void)
{
    int n, i;
    for (n = 0; n < 4; n++) {
        Stage *t = &s_stage[n];
        t->cube_rt = 0;
        for (i = 0; t->cube && i < RT_MAX && !t->cube_rt; i++)
            t->cube_rt = s_rt[i].tex.img
                         && nv2a_stage_cube_over_rt(t, s_rt[i].addr, s_rt[i].pitch * s_rt[i].h)
                         && !rt_drop_if_stale(&s_rt[i], "cube");
    }
}

/* A sampled texture of `levels` levels from src: BGRA8 from A8R8G8B8 texels,
 * or a pack entry's BC1/BC3 blocks. */
static int tex_make(Img *im, uint32_t width, uint32_t height, uint32_t levels,
                    const uint32_t *src, int bc)
{
    const uint8_t *p = (const uint8_t *)src;
    VkFormat f = bc == XRTP_BC_3 ? VK_FORMAT_BC3_UNORM_BLOCK
               : bc == XRTP_BC_1 ? VK_FORMAT_BC1_RGBA_UNORM_BLOCK : VK_FORMAT_B8G8R8A8_UNORM;
    uint32_t l;
    if (!img_make(im, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, f, width, height, 1, levels, 1,
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return 0;
    img_layout(up_cmd(), im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    for (l = 0; l < levels; l++) {
        uint32_t w = nv2a_tex_level_dim(width, l), h = nv2a_tex_level_dim(height, l);
        uint32_t row = bc ? (w + 3u) / 4u * (bc == XRTP_BC_1 ? 8u : 16u) : w * 4u;
        uint32_t rows = bc ? (h + 3u) / 4u : h;
        if (!img_upload(im, l, 0, w, h, p, (VkDeviceSize)row * rows, 0,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)) {
            img_destroy_now(im);
            return 0;
        }
        p += (size_t)row * rows;
    }
    img_layout(up_cmd(), im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return 1;
}

static int tex_make_cube(Img *im, uint32_t size, uint32_t levels, const uint32_t *src)
{
    const uint8_t *p = (const uint8_t *)src;
    uint32_t f, l;
    if (!img_make(im, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, VK_FORMAT_B8G8R8A8_UNORM,
                  size, size, 1, levels, 6,
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
        return 0;
    img_layout(up_cmd(), im, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    for (f = 0; f < 6; f++)
        for (l = 0; l < levels; l++) {
            uint32_t w = nv2a_tex_level_dim(size, l);
            if (!img_upload(im, l, f, w, w, p, (VkDeviceSize)w * w * 4u, 0,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)) {
                img_destroy_now(im);
                return 0;
            }
            p += (size_t)w * w * 4u;
        }
    img_layout(up_cmd(), im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return 1;
}

static uint32_t s_up_scale, s_up_levels;

static int tex_decode_guest(const Stage *t, uint32_t faces)
{
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    size_t n = nv2a_tex_texels(t->width, t->height, t->levels);
    const uint32_t *pal = NULL;
    uint32_t f, stride = faces > 1 ? nv2a_stage_face_stride(t) : 0;
    if (n * faces > s_texels_cap) {
        uint32_t *p = (uint32_t *)realloc(s_texels, n * faces * 4);
        if (!p)
            return 0;
        s_texels = p;
        s_texels_cap = n * faces;
    }
    if (t->color == NV2A_TEX_P8)
        pal = (const uint32_t *)(mem + t->palette);
    for (f = 0; f < faces; f++)
        nv2a_tex_decode(mem + t->offset + f * stride, t->color, t->width, t->height, t->pitch,
                        t->levels, pal, pal ? t->pal_len : 0, s_texels + n * f);
    return 1;
}

static int tex_upload(const Stage *t, uint64_t hash, int cube, Img *out)
{
    int ok = 0;
    s_up_scale = 1;
    s_up_levels = t->levels;
    if (cube) {
        if (!tex_decode_guest(t, 6))
            return 0;
        if (nv2a_tex_dump_on())
            nv2a_tex_dump_upload(t, hash, s_texels);
        if (!tex_make_cube(out, t->width, t->levels, s_texels))
            return 0;
        s_tex_uploads++;
        nv2a_stats_add(NV2A_STAT_TEX_UPLOADS, 1);
        return 1;
    }
    if (nv2a_tex_pack_count()) {
        struct nv2a_tex_pack_entry *pe = nv2a_tex_pack_find(hash, t->width, t->height);
        const uint32_t *src = NULL;
        if (pe && pe->bc && !s_bc) {
            static int said;
            if (!said++)
                fprintf(stderr, LOGP "textures.pack: this GPU has no BC textures; BC1 and BC3"
                                     " entries go up from the guest\n");
        } else if (pe) {
            src = nv2a_tex_pack_texels(pe);
        }
        if (src)
            ok = tex_make(out, pe->width, pe->height, pe->levels, src, pe->bc);
        if (ok) {
            s_up_scale = nv2a_stage_pack_scale(t->width, pe->width);
            s_up_levels = pe->levels;
            nv2a_tex_pack_release(pe);
        }
        if (!ok)
            nv2a_tex_pack_miss(hash, t->width, t->height, t->color, t->levels);
        else if (nv2a_tex_dump_on())
            nv2a_tex_dump_upload(t, hash, nv2a_tex_dump_wants(t, hash)
                                          && tex_decode_guest(t, 1) ? s_texels : NULL);
    }
    if (!ok) {
        if (!tex_decode_guest(t, 1))
            return 0;
        if (nv2a_tex_dump_on())
            nv2a_tex_dump_upload(t, hash, s_texels);
        if (!tex_make(out, t->width, t->height, t->levels, s_texels, 0))
            return 0;
    }
    s_tex_uploads++;
    nv2a_stats_add(NV2A_STAT_TEX_UPLOADS, 1);
    return 1;
}

static TexEntry *s_tex_last;

static Img *tex_bind_body(const Stage *t, int cube);

static Img *tex_bind(const Stage *t, int cube)
{
    Img *im = tex_bind_body(t, cube);
    return cube && im->vtype != VK_IMAGE_VIEW_TYPE_CUBE ? &s_black_cube : im;
}

static Img *tex_bind_body(const Stage *t, int cube)
{
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const uint8_t *mem = (const uint8_t *)xbox_GetMemoryOffset();
    uint32_t bytes = 0, i, n;
    TexEntry *e = NULL, *victim = &s_tc[0];

    s_tex_last = NULL;
    if (t->valid && (uint64_t)t->width * t->height <= TEX_MAX_TEXELS)
        bytes = cube ? nv2a_stage_cube_extent(t)
                     : nv2a_tex_extent(t->color, t->width, t->height, t->pitch, t->levels);
    if (bytes) {
        uint32_t lo = t->offset & 0x07FFFFFFu, hi = lo + bytes;
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            uint32_t rlo = rt->addr & 0x07FFFFFFu;
            if (rt->tex.img && rt->dirty && rt->pitch
                    && lo < rlo + rt->pitch * rt->h && rlo < hi) {
                if (rt_drop_if_stale(rt, "sync"))
                    continue;
                if (nv2a_rt_pitch_alias(t->color, t->pitch, t->height, rt->fmt, rt->pitch)) {
                    s_pitch_alias_binds++;
                    if (recomp_env(RENV_METAL_TRACE)
                            && nv2a_pair_new(&s_pitch_alias_seen, rt->fmt, rt->pitch, t->color,
                                             t->pitch))
                        fprintf(stderr, LOGP "pitch alias: target 0x%08X fmt 0x%X pitch %u,"
                                " texture 0x%08X fmt 0x%02X pitch %u decoded without a"
                                " write-back\n", rt->offset, rt->fmt, rt->pitch, t->offset,
                                t->color, t->pitch);
                    continue;
                }
                s_decode_syncs++;
                rt_sync(rt, rt->pitch);
            }
        }
    }
    if (!bytes) {
        static uint8_t said[256];
        if ((uint64_t)t->width * t->height > TEX_MAX_TEXELS)
            return &s_black;
        if (!said[t->color & 0xFF]) {
            said[t->color & 0xFF] = 1;
            fprintf(stderr, LOGP "\x1b[1;35mtexture format 0x%02X (%ux%u at 0x%08X) not"
                    " decoded: drawn magenta\x1b[0m\n", t->color, t->width, t->height,
                    t->offset);
        }
        return &s_magenta;
    }
    n = tex_cache_n();
    for (i = 0; i < n; i++) {
        TexEntry *c = &s_tc[i];
        if (c->live && c->offset == t->offset && c->color == t->color
                && c->width == t->width && c->height == t->height
                && c->pitch == t->pitch && c->palette == t->palette
                && c->levels == t->levels && c->cube == cube) {
            e = c;
            break;
        }
        if (!c->live || (victim->live && c->used < victim->used))
            victim = c;
    }
    if (e) {
        uint32_t pg = nv2a_tex_pack_gen();
        if (e->gen != g->flips || e->pack_gen != pg) {
            uint64_t h = nv2a_stage_hash(mem, t, bytes);
            s_tex_rehash++;
            e->gen = g->flips;
            if (h != e->hash || e->pack_gen != pg) {
                Img nt;
                s_tex_changed++;
                if (!tex_upload(t, h, cube, &nt))
                    return &s_black;
                e->hash = h;
                e->pack_gen = pg;
                e->pack_scale = s_up_scale;
                e->pack_levels = s_up_levels;
                img_release(&e->tex);
                e->tex = nt;
            }
        }
    } else {
        Img nt;
        uint64_t h = nv2a_stage_hash(mem, t, bytes);
        e = victim;
        img_release(&e->tex);
        e->live = 0;
        if (!tex_upload(t, h, cube, &nt))
            return &s_black;
        e->offset = t->offset;
        e->color = t->color;
        e->width = t->width;
        e->height = t->height;
        e->pitch = t->pitch;
        e->palette = t->palette;
        e->levels = t->levels;
        e->cube = cube;
        e->bytes = bytes;
        e->gen = g->flips;
        e->hash = h;
        e->pack_gen = nv2a_tex_pack_gen();
        e->pack_scale = s_up_scale;
        e->pack_levels = s_up_levels;
        e->tex = nt;
        e->live = 1;
    }
    e->used = ++s_tc_clock;
    s_tex_last = e;
    return &e->tex;
}

#define filter_linear nv2a_tex_filter_linear
#define filter_mip    nv2a_tex_filter_mip
#define lod_bias      nv2a_tex_lod_bias

static uint32_t s_stage_pack[4][2];
static uint64_t s_stage_hash[4];

static void stage_from_cache(int k)
{
    if (s_tex_last) {
        s_stage_pack[k][0] = s_tex_last->pack_scale;
        s_stage_pack[k][1] = s_tex_last->pack_levels;
        s_stage_hash[k] = s_tex_last->hash;
    }
}

static VkSamplerAddressMode address_mode(uint32_t m, int border)
{
    switch (m) {
    case 1:  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2:  return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case NV2A_ADDR_BORDER:
        return border ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
                      : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}

/* VkBorderColor + 1 for an A8R8G8B8 border Vulkan can give, else 0 (the
 * same three colours as Metal). */
static int border_colour(uint32_t argb)
{
    static int said;
    switch (argb) {
    case 0x00000000u: return 1 + VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    case 0xFF000000u: return 1 + VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    case 0xFFFFFFFFu: return 1 + VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    }
    if (!said) {
        said = 1;
        fprintf(stderr, LOGP "texture border colour 0x%08X has no Vulkan sampler colour:"
                " clamping to the edge instead\n", argb);
    }
    return 0;
}

#define SMP_MAX 128
static struct { uint64_t key; VkSampler s; } s_smp[SMP_MAX];
static int s_smp_n;

static VkSampler sampler(const Stage *t, int bump_source, uint32_t scale, uint32_t entry_levels)
{
    static int point = -1;
    VkSamplerCreateInfo sd = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    VkSampler st;
    uint32_t u = t->addr_u & 0xF, v = t->addr_v & 0xF;
    int mn, mg, mip, guest_mip, i, bc = 0;
    unsigned an;
    uint64_t key;
    struct nv2a_stage_lod lod;

    if (point < 0)
        point = recomp_env(RENV_METAL_POINT) != NULL;
    if (u == NV2A_ADDR_BORDER || v == NV2A_ADDR_BORDER)
        bc = border_colour(t->border);
    mn = !point && filter_linear(t->filter, 0);
    mg = !point && filter_linear(t->filter, 1);
    guest_mip = t->levels > 1 ? filter_mip(t->filter) : 0;
    lod = nv2a_stage_lod(guest_mip, t->lod_min, t->lod_max, scale, entry_levels);
    mip = lod.mip;
    an = s_aniso ? nv2a_stage_aniso(t, guest_mip, mn, bump_source) : 0;
    key = u | v << 4 | (uint64_t)mn << 8 | (uint64_t)mg << 9 | (uint64_t)mip << 10
        | (uint64_t)an << 35 | (uint64_t)bc << 40 | lod.key;
    if (guest_mip)
        key |= (uint64_t)t->lod_min << 25 | (uint64_t)t->lod_max << 29;
    for (i = 0; i < s_smp_n; i++)
        if (s_smp[i].key == key)
            return s_smp[i].s;
    sd.minFilter = mn ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sd.magFilter = mg ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    sd.mipmapMode = mip == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sd.addressModeU = address_mode(u, bc);
    sd.addressModeV = address_mode(v, bc);
    sd.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (bc)
        sd.borderColor = (VkBorderColor)(bc - 1);
    sd.minLod = lod.min;
    sd.maxLod = lod.max;
    if (!mip) {
        /* Metal's NotMipmapped: level 0, with the min/mag choice intact. */
        sd.minLod = 0.0f;
        sd.maxLod = 0.25f;
    }
    if (an) {
        sd.minFilter = sd.magFilter = VK_FILTER_LINEAR;
        sd.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sd.anisotropyEnable = VK_TRUE;
        sd.maxAnisotropy = (float)an;
        if (sd.maxAnisotropy > s_props.limits.maxSamplerAnisotropy)
            sd.maxAnisotropy = s_props.limits.maxSamplerAnisotropy;
    }
    if (vkCreateSampler(s_dev, &sd, NULL, &st) != VK_SUCCESS)
        return s_smp_n ? s_smp[0].s : VK_NULL_HANDLE;
    if (s_smp_n < SMP_MAX) {
        i = s_smp_n++;
    } else {
        static unsigned next;
        i = (int)(next++ % SMP_MAX);
        flush_keep();
        vkDestroySampler(s_dev, s_smp[i].s, NULL);
    }
    s_smp[i].key = key;
    s_smp[i].s = st;
    return st;
}

/* ---- shaders: vertex programs and register combiners -------------------------- */

#define VS_MAX 512
typedef struct {
    uint32_t       hash;
    uint16_t       inputs;
    int            used, bad;
    struct nv2a_aspect_gate hp;
    VkShaderModule fn;
} VsEntry;
static VsEntry s_vs[VS_MAX];

static VsEntry *vs_cache(void)
{
    static NV2AVshProgram parsed;
    static char src[65536];
    static struct { int valid; uint32_t dwords, start; VsEntry *e; } memo;
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    uint32_t hash;
    VsEntry *e = NULL;
    char what[32];
    int i;

    if (memo.valid && memo.dwords == v->prog_dwords && memo.start == v->start)
        return memo.e;
    hash = nv2a_vsh_program_hash(v, NULL);
    if (!hash)
        hash = 1;
    for (i = 0; i < VS_MAX; i++)
        if (s_vs[i].used && s_vs[i].hash == hash) {
            e = &s_vs[i];
            break;
        }
    if (!e) {
        for (i = 0; i < VS_MAX && s_vs[i].used; i++)
            ;
        if (i == VS_MAX)
            return NULL;
        e = &s_vs[i];
        e->used = 1;
        e->hash = hash;
        e->hp.two_d = -1;
        e->hp.n = 0;
        e->bad = 1;
        d3d8_vsh_parse(&v->prog[v->start][0], NV2A_VSH_SLOTS - (int)v->start, &parsed);
        snprintf(what, sizeof what, "program %08X", hash);
        if (d3d8_vsh_generate_glsl(&parsed, D3D8_GLSL_SCREEN_SPACE, src, (int)sizeof src) <= 0) {
            s_compile_errors++;
            fprintf(stderr, LOGP "%s: no GLSL\n", what);
        } else if ((e->fn = compile_glsl(src, 0, what)) != VK_NULL_HANDLE) {
            e->inputs = parsed.inputs_read;
            e->bad = 0;
            s_vs_compiled++;
        }
    }
    memo.valid = 1;
    memo.dwords = v->prog_dwords;
    memo.start = v->start;
    memo.e = e->bad ? NULL : e;
    return memo.e;
}

/* What each stage's sampler is declared as in a fragment shader: 0 none,
 * 1 sampler2D, 2 samplerCube, 3 sampler3D. */
typedef struct { uint8_t decl[4]; } FsDecl;

#define FS_MAX 512
typedef struct {
    uint32_t          hash;
    int               bad;
    NV2ACombinerState st;
    FsDecl            d;
    VkShaderModule    fn;
} FsEntry;
static FsEntry *s_fs;
static int      s_fs_n;

static uint32_t fnv(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint32_t h = 0x811c9dc5u;
    while (n--) {
        h ^= *b++;
        h *= 0x01000193u;
    }
    return h;
}

static const FsDecl s_basic_decl = { { 1, 0, 0, 0 } };
static FsDecl s_fs_decl;   /* the selected shader's */

static VkShaderModule fs_basic(void)
{
    static int tried;
    static VkShaderModule fn;
    if (!nv2a_snap_2d_on(nv2a_host_render_scale()))
        return s_fs_basic;
    if (!tried) {
        tried = 1;
        fn = compile_glsl(NV2A_GLSL_FS_BASIC_SNAP, 1, "fs_basic_snap");
    }
    return fn ? fn : s_fs_basic;
}

static int s_fs_comb, s_fs_w;

static VkShaderModule fs_select_body(const struct nv2a_pb_vsh *v, int w);

static VkShaderModule fs_select(const struct nv2a_pb_vsh *v, int w)
{
    VkShaderModule fn = fs_select_body(v, w);
    s_fs_comb = fn != s_fs_basic && fn != fs_basic();
    s_fs_w = w && s_fs_comb;
    if (!s_fs_comb)
        s_fs_decl = s_basic_decl;
    return fn;
}

static VkShaderModule fs_select_body(const struct nv2a_pb_vsh *v, int w)
{
    static char src[65536];
    static struct { int valid; uint32_t r[38]; VkShaderModule fn; FsDecl d; } memo;
    NV2ACombinerState st;
    uint32_t key[38], modes = 0, inputs = 0, hash;
    FsEntry *e = NULL;
    char what[32];
    int n;

    if (!(v->rc_set && (v->rc_ctl & 0xFF)))
        return fs_basic();
    nv2a_ps_stage_key(s_stage, v->other_stage_input, v->dot_rgbmapping, &modes, &inputs);
    memcpy(key, v->rc_cicw, 32);
    memcpy(key + 8, v->rc_aicw, 32);
    memcpy(key + 16, v->rc_cocw, 32);
    memcpy(key + 24, v->rc_aocw, 32);
    key[32] = v->rc_ctl;
    key[33] = v->rc_fcw0;
    key[34] = v->rc_fcw1;
    key[35] = modes;
    key[36] = (nv2a_host_render_scale() > 1) | (nv2a_host_opts()->aspect_w ? 2u : 0u)
            | (nv2a_snap_2d_on(nv2a_host_render_scale()) ? 4u : 0u)
            | (w ? 8u : 0u);
    key[37] = inputs;
    if (memo.valid && !memcmp(memo.r, key, sizeof key)) {
        s_fs_decl = memo.d;
        return memo.fn;
    }
    memset(&st, 0, sizeof st);
    d3d8_combiners_from_regs(v->rc_cicw, v->rc_aicw, v->rc_cocw, v->rc_aocw,
                             v->rc_ctl, v->rc_fcw0, v->rc_fcw1, modes, &st);
    st.fog_input = 1;
    st.rect_snap = (int)(key[36] & 1u);
    st.rt_offset = (int)(key[36] >> 1 & 1u);
    st.snap_2d = (int)(key[36] >> 2 & 1u);
    st.w_depth = (int)(key[36] >> 3 & 1u);
    for (n = 0; n < 4; n++) {
        st.tex_input[n] = (int)((inputs >> (4 * n)) & 0xF);
        st.dot_map[n] = n ? (int)((inputs >> (16 + 4 * (n - 1))) & 0xF) : 0;
        st.tex_cube[n] = (int)((inputs >> (28 + n)) & 1u);
    }
    st.bump_on = 1;
    hash = fnv(&st, sizeof st);
    if (!s_fs && !(s_fs = (FsEntry *)calloc(FS_MAX, sizeof *s_fs)))
        return fs_basic();
    for (n = 0; n < s_fs_n; n++)
        if (s_fs[n].hash == hash && !memcmp(&s_fs[n].st, &st, sizeof st)) {
            e = &s_fs[n];
            break;
        }
    if (!e) {
        if (s_fs_n == FS_MAX)
            return fs_basic();
        e = &s_fs[s_fs_n++];
        e->hash = hash;
        e->st = st;
        e->bad = 1;
        /* The declarations d3d8_combiners_generate_glsl makes. */
        for (n = 0; n < NV2A_MAX_TEXTURES; n++) {
            int m = (int)st.tex_mode[n];
            e->d.decl[n] = 0;
            if (m == NV2A_TEXMODE_NONE || m == NV2A_TEXMODE_PASSTHRU
                    || m == NV2A_TEXMODE_CLIPPLANE
                    || (NV2A_TEXMODE_DOT_ON(&st, n) && !nv2a_texmode_samples(m)))
                continue;
            e->d.decl[n] = m == NV2A_TEXMODE_3D ? 3
                         : nv2a_texmode_cube_decl(m, st.bump_on && st.tex_cube[n]) ? 2 : 1;
        }
        snprintf(what, sizeof what, "combiners %08X", hash);
        if (d3d8_combiners_generate_glsl(&st, src, (int)sizeof src) <= 0) {
            s_compile_errors++;
            fprintf(stderr, LOGP "%s: no GLSL\n", what);
        } else if ((e->fn = compile_glsl(src, 1, what)) != VK_NULL_HANDLE) {
            e->bad = 0;
            s_fs_compiled++;
        }
    }
    memcpy(memo.r, key, sizeof key);
    memo.valid = 1;
    memo.fn = e->bad ? fs_basic() : e->fn;
    memo.d = e->bad ? s_basic_decl : e->d;
    s_fs_decl = memo.d;
    return memo.fn;
}

/* ---- render-to-texture (nv2a_pb_metal.m's rt_texture) ---------------------------- */

static RenderTarget s_self;

static RenderTarget *rt_texture(const Stage *t, RenderTarget *cur)
{
    static int off = -1;
    RenderTarget *self = NULL;
    int i;

    if (off < 0)
        off = recomp_env(RENV_METAL_NO_RTT) != NULL;
    if (off || !t->offset)
        return NULL;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex.img || ((rt->addr ^ t->offset) & 0x07FFFFFFu))
            continue;
        if (!nv2a_rt_tex_reads(t->color, t->pitch, rt->fmt, rt->pitch))
            continue;
        if (rt_drop_if_stale(rt, "bind"))
            continue;
        if (rt == cur) {
            self = rt;
            continue;
        }
        return rt;
    }
    if (!self)
        return NULL;
    if (!s_self.tex.img || s_self.w != self->w || s_self.h != self->h
            || s_self.hw != self->hw || s_self.hh != self->hh) {
        img_release(&s_self.tex);
        if (!rt_image_new(&s_self.tex, self->hw, self->hh))
            return NULL;
        s_self.w = self->w;
        s_self.h = self->h;
        s_self.hw = self->hw;
        s_self.hh = self->hh;
    }
    copy_image(&self->tex, 0, 0, &s_self.tex, 0, 0, self->hw, self->hh);
    s_self.offset = self->offset;
    s_self.pad = self->pad;
    s_self.n = self->n;
    s_self_copies++;
    {
        static int said;
        if (!said++)
            fprintf(stderr, LOGP "surface 0x%08X sampled while drawn into: bound as a copy\n",
                    self->offset);
    }
    return &s_self;
}

/* ---- draws (nv2a_pb_metal.m's metal_on_draw) ------------------------------------- */

static int aspect_copy_tracks(const RenderTarget *src, const Stage *t, unsigned k,
                              const uint32_t (*prog)[4], uint32_t start, float (*c)[4],
                              const float *vf, unsigned nf4, const int *slot, uint32_t count)
{
    float xy[NV2A_ASPECT_UV_MAX][2], uv[NV2A_ASPECT_UV_MAX][2];
    uint32_t n;
    int norm;
    if (!src->copy || !src->pad || src->copy_flip != (uint32_t)s_presents)
        return 0;
    n = nv2a_aspect_uv_sample(prog, start, c, vf, nf4, slot, count, k, xy, uv,
                              NV2A_ASPECT_UV_MAX);
    norm = nv2a_tex_size_from_format(t->color);
    return nv2a_aspect_uv_tracks((const float (*)[2])xy, (const float (*)[2])uv, n,
                                 norm ? (float)t->width : 1.0f, norm ? (float)t->height : 1.0f,
                                 src->w, src->h);
}

/* A uniform block from the constant ring. */
static VkDeviceSize ub_put(const void *p, size_t n)
{
    VkDeviceSize off = 0;
    void *d = ring_alloc(&s_ub, n, s_ubo_align, &off);
    if (d)
        memcpy(d, p, n);
    return off;
}

static VkSampler s_smp_cur[4];

static void vk_on_draw(const struct nv2a_pb_draw *d)
{
    static uint32_t *idx;
    static uint32_t idx_cap;
    static float (*hxy)[2];
    static uint32_t hxy_cap;
    const struct nv2a_pb_gpu *g = nv2a_pb_gpu_state();
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    enum nv2a_topology topo;
    RenderTarget *rt;
    VkPipeline pso;
    VkShaderModule vfn, ffn;
    struct nv2a_vp_consts vc;
    VkPsConsts mpc;
    VkDeviceSize vb_off, ib_off, vbytes, c_off = 0, vc_off, pc_off, wd_off = 0;
    uint32_t lo = 0xFFFF, hi = 0, count, nidx, i, k, nf4 = 0;
    int slot[NV2A_VS_MAX_INPUTS];
    float *vf;
    void *ip;
    int cull = 0, cull_all = 0, ptopo;
    unsigned bump_src;
    int full_w = 0, guest_tex = 0, flat_rt = 0, two_d = 0, have_x = 0, tris = 0, flat = 0;
    int padded_rt = 0, lift = 0, wide = 0, vary_z = 0, scan;
    int place = NV2A_PLACE_CENTRE;
    float lo_x = 0, hi_x = 0;
    const RenderTarget *srcs[4] = { NULL, NULL, NULL, NULL };
    VsEntry *ve = NULL;
    int use_s = v->stencil_enable && ((g->format >> 4) & 0xF) == 2;
    int use_z = v->depth_enable || use_s;
    Img *stex[4];
    uint64_t dss;

    if (!init())
        return;
    if (nv2a_batch_skip(d, g, v))
        return;
    if (g->color_keep == 0xFFFFFFFFu && !(v->depth_enable && v->depth_mask) && !use_s
            && !s_occ_q)
        return;
    if (!d->idx_count) {
        s_draws_skipped++;
        return;
    }
    if (d->program) {
        VsEntry *e = vs_cache();
        s_draws_prog += d->batches;
        if (!e) {
            s_draws_skipped++;
            return;
        }
        ve = e;
        vfn = e->fn;
        for (i = 0; i < NV2A_VS_MAX_INPUTS; i++)
            if (e->inputs & (1u << i))
                slot[nf4++] = (int)i;
    } else {
        vfn = s_vs_pass;
        for (k = 0; k < VTX_F4; k++)
            slot[nf4++] = s_vtx_attr[k];
    }
    rt = surface();
    if (!rt) {
        s_draws_skipped++;
        return;
    }
    build_stages();
    stages_cube_rt();
    bump_src = nv2a_bump_sources(s_stage, v->rc_set && (v->rc_ctl & 0xFF), v->other_stage_input);
    nv2a_ps_consts_fill(v, s_stage, &mpc.b);
    for (k = 0; k < 4; k++) {
        const Stage *t = &s_stage[k];
        float b = t->levels > 1 && filter_mip(t->filter) ? lod_bias(t->filter) : 0.0f;
        mpc.tex_bias[k] = b > 15.99f ? 15.99f : b < -16.0f ? -16.0f : b;
    }
    ffn = fs_select(v, nv2a_w_depth_on(v, d->program != 0, v->rc_set && (v->rc_ctl & 0xFF)));
    if (!ffn) {
        s_draws_skipped++;
        return;
    }
    for (i = 0; i < d->idx_count; i++) {
        if (d->idx[i] < lo) lo = d->idx[i];
        if (d->idx[i] > hi) hi = d->idx[i];
    }
    count = hi - lo + 1;
    if (d->idx_count * 3u > idx_cap) {
        uint32_t *grown = (uint32_t *)realloc(idx, (size_t)d->idx_count * 12u);
        if (!grown) {
            s_draws_skipped++;
            return;
        }
        idx = grown;
        idx_cap = d->idx_count * 3u;
    }
    nidx = nv2a_prim_to_list(d->prim, d->idx, d->idx_count, lo, idx, &topo);
    if (!nidx) {
        s_draws_skipped++;
        return;
    }
    ptopo = topo == NV2A_TOPO_POINTS ? 2 : topo == NV2A_TOPO_LINES ? 1 : 0;
    if (v->cull_enable) {
        if (v->cull_face == 0x405)      cull = 1;
        else if (v->cull_face == 0x404) cull = 2;
        else if (v->cull_face == 0x408) cull_all = 1;
    }
    if (cull_all && ptopo == 0) {
        s_draws_culled++;
        return;
    }

    vbytes = (VkDeviceSize)count * nf4 * 16u;
    if (vbytes < 16)
        vbytes = 16;
    /* Room for this draw's constants first: their ring's overflow flush
     * would reset the vertex ring under the vertices placed below. */
    if (s_ub.pos + 4u * s_ubo_align + 8192u > s_ub.size)
        flush(1);
    vf = (float *)ring_alloc(&s_vb, vbytes, s_ssbo_align, &vb_off);
    {
        uint64_t fl = s_flushes;
        ip = vf ? ring_alloc(&s_ib, (VkDeviceSize)nidx * 4u, 16, &ib_off) : NULL;
        if (vf && ip && fl != s_flushes)
            vf = (float *)ring_alloc(&s_vb, vbytes, s_ssbo_align, &vb_off);
    }
    if (!vf || !ip) {
        s_draws_skipped++;
        return;
    }
    for (k = 0; k < nf4; k++) {
        const VertexAttr *a = &d->attr[slot[k]];
        int on = a->size != 0;
        for (i = 0; i < count; i++) {
            float *o = vf + ((size_t)i * nf4 + k) * 4;
            if (!on || !fetch(d, a, lo + i, o))
                memcpy(o, v->inl[slot[k]], 16);
        }
    }
    memcpy(ip, idx, (size_t)nidx * 4u);

    /* Hor+ and render.snap_2d's 2D test, as the Metal backend. */
    full_w = 0;
    scan = rt->pad || nv2a_snap_2d_on(rt->n);
    tris = rt->pad && topo == NV2A_TOPO_TRIANGLES;
    if (tris && count > hxy_cap) {
        float (*grown)[2] = (float (*)[2])realloc(hxy, (size_t)count * sizeof *hxy);
        if (grown) {
            hxy = grown;
            hxy_cap = count;
        } else {
            tris = 0;
        }
    }
    if (rt->pad && !d->program && nf4 > 0 && s_vtx_attr[0] == 0 && count) {
        lo_x = hi_x = vf[0];
        for (i = 0; i < count; i++) {
            float x = vf[(size_t)i * nf4 * 4u];
            if (x < lo_x) lo_x = x;
            if (x > hi_x) hi_x = x;
            if (tris) {
                hxy[i][0] = x;
                hxy[i][1] = vf[(size_t)i * nf4 * 4u + 1];
            }
        }
        two_d = have_x = 1;
    } else if (!rt->pad && scan && !d->program) {
        two_d = 1;
    } else if (scan && d->program && ve && count) {
        two_d = have_x = nv2a_aspect_scan(&ve->hp, (const uint32_t (*)[4])v->prog, v->start,
                                          (float (*)[4])(uintptr_t)v->c, vf, nf4, slot,
                                          count, NULL, &lo_x, &hi_x);
        vary_z = have_x && ve->hp.vary_z;
        if (have_x && tris && nv2a_aspect_full_width(lo_x, hi_x, rt->w))
            nv2a_aspect_scan(&ve->hp, (const uint32_t (*)[4])v->prog, v->start,
                             (float (*)[4])(uintptr_t)v->c, vf, nf4, slot, count, hxy,
                             &lo_x, &hi_x);
    }
    if (have_x)
        full_w = nv2a_aspect_full(lo_x, hi_x, tris ? (const float (*)[2])hxy : NULL,
                                  tris ? idx : NULL, nidx / 3u, rt->w, rt->h, &wide);

    if (use_z) {
        DepthTarget *dt = depth_target(rt->w, rt->h);
        if (dt && rt->dt != dt) {
            rt->dt = dt;
            if (s_enc_rt == rt)
                end_encoder();
        }
    }
    {
        int textured = 0;
        for (k = 0; k < 4; k++) {
            const Stage *t = &s_stage[k];
            RenderTarget *src;
            stex[k] = &s_black;
            s_stage_pack[k][0] = s_stage_pack[k][1] = 1;
            s_stage_hash[k] = 0;
            if (!t->mode || t->mode == 4)
                continue;
            textured = 1;
            if (s_fs_comb && !nv2a_texmode_samples((int)nv2a_stage_ps_mode(t, k))) {
                continue;
            } else if (s_fs_comb && nv2a_stage_cube_bind(t, nv2a_stage_ps_mode(t, k))) {
                stex[k] = tex_bind(t, 1);
                stage_from_cache((int)k);
            } else if ((src = rt_texture(t, rt)) != NULL) {
                stex[k] = &src->tex;
                nv2a_ps_consts_rt(&mpc.b, k, t, src->w, src->h);
                if (nv2a_rt_no_alpha(nv2a_tex_linear_twin(t->color)))
                    mpc.b.alpha_only[k] = 2;
                srcs[k] = src;
                if (!src->pad)
                    flat_rt = 1;
                else
                    padded_rt = 1;
                s_rtt_binds++;
                nv2a_tex_dump_rt(t);
            } else {
                stex[k] = tex_bind(t, 0);
                stage_from_cache((int)k);
                if (nv2a_aspect_picture(t->width, rt->w, bump_src >> k & 1))
                    guest_tex = 1;
            }
        }
        if (textured)
            s_draws_tex++;
    }
    if (rt->pad && d->program && ve && topo == NV2A_TOPO_TRIANGLES && !two_d) {
        uint32_t flips = nv2a_pb_gpu_state()->flips;
        unsigned bars = nv2a_aspect_bars(rt->bar_flip, flips);
        int quad = nv2a_aspect_quad_ok(srcs[0] || srcs[1] || srcs[2] || srcs[3], flat_rt,
                                       guest_tex);
        int band = bars && !v->depth_enable && !flat_rt && !guest_tex;
        float qxy[NV2A_ASPECT_QUAD_MAX][2];
        if ((quad || band)
            && nv2a_aspect_screen_quad((const uint32_t (*)[4])v->prog, v->start,
                                       (float (*)[4])(uintptr_t)v->c, vf, nf4, slot, count,
                                       qxy, &lo_x, &hi_x)) {
            if (quad && nv2a_aspect_quad_span(lo_x, hi_x, rt->w)
                && nv2a_aspect_full_tris((const float (*)[2])qxy, idx, nidx / 3u, rt->w)) {
                two_d = full_w = 1;
                nv2a_aspect_bar_mark(rt->bar_flip,
                                     nv2a_aspect_third((const float (*)[2])qxy, count, rt->h),
                                     flips);
            } else if (band && nv2a_aspect_band((const float (*)[2])qxy, count, rt->w, rt->h,
                                                bars)) {
                two_d = full_w = 1;
            }
        }
    }
    flat = 0;
    if (rt->pad && d->program && ve && !two_d && !v->depth_enable && count) {
        float fw;
        flat = nv2a_aspect_flat((const uint32_t (*)[4])v->prog, v->start,
                                (float (*)[4])(uintptr_t)v->c, vf, nf4, slot, count, &fw);
    }
    place = rt->pad ? nv2a_aspect_place(two_d, full_w,
                                        nv2a_aspect_place_tex(guest_tex, padded_rt), flat_rt,
                                        rt->w == rt->h ? NV2A_MOVIE_BLACK
                                                       : nv2a_host_opts()->movie_mode)
                    : NV2A_PLACE_CENTRE;
    if (rt->pad && two_d && place == NV2A_PLACE_CENTRE) {
        int copy_rt = 0;
        for (k = 0; k < 4 && !copy_rt; k++)
            copy_rt = srcs[k]
                      && aspect_copy_tracks(srcs[k], &s_stage[k], k,
                                            d->program ? (const uint32_t (*)[4])v->prog : NULL,
                                            v->start, (float (*)[4])(uintptr_t)v->c, vf, nf4,
                                            slot, count);
        place = nv2a_aspect_place_copy(place, two_d, copy_rt);
    }
    if (nv2a_snap_2d_on(rt->n)) {
        unsigned rt_mask = 0;
        for (k = 0; k < 4; k++)
            if (srcs[k])
                rt_mask |= 1u << k;
        nv2a_ps_consts_snap(&mpc.b, s_stage, rt_mask, rt->n,
                            nv2a_snap_flat2d(two_d && !vary_z, place));
    }
    if (place != NV2A_PLACE_CENTRE)
        s_draws_stretch++;
    lift = rt->pad && nv2a_aspect_hud_lift(rt->w, rt->h, two_d, flat, full_w,
                                           have_x ? lo_x : 0.0f, have_x ? hi_x : (float)rt->w);
    (void)wide;
    for (k = 0; k < 4; k++) {
        const RenderTarget *src = srcs[k];
        float sc, off;
        if (!src || !src->pad)
            continue;
        if (place != NV2A_PLACE_STRETCH && place != NV2A_PLACE_COVER)
            nv2a_rt_guest_read(src->w, src->pad, &sc, &off);
        else if (!nv2a_rt_stretch_read(src->w, src->h, src->pad, &sc, &off))
            continue;
        mpc.b.tex_scale[k][0] *= sc;
        mpc.b.tex_scale[k][3] = off;
    }
    nv2a_vp_consts_fill(v, g->format, rt->w, rt->h, rt->pad, &vc);
    nv2a_aspect_map(&vc, rt->w, rt->h, rt->pad, place);
    nv2a_vp_tie_fill(&vc, rt->hw, rt->hh);
    vc.u[7] = s_fs_w ? 1u : 0u;

    /* The constants before the pass: a ring overflow flushes, which ends it. */
    if (d->program) {
        uint32_t cg = nv2a_pb_vsh_consts_gen();
        if (!memo_on() || !s_cmemo.ok || s_cmemo.gen != cg) {
            s_cmemo.off = ub_put(v->c, sizeof v->c);
            s_cmemo.gen = cg;
            s_cmemo.ok = 1;
        }
        c_off = s_cmemo.off;
    }
    vc_off = ub_put(&vc, sizeof vc);
    pc_off = ub_put(&mpc, sizeof mpc);
    if (s_fs_w) {
        struct nv2a_wdepth_consts wc;
        nv2a_wdepth_consts_fill(v, d->prim, g->format, nv2a_host_render_scale(), &wc);
        wd_off = ub_put(&wc, sizeof wc);
    }
    for (k = 0; k < 4; k++)
        s_smp_cur[k] = sampler(&s_stage[k], bump_src >> k & 1, s_stage_pack[k][0],
                               s_stage_pack[k][1]);

    if (!s_in_pass || s_enc_rt != rt)
        begin_encoder(rt, 0, NULL);
    dss = s_enc_dt ? (use_z ? depth_key(v->depth_enable, use_s) : depth_key(0, 0)) : 0;
    pso = pipeline(vfn, ffn, ptopo, cull, v->front_face != 0x900, dss);
    if (!pso || !s_in_pass) {
        s_draws_skipped++;
        return;
    }
    vkCmdBindPipeline(s_cur->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pso);
    if (s_enc_dt) {
        vkCmdSetStencilReference(s_cur->cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                 use_s ? (v->stencil_ref & 0xFFu) : 0u);
        if (use_z)
            s_draws_z++;
    }
    {
        VkViewport vp;
        VkRect2D sr;
        uint32_t b[4], hb[4];
        /* Negative height: clip space y up, as Metal's. */
        vp.x = 0.0f;
        vp.y = (float)rt->hh;
        vp.width = (float)rt->hw;
        vp.height = -(float)rt->hh;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(s_cur->cmd, 0, 1, &vp);
        if (!nv2a_draw_box(g, rt->w, rt->h, b)) {
            s_draws_skipped++;
            return;
        }
        nv2a_aspect_box(b, rt->w, rt->h, rt->pad, rt->n, place, two_d || flat, lift, hb);
        sr.offset.x = (int32_t)hb[0];
        sr.offset.y = (int32_t)hb[1];
        sr.extent.width = hb[2] - hb[0];
        sr.extent.height = hb[3] - hb[1];
        vkCmdSetScissor(s_cur->cmd, 0, 1, &sr);
    }
    {
        float zb = 0.0f, zs = 0.0f, bf[4];
        if (!s_fs_w)
            nv2a_depth_bias(v, d->prim, g->format, nv2a_host_render_scale(), &zb, &zs);
        vkCmdSetDepthBias(s_cur->cmd, zb, 0.0f, zs);
        (void)nv2a_blend_constant(g->blend_enable, g->blend_sfactor, g->blend_dfactor,
                                  g->blend_color, bf, &(int){0});
        vkCmdSetBlendConstants(s_cur->cmd, bf);
    }
    {
        VkDescriptorBufferInfo bi[5];
        VkDescriptorImageInfo ii[4];
        VkWriteDescriptorSet w[10];
        uint32_t nw = 0;
        memset(w, 0, sizeof w);
        memset(bi, 0, sizeof bi);
#define WBUF(bind, type, buf_, off_, range_) do { \
        VkDescriptorBufferInfo *b_ = &bi[nw < 5 ? nw : 4]; \
        b_->buffer = (buf_); b_->offset = (off_); b_->range = (range_); \
        w[nw].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; \
        w[nw].dstBinding = (bind); w[nw].descriptorCount = 1; \
        w[nw].descriptorType = (type); w[nw].pBufferInfo = b_; nw++; } while (0)
        WBUF(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, s_vb.buf, vb_off, vbytes);
        if (d->program)
            WBUF(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, s_ub.buf, c_off, sizeof v->c);
        WBUF(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, s_ub.buf, vc_off, sizeof vc);
        WBUF(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, s_ub.buf, pc_off, sizeof mpc);
        if (s_fs_w)
            WBUF(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, s_ub.buf, wd_off,
                 sizeof(struct nv2a_wdepth_consts));
#undef WBUF
        for (k = 0; k < 4; k++) {
            int dcl = s_fs_decl.decl[k];
            Img *im = stex[k];
            if (!dcl)
                continue;
            /* The view's type must be the declaration's. */
            if (dcl == 2 && im->vtype != VK_IMAGE_VIEW_TYPE_CUBE)
                im = &s_black_cube;
            else if (dcl == 3)
                im = &s_black_3d;
            else if (dcl == 1 && im->vtype != VK_IMAGE_VIEW_TYPE_2D)
                im = &s_black;
            ii[k].sampler = s_smp_cur[k];
            ii[k].imageView = im->view;
            ii[k].imageLayout = (im == &s_self.tex || (im >= &s_rt[0].tex && im <= &s_rt[RT_MAX - 1].tex))
                                ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            w[nw].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[nw].dstBinding = 8 + k;
            w[nw].descriptorCount = 1;
            w[nw].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[nw].pImageInfo = &ii[k];
            nw++;
        }
        if (s_push_desc) {
            s_push_desc(s_cur->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_playout, 0, nw, w);
        } else {
            VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            VkDescriptorSet set;
            uint32_t j;
            ai.descriptorPool = s_cur->dpool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &s_dsl;
            if (vkAllocateDescriptorSets(s_dev, &ai, &set) != VK_SUCCESS) {
                /* the slot's pool is spent: this draw is dropped, and the
                 * next one starts a fresh slot */
                s_draws_skipped++;
                flush(0);
                return;
            }
            for (j = 0; j < nw; j++)
                w[j].dstSet = set;
            vkUpdateDescriptorSets(s_dev, nw, w, 0, NULL);
            vkCmdBindDescriptorSets(s_cur->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_playout, 0,
                                    1, &set, 0, NULL);
        }
    }
    if (nv2a_tex_dump_on()) {
        int dump_2d = !d->program;
        if (d->program && nv2a_tex_dump_draw_open(s_stage_hash))
            dump_2d = nv2a_aspect_batch_2d((const uint32_t (*)[4])v->prog, v->start,
                                           (float (*)[4])(uintptr_t)v->c, vf, nf4, slot,
                                           count, v->depth_enable);
        nv2a_tex_dump_draw(s_stage_hash, bump_src, dump_2d);
    }
    vkCmdBindIndexBuffer(s_cur->cmd, s_ib.buf, ib_off, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(s_cur->cmd, nidx, 1, 0, 0, 0);
    s_draws += d->batches;
}

/* ---- flip ------------------------------------------------------------------------ */

static RenderTarget *present_target(uint32_t surface_offset)
{
    const struct nv2a_pb_present *p = nv2a_pb_present_state();
    int i;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex.img && surface_offset && rt->offset == surface_offset
                && (!p->offset || rt->pitch == p->pitch)) {
            rt->presented = 1;
            return rt;
        }
    }
    return NULL;
}

static void present_extent(RenderTarget *rt, uint32_t *w, uint32_t *h)
{
    const struct nv2a_pb_present *p = nv2a_pb_present_state();
    if (p->offset && p->offset == rt->offset && p->pitch == rt->pitch && p->w && p->h
            && (p->w > rt->w || p->h > rt->h))
        rt_grow(rt, rt->w > p->w ? rt->w : p->w, rt->h > p->h ? rt->h : p->h);
    *w = rt->w;
    *h = rt->h;
    if (p->offset && p->offset == rt->offset && p->pitch == rt->pitch && p->w && p->h) {
        if (p->w < *w) *w = p->w;
        if (p->h < *h) *h = p->h;
    }
}

static void pack16(const RenderTarget *rt, const uint32_t *row, uint16_t *o, uint32_t w)
{
    uint32_t f = rt->fmt & 0xFu, x;
    if (f == 0x1 || f == 0x2 || f == 0x3) {
        nv2a_surf_pack_row(f, (const uint8_t *)row, w, (uint8_t *)o);
        return;
    }
    for (x = 0; x < w; x++) {
        uint32_t c = row[x];
        o[x] = (uint16_t)(((c >> 8) & 0xF800) | ((c >> 5) & 0x07E0) | ((c >> 3) & 0x001F));
    }
}

static void fix32(const RenderTarget *rt, uint8_t *dst, uint32_t pitch, uint32_t h)
{
    uint32_t f = rt->fmt & 0xFu, y;
    if (f < 0x4 || f > 0x7)
        return;
    for (y = 0; y < h; y++)
        nv2a_surf_pack_row(f, dst + (size_t)y * pitch, rt->w, dst + (size_t)y * pitch);
}

/* rt's pixels into the guest surface at addr: the host pixels px (hw x hh,
 * tightly packed BGRA8), box-filtered down at render.scale > 1, packed as
 * the surface's format. */
static void write_back_px(RenderTarget *rt, const uint8_t *px, uint32_t addr, uint32_t pitch)
{
    uint8_t *dst = (uint8_t *)xbox_GetMemoryOffset() + addr;
    uint32_t bpp = pitch / rt->w, w = rt->w, n = rt->n, y;
    static uint32_t *row;
    static uint32_t row_cap;

    px += (size_t)rt->pad * n * 4u;
    if (bpp >= 4) {
        if (rt->hw == rt->w) {
            for (y = 0; y < rt->h; y++)
                memcpy(dst + (size_t)y * pitch, px + (size_t)y * rt->hw * 4u, (size_t)w * 4u);
        } else {
            nv2a_downscale_box32(px, rt->hw * 4u, w, rt->h, n, dst, pitch);
        }
        fix32(rt, dst, pitch, rt->h);
    } else if (bpp == 2) {
        if (row_cap < w) {
            uint32_t *nr = (uint32_t *)realloc(row, (size_t)w * 4);
            if (!nr)
                return;
            row = nr;
            row_cap = w;
        }
        for (y = 0; y < rt->h; y++) {
            uint16_t *o = (uint16_t *)(dst + (size_t)y * pitch);
            if (rt->hw == rt->w)
                memcpy(row, px + (size_t)y * rt->hw * 4u, (size_t)w * 4u);
            else
                nv2a_downscale_box32(px + (size_t)y * n * rt->hw * 4u, rt->hw * 4u, w, 1, n,
                                     (uint8_t *)row, w * 4u);
            pack16(rt, row, o, w);
        }
    } else {
        return;
    }
    tex_cache_stale(addr, pitch * rt->h);
    s_writebacks++;
}

static void rt_sync(RenderTarget *rt, uint32_t pitch)
{
    const uint8_t *px;
    if (!rt->tex.img || !rt->dirty || !pitch || pitch < rt->w * 2u)
        return;
    px = img_readback(&rt->tex, 0, 0, rt->hw, rt->hh);
    if (px)
        write_back_px(rt, px, rt->addr, pitch);
    rt->dirty = 0;
    rt_own_reset(rt);
}

static void vk_sync_guest(uint32_t offset, uint32_t pitch, uint32_t w, uint32_t h)
{
    RenderTarget *rt = NULL;
    uint32_t addr = nv2a_pb_dma_resolve(offset) & 0x07FFFFFFu;
    int i;
    if (s_failed)
        return;
    for (i = 0; i < RT_MAX && !rt; i++)
        if (s_rt[i].tex.img && s_rt[i].offset == offset && s_rt[i].w >= w && s_rt[i].h >= h)
            rt = &s_rt[i];
    if (!rt)
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *c = &s_rt[i];
            if (c->tex.img && (c->addr & 0x07FFFFFFu) == addr
                    && (!rt || c->last_use > rt->last_use))
                rt = c;
        }
    if (rt && !rt_drop_if_stale(rt, "sync"))
        rt_sync(rt, pitch);
}

static int wb_always(void)
{
    const char *e = recomp_env(RENV_METAL_WRITEBACK);
    return (e && !strcasecmp(e, "always")) || recomp_env(RENV_METAL_NO_RTT) != NULL;
}

/* ---- visibility tests: occlusion queries, a slot per pass (Metal's design) ---- */

#define OCC_QSLOTS 16
#define OCC_POOL   (NV2A_OCC_PENDING * 3 + NV2A_OCC_SEG)
typedef struct OccQuery {
    int      slot[OCC_QSLOTS];
    uint64_t cmd[OCC_QSLOTS];   /* the serial of the slot's command buffer */
    int      n, ovf;
    struct OccQuery *next;
} OccQuery;
static OccQuery        s_qp[OCC_POOL];
static OccQuery       *s_qfree;
static int             s_qmade;
static unsigned char   s_vis_used[VIS_SLOTS];
static int             s_vis_next;
static int             s_retire[VIS_SLOTS], s_nretire;
static struct nv2a_occ s_occ;
static int             s_occ_mode = -1;
static uint64_t        s_occ_ovf;

static int occ_mode(void)
{
    if (s_occ_mode < 0) {
        const char *e = recomp_env(RENV_METAL_OCC);
        s_occ_mode = !e ? 0 : !strcasecmp(e, "sync") ? 1 : !strcasecmp(e, "fixed") ? 2 : 0;
    }
    return s_occ_mode;
}

/* A free query slot, reset in the upload buffer (outside any pass, ahead
 * of the frame's commands). */
static int vis_alloc(void)
{
    int i, k;
    for (k = 0; k < VIS_SLOTS; k++) {
        i = (s_vis_next + k) % VIS_SLOTS;
        if (!s_vis_used[i]) {
            s_vis_used[i] = 1;
            s_vis_next = (i + 1) % VIS_SLOTS;
            vkCmdResetQueryPool(up_cmd(), s_qpool, (uint32_t)i, 1);
            return i;
        }
    }
    return -1;
}

static void occ_pass_begin(void)
{
    OccQuery *q = s_occ_q;
    int i;
    if (!q || !s_in_pass || s_occ_slot >= 0)
        return;
    if (q->n >= OCC_QSLOTS || (i = vis_alloc()) < 0) {
        q->ovf = 1;
        s_occ_ovf++;
        return;
    }
    q->slot[q->n] = i;
    q->cmd[q->n] = s_cur->serial;
    q->n++;
    vkCmdBeginQuery(s_cur->cmd, s_qpool, (uint32_t)i,
                    s_precise_occ ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
    s_occ_slot = i;
}

static void occ_pass_end(void)
{
    if (s_occ_slot >= 0 && s_in_pass)
        vkCmdEndQuery(s_cur->cmd, s_qpool, (uint32_t)s_occ_slot);
    s_occ_slot = -1;
}

static uint64_t occ_now_us(void *ctx)
{
    struct timespec ts;
    (void)ctx;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

static void *occ_get(void *ctx)
{
    OccQuery *q;
    (void)ctx;
    if (s_qfree) {
        q = s_qfree;
        s_qfree = q->next;
    } else if (s_qmade < OCC_POOL) {
        q = &s_qp[s_qmade++];
    } else {
        return NULL;
    }
    q->n = 0;
    q->ovf = 0;
    return q;
}

static void occ_put(void *ctx, void *qv)
{
    OccQuery *q = (OccQuery *)qv;
    int i;
    (void)ctx;
    if (!q)
        return;
    if (q == s_occ_q) {
        occ_pass_end();
        s_occ_q = NULL;
    }
    for (i = 0; i < q->n; i++) {
        if (serial_done(q->cmd[i]))
            s_vis_used[q->slot[i]] = 0;
        else
            s_retire[s_nretire++] = q->slot[i];
    }
    q->n = 0;
    q->next = s_qfree;
    s_qfree = q;
}

static void occ_begin(void *ctx, void *qv)
{
    (void)ctx;
    occ_pass_end();
    s_occ_q = (OccQuery *)qv;
    occ_pass_begin();
}

static void occ_end(void *ctx, void *qv)
{
    (void)ctx;
    if (s_occ_q == (OccQuery *)qv) {
        occ_pass_end();
        s_occ_q = NULL;
    }
}

static int occ_result(void *ctx, void *qv, int flush_gpu, uint64_t *count)
{
    OccQuery *q = (OccQuery *)qv;
    uint64_t sum = 0, r;
    int i;
    (void)ctx;
    for (i = 0; i < q->n; i++) {
        if (s_cur && q->cmd[i] == s_cur->serial) {
            if (!flush_gpu)
                return 0;
            flush(0);
        }
        if (!serial_done(q->cmd[i]))
            return 0;
    }
    for (i = 0; i < q->n; i++) {
        r = 0;
        if (vkGetQueryPoolResults(s_dev, s_qpool, (uint32_t)q->slot[i], 1, sizeof r, &r,
                                  sizeof r, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            sum += r;
    }
    *count = q->ovf ? NV2A_OCC_VISIBLE : sum;
    return 1;
}

static const struct nv2a_occ_ops s_occ_ops = {
    occ_get, occ_put, occ_begin, occ_end, occ_result, occ_now_us, NULL
};

static uint64_t occ_env_us(recomp_env_id id, uint64_t def)
{
    long v = recomp_env_int(id, (long)def);
    return v > 0 ? (uint64_t)v : 0;
}

static int occ_ready_on(void)
{
    if (occ_mode() == 2 || !init())
        return 0;
    if (!s_occ.ops) {
        nv2a_occ_init(&s_occ, &s_occ_ops, (uint8_t *)xbox_GetMemoryOffset());
        nv2a_occ_set_delay(&s_occ, occ_env_us(RENV_OCC_DELAY_US, 0));
        nv2a_occ_set_settle(&s_occ, occ_env_us(RENV_OCC_SETTLE_US, NV2A_OCC_SETTLE_US));
        nv2a_occ_set_trace(&s_occ, recomp_env(RENV_ZPASS_TRACE) != NULL);
    }
    return 1;
}

static void vk_on_zpass(int enable)
{
    if (occ_ready_on())
        nv2a_occ_zpass(&s_occ, enable);
}

static void vk_on_zpass_clear(void)
{
    if (occ_ready_on())
        nv2a_occ_zpass_clear(&s_occ);
}

static int report_in_cmd(uint32_t va)
{
    const struct nv2a_occ_report *p;
    int k, i;
    if (!s_cur || !s_occ.npend || s_occ.pend[s_occ.npend - 1].va != va)
        return 0;
    p = &s_occ.pend[s_occ.npend - 1];
    for (k = 0; k < p->n; k++) {
        const OccQuery *q = (const OccQuery *)p->q[k];
        for (i = 0; i < q->n; i++)
            if (q->cmd[i] == s_cur->serial)
                return 1;
    }
    return 0;
}

static int vk_on_report(uint32_t va)
{
    if (!occ_ready_on())
        return 0;
    nv2a_occ_report(&s_occ, va, occ_mode() == 1);
    if (occ_mode() == 0 && s_cur && report_in_cmd(va)) {
        s_report_commits++;
        flush(0);
    }
    return 1;
}

static void occ_settle(int flip)
{
    if (!s_occ.ops)
        return;
    if (flip)
        nv2a_occ_flip(&s_occ);
    else
        nv2a_occ_settle(&s_occ);
}

static void vk_on_poll(void)
{
    if (s_occ.npend || s_occ.nret)
        occ_settle(0);
}

static void occ_after_wait(void)
{
    while (s_nretire)
        s_vis_used[s_retire[--s_nretire]] = 0;
    if (s_occ.ops)
        nv2a_occ_poll(&s_occ, 0);
}

/* ---- flip ---------------------------------------------------------------------- */

static uint32_t s_shown_w, s_shown_h;
static uint64_t s_window_presents;

/* In an event, not a menu (nv2a_vk_in_event): Burnout 3's races draw
 * 650-1250 depth-tested batches a frame, its menus 0-300. On above 500,
 * off below 300, each held 30 flips, so neither a busy menu nor a quiet
 * moment of a race flips it. */
static int s_in_event;

int nv2a_vk_in_event(void) { return __atomic_load_n(&s_in_event, __ATOMIC_RELAXED); }

static void event_track(void)
{
    static uint64_t last_z;
    static int run;
    uint64_t z = s_draws_z - last_z;
    int want = s_in_event ? z >= 300 : z > 500;
    last_z = s_draws_z;
    if (want == s_in_event) {
        run = 0;
        return;
    }
    if (++run >= 30) {
        run = 0;
        __atomic_store_n(&s_in_event, want, __ATOMIC_RELAXED);
        fprintf(stderr, LOGP "%s (%llu depth draws a frame), flip %llu\n",
                want ? "in an event" : "in the menus", (unsigned long long)z,
                (unsigned long long)s_presents);
    }
}

/* The VR runtime's copy of the latest frame (nv2a_vk_frame). */
static Img      s_xr_img;
static uint64_t s_xr_gen;

int nv2a_vk_frame(VkImage *img, uint32_t *w, uint32_t *h, uint64_t *gen)
{
    if (!s_xr_img.img || !s_xr_gen)
        return 0;
    *img = s_xr_img.img;
    *w = s_xr_img.w;
    *h = s_xr_img.h;
    *gen = s_xr_gen;
    return 1;
}

/* ---- stereo from depth (in-world mode) ----------------------------------------
 *
 * The frame and its depth buffer make a side-by-side pair, left eye then
 * right, each pixel shifted by half its parallax: s * (1 - (1 - d) / (1 - c))
 * host pixels, d the stored depth (z / zmax), c the depth that sits on the
 * screen plane. For a D3D projection 1 - d is about near / w, so that is
 * s * (1 - w_c / w): 0 at the screen plane, s at infinity, negative nearer.
 * A gather: each output pixel reads the depth under itself, then the colour
 * that far across. RECOMP_XR_STEREO=s,c sets them (default 2% of the width,
 * 0.98). */
static const char s_stereo_glsl[] =
    "#version 450\n"
    "layout(local_size_x = 8, local_size_y = 8) in;\n"
    "layout(set = 0, binding = 0) uniform sampler2D col;\n"
    "layout(set = 0, binding = 1) uniform sampler2D dep;\n"
    "layout(set = 0, binding = 2, rgba8) writeonly uniform image2D outi;\n"
    "layout(push_constant) uniform P { vec4 a; } pc;   /* s, c, w, h */\n"
    "void main() {\n"
    "    ivec2 g = ivec2(gl_GlobalInvocationID.xy);\n"
    "    int w = int(pc.a.z), h = int(pc.a.w);\n"
    "    if (g.x >= 2 * w || g.y >= h) return;\n"
    "    float eye = g.x < w ? -1.0 : 1.0;\n"
    "    int x = g.x < w ? g.x : g.x - w;\n"
    "    float d = texelFetch(dep, ivec2(x, g.y), 0).r;\n"
    "    float k = (1.0 - d) / max(1.0 - pc.a.y, 1e-5);\n"
    "    float p = clamp(pc.a.x * (1.0 - k), -2.0 * pc.a.x, pc.a.x);\n"
    "    int sx = clamp(int(float(x) - eye * 0.5 * p + 0.5), 0, w - 1);\n"
    "    imageStore(outi, g, texelFetch(col, ivec2(sx, g.y), 0));\n"
    "}\n";

static VkPipeline s_st_pso;
static VkPipelineLayout s_st_layout;
static VkDescriptorSetLayout s_st_dsl;
static VkDescriptorPool s_st_pool;
static VkDescriptorSet s_st_set;
static VkSampler s_st_smp;
static Img      s_xr_sbs;        /* RGBA8, 2w x h: left | right */
static Img      s_xr_sbs_srgb;   /* its bytes, tagged sRGB (see xr_frame_copy) */
static uint64_t s_xr_sbs_gen;
static float    s_st_s = -1.0f, s_st_c = 0.98f;

int nv2a_vk_frame_stereo(VkImage *img, uint32_t *w, uint32_t *h, uint64_t *gen)
{
    if (!s_xr_sbs_srgb.img || !s_xr_sbs_gen)
        return 0;
    *img = s_xr_sbs_srgb.img;
    *w = s_xr_sbs.w;
    *h = s_xr_sbs.h;
    *gen = s_xr_sbs_gen;
    return 1;
}

static int stereo_init(void)
{
    static int tried;
    VkDescriptorSetLayoutBinding b[3];
    VkDescriptorSetLayoutCreateInfo dl = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    VkPushConstantRange pr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo pl = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
                                   { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 } };
    VkDescriptorPoolCreateInfo dp = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    VkDescriptorSetAllocateInfo da = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    VkSamplerCreateInfo sc = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    shaderc_compilation_result_t r;
    VkShaderModuleCreateInfo mi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkShaderModule m = VK_NULL_HANDLE;
    const char *e;
    int i;

    if (tried)
        return s_st_pso != VK_NULL_HANDLE;
    tried = 1;
    e = recomp_env(RENV_XR_STEREO);
    if (e)
        sscanf(e, "%f,%f", &s_st_s, &s_st_c);
    memset(b, 0, sizeof b);
    for (i = 0; i < 3; i++) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                    : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    dl.bindingCount = 3;
    dl.pBindings = b;
    if (vkCreateDescriptorSetLayout(s_dev, &dl, NULL, &s_st_dsl) != VK_SUCCESS)
        return 0;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &s_st_dsl;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(s_dev, &pl, NULL, &s_st_layout) != VK_SUCCESS)
        return 0;
    dp.maxSets = 1;
    dp.poolSizeCount = 2;
    dp.pPoolSizes = ps;
    if (vkCreateDescriptorPool(s_dev, &dp, NULL, &s_st_pool) != VK_SUCCESS)
        return 0;
    da.descriptorPool = s_st_pool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &s_st_dsl;
    if (vkAllocateDescriptorSets(s_dev, &da, &s_st_set) != VK_SUCCESS)
        return 0;
    sc.magFilter = sc.minFilter = VK_FILTER_NEAREST;
    sc.addressModeU = sc.addressModeV = sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(s_dev, &sc, NULL, &s_st_smp) != VK_SUCCESS)
        return 0;
    r = shaderc_compile_into_spv(s_cc, s_stereo_glsl, strlen(s_stereo_glsl),
                                 shaderc_compute_shader, "stereo", "main", s_copt);
    if (shaderc_result_get_compilation_status(r) != shaderc_compilation_status_success) {
        fprintf(stderr, LOGP "stereo shader: %s\n", shaderc_result_get_error_message(r));
        shaderc_result_release(r);
        return 0;
    }
    mi.codeSize = shaderc_result_get_length(r);
    mi.pCode = (const uint32_t *)shaderc_result_get_bytes(r);
    vkCreateShaderModule(s_dev, &mi, NULL, &m);
    shaderc_result_release(r);
    if (!m)
        return 0;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m;
    ci.stage.pName = "main";
    ci.layout = s_st_layout;
    if (vkCreateComputePipelines(s_dev, s_pcache, 1, &ci, NULL, &s_st_pso) != VK_SUCCESS)
        s_st_pso = VK_NULL_HANDLE;
    vkDestroyShaderModule(s_dev, m, NULL);
    fprintf(stderr, LOGP "stereo from depth: %s\n", s_st_pso ? "ready" : "failed");
    return s_st_pso != VK_NULL_HANDLE;
}

/* The pair from rt's frame (w x h host pixels) and its depth buffer. */
static void stereo_pass(RenderTarget *rt, uint32_t w, uint32_t h)
{
    VkDescriptorImageInfo ii[3];
    VkWriteDescriptorSet wr[3];
    float pc[4];
    int i;

    if (!rt->dt || !rt->dt->dview || !stereo_init())
        return;
    if (!s_xr_sbs.img || s_xr_sbs.w != 2u * w || s_xr_sbs.h != h) {
        if (s_xr_sbs.img) {
            nv2a_vk_queue_lock();
            vkQueueWaitIdle(s_queue);
            nv2a_vk_queue_unlock();
            img_destroy_now(&s_xr_sbs);
            img_destroy_now(&s_xr_sbs_srgb);
        }
        if (!img_make(&s_xr_sbs, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
                      VK_FORMAT_R8G8B8A8_UNORM, 2u * w, h, 1, 1, 1,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT))
            return;
        img_layout(up_cmd(), &s_xr_sbs, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        if (!img_make(&s_xr_sbs_srgb, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
                      VK_FORMAT_R8G8B8A8_SRGB, 2u * w, h, 1, 1, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT)) {
            img_destroy_now(&s_xr_sbs);
            return;
        }
        img_layout(up_cmd(), &s_xr_sbs_srgb, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_GENERAL);
    }
    memset(ii, 0, sizeof ii);
    memset(wr, 0, sizeof wr);
    ii[0].sampler = ii[1].sampler = s_st_smp;
    ii[0].imageView = rt->tex.view;
    ii[1].imageView = rt->dt->dview;
    ii[2].imageView = s_xr_sbs.view;
    for (i = 0; i < 3; i++) {
        ii[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = s_st_set;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                     : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(s_dev, 3, wr, 0, NULL);
    pc[0] = s_st_s >= 0.0f ? s_st_s : 0.02f * (float)w;
    pc[1] = s_st_c;
    pc[2] = (float)w;
    pc[3] = (float)h;
    end_encoder();
    ensure_cmd();
    barrier_all(s_cur->cmd);
    vkCmdBindPipeline(s_cur->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_st_pso);
    vkCmdBindDescriptorSets(s_cur->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_st_layout, 0, 1,
                            &s_st_set, 0, NULL);
    vkCmdPushConstants(s_cur->cmd, s_st_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(s_cur->cmd, (2u * w + 7u) / 8u, (h + 7u) / 8u, 1);
    barrier_all(s_cur->cmd);
    copy_image(&s_xr_sbs, 0, 0, &s_xr_sbs_srgb, 0, 0, 2u * w, h);
}

/* At the flip, before its wait: the frame's host pixels into s_xr_img. */
static void xr_frame_copy(RenderTarget *rt, uint32_t w, uint32_t h)
{
    if (!xr_on() || !rt || !w || !h)
        return;
    nv2a_vk_frame_lock();
    if (!s_xr_img.img || s_xr_img.w != w || s_xr_img.h != h) {
        if (s_xr_img.img) {
            /* the runtime may be reading it: let the queue drain */
            nv2a_vk_queue_lock();
            vkQueueWaitIdle(s_queue);
            nv2a_vk_queue_unlock();
            img_destroy_now(&s_xr_img);
        }
        /* sRGB-tagged: the runtime then shows the bytes as they are
         * (display values), where a UNORM image is taken as linear light
         * and comes out washed out. The copy moves the bytes unchanged. */
        if (!img_make(&s_xr_img, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D,
                      VK_FORMAT_B8G8R8A8_SRGB, w, h, 1, 1, 1,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT)) {
            nv2a_vk_frame_unlock();
            return;
        }
        img_layout(up_cmd(), &s_xr_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    }
    copy_image(&rt->tex, 0, 0, &s_xr_img, 0, 0, w, h);
    stereo_pass(rt, w, h);
    nv2a_vk_frame_unlock();
}

/* RECOMP_DEBUG=metal_host_dump=<prefix>: the frame as presented, host
 * size, every fb_dump_flips flip (20), <prefix>host_NNNNN.bmp, as the Metal
 * backend's host_dump. */
static int host_dump_due(void)
{
    static const char *prefix;
    static int every = -1;
    if (every < 0) {
        const char *e = recomp_env(RENV_FB_DUMP_FLIPS);
        prefix = recomp_env(RENV_METAL_HOST_DUMP);
        every = e ? atoi(e) : 0;
        if (every <= 0)
            every = 20;
    }
    return prefix && ((s_presents + 1) % (uint64_t)every) == 0;
}

static void host_dump(const uint8_t *px, uint32_t stride, uint32_t w, uint32_t h)
{
    char path[1024];
    uint8_t hdr[54];
    uint32_t row = w * 3u, padb = (4u - (row & 3u)) & 3u, size, y, x;
    FILE *f;
    snprintf(path, sizeof path, "%shost_%05llu.bmp", recomp_env(RENV_METAL_HOST_DUMP),
             (unsigned long long)(s_presents + 1));
    if (!(f = fopen(path, "wb")))
        return;
    size = 54u + (row + padb) * h;
    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &size, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1; hdr[28] = 24;
    fwrite(hdr, 1, sizeof hdr, f);
    for (y = h; y-- > 0; ) {
        const uint8_t *sp = px + (size_t)y * stride;
        for (x = 0; x < w; x++)
            fwrite(sp + (size_t)x * 4u, 1, 3, f);
        if (padb)
            fwrite(hdr + 6, 1, padb, f);
    }
    fclose(f);
}

/* ---- the window's swapchain (Android) ------------------------------------------
 *
 * With the presenter's Vulkan window (fb_present_sdl.c, Android) the frame
 * goes to the screen from here: at the flip the target is blitted into the
 * swapchain image, letterboxed at its own shape, the touch controls drawn
 * over it (touch_pad_shapes), and the image presented. No readback, no SDL
 * renderer, and no wait for the GPU at each flip: the readback path held
 * the guest for every frame. A lost or out-of-date swapchain is made again
 * at a later flip (Android takes the surface away while the app is paused). */
typedef struct { float x, y, r, c[4]; int fill; } VkTouchShape;
int touch_pad_shapes(VkTouchShape *out, int max, int w, int h);

static VkSwapchainKHR s_swap;
static VkImage        s_sw_img[8];
static VkImageView    s_sw_view[8];
static VkFramebuffer  s_sw_fb[8];      /* render pass objects only */
static uint32_t       s_sw_n;
static VkExtent2D     s_sw_ext;
static VkFormat       s_sw_fmt;
static VkSemaphore    s_sw_acq[SLOTS], s_sw_done[8];
static VkPipeline     s_ov_pso;
static VkPipelineLayout s_ov_layout;
static VkRenderPass   s_ov_rp;
static uint64_t       s_sw_retry, s_sw_presents;

static const char s_ov_vs[] =
    "#version 450\n"
    "layout(push_constant) uniform P { vec4 c; vec4 col; vec4 scr; } p;\n"
    "layout(location = 0) out vec2 v_pos;\n"
    "void main() {\n"
    "    vec2 k = vec2((gl_VertexIndex & 1) != 0 ? 1.0 : -1.0,\n"
    "                  (gl_VertexIndex == 2 || gl_VertexIndex >= 4) ? 1.0 : -1.0);\n"
    "    if (gl_VertexIndex == 3) k = vec2(1.0, -1.0);\n"
    "    if (gl_VertexIndex == 4) k = vec2(-1.0, 1.0);\n"
    "    if (gl_VertexIndex == 5) k = vec2(1.0, 1.0);\n"
    "    v_pos = p.c.xy + k * (p.c.z + 2.0);\n"
    "    gl_Position = vec4(v_pos / p.scr.xy * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";
static const char s_ov_fs[] =
    "#version 450\n"
    "layout(push_constant) uniform P { vec4 c; vec4 col; vec4 scr; } p;\n"
    "layout(location = 0) in vec2 v_pos;\n"
    "layout(location = 0) out vec4 o;\n"
    "void main() {\n"
    "    float d = length(v_pos - p.c.xy);\n"
    "    float a = p.c.w != 0.0 ? clamp(p.c.z - d + 0.5, 0.0, 1.0)\n"
    "                           : clamp(1.5 - abs(d - p.c.z), 0.0, 1.0);\n"
    "    o = vec4(p.col.rgb, p.col.a * a);\n"
    "}\n";

static void sw_destroy(void)
{
    uint32_t i;
    if (!s_swap)
        return;
    nv2a_vk_queue_lock();
    vkDeviceWaitIdle(s_dev);
    nv2a_vk_queue_unlock();
    for (i = 0; i < s_sw_n; i++) {
        if (s_sw_fb[i]) vkDestroyFramebuffer(s_dev, s_sw_fb[i], NULL);
        if (s_sw_view[i]) vkDestroyImageView(s_dev, s_sw_view[i], NULL);
        s_sw_fb[i] = VK_NULL_HANDLE;
        s_sw_view[i] = VK_NULL_HANDLE;
    }
    vkDestroySwapchainKHR(s_dev, s_swap, NULL);
    s_swap = VK_NULL_HANDLE;
    s_sw_n = 0;
}

static VkShaderModule sw_module(const char *src, shaderc_shader_kind kind)
{
    shaderc_compilation_result_t r = shaderc_compile_into_spv(s_cc, src, strlen(src), kind,
                                                              "overlay", "main", s_copt);
    VkShaderModuleCreateInfo mi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    VkShaderModule m = VK_NULL_HANDLE;
    if (shaderc_result_get_compilation_status(r) == shaderc_compilation_status_success) {
        mi.codeSize = shaderc_result_get_length(r);
        mi.pCode = (const uint32_t *)shaderc_result_get_bytes(r);
        vkCreateShaderModule(s_dev, &mi, NULL, &m);
    } else {
        fprintf(stderr, LOGP "overlay shader: %s\n", shaderc_result_get_error_message(r));
    }
    shaderc_result_release(r);
    return m;
}

/* The touch overlay's pipeline for the swapchain's format. */
static int ov_init(void)
{
    VkShaderModule vs, fs;
    VkPushConstantRange pr = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 48 };
    VkPipelineLayoutCreateInfo pl = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    VkPipelineShaderStageCreateInfo st[2] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
    VkPipelineVertexInputStateCreateInfo vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    VkPipelineViewportStateCreateInfo vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    VkPipelineColorBlendAttachmentState cb;
    VkPipelineColorBlendStateCreateInfo cbs = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    static const VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    VkPipelineRenderingCreateInfoKHR rci = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
    VkGraphicsPipelineCreateInfo pci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };

    if (s_ov_pso)
        return 1;
    vs = sw_module(s_ov_vs, shaderc_vertex_shader);
    fs = sw_module(s_ov_fs, shaderc_fragment_shader);
    if (!vs || !fs)
        return 0;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pr;
    if (!s_ov_layout && vkCreatePipelineLayout(s_dev, &pl, NULL, &s_ov_layout) != VK_SUCCESS)
        return 0;
    if (s_legacy && !s_ov_rp) {
        VkAttachmentDescription a;
        VkAttachmentReference r = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkSubpassDescription sp;
        VkRenderPassCreateInfo rc = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        memset(&a, 0, sizeof a);
        a.format = s_sw_fmt;
        a.samples = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout = a.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        memset(&sp, 0, sizeof sp);
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1;
        sp.pColorAttachments = &r;
        rc.attachmentCount = 1;
        rc.pAttachments = &a;
        rc.subpassCount = 1;
        rc.pSubpasses = &sp;
        vkCreateRenderPass(s_dev, &rc, NULL, &s_ov_rp);
    }
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    st[0].module = vs;
    st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    st[1].module = fs;
    st[1].pName = "main";
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    vp.viewportCount = vp.scissorCount = 1;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.0f;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    memset(&cb, 0, sizeof cb);
    cb.blendEnable = VK_TRUE;
    cb.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cb.colorWriteMask = 0xF;
    cbs.attachmentCount = 1;
    cbs.pAttachments = &cb;
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyn;
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &s_sw_fmt;
    pci.stageCount = 2;
    pci.pStages = st;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &cbs;
    pci.pDynamicState = &dy;
    pci.layout = s_ov_layout;
    if (s_legacy)
        pci.renderPass = s_ov_rp;
    else
        pci.pNext = &rci;
    if (vkCreateGraphicsPipelines(s_dev, s_pcache, 1, &pci, NULL, &s_ov_pso) != VK_SUCCESS)
        s_ov_pso = VK_NULL_HANDLE;
    vkDestroyShaderModule(s_dev, vs, NULL);
    vkDestroyShaderModule(s_dev, fs, NULL);
    return s_ov_pso != VK_NULL_HANDLE;
}

static int sw_create(void)
{
    VkSurfaceCapabilitiesKHR caps;
    VkSurfaceFormatKHR fmts[32];
    VkSwapchainCreateInfoKHR ci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    VkSemaphoreCreateInfo si = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    uint32_t nf = 32, i;
    int w = 0, h = 0;

    if (!s_surf)
        return 0;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(s_pdev, s_surf, &caps) != VK_SUCCESS)
        return 0;
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        return 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(s_pdev, s_surf, &nf, fmts);
    s_sw_fmt = VK_FORMAT_UNDEFINED;
    for (i = 0; i < nf && !s_sw_fmt; i++)
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM)
            s_sw_fmt = fmts[i].format;
    for (i = 0; i < nf && !s_sw_fmt; i++)
        if (fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM)
            s_sw_fmt = fmts[i].format;
    if (!s_sw_fmt)
        return 0;
    s_sw_ext = caps.currentExtent;
    if (s_sw_ext.width == 0xFFFFFFFFu) {
        SDL_GetWindowSizeInPixels(s_win, &w, &h);
        s_sw_ext.width = (uint32_t)w;
        s_sw_ext.height = (uint32_t)h;
    }
    if (!s_sw_ext.width || !s_sw_ext.height)
        return 0;
    ci.surface = s_surf;
    ci.minImageCount = caps.minImageCount < 3 ? 3 : caps.minImageCount;
    if (caps.maxImageCount && ci.minImageCount > caps.maxImageCount)
        ci.minImageCount = caps.maxImageCount;
    ci.imageFormat = s_sw_fmt;
    ci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    ci.imageExtent = s_sw_ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    /* The surface's own transform: the compositor does no rotation. */
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                        ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                        : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;
    if (vkCreateSwapchainKHR(s_dev, &ci, NULL, &s_swap) != VK_SUCCESS) {
        s_swap = VK_NULL_HANDLE;
        return 0;
    }
    s_sw_n = 8;
    vkGetSwapchainImagesKHR(s_dev, s_swap, &s_sw_n, s_sw_img);
    for (i = 0; i < s_sw_n; i++) {
        VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vi.image = s_sw_img[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = s_sw_fmt;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        vkCreateImageView(s_dev, &vi, NULL, &s_sw_view[i]);
        if (!s_sw_done[i])
            vkCreateSemaphore(s_dev, &si, NULL, &s_sw_done[i]);
    }
    for (i = 0; i < SLOTS; i++)
        if (!s_sw_acq[i])
            vkCreateSemaphore(s_dev, &si, NULL, &s_sw_acq[i]);
    if (ov_init() && s_legacy)
        for (i = 0; i < s_sw_n; i++) {
            VkFramebufferCreateInfo fi = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
            fi.renderPass = s_ov_rp;
            fi.attachmentCount = 1;
            fi.pAttachments = &s_sw_view[i];
            fi.width = s_sw_ext.width;
            fi.height = s_sw_ext.height;
            fi.layers = 1;
            vkCreateFramebuffer(s_dev, &fi, NULL, &s_sw_fb[i]);
        }
    fprintf(stderr, LOGP "swapchain %ux%u, %u images, format %d, transform 0x%X\n",
            s_sw_ext.width, s_sw_ext.height, s_sw_n, (int)s_sw_fmt,
            (unsigned)caps.currentTransform);
    return 1;
}

static void sw_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to)
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
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

/* rt's frame (hcw x hch host pixels) to the screen. 1 if presented. */
static int sw_present(RenderTarget *rt, uint32_t hcw, uint32_t hch)
{
    VkCommandBuffer cb;
    VkSemaphore acq;
    uint32_t idx = 0;
    VkResult r;
    VkClearColorValue black = { { 0, 0, 0, 1 } };
    VkImageSubresourceRange rr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageBlit bl;
    float sx, sy, sc;
    VkTouchShape sh[32];
    int nsh, i;
    VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };

    if (!s_surf || !rt || !hcw || !hch)
        return 0;
    if (!s_swap) {
        if (s_sw_retry && s_presents < s_sw_retry)
            return 0;
        if (!sw_create()) {
            s_sw_retry = s_presents + 60;
            return 0;
        }
    }
    end_encoder();
    ensure_cmd();
    acq = s_sw_acq[s_cur - s_slot];
    nv2a_vk_queue_lock();
    r = vkAcquireNextImageKHR(s_dev, s_swap, 100000000ull, acq, VK_NULL_HANDLE, &idx);
    nv2a_vk_queue_unlock();
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        /* out of date, the surface gone (paused), or a timeout */
        if (r != VK_TIMEOUT && r != VK_NOT_READY) {
            sw_destroy();
            if (r == VK_ERROR_SURFACE_LOST_KHR) {
                vkDestroySurfaceKHR(s_inst, s_surf, NULL);
                if (!SDL_Vulkan_CreateSurface(s_win, s_inst, NULL, &s_surf))
                    s_surf = VK_NULL_HANDLE;
            }
            s_sw_retry = s_presents + 10;
        }
        return 0;
    }
    cb = s_cur->cmd;
    barrier_all(cb);
    sw_barrier(cb, s_sw_img[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdClearColorImage(cb, s_sw_img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr);
    /* letterboxed: the frame's shape, as large as the screen takes */
    sx = (float)s_sw_ext.width / (float)hcw;
    sy = (float)s_sw_ext.height / (float)hch;
    sc = sx < sy ? sx : sy;
    memset(&bl, 0, sizeof bl);
    bl.srcSubresource.aspectMask = bl.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bl.srcSubresource.layerCount = bl.dstSubresource.layerCount = 1;
    bl.srcOffsets[1].x = (int32_t)hcw;
    bl.srcOffsets[1].y = (int32_t)hch;
    bl.srcOffsets[1].z = 1;
    bl.dstOffsets[0].x = (int32_t)(((float)s_sw_ext.width - hcw * sc) * 0.5f);
    bl.dstOffsets[0].y = (int32_t)(((float)s_sw_ext.height - hch * sc) * 0.5f);
    bl.dstOffsets[1].x = bl.dstOffsets[0].x + (int32_t)(hcw * sc);
    bl.dstOffsets[1].y = bl.dstOffsets[0].y + (int32_t)(hch * sc);
    bl.dstOffsets[1].z = 1;
    vkCmdBlitImage(cb, rt->tex.img, VK_IMAGE_LAYOUT_GENERAL, s_sw_img[idx],
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    sw_barrier(cb, s_sw_img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    nsh = touch_pad_shapes(sh, 32, (int)s_sw_ext.width, (int)s_sw_ext.height);
    if (nsh && s_ov_pso) {
        VkViewport vp = { 0, 0, (float)s_sw_ext.width, (float)s_sw_ext.height, 0, 1 };
        VkRect2D sci = { { 0, 0 }, s_sw_ext };
        if (s_legacy) {
            VkRenderPassBeginInfo bi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            bi.renderPass = s_ov_rp;
            bi.framebuffer = s_sw_fb[idx];
            bi.renderArea = sci;
            vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
        } else {
            VkRenderingAttachmentInfoKHR ca = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR };
            VkRenderingInfoKHR ri = { VK_STRUCTURE_TYPE_RENDERING_INFO_KHR };
            ca.imageView = s_sw_view[idx];
            ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            ri.renderArea = sci;
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &ca;
            s_begin_rendering(cb, &ri);
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s_ov_pso);
        vkCmdSetViewport(cb, 0, 1, &vp);
        vkCmdSetScissor(cb, 0, 1, &sci);
        for (i = 0; i < nsh; i++) {
            float pc[12] = { sh[i].x, sh[i].y, sh[i].r, sh[i].fill ? 1.0f : 0.0f,
                             sh[i].c[0], sh[i].c[1], sh[i].c[2], sh[i].c[3],
                             (float)s_sw_ext.width, (float)s_sw_ext.height, 0, 0 };
            vkCmdPushConstants(cb, s_ov_layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof pc, pc);
            vkCmdDraw(cb, 6, 1, 0, 0);
        }
        if (s_legacy)
            vkCmdEndRenderPass(cb);
        else
            s_end_rendering(cb);
    }
    sw_barrier(cb, s_sw_img[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    s_sub_wait = acq;
    s_sub_signal = s_sw_done[idx];
    flush(0);
    s_sub_wait = s_sub_signal = VK_NULL_HANDLE;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &s_sw_done[idx];
    pi.swapchainCount = 1;
    pi.pSwapchains = &s_swap;
    pi.pImageIndices = &idx;
    nv2a_vk_queue_lock();
    r = vkQueuePresentKHR(s_queue, &pi);
    nv2a_vk_queue_unlock();
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) {
        sw_destroy();
        s_sw_retry = s_presents + 10;
    }
    s_sw_presents++;
    return 1;
}

static void vk_flip(uint32_t surface_offset, uint32_t pitch)
{
    RenderTarget *rt = NULL;
    uint32_t addr = surface_offset ? nv2a_pb_dma_resolve(surface_offset) : 0;
    const uint8_t *px = NULL;
    int always = wb_always(), win = 0, wb = 0;
    uint32_t cw = 0, ch = 0, hcw = 0, hch = 0;

    if (init()) {
        rt = present_target(surface_offset);
        if (rt) {
            nv2a_aspect_frame_publish(rt->w, rt->h);
            present_extent(rt, &cw, &ch);
            hcw = (cw + 2u * rt->pad) * rt->n;
            hch = ch * rt->n;
            s_shown_w = cw;
            s_shown_h = ch;
        }
        if (rt)
            xr_frame_copy(rt, hcw, hch);
        win = rt && (xbox_FramebufferWindowRunning() || host_dump_due());
        wb = rt && addr && pitch && pitch >= rt->w * 2u && always;
        /* The window's swapchain: the frame goes from the GPU to the screen,
         * and the flip waits for nothing (no readback, no flush(1)). */
        if (rt && s_surf && !wb && !host_dump_due() && sw_present(rt, hcw, hch)) {
            occ_settle(1);
            s_presents++;
            if (s_xr && s_xr_img.img)
                __atomic_store_n(&s_xr_gen, s_xr_gen + 1, __ATOMIC_RELEASE);
            if (s_presents == 1 || (s_presents % 300) == 0)
                fprintf(stderr, LOGP "present #%llu: swapchain (%llu window presents),"
                        " %llu draws, %llu texture uploads, %llu flushes\n",
                        (unsigned long long)s_presents, (unsigned long long)s_sw_presents,
                        (unsigned long long)s_draws, (unsigned long long)s_tex_uploads,
                        (unsigned long long)s_flushes);
            return;
        }
        /* One readback of the whole target serves the window and the
         * write-back; it waits for the frame. */
        if (rt && (win || wb))
            px = img_readback(&rt->tex, 0, 0, rt->hw, rt->hh);
        else
            flush(1);
        occ_settle(1);
        if (s_xr && rt && s_xr_img.img)
            __atomic_store_n(&s_xr_gen, s_xr_gen + 1, __ATOMIC_RELEASE);
        if (s_xr && rt && s_xr_sbs.img)
            __atomic_store_n(&s_xr_sbs_gen, s_xr_sbs_gen + 1, __ATOMIC_RELEASE);
        if (px && host_dump_due())
            host_dump(px, rt->hw * 4u, hcw, hch);
        if (!s_xr && s_xr_sbs.img && host_dump_due()) {
            /* the stereo test: the pair beside the frame, RGBA as BGRA */
            const uint8_t *sp = img_readback(&s_xr_sbs, 0, 0, s_xr_sbs.w, s_xr_sbs.h);
            if (sp) {
                static uint8_t *bg;
                size_t n = (size_t)s_xr_sbs.w * s_xr_sbs.h, k;
                if ((bg = (uint8_t *)realloc(bg, n * 4u)) != NULL) {
                    for (k = 0; k < n; k++) {
                        bg[k * 4] = sp[k * 4 + 2];
                        bg[k * 4 + 1] = sp[k * 4 + 1];
                        bg[k * 4 + 2] = sp[k * 4];
                        bg[k * 4 + 3] = sp[k * 4 + 3];
                    }
                    s_presents += 500000;   /* its own file name */
                    host_dump(bg, s_xr_sbs.w * 4u, s_xr_sbs.w, s_xr_sbs.h);
                    s_presents -= 500000;
                }
            }
            /* the shared readback buffer now holds the pair: the frame again */
            px = img_readback(&rt->tex, 0, 0, rt->hw, rt->hh);
        }
        if (wb && px) {
            write_back_px(rt, px, addr, pitch);
            rt->dirty = 0;
            if (addr == rt->addr && pitch == rt->pitch)
                rt_own_reset(rt);
        }
    }
    s_presents++;
    if (xr_on())
        event_track();
    if (s_presents == 1 || (s_presents % 300) == 0)
        fprintf(stderr, LOGP "present #%llu: surface 0x%08X (%s), %llu clears, %llu draws"
                " (%llu vertex-program), %llu skipped, %llu culled, %llu texture uploads,"
                " %llu rehashes, %llu write-backs, %llu decode syncs, %llu stale drops,"
                " %llu grows, %llu partial clears; shaders %llu vs + %llu fs, %llu compile"
                " errors; %llu z-clears, %llu depth draws, %llu rt binds (%llu self copies);"
                " %llu window presents; %llu flushes, %llu at GET_REPORT\n",
                (unsigned long long)s_presents, surface_offset, rt ? "drawn here" : "guest bytes",
                (unsigned long long)s_clears, (unsigned long long)s_draws,
                (unsigned long long)s_draws_prog, (unsigned long long)s_draws_skipped,
                (unsigned long long)s_draws_culled, (unsigned long long)s_tex_uploads,
                (unsigned long long)s_tex_rehash, (unsigned long long)s_writebacks,
                (unsigned long long)s_decode_syncs, (unsigned long long)s_stale_drops,
                (unsigned long long)s_grows, (unsigned long long)s_partial_clears,
                (unsigned long long)s_vs_compiled, (unsigned long long)s_fs_compiled,
                (unsigned long long)s_compile_errors, (unsigned long long)s_zclears,
                (unsigned long long)s_draws_z, (unsigned long long)s_rtt_binds,
                (unsigned long long)s_self_copies, (unsigned long long)s_window_presents,
                (unsigned long long)s_flushes, (unsigned long long)s_report_commits);
    nv2a_tex_dump_flip(s_presents);
    if (!pitch || !addr)
        return;
    xbox_FramebufferWindowSet(addr, pitch);
    if (rt && win && px) {
        /* The frame's part of the host pixels: the guest columns plus pads. */
        xbox_FramebufferWindowPresentPixels(px, hcw, hch, rt->hw * 4u, 4u);
        s_window_presents++;
        return;
    }
    xbox_FramebufferWindowPresent(addr, pitch);
}

static void scale_drop(void)
{
    int i;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (!rt->tex.img)
            continue;
        if (!rt_drop_if_stale(rt, "scale")) {
            rt_sync(rt, rt->pitch);
            rt_free(rt);
        }
    }
    for (i = 0; i < DT_MAX; i++) {
        DepthTarget *dd = &s_dt[i];
        if (!dd->tex.img)
            continue;
        if (s_enc_dt == dd)
            end_encoder();
        if (dd->dview) {
            vkDeviceWaitIdle(s_dev);
            vkDestroyImageView(s_dev, dd->dview, NULL);
        }
        img_release(&dd->tex);
        memset(dd, 0, sizeof *dd);
    }
    if (s_occ.ops) {
        occ_settle(0);
        nv2a_occ_set_scale(&s_occ, nv2a_host_render_scale());
    }
    fprintf(stderr, LOGP "render.scale=%u from this flip: targets dropped\n",
            nv2a_host_render_scale());
}

static void vk_on_flip(uint32_t surface_offset, uint32_t pitch)
{
    unsigned changed;
    vk_flip(surface_offset, pitch);
    changed = nv2a_host_opts_apply();
    if (changed & NV2A_OPT_SCALE)
        scale_drop();
}

/* ---- the 2D engine (nv2a_pb_metal.m's metal_on_blit) ------------------------------ */

static RenderTarget *blit_source(const struct nv2a_pb_blit *b)
{
    int i;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex.img && !((rt->addr ^ b->src) & 0x07FFFFFFu) && !rt_drop_if_stale(rt, "blit"))
            return rt;
    }
    return NULL;
}

static int blit_gpu(const struct nv2a_pb_blit *b, RenderTarget *src)
{
    RenderTarget *dst = NULL, *free_rt = NULL, *lru = NULL;
    uint32_t box[6];
    int i, whole;

    if (b->operation != NV09F_OP_SRCCOPY || b->src_pitch != src->pitch
            || b->dst_pitch < src->w * b->bpp || b->bpp != nv2a_surf_color_bytes(src->fmt)
            || nv2a_rt_overlap(b->src, src->pitch * src->h, b->dst, b->dst_pitch * src->h)
            || !nv2a_2d_host_box(b, src->w, src->h, src->pad, src->n, src->hw, src->hh, box))
        return 0;
    whole = !b->in_x && !b->in_y && !b->out_x && !b->out_y && b->w == src->w && b->h == src->h;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt->tex.img && !((rt->addr ^ b->dst) & 0x07FFFFFFu) && rt->w == src->w
                && rt->h == src->h && rt->hw == src->hw && rt->hh == src->hh
                && rt->pad == src->pad && rt->n == src->n && rt->pitch == b->dst_pitch
                && rt->fmt == src->fmt && !rt_drop_if_stale(rt, "blit")) {
            dst = rt;
            break;
        }
    }
    if (!dst && !whole)
        return 0;
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt == dst || rt == src || !rt->tex.img
                || !nv2a_rt_overlap(rt->addr, rt->pitch * rt->h, b->dst, b->dst_pitch * src->h))
            continue;
        if (!rt_drop_if_stale(rt, "overlap")) {
            rt_sync(rt, rt->pitch);
            rt_free(rt);
        }
    }
    if (!dst) {
        for (i = 0; i < RT_MAX; i++) {
            RenderTarget *rt = &s_rt[i];
            if (!rt->tex.img && !free_rt)
                free_rt = rt;
            if (rt != src && (!lru || rt->last_use < lru->last_use))
                lru = rt;
        }
        if (!free_rt) {
            free_rt = lru;
            if (!rt_drop_if_stale(free_rt, "evict")) {
                rt_sync(free_rt, free_rt->pitch);
                rt_free(free_rt);
            }
        }
        if (!rt_image_new(&free_rt->tex, src->hw, src->hh))
            return 0;
        dst = free_rt;
        dst->offset = b->dst_offset;
        dst->addr = b->dst;
        dst->w = src->w;
        dst->h = src->h;
        dst->hw = src->hw;
        dst->hh = src->hh;
        dst->pad = src->pad;
        dst->n = src->n;
        dst->pitch = b->dst_pitch;
        dst->fmt = src->fmt;
        fprintf(stderr, LOGP "surface 0x%08X: %ux%u render target (%ux%u host), the 2D blit"
                " copy of 0x%08X\n", dst->offset, dst->w, dst->h, dst->hw, dst->hh,
                src->offset);
    }
    copy_image(&src->tex, (int32_t)box[0], (int32_t)box[1], &dst->tex, (int32_t)box[2],
               (int32_t)box[3], box[4], box[5]);
    s_tick++;
    dst->last_use = s_tick;
    dst->dirty = 1;
    dst->copy = 1;
    dst->copy_flip = (uint32_t)s_presents;
    rt_own_reset(dst);
    return 1;
}

static void vk_on_blit(const struct nv2a_pb_blit *b)
{
    RenderTarget *src;
    uint64_t span = (uint64_t)(b->out_y + b->h) * b->dst_pitch;
    int i;

    if (!init()) {
        nv2a_pb_blit_guest(b);
        return;
    }
    src = blit_source(b);
    if (src && blit_gpu(b, src)) {
        s_blits_gpu++;
        return;
    }
    if (src)
        rt_sync(src, src->pitch);
    for (i = 0; i < RT_MAX; i++) {
        RenderTarget *rt = &s_rt[i];
        if (rt == src || !rt->tex.img
                || !nv2a_rt_overlap(rt->addr, rt->pitch * rt->h, b->dst, (uint32_t)span))
            continue;
        if (!rt_drop_if_stale(rt, "overlap")) {
            rt_sync(rt, rt->pitch);
            rt_free(rt);
        }
    }
    nv2a_pb_blit_guest(b);
    tex_cache_stale(b->dst, (uint32_t)span);
    s_blits_guest++;
}

/* ---- registration ------------------------------------------------------------------ */

static int vk_merge_placed(int program)
{
    const struct nv2a_pb_vsh *v = nv2a_pb_vsh_state();
    const VsEntry *ve;
    int n, i, rt = 0;

    if (!program || !v->depth_enable || !init())
        return 0;
    ve = vs_cache();
    if (!ve)
        return 0;
    build_stages();
    for (n = 0; n < 4 && !rt; n++) {
        const Stage *t = &s_stage[n];
        if (!t->mode || t->mode == 4 || !t->offset)
            continue;
        for (i = 0; i < RT_MAX && !rt; i++)
            rt = s_rt[i].tex.img && !((s_rt[i].addr ^ t->offset) & 0x07FFFFFFu);
    }
    return nv2a_merge_placed(program, v->depth_enable, &ve->hp, rt);
}

static const struct nv2a_pb_backend s_vk_backend = {
    vk_on_clear,
    vk_on_zclear,
    vk_on_draw,
    vk_on_flip,
    vk_on_zpass,
    vk_on_zpass_clear,
    vk_on_report,
    vk_on_poll,
    1,                  /* writes_back: dumps read guest memory, after sync_guest */
    vk_sync_guest,
    1,                  /* merge_draws */
    vk_merge_placed,
    vk_on_blit,
};

const struct nv2a_pb_backend *nv2a_pb_vulkan_backend(void)
{
    if (!s_black.img && init()) {
        solid_make(&s_black, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, 0xFF000000u);
        solid_make(&s_magenta, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, NV2A_TEX_MAGENTA);
        solid_make(&s_black_cube, VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, 0xFF000000u);
        solid_make(&s_black_3d, VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, 0xFF000000u);
    }
    return &s_vk_backend;
}

void nv2a_pb_vulkan_register_from_env(void)
{
    const char *b = recomp_env(RENV_PB_BACKEND);
    if (b && !strcasecmp(b, "vulkan")) {
        const struct nv2a_pb_backend *vb = nv2a_pb_vulkan_backend();
        if (s_failed) {
            fprintf(stderr, LOGP "RECOMP_PB_BACKEND=vulkan: no device, staying on the CPU"
                    " rasteriser\n");
            return;
        }
        nv2a_pb_set_backend(vb);
        fprintf(stderr, LOGP "RECOMP_PB_BACKEND=vulkan: executor draws through Vulkan\n");
    }
}

#endif /* XBR_HAVE_VULKAN */

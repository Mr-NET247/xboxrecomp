/* nv2a_vk_xr.h - what the Vulkan backend (nv2a_pb_vulkan.c) offers a VR
 * runtime (src/xr/xr_openxr.c): the runtime makes the instance and device
 * (XR_KHR_vulkan_enable2 wants them made through it), shares the queue under
 * a lock, and reads the latest presented frame from an image the backend
 * fills at each flip. All of it is optional: without nv2a_pb_vulkan_set_xr
 * the backend makes its own device and nothing here is used. */
#ifndef NV2A_VK_XR_H
#define NV2A_VK_XR_H

#include <stdint.h>
#include <vulkan/vulkan.h>

struct nv2a_vk_xr {
    VkResult (*create_instance)(const VkInstanceCreateInfo *ci, VkInstance *out);
    VkPhysicalDevice (*physical_device)(VkInstance inst);
    VkResult (*create_device)(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci,
                              VkDevice *out);
    /* The device is up: the runtime makes its session now. */
    void (*device_ready)(VkInstance inst, VkPhysicalDevice pd, VkDevice dev,
                         uint32_t queue_family, VkQueue queue);
};

/* Before the backend's first use (before the guest starts). */
void nv2a_pb_vulkan_set_xr(const struct nv2a_vk_xr *xr);

/* The queue is shared: every submit, and the runtime's frame calls that
 * touch it, hold this lock. */
void nv2a_vk_queue_lock(void);
void nv2a_vk_queue_unlock(void);

/* The latest frame: a BGRA8 image in GENERAL, w x h, written by the
 * backend's flip and complete by the time gen moves. Call with
 * nv2a_vk_frame_lock held while recording and submitting a read of it. */
int  nv2a_vk_frame(VkImage *img, uint32_t *w, uint32_t *h, uint64_t *gen);
/* The stereo pair (in-world mode): RGBA8, left eye then right, 2w x h,
 * in GENERAL; under the same lock. */
int  nv2a_vk_frame_stereo(VkImage *img, uint32_t *w, uint32_t *h, uint64_t *gen);
/* Nonzero while the title is in an event (a race), not its menus: the
 * backend's guess from the depth-tested draws a frame. */
int  nv2a_vk_in_event(void);
void nv2a_vk_frame_lock(void);
void nv2a_vk_frame_unlock(void);

#endif

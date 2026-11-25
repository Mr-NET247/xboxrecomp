/* xr_openxr.h - immersive VR on an OpenXR runtime (Meta Quest): the game's
 * frame on a large virtual screen in front of the player, and the motion
 * controllers as player 1's pad. */
#ifndef XR_OPENXR_H
#define XR_OPENXR_H

/* Start OpenXR before the guest: 1 when a runtime and a headset are there
 * (the Vulkan backend then makes its device through it and the controllers
 * drive port 0), 0 otherwise (a phone: the game stays a flat window).
 * RECOMP_XR=0 turns it off. */
int xbox_XrInit(void);

#endif

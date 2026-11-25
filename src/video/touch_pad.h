/* touch_pad.h - on-screen controls for a touch screen (phones, tablets):
 * player 1's pad drawn over the game in the SDL presenter window. */
#ifndef TOUCH_PAD_H
#define TOUCH_PAD_H

#include <SDL3/SDL.h>

/* On by default on iOS and Android, off elsewhere; RECOMP_TOUCH=1 or 0
 * forces it. Call once the window is up; installs the external pad. */
int  touch_pad_init(void);
/* A window event; nonzero when the controls need drawing again. */
int  touch_pad_event(const SDL_Event *ev, int out_w, int out_h);
/* Over the frame, before the present. */
void touch_pad_draw(SDL_Renderer *ren, int out_w, int out_h);

/* The controls as circles in window pixels, for a presenter that draws
 * them itself (the Vulkan swapchain): x, y, r, rgba, fill. Returns the
 * count (at most max); 0 when the controls are off. */
typedef struct { float x, y, r, c[4]; int fill; } TouchShape;
int  touch_pad_shapes(TouchShape *out, int max, int w, int h);

#endif

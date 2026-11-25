/*
 * touch_pad.c - on-screen controls (see touch_pad.h).
 *
 * Laid out for a landscape phone, as fractions of the window, so any size
 * and aspect works:
 *   left half            a floating stick: it centres where the thumb
 *                        lands, and the drag is the left stick
 *   bottom, right of it  LT (brake) and RT (accelerate), large pads
 *   right edge           A B X Y, Xbox placed and coloured (Burnout 3:
 *                        A boost, X look behind, Y camera)
 *   top centre           BACK and START
 * Each finger keeps the control it went down on until it lifts. The pad
 * reaches the game as the external pad on port 0 (xbox_InputSetExtPad),
 * merged with any controller.
 */
#include "touch_pad.h"
#include "host_pad.h"
#include "xinput_xbox.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

enum { C_NONE, C_STICK, C_A, C_B, C_X, C_Y, C_LT, C_RT, C_START, C_BACK, C_N };

typedef struct { float x, y, r; Uint8 cr, cg, cb; } Ctl;   /* fractions of h for r */

/* Centres in fractions of the window; radii in fractions of its height. */
static const Ctl s_ctl[C_N] = {
    [C_A]     = { 0.88f, 0.80f, 0.075f,  90, 200,  60 },
    [C_B]     = { 0.95f, 0.62f, 0.075f, 220,  60,  60 },
    [C_X]     = { 0.81f, 0.62f, 0.075f,  60, 120, 230 },
    [C_Y]     = { 0.88f, 0.44f, 0.075f, 230, 200,  50 },
    [C_LT]    = { 0.56f, 0.84f, 0.11f,  180, 180, 180 },
    [C_RT]    = { 0.70f, 0.80f, 0.13f,  230, 230, 230 },
    [C_START] = { 0.56f, 0.07f, 0.05f,  230, 230, 230 },
    [C_BACK]  = { 0.44f, 0.07f, 0.05f,  230, 230, 230 },
};

#define FINGERS 10
static struct { SDL_FingerID id; int ctl; float ox, oy, x, y; } s_f[FINGERS];
static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;
static int s_on;

static int touch_ext_pad(XBOX_INPUT_STATE *st)
{
    static DWORD packet;
    XBOX_INPUT_STATE p;
    int i;
    memset(&p, 0, sizeof p);
    pthread_mutex_lock(&s_lock);
    for (i = 0; i < FINGERS; i++) {
        float dx, dy, m;
        switch (s_f[i].ctl) {
        case C_STICK:
            /* full deflection at 12% of the height from the origin */
            dx = (s_f[i].x - s_f[i].ox) / 0.12f;
            dy = (s_f[i].oy - s_f[i].y) / 0.12f;
            m = sqrtf(dx * dx + dy * dy);
            if (m > 1.0f) { dx /= m; dy /= m; }
            p.Gamepad.sThumbLX = (SHORT)(dx * 32767.0f);
            p.Gamepad.sThumbLY = (SHORT)(dy * 32767.0f);
            break;
        case C_A:  p.Gamepad.bAnalogButtons[XBOX_BUTTON_A] = 255; break;
        case C_B:  p.Gamepad.bAnalogButtons[XBOX_BUTTON_B] = 255; break;
        case C_X:  p.Gamepad.bAnalogButtons[XBOX_BUTTON_X] = 255; break;
        case C_Y:  p.Gamepad.bAnalogButtons[XBOX_BUTTON_Y] = 255; break;
        case C_LT: p.Gamepad.bAnalogButtons[XBOX_BUTTON_LTRIGGER] = 255; break;
        case C_RT: p.Gamepad.bAnalogButtons[XBOX_BUTTON_RTRIGGER] = 255; break;
        case C_START: p.Gamepad.wButtons |= XBOX_GAMEPAD_START; break;
        case C_BACK:  p.Gamepad.wButtons |= XBOX_GAMEPAD_BACK; break;
        default: break;
        }
    }
    pthread_mutex_unlock(&s_lock);
    p.dwPacketNumber = ++packet;
    *st = p;
    return 1;
}

int touch_pad_init(void)
{
    const char *e = getenv("RECOMP_TOUCH");
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IPHONE)
    s_on = !(e && e[0] == '0');
#else
    s_on = e && e[0] == '1';
#endif
    if (s_on)
        xbox_InputSetExtPad(touch_ext_pad);
    return s_on;
}

/* The control under (x, y), fractions of a w x h window. */
static int hit(float x, float y, int w, int h)
{
    float aspect = (float)w / (float)h;
    int c;
    for (c = C_A; c < C_N; c++) {
        float dx = (x - s_ctl[c].x) * aspect, dy = y - s_ctl[c].y;
        /* a little larger than drawn: thumbs miss */
        if (dx * dx + dy * dy <= (s_ctl[c].r * 1.25f) * (s_ctl[c].r * 1.25f))
            return c;
    }
    return x < 0.45f ? C_STICK : C_NONE;
}

int touch_pad_event(const SDL_Event *ev, int w, int h)
{
    int i, changed = 0;
    if (!s_on || w <= 0 || h <= 0)
        return 0;
    if (ev->type != SDL_EVENT_FINGER_DOWN && ev->type != SDL_EVENT_FINGER_UP
            && ev->type != SDL_EVENT_FINGER_MOTION && ev->type != SDL_EVENT_FINGER_CANCELED)
        return 0;
    pthread_mutex_lock(&s_lock);
    for (i = 0; i < FINGERS; i++)
        if (s_f[i].ctl && s_f[i].id == ev->tfinger.fingerID)
            break;
    if (ev->type == SDL_EVENT_FINGER_DOWN && i == FINGERS) {
        for (i = 0; i < FINGERS && s_f[i].ctl; i++)
            ;
        if (i < FINGERS) {
            int c = hit(ev->tfinger.x, ev->tfinger.y, w, h);
            if (c) {
                s_f[i].id = ev->tfinger.fingerID;
                s_f[i].ctl = c;
                s_f[i].ox = s_f[i].x = ev->tfinger.x;
                s_f[i].oy = s_f[i].y = ev->tfinger.y;
                changed = 1;
            }
        }
    } else if (i < FINGERS) {
        if (ev->type == SDL_EVENT_FINGER_MOTION) {
            s_f[i].x = ev->tfinger.x;
            s_f[i].y = ev->tfinger.y;
            changed = s_f[i].ctl == C_STICK;
        } else {
            s_f[i].ctl = C_NONE;
            changed = 1;
        }
    }
    pthread_mutex_unlock(&s_lock);
    return changed;
}

static void circle(SDL_Renderer *ren, float cx, float cy, float r, SDL_FColor c, int fill)
{
    enum { SEG = 32 };
    SDL_Vertex v[SEG + 2];
    int idx[SEG * 3], i;
    v[0].position.x = cx;
    v[0].position.y = cy;
    v[0].color = c;
    for (i = 0; i <= SEG; i++) {
        float a = (float)i * 6.2831853f / SEG;
        v[i + 1].position.x = cx + r * cosf(a);
        v[i + 1].position.y = cy + r * sinf(a);
        v[i + 1].color = c;
        v[i + 1].tex_coord.x = v[i + 1].tex_coord.y = 0;
    }
    v[0].tex_coord.x = v[0].tex_coord.y = 0;
    if (fill) {
        for (i = 0; i < SEG; i++) {
            idx[i * 3] = 0;
            idx[i * 3 + 1] = i + 1;
            idx[i * 3 + 2] = i + 2;
        }
        SDL_RenderGeometry(ren, NULL, v, SEG + 2, idx, SEG * 3);
    } else {
        SDL_FPoint pts[SEG + 1];
        for (i = 0; i <= SEG; i++) {
            pts[i].x = v[i + 1].position.x;
            pts[i].y = v[i + 1].position.y;
        }
        SDL_SetRenderDrawColorFloat(ren, c.r, c.g, c.b, c.a);
        SDL_RenderLines(ren, pts, SEG + 1);
    }
}

static void draw_controls(SDL_Renderer *ren, int w, int h, const int *down, int have_stick,
                          const float *stick);

void touch_pad_draw(SDL_Renderer *ren, int w, int h)
{
    int down[C_N] = { 0 }, i;
    float stick[4] = { 0 };
    int have_stick = 0;
    if (!s_on || w <= 0 || h <= 0)
        return;
    pthread_mutex_lock(&s_lock);
    for (i = 0; i < FINGERS; i++) {
        down[s_f[i].ctl] = 1;
        if (s_f[i].ctl == C_STICK) {
            stick[0] = s_f[i].ox; stick[1] = s_f[i].oy;
            stick[2] = s_f[i].x;  stick[3] = s_f[i].y;
            have_stick = 1;
        }
    }
    pthread_mutex_unlock(&s_lock);
    /* The presenter draws the frame through a logical presentation at the
     * frame's size: the controls are in window pixels, so it is off while
     * they draw (as the stats overlay does), and the draw colour and blend
     * mode are put back, since the next frame clears with them. */
    {
        int lw = 0, lh = 0;
        SDL_RendererLogicalPresentation mode = SDL_LOGICAL_PRESENTATION_DISABLED;
        float r0, g0, b0, a0;
        SDL_BlendMode bm;
        SDL_GetRenderLogicalPresentation(ren, &lw, &lh, &mode);
        SDL_GetRenderDrawColorFloat(ren, &r0, &g0, &b0, &a0);
        SDL_GetRenderDrawBlendMode(ren, &bm);
        if (mode != SDL_LOGICAL_PRESENTATION_DISABLED)
            SDL_SetRenderLogicalPresentation(ren, 0, 0, SDL_LOGICAL_PRESENTATION_DISABLED);
        draw_controls(ren, w, h, down, have_stick, stick);
        if (mode != SDL_LOGICAL_PRESENTATION_DISABLED)
            SDL_SetRenderLogicalPresentation(ren, lw, lh, mode);
        SDL_SetRenderDrawColorFloat(ren, r0, g0, b0, a0);
        SDL_SetRenderDrawBlendMode(ren, bm);
    }
}

static void draw_controls(SDL_Renderer *ren, int w, int h, const int *down, int have_stick,
                          const float *stick)
{
    int c;
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    for (c = C_A; c < C_N; c++) {
        const Ctl *k = &s_ctl[c];
        SDL_FColor col = { k->cr / 255.0f, k->cg / 255.0f, k->cb / 255.0f,
                           down[c] ? 0.55f : 0.22f };
        circle(ren, k->x * w, k->y * h, k->r * h, col, 1);
        col.a = 0.6f;
        circle(ren, k->x * w, k->y * h, k->r * h, col, 0);
    }
    {
        /* the stick: its ring where the thumb landed (or a resting spot),
         * its knob under the thumb */
        SDL_FColor ring = { 1, 1, 1, 0.35f }, knob = { 1, 1, 1, 0.45f };
        float ox = have_stick ? stick[0] : 0.16f, oy = have_stick ? stick[1] : 0.72f;
        float kx = have_stick ? stick[2] : ox, ky = have_stick ? stick[3] : oy;
        float dx = (kx - ox) * w, dy = (ky - oy) * h, R = 0.12f * h, m = sqrtf(dx * dx + dy * dy);
        if (m > R) { dx *= R / m; dy *= R / m; }
        circle(ren, ox * w, oy * h, R, ring, 0);
        circle(ren, ox * w + dx, oy * h + dy, 0.06f * h, knob, 1);
    }
}

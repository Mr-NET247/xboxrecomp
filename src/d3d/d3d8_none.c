/*
 * d3d8_none.c - the D3D8 compat layer's entry points on a host with no
 * OpenGL (iOS). d3d8_gl.c stands behind them on Linux and macOS, but no title
 * on the pushbuffer path calls them: the Metal backend draws the guest's own
 * pushbuffer (RECOMP_PB_BACKEND=metal), and the framebuffer window presents
 * it. Each one reports that there is no device, so a caller that does reach
 * one takes its failure path instead of a missing GL context.
 */
#include "d3d8_xbox.h"
#include "d3d8_internal.h"
#include <stdio.h>

volatile int g_suppress_present = 0;

void xbox_D3D8SetWindowTitle(const char *title) { (void)title; }

IDirect3D8 *xbox_Direct3DCreate8(UINT SDKVersion)
{
    (void)SDKVersion;
    fprintf(stderr, "[d3d8] no OpenGL on this host: Direct3DCreate8 has no device\n");
    return NULL;
}

IDirect3DDevice8 *xbox_GetD3DDevice(void) { return NULL; }
IDirect3DDevice8 *d3d8_GetDevice(void)    { return NULL; }
void d3d8_PresentFrame(void) {}

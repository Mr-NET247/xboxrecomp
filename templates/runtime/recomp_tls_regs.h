/*
 * recomp_tls_regs.h - the recompiled register set, per thread.
 *
 * The MMX and XMM registers are one thread-local block, g_vregs; the
 * integer registers, flags and x87 stack stay one thread-local variable
 * each. In a shared library on Android every access to a thread-local
 * variable is a call to the dynamic TLS resolver, and the compiler cannot
 * share one lookup between two variables: Burnout 3's movie decoder (MMX
 * and SSE in a tight loop) spent 7.6% of its thread in the resolver on a
 * Pixel 7 Pro. With the vectors in one block a function needs one lookup
 * for all sixteen. The old names stay: g_mm0 is (g_vregs.r_mm[0]).
 *
 * The integer registers in a block of their own (g_regs) were tried, and a
 * guest DPC on the kernel's timer thread then read a bad pointer in a
 * Burnout 3 race; the cause was not found, so they stay separate.
 *
 * Include once with RECOMP_TLS defined; include again once RecompXmm and
 * RecompMmx exist, with RECOMP_TLS_REGS_VECTORS defined, for g_vregs.
 */
#ifndef RECOMP_TLS_REGS_H
#define RECOMP_TLS_REGS_H
#include <stdint.h>

/* The integer registers, flags and x87 stack stay one thread-local
 * variable each (see the note on g_vregs below). */
extern RECOMP_TLS uint32_t g_eax, g_ecx, g_edx, g_esp, g_ebx, g_esi, g_edi, g_ebp;
extern RECOMP_TLS uint32_t g_fs_base, g_seh_ebp, g_eflags, g_mxcsr;
extern RECOMP_TLS int g_df, g_fp_top, g_fp_cmp;
extern RECOMP_TLS uint16_t g_fp_control_word, g_fp_cc;
extern RECOMP_TLS double g_fp_stack[8];
#endif /* RECOMP_TLS_REGS_H */

#if defined(RECOMP_TLS_REGS_VECTORS) && !defined(RECOMP_TLS_VREGS_H)
#define RECOMP_TLS_VREGS_H
typedef struct RecompTlsVRegs {
    RecompXmm r_xmm[8];
    RecompMmx r_mm[8];
} RecompTlsVRegs;

extern RECOMP_TLS RecompTlsVRegs g_vregs;

#define g_xmm0 (g_vregs.r_xmm[0])
#define g_xmm1 (g_vregs.r_xmm[1])
#define g_xmm2 (g_vregs.r_xmm[2])
#define g_xmm3 (g_vregs.r_xmm[3])
#define g_xmm4 (g_vregs.r_xmm[4])
#define g_xmm5 (g_vregs.r_xmm[5])
#define g_xmm6 (g_vregs.r_xmm[6])
#define g_xmm7 (g_vregs.r_xmm[7])
#define g_mm0  (g_vregs.r_mm[0])
#define g_mm1  (g_vregs.r_mm[1])
#define g_mm2  (g_vregs.r_mm[2])
#define g_mm3  (g_vregs.r_mm[3])
#define g_mm4  (g_vregs.r_mm[4])
#define g_mm5  (g_vregs.r_mm[5])
#define g_mm6  (g_vregs.r_mm[6])
#define g_mm7  (g_vregs.r_mm[7])
#endif

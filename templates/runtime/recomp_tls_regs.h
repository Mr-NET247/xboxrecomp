/*
 * recomp_tls_regs.h - the recompiled register set, per thread.
 *
 * The MMX and XMM registers are one thread-local block, g_vregs, and the
 * data registers another, g_dregs (below). In a shared library on Android every access to a thread-local
 * variable is a call to the dynamic TLS resolver, and the compiler cannot
 * share one lookup between two variables: Burnout 3's movie decoder (MMX
 * and SSE in a tight loop) spent 7.6% of its thread in the resolver on a
 * Pixel 7 Pro. With the vectors in one block a function needs one lookup
 * for all sixteen. The old names stay: g_mm0 is (g_vregs.r_mm[0]).
 *
 * The six data registers (eax ecx edx ebx esi edi) are a second block,
 * g_dregs. All the integer registers in one block were tried first, and a
 * guest DPC on the kernel's timer thread then read a bad pointer in a
 * Burnout 3 race; with esp, ebp, fs_base, seh_ebp, the flags and the x87
 * state left one variable each the race runs clean. Which of those it was
 * is not known.
 *
 * Include once with RECOMP_TLS defined; include again once RecompXmm and
 * RecompMmx exist, with RECOMP_TLS_REGS_VECTORS defined, for g_vregs.
 */
#ifndef RECOMP_TLS_REGS_H
#define RECOMP_TLS_REGS_H
#include <stdint.h>

/* The integer registers, flags and x87 stack stay one thread-local
 * variable each (see the note on g_vregs below). */
/* The six data registers as one block, g_dregs. */
typedef struct RecompTlsDRegs { uint32_t r_eax, r_ecx, r_edx, r_ebx, r_esi, r_edi; } RecompTlsDRegs;
extern RECOMP_TLS RecompTlsDRegs g_dregs;
#define g_eax (g_dregs.r_eax)
#define g_ecx (g_dregs.r_ecx)
#define g_edx (g_dregs.r_edx)
#define g_ebx (g_dregs.r_ebx)
#define g_esi (g_dregs.r_esi)
#define g_edi (g_dregs.r_edi)
extern RECOMP_TLS uint32_t g_esp, g_ebp;
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

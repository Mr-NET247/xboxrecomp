/*
 * d3d8_glsl_split: the GLSL emitters (d3d8_glsl.h) on d3d8_msl_split's
 * inputs -- NPROG pseudo-random vertex programs through d3d8_vsh_parse and
 * d3d8_vsh_generate_glsl (both flag settings), NCOMB pseudo-random combiner
 * register sets through d3d8_combiners_from_regs and
 * d3d8_combiners_generate_glsl, plus the dot-product cases and each
 * combiner variant flag -- each compiled to SPIR-V with shaderc, as
 * nv2a_pb_vulkan.c compiles them. Any compile error fails the test.
 * GLSL_DUMP=<path> writes all the text.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <shaderc/shaderc.h>

#include "d3d8_glsl.h"
#include "../d3d8_hlsl_split/dot_cases.h"

#define NPROG 400
#define NCOMB 400

static uint32_t s_seed = 0x2f6b1d3u;
static uint32_t rnd(void)
{
    s_seed = s_seed * 1664525u + 1013904223u;
    return s_seed ^ (s_seed >> 15);
}

static shaderc_compiler_t s_cc;
static shaderc_compile_options_t s_opt;
static FILE *s_dump;
static int s_cases, s_bad;

static void check(const char *what, int idx, const char *text, int n, int frag)
{
    shaderc_compilation_result_t r;
    s_cases++;
    if (n <= 0) {
        if (++s_bad <= 5)
            printf("FAIL: %s %d: no text\n", what, idx);
        return;
    }
    if (s_dump)
        fprintf(s_dump, "// %s %d\n%s\n", what, idx, text);
    r = shaderc_compile_into_spv(s_cc, text, (size_t)n,
                                 frag ? shaderc_fragment_shader : shaderc_vertex_shader,
                                 what, "main", s_opt);
    if (shaderc_result_get_compilation_status(r) != shaderc_compilation_status_success
            && ++s_bad <= 5)
        printf("FAIL: %s %d does not compile:\n%s\n%s\n", what, idx,
               shaderc_result_get_error_message(r), s_bad == 1 ? text : "");
    shaderc_result_release(r);
}

int main(void)
{
    static uint32_t prog[NV2A_VS_MAX_INSTRUCTIONS * 4];
    static NV2AVshProgram parsed;
    static char buf[256 * 1024];
    const char *dump = getenv("GLSL_DUMP");
    int i, k, n, v;

    s_cc = shaderc_compiler_initialize();
    s_opt = shaderc_compile_options_initialize();
    shaderc_compile_options_set_target_env(s_opt, shaderc_target_env_vulkan,
                                           shaderc_env_version_vulkan_1_1);
    if (dump)
        s_dump = fopen(dump, "wb");

    for (i = 0; i < NPROG; i++) {
        int len = 1 + (int)(rnd() % 40), fin = (int)(rnd() % (uint32_t)len);
        for (k = 0; k < len * 4; k++)
            prog[k] = rnd();
        for (k = 0; k < len; k++)
            prog[k * 4 + 3] &= ~1u;
        prog[fin * 4 + 3] |= 1u;
        d3d8_vsh_parse(prog, len, &parsed);
        n = d3d8_vsh_generate_glsl(&parsed, 0, buf, (int)sizeof buf);
        check("vsh", i, buf, n, 0);
        n = d3d8_vsh_generate_glsl(&parsed, D3D8_GLSL_SCREEN_SPACE, buf, (int)sizeof buf);
        check("vsh-ss", i, buf, n, 0);
    }
    for (i = 0; i < NCOMB; i++) {
        uint32_t cicw[8], aicw[8], cocw[8], aocw[8];
        uint32_t control, fcw0, fcw1, modes;
        NV2ACombinerState st;
        for (k = 0; k < 8; k++) {
            cicw[k] = rnd(); aicw[k] = rnd(); cocw[k] = rnd(); aocw[k] = rnd();
        }
        control = (rnd() & ~0xFFu) | (1 + rnd() % 8);
        fcw0 = rnd(); fcw1 = rnd(); modes = rnd();
        d3d8_combiners_from_regs(cicw, aicw, cocw, aocw, control, fcw0, fcw1, modes, &st);
        st.fog_input = (int)(rnd() & 1);
        /* Every variant flag in turn, as the backend's key sets them. */
        v = i % 6;
        st.rect_snap = v >= 1;
        st.snap_2d = v == 2;
        st.rt_offset = v == 3;
        st.w_depth = v == 4;
        st.tex_mag = v == 5;
        n = d3d8_combiners_generate_glsl(&st, buf, (int)sizeof buf);
        check("rc", i, buf, n, 1);
    }
    for (i = 0; i < DOT_NCASE; i++)
        for (v = 0; v < 2; v++) {
            NV2ACombinerState st;
            dot_case_state(i, v, &st);
            n = d3d8_combiners_generate_glsl(&st, buf, (int)sizeof buf);
            check(k_dot_cases[i].name, i, buf, n, 1);
        }
    if (s_dump)
        fclose(s_dump);
    printf("glsl_split: %d cases compiled to SPIR-V, %d failed\n", s_cases, s_bad);
    return s_bad ? 1 : 0;
}

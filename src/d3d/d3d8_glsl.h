/**
 * Vulkan GLSL emitters for NV2A vertex programs (d3d8_vsh_glsl.c) and
 * register combiners (d3d8_combiners_glsl.c): the Vulkan backend's
 * (nv2a_pb_vulkan.c) counterparts of d3d8_msl.h, statement for statement.
 * The same parsed state goes in and GLSL 4.50 text comes out, which the
 * backend compiles to SPIR-V at run time. They build strings only and use
 * no graphics API, so they build and can be tested on every host.
 *
 * The shaders keep the MSL spelling where GLSL can take it as a macro
 * (D3D8_GLSL_PRELUDE: float4 for vec4, saturate, dfdx), so the two
 * emitters read the same and a fix to one maps onto the other line for line.
 *
 * Resources, all in descriptor set 0 (pushed per draw):
 *   0  storage buffer  vb: one vec4 per input the program reads, per vertex
 *   1  uniform         c[192]: the program's constants
 *   2  uniform         VpConsts: struct nv2a_vp_consts
 *   3  uniform         PsConsts: struct nv2a_ps_consts + tex_bias
 *   4  uniform         WDepth: struct nv2a_wdepth_consts (W-buffered batches)
 *   5  storage buffer  mag: RECOMP_TRACE=tex_mag's slots
 *   8-11 combined image samplers tex0..tex3
 * The stages meet in VOut, passed as locations 0-7.
 */
#ifndef XBOXRECOMP_D3D8_GLSL_H
#define XBOXRECOMP_D3D8_GLSL_H

#include "d3d8_vsh_parse.h"
#include "d3d8_combiners_parse.h"

#ifdef __cplusplus
extern "C" {
#endif

/** MSL type names and helpers as GLSL macros. */
#define D3D8_GLSL_PRELUDE \
    "#define float2 vec2\n" \
    "#define float3 vec3\n" \
    "#define float4 vec4\n" \
    "#define uint2 uvec2\n" \
    "#define uint3 uvec3\n" \
    "#define uint4 uvec4\n" \
    "#define int2 ivec2\n" \
    "#define bool4 bvec4\n" \
    "#define saturate(x) clamp((x), 0.0, 1.0)\n" \
    "#define fabs abs\n" \
    "#define rsqrt inversesqrt\n" \
    "#define dfdx dFdx\n" \
    "#define dfdy dFdy\n"

/** VOut as a struct both stages build (the vertex stage fills one and
 * D3D8_GLSL_VS_EMIT writes it out; the fragment stage reads one back with
 * vout_in()), so the emitted bodies use i.t0 and o.pos as the MSL does. */
#define D3D8_GLSL_VOUT_DECL \
    "struct VOut {\n" \
    "    float4 pos; float4 d0; float4 d1;\n" \
    "    float4 t0; float4 t1; float4 t2; float4 t3;\n" \
    "    float fog; float psize; float zw;\n" \
    "};\n"

#define D3D8_GLSL_VS_OUT \
    "layout(location = 0) out float4 v_d0;\n" \
    "layout(location = 1) out float4 v_d1;\n" \
    "layout(location = 2) out float4 v_t0;\n" \
    "layout(location = 3) out float4 v_t1;\n" \
    "layout(location = 4) out float4 v_t2;\n" \
    "layout(location = 5) out float4 v_t3;\n" \
    "layout(location = 6) out float v_fog;\n" \
    "layout(location = 7) out float v_zw;\n"

#define D3D8_GLSL_VS_EMIT \
    "    gl_Position = o.pos;\n" \
    "    gl_PointSize = o.psize;\n" \
    "    v_d0 = o.d0; v_d1 = o.d1;\n" \
    "    v_t0 = o.t0; v_t1 = o.t1; v_t2 = o.t2; v_t3 = o.t3;\n" \
    "    v_fog = o.fog; v_zw = o.zw;\n"

/** The fragment side: the varyings, vout_in(), and the colour output. */
#define D3D8_GLSL_FS_IN \
    "layout(location = 0) in float4 v_d0;\n" \
    "layout(location = 1) in float4 v_d1;\n" \
    "layout(location = 2) in float4 v_t0;\n" \
    "layout(location = 3) in float4 v_t1;\n" \
    "layout(location = 4) in float4 v_t2;\n" \
    "layout(location = 5) in float4 v_t3;\n" \
    "layout(location = 6) in float v_fog;\n" \
    "layout(location = 7) in float v_zw;\n" \
    "layout(location = 0) out float4 o_c;\n" \
    "VOut vout_in() {\n" \
    "    VOut i;\n" \
    "    i.pos = gl_FragCoord; i.d0 = v_d0; i.d1 = v_d1;\n" \
    "    i.t0 = v_t0; i.t1 = v_t1; i.t2 = v_t2; i.t3 = v_t3;\n" \
    "    i.fog = v_fog; i.psize = 1.0; i.zw = v_zw;\n" \
    "    return i;\n" \
    "}\n"

/** The fragment constants: MsPsConsts (nv2a_pb_metal.m) in std140, every
 * member a vec4 or uvec4, so the layout is the C struct's. */
#define D3D8_GLSL_PSCONSTS \
    "layout(std140, set = 0, binding = 3) uniform PsConsts {\n" \
    "    float4 c0[8]; float4 c1[8]; float4 fc0; float4 fc1; float4 fog_color;\n" \
    "    uint4 atest; uint4 alpha_only; float4 tex_scale[4]; uint4 tex_mode;\n" \
    "    float4 bump[4]; float4 bumpls[4]; uint4 clip_mode;\n" \
    "    float4 tex_bias;\n" \
    "} pc;\n"

/** D3D8_MSL_WDEPTH and its BEGIN / END (d3d8_msl.h) in GLSL. */
#define D3D8_GLSL_WDEPTH \
    "layout(std140, set = 0, binding = 4) uniform WDepth { float4 w0; float4 w1; } wd;\n"

#define D3D8_GLSL_WDEPTH_BEGIN \
    "    float zv = i.zw;\n" \
    "    float zs = max(abs(dfdx(i.zw)), abs(dfdy(i.zw)));\n" \
    "    if (zv > 0.0) zv += wd.w0.x + wd.w0.y * zs;\n" \
    "    else zv = uintBitsToFloat(0x7F7FFFFFu);\n" \
    "    if ((floatBitsToUint(zv) & 0x7FFFFFFFu) > 0x7F800000u) zv = uintBitsToFloat(0x7F7FFFFFu);\n" \
    "    bool wkill = wd.w1.y != 0.0 && (zv < wd.w0.z || wd.w0.w < zv);\n" \
    "    zv = clamp(zv, wd.w0.z, wd.w0.w);\n"

#define D3D8_GLSL_WDEPTH_END \
    "    if (wkill) discard;\n" \
    "    o_c = res;\n" \
    "    gl_FragDepth = saturate(zv * wd.w1.x);\n"

/** D3D8_MSL_CLIP_KILLS in GLSL. */
#define D3D8_GLSL_CLIP_KILLS \
    "bool nv2a_clip_kills(uint cm, float4 t) {\n" \
    "    bool4 ge = notEqual((uint4(cm) >> uint4(0u, 1u, 2u, 3u)) & 1u, uint4(0u));\n" \
    "    return (cm & 16u) != 0u\n" \
    "        && any(mix(lessThan(t, float4(0.0)), greaterThanEqual(t, float4(0.0)), ge));\n" \
    "}\n"

/** D3D8_MSL_SNAP_UV in GLSL; pos is gl_FragCoord.xy. */
#define D3D8_GLSL_SNAP_UV \
    "float2 nv2a_snap_uv(float2 uv, float2 pos, float4 ts) {\n" \
    "    float2 dx = dfdx(uv), dy = dfdy(uv);\n" \
    "    float n = ts.z, eps = 1.0 / 64.0;\n" \
    "    if (n > 1.0 && abs(dx.y) <= eps && abs(dy.x) <= eps\n" \
    "            && abs(abs(dx.x) * n - 1.0) <= eps && abs(abs(dy.y) * n - 1.0) <= eps) {\n" \
    "        float2 d = (floor(pos / n) + 0.5) * n - pos;\n" \
    "        uv += dx * d.x + dy * d.y;\n" \
    "    }\n" \
    "    return uv;\n" \
    "}\n"

/** D3D8_MSL_SNAP_2D in GLSL. */
#define D3D8_GLSL_SNAP_2D \
    "bool nv2a_snap_ok(float2 dx, float2 dy, float4 ts) {\n" \
    "    float n = abs(ts.z), eps = 1.0 / 64.0;\n" \
    "    return n > 1.0 && (ts.z < 0.0\n" \
    "        ? abs(dx.y) <= abs(dx.x) * eps && abs(dy.x) <= abs(dy.y) * eps\n" \
    "        : abs(dx.y) <= eps && abs(dy.x) <= eps\n" \
    "          && abs(abs(dx.x) * n - 1.0) <= eps && abs(abs(dy.y) * n - 1.0) <= eps);\n" \
    "}\n" \
    "float2 nv2a_snap_uv2d(float2 uv, float2 pos, float4 ts) {\n" \
    "    float2 dx = dfdx(uv), dy = dfdy(uv);\n" \
    "    if (nv2a_snap_ok(dx, dy, ts)) {\n" \
    "        float n = abs(ts.z);\n" \
    "        float2 d = (floor(pos / n) + 0.5) * n - pos;\n" \
    "        uv += dx * d.x + dy * d.y;\n" \
    "    }\n" \
    "    return uv;\n" \
    "}\n" \
    "float4 nv2a_snap_grad(float2 uv, float4 ts) {\n" \
    "    float2 dx = dfdx(uv), dy = dfdy(uv);\n" \
    "    return float4(dx, dy) * (nv2a_snap_ok(dx, dy, ts) ? abs(ts.z) : 1.0);\n" \
    "}\n"

/** D3D8_MSL_SCREEN_SPACE: the same epilogue (see d3d8_msl.h). */
#define D3D8_GLSL_SCREEN_SPACE 0x1

/** The vertex shader (main): d3d8_vsh_generate_msl's vs_main in GLSL. */
int d3d8_vsh_generate_glsl(const NV2AVshProgram *program, unsigned flags,
                           char *buf, int bufsize);

/** The fragment shader (main): d3d8_combiners_generate_msl's fs_main. */
int d3d8_combiners_generate_glsl(const NV2ACombinerState *state,
                                 char *buf, int bufsize);

#ifdef __cplusplus
}
#endif

#endif /* XBOXRECOMP_D3D8_GLSL_H */

/* nv2a_basic_glsl.h - the Vulkan backend's basic shaders as GLSL text, the
 * counterparts of nv2a_basic_msl.h: vs_pass (pre-transformed batches),
 * fs_basic (batches without register combiners) and render.snap_2d's
 * fs_basic_snap. Each is a whole shader with main(); the backend compiles
 * them to SPIR-V at run time (nv2a_pb_vulkan.c). */
#ifndef NV2A_BASIC_GLSL_H
#define NV2A_BASIC_GLSL_H

#include "d3d8_glsl.h"

/* fs_basic's text: SAMPLE is stage 0's sample into t. */
#define GLSL_FS_BASIC(SAMPLE) \
    "layout(set = 0, binding = 8) uniform sampler2D tex0;\n" \
    "void main() {\n" \
    "    VOut i = vout_in();\n" \
    "    float4 c = i.d0;\n" \
    "    if (pc.tex_mode.x == 4u) {\n" \
    "        c *= saturate(i.t0);\n" \
    "    } else if (pc.tex_mode.x != 0u) {\n" \
    "        float2 uv = i.t0.xy;\n" \
    "        if (pc.tex_mode.x == 1u && i.t0.w != 0.0 && i.t0.w != 1.0) uv /= i.t0.w;\n" \
    SAMPLE \
    "        if (pc.alpha_only.x == 2u) t.a = 1.0;\n" \
    "        c *= t;\n" \
    "    }\n" \
    "    if (nv2a_clip_kills(pc.clip_mode.x, i.t0)) discard;\n" \
    "    if (pc.atest.z != 0u) {\n" \
    "        float ref = uintBitsToFloat(pc.atest.x);\n" \
    "        uint a = uint(saturate(c.a) * 255.0 + 0.5), r = uint(ref * 255.0 + 0.5);\n" \
    "        bool ok = true;\n" \
    "        uint f = pc.atest.y;\n" \
    "        if      (f == 1u) ok = false;\n" \
    "        else if (f == 2u) ok = a <  r;\n" \
    "        else if (f == 3u) ok = a == r;\n" \
    "        else if (f == 4u) ok = a <= r;\n" \
    "        else if (f == 5u) ok = a >  r;\n" \
    "        else if (f == 6u) ok = a != r;\n" \
    "        else if (f == 7u) ok = a >= r;\n" \
    "        if (!ok) discard;\n" \
    "    }\n" \
    "    o_c = c;\n" \
    "}\n"

#define NV2A_GLSL_FS_HEAD \
    "#version 450\n" \
    D3D8_GLSL_PRELUDE \
    D3D8_GLSL_PSCONSTS \
    D3D8_GLSL_VOUT_DECL \
    D3D8_GLSL_FS_IN \
    D3D8_GLSL_SNAP_UV \
    D3D8_GLSL_CLIP_KILLS

#define NV2A_GLSL_FS_BASIC \
    NV2A_GLSL_FS_HEAD \
    GLSL_FS_BASIC( \
        "        uv = nv2a_snap_uv(uv, i.pos.xy, pc.tex_scale[0]);\n" \
        "        float4 t = texture(tex0, uv * pc.tex_scale[0].xy" \
        " + float2(pc.tex_scale[0].w, 0.0), pc.tex_bias.x);\n")

#define NV2A_GLSL_FS_BASIC_SNAP \
    NV2A_GLSL_FS_HEAD \
    D3D8_GLSL_SNAP_2D \
    GLSL_FS_BASIC( \
        "        float4 g = nv2a_snap_grad(uv, pc.tex_scale[0]) * pc.tex_scale[0].xyxy" \
        " * exp2(pc.tex_bias.x);\n" \
        "        uv = nv2a_snap_uv2d(uv, i.pos.xy, pc.tex_scale[0]);\n" \
        "        uv = uv * pc.tex_scale[0].xy + float2(pc.tex_scale[0].w, 0.0);\n" \
        "        float4 t;\n" \
        "        if (pc.tex_scale[0].z < 0.0) t = textureGrad(tex0, uv, g.xy, g.zw);\n" \
        "        else t = texture(tex0, uv, pc.tex_bias.x);\n")

/* vs_pass: attribute 0 in surface pixels (x, y, z, rhw), as nv2a_basic_msl.h;
 * VpConsts here is c[0] (2/w, -2/h, 1/zmax, 0) and c[1] (-1, 1, tie x, y). */
#define NV2A_GLSL_VS_PASS \
    "#version 450\n" \
    D3D8_GLSL_PRELUDE \
    "layout(std430, set = 0, binding = 0) readonly buffer VB { vec4 vb[]; };\n" \
    "layout(std140, set = 0, binding = 2) uniform VpConsts { float4 c[3]; uint4 u[2]; } vp;\n" \
    D3D8_GLSL_VOUT_DECL \
    D3D8_GLSL_VS_OUT \
    "void main() {\n" \
    "    uint b = uint(gl_VertexIndex) * 7u;\n" \
    "    VOut o;\n" \
    "    float2 p = trunc(vb[b].xy * 16.0) / 16.0;\n" \
    "    p -= vp.c[1].zw * float2(equal(fract(p), float2(0.5))) * float(vb[b].w == 1.0);\n" \
    "    o.pos = float4(p * vp.c[0].xy + vp.c[1].xy,\n" \
    "                   saturate(vb[b].z * vp.c[0].z), 1.0);\n" \
    "    o.d0 = vb[b + 1u]; o.d1 = vb[b + 2u];\n" \
    "    o.t0 = vb[b + 3u]; o.t1 = vb[b + 4u]; o.t2 = vb[b + 5u]; o.t3 = vb[b + 6u];\n" \
    "    o.fog = 1.0; o.psize = 1.0; o.zw = 1.0;\n" \
    D3D8_GLSL_VS_EMIT \
    "}\n"

#endif

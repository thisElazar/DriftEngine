// river_overlay.fs.hlsl — samples the global hydrology field and draws animated
// flowing water where river_strength exceeds the threshold. The stripe scrolls
// along the per-cell flow direction (time-driven) so it reads as motion at any
// planet scale, independent of the (too-slow) SWE timescale.

[[vk::binding(2, 0)]] Texture2DArray<float4> hydrology;  // .r=strength .g=moist .ba=flow
[[vk::binding(3, 0)]] SamplerState samp;

[[vk::push_constant]]
cbuffer RiverPC {
    float rel_x, rel_y, rel_z;
    float u_min, v_min, tile_size;
    uint  face;
    uint  pool_index;
    float planet_radius;
    float heightmap_texel;
    float time;
    float river_threshold;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 hyd_uv  : TEXCOORD0;
    [[vk::location(1)]] nointerpolation uint face_out : TEXCOORD1;
    [[vk::location(2)]] float2 tile_uv : TEXCOORD2;
};

float4 main(PSInput input) : SV_Target
{
    // r=river strength (flowing water), g=moisture, b=flow angle, a=lake depth (standing).
    float4 h = hydrology.SampleLevel(samp, float3(input.hyd_uv, float(input.face_out)), 0);
    float flow = h.r;   // flowing component (rivers)
    float pool = h.a;   // standing component (lakes)

    // ONE water body, not two. Rivers and lakes are the same surface sampled at
    // different points on a continuum: "how much water" (depth → colour) and "how
    // fast it moves" (flow → animation). Soft presence with smoothstep edges so a
    // river deepens smoothly into the lake it feeds instead of ending at a seam.
    // The flow component is gated by river_threshold so the ambient rain-spill on
    // every cell doesn't paint the whole planet blue — only real channels show.
    float flowv = smoothstep(river_threshold * 0.7, river_threshold * 1.15, flow); // 0..1 river presence
    float poolv = smoothstep(0.02, 0.08, pool);                                    // 0..1 lake presence
    float present = max(flowv, poolv);
    if (present < 0.01)
        discard;

    // Single flow animation along the downstream direction. Fixed freq/speed keeps
    // the wave phase continuous from river into lake; only its CONTRAST scales with
    // flow, so the same travelling crests fade from bright river ripples to a faint
    // lake drift rather than switching to a different-looking material.
    float ang   = (h.b - 0.5) * 6.2831853;
    float2 flowdir = float2(cos(ang), sin(ang));
    float along = dot(input.hyd_uv, flowdir);
    static const float FREQ  = 60.0;
    static const float SPEED = 0.45;
    float stripe  = 0.5 + 0.5 * sin((along * FREQ - time * SPEED) * 6.2831853);
    // broad slow shimmer so large calm lakes still live between crests
    float shimmer = 0.5 + 0.5 * sin(((input.hyd_uv.x + input.hyd_uv.y) * 26.0 - time * 0.18) * 6.2831853);

    // motion = directional crests where it flows, faint shimmer where it stands.
    float motion    = lerp(shimmer, stripe, flowv);
    float crest_amt = 0.12 + 0.55 * flowv;

    // Single depth-driven palette shared by all water: shallow/bright → deep/dark.
    float  depth   = saturate(pool + 0.25 * flowv);
    float3 shallow = float3(0.35, 0.60, 0.85);
    float3 deep    = float3(0.05, 0.18, 0.42);
    float3 crest   = float3(0.60, 0.82, 1.00);
    float3 base = lerp(shallow, deep, depth);
    float3 col  = lerp(base, crest, crest_amt * motion);

    float alpha = saturate(present * 1.6) * (0.80 + 0.20 * motion);
    return float4(col, alpha);
}

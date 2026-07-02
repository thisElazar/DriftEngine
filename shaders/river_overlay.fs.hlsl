// river_overlay.fs.hlsl — samples the global hydrology field and draws animated
// flowing water where river_strength exceeds the threshold. The stripe scrolls
// along the per-cell flow direction (time-driven) so it reads as motion at any
// planet scale, independent of the (too-slow) SWE timescale.

#include "atmosphere_planet.hlsli"

[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float _cam_time;   // (=time; FS uses RiverPC.time)
    // Repurposed under camera-relative rendering: planet center rel. camera.
    float3 planet_center; float _pad2;
};
[[vk::binding(2, 0)]] Texture2DArray<float4> hydrology;  // .r=strength .g=moist .b=flow .a=surface elevation
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
    float atmo_density;  // aerial perspective density (0 = off)
    float sun_intensity; // sun radiance scale, shared with the sky pass
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 hyd_uv  : TEXCOORD0;
    [[vk::location(1)]] nointerpolation uint face_out : TEXCOORD1;
    [[vk::location(2)]] float2 tile_uv : TEXCOORD2;
    [[vk::location(3)]] float3 world_pos : TEXCOORD3;   // camera-relative
    [[vk::location(4)]] float3 normal    : TEXCOORD4;   // surface (radial) normal
};

float4 main(PSInput input) : SV_Target
{
    // r=river strength (flowing), g=moisture, b=flow angle, a=water-surface
    // elevation in metres (rendered by the terrain/SWE pass, not here).
    float4 h = hydrology.SampleLevel(samp, float3(input.hyd_uv, float(input.face_out)), 0);
    float flow = h.r;

    // FLOW ONLY. Standing water (ocean + lakes) is seeded into the SWE state
    // from this same field (planet_swe_init) and rendered by the terrain pass,
    // so the overlay draws just the channels too thin to carry as SWE depth.
    // Gated by river_threshold so the ambient rain-spill on every cell doesn't
    // paint the whole planet blue — only real channels show.
    float flowv = smoothstep(river_threshold * 0.7, river_threshold * 1.15, flow); // 0..1 river presence
    float present = flowv;
    if (present < 0.01)
        discard;

    // Flow animation along the downstream direction: travelling crests whose
    // contrast scales with flow strength.
    float ang   = (h.b - 0.5) * 6.2831853;
    float2 flowdir = float2(cos(ang), sin(ang));
    float along = dot(input.hyd_uv, flowdir);
    static const float FREQ  = 60.0;
    static const float SPEED = 0.45;
    float motion    = 0.5 + 0.5 * sin((along * FREQ - time * SPEED) * 6.2831853);
    float crest_amt = 0.12 + 0.55 * flowv;

    // Depth palette shared with the ocean/lake water: stronger channels read deeper.
    float  depth   = saturate(0.25 * flowv);
    float3 shallow = float3(0.35, 0.60, 0.85);
    float3 deep    = float3(0.05, 0.18, 0.42);
    float3 crest   = float3(0.60, 0.82, 1.00);
    float3 base = lerp(shallow, deep, depth);
    float3 col  = lerp(base, crest, crest_amt * motion);

    // --- Lighting parity with the ocean (terrain.fs) -------------------------
    // Same fresnel + sky reflection + specular so a lake catches the sun exactly
    // like the sea. The surface normal is perturbed slightly along the flow so the
    // sun glints travel with the current (matching the ocean's swell glints).
    float3 Ngeo = normalize(input.normal);
    float3 up = (abs(Ngeo.y) < 0.99) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 T  = normalize(cross(up, Ngeo));
    float3 Bt = cross(Ngeo, T);
    float  bump = (motion - 0.5) * (0.5 + flowv);          // wave-slope proxy
    float3 Nw = normalize(Ngeo + 0.05 * bump * (flowdir.x * T + flowdir.y * Bt));

    float3 V = normalize(-input.world_pos);
    float3 L = normalize(sun_dir);
    float  NdotV = saturate(dot(Nw, V));
    float  fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);

    float3 Rr = reflect(-V, Nw);
    float  sky_t = saturate(Rr.y * 0.5 + 0.5);
    float3 sky = lerp(float3(0.85, 0.88, 0.92), float3(0.30, 0.55, 0.85), sky_t);

    float3 H = normalize(L + V);
    float  spec = pow(saturate(dot(Nw, H)), 64.0);
    float3 specular = sun_color * spec * 0.8;

    float  NdotL = saturate(dot(Nw, L));
    float3 lit = col * (0.4 + 0.6 * NdotL);
    float3 outc = lerp(lit, sky, fresnel) + specular;

    // Aerial perspective, matching the terrain beneath so distant rivers haze
    // out with the landscape instead of staying saturated.
    if (atmo_density > 0.0) {
        float dist = length(input.world_pos);
        float3 rd = input.world_pos / max(dist, 1.0);
        AtmoResult ar = atmo_integrate(-planet_center, rd, dist, L,
                                       planet_radius, atmo_density, sun_intensity,
                                       6, 2);
        outc = outc * ar.transmittance + ar.inscatter;
    }

    // Reflective water reads more opaque at grazing angles (like the sea).
    float alpha = saturate(present * 1.6) * saturate(0.72 + 0.22 * motion + 0.28 * fresnel);
    return float4(outc, alpha);
}

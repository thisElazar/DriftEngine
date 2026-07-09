// sky.fs.hlsl — raymarched planetary atmosphere, drawn as a fullscreen
// triangle with depth-EQUAL against the reverse-Z clear value, so it fills
// exactly the pixels nothing else rendered: the sky dome from the ground,
// the blue limb from orbit, and black space beyond.

#include "atmosphere_planet.hlsli"

[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float time;
    // Repurposed under camera-relative rendering: position of the PLANET
    // CENTER relative to the camera (metres). See CameraData in camera.h.
    float3 planet_center; float _pad2;
    float4 brush_world;
    float4 brush_color;
    float4x4 inv_view_proj;
};

[[vk::push_constant]]
cbuffer SkyPC {
    float planet_radius;
    float density;        // atmosphere density scale (UI)
    float sun_intensity;  // sun radiance scale (UI)
    float _pad;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 ndc : TEXCOORD0;
    [[vk::location(1)]] float2 uv  : TEXCOORD1;
};

float4 main(PSInput input) : SV_Target
{
    // Unproject the pixel to a camera-relative world direction (the camera sits
    // at the origin of camera-relative space, so the unprojected point IS the
    // ray direction).
    float4 p = mul(inv_view_proj, float4(input.ndc, 0.5, 1.0));
    float3 rd = normalize(p.xyz / p.w);

    // Atmosphere frame: origin at the planet center.
    float3 ro = -planet_center;   // camera position relative to the planet center
    float3 L  = normalize(sun_dir);

    // Rays that hit the planet datum stop there (terrain pixels have their own
    // aerial perspective; this only covers sub-horizon gaps), others march the
    // whole shell. The clip is SOFT — feathered on the fp32-stable impact
    // parameter — so the limb never sparkles under camera motion. See
    // atmo_sky_tmax in atmosphere_planet.hlsli.
    float miss_w;
    float t_max = atmo_sky_tmax(ro, rd, planet_radius, miss_w);

    AtmoResult ar = atmo_integrate(ro, rd, t_max, L, planet_radius,
                                   density, sun_intensity, 24);

    // Sun disk + glow, attenuated by the atmosphere in front of it and faded
    // by the same miss weight — no binary occlusion gate to pop.
    float3 col = ar.inscatter;
    float mu = dot(rd, L);
    float disk = smoothstep(0.99995, 0.99999, mu);
    float glow = pow(saturate(mu), 2000.0) * 0.4;
    col += (disk * 4.0 + glow) * sun_color * ar.transmittance * miss_w;

    return float4(col, 1.0);
}

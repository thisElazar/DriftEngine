// planet_climate.hlsli — deterministic climate + biome weights from a sphere
// direction and seed. Shared by planet_gen.cs.hlsl (terrain CHARACTER varies
// by biome) and terrain.fs.hlsl (surface PALETTE varies by biome) so geology
// and coloring agree by construction — the same pure-function discipline the
// whole planet runs on (docs/ROADMAP_PLANET_SCALE.md).
//
// Everything is pc_-prefixed so including shaders keep their own noise
// helpers without collisions. Functions take the seed explicitly — no cbuffer
// dependencies.

float pc_hash31(float3 p)
{
    p = frac(p * float3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 31.32);
    return frac((p.x + p.y) * p.z);
}

float pc_noise3(float3 p)
{
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    return lerp(
        lerp(lerp(pc_hash31(i + float3(0,0,0)), pc_hash31(i + float3(1,0,0)), u.x),
             lerp(pc_hash31(i + float3(0,1,0)), pc_hash31(i + float3(1,1,0)), u.x), u.y),
        lerp(lerp(pc_hash31(i + float3(0,0,1)), pc_hash31(i + float3(1,0,1)), u.x),
             lerp(pc_hash31(i + float3(0,1,1)), pc_hash31(i + float3(1,1,1)), u.x), u.y),
        u.z);
}

float pc_fbm3(float3 p, int octaves)
{
    float sum = 0.0, amp = 1.0, freq = 1.0, norm = 0.0;
    for (int i = 0; i < octaves; i++) {
        sum  += pc_noise3(p * freq) * amp;
        norm += amp;
        freq *= 2.0;
        amp  *= 0.5;
    }
    return sum / norm;
}

struct PlanetClimate {
    float temperature;   // 0 polar .. 1 equatorial, at sea level (apply an
                         // altitude lapse at the use site when height is known)
    float moisture;      // 0 arid .. 1 wet
};

PlanetClimate planet_climate(float3 n, float seed_f)
{
    float3 so = float3(seed_f * 0.0173, seed_f * 0.0091, seed_f * 0.0047);

    // Latitude bands, warmest at the equator, plus a low-frequency wobble so
    // climate zones meander instead of being perfect rings.
    float cos_lat = sqrt(saturate(1.0 - n.y * n.y));
    float t = pow(cos_lat, 1.5);
    t += 0.25 * (pc_fbm3(n * 2.1 + so, 4) - 0.5);

    // Moisture provinces: domain-warped fbm so wet/dry regions have organic,
    // continent-scale shapes (same warping trick as the coastlines).
    float3 w = float3(pc_fbm3(n * 1.3 + so + 11.7, 3),
                      pc_fbm3(n * 1.3 + so + 31.9, 3),
                      pc_fbm3(n * 1.3 + so + 57.3, 3)) - 0.5;
    float m = pc_fbm3((n + 0.55 * w) * 1.7 + so, 4);
    m = saturate((m - 0.30) * 2.2);

    PlanetClimate c;
    c.temperature = saturate(t);
    c.moisture    = m;
    return c;
}

// Biome weights from climate: x = desert, y = grass/savanna, z = forest,
// w = tundra. Always sums to 1. Snow/ice is NOT a biome here — derive it at
// the use site from temperature + altitude (the snowline falls toward the
// poles).
float4 planet_biome_weights(PlanetClimate c)
{
    float warm = smoothstep(0.25, 0.45, c.temperature);
    float hot  = smoothstep(0.55, 0.75, c.temperature);
    float wet  = smoothstep(0.30, 0.55, c.moisture);

    float4 wgt;
    wgt.x = hot * (1.0 - wet);                 // desert: hot + dry
    wgt.w = 1.0 - warm;                        // tundra: cold
    wgt.z = warm * wet * (1.0 - wgt.x);        // forest: mild-to-warm + wet
    wgt.y = max(1.0 - wgt.x - wgt.z - wgt.w, 0.0);   // grass takes the rest
    return wgt / max(wgt.x + wgt.y + wgt.z + wgt.w, 1e-3);
}

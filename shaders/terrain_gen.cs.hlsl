// terrain_gen.cs.hlsl — Procedural heightmap generation for geometry clipmap
// Writes to one layer of a Texture2DArray per dispatch.
// Uses FBM noise with biome-based terrain blending.

[[vk::binding(0, 0)]] RWTexture2DArray<float> clipmap_heights;

[[vk::push_constant]]
cbuffer TerrainGenPC {
    float    origin_x;
    float    origin_z;
    float    cell_size;
    uint     tex_res;
    uint     level;
    uint     seed;
    float    _pad0;
    float    _pad1;
};

float hash2(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float hash3(float3 p)
{
    p = frac(p * float3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 31.32);
    return frac((p.x + p.y) * p.z);
}

float2 hash22(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float value_noise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash2(i);
    float b = hash2(i + float2(1, 0));
    float c = hash2(i + float2(0, 1));
    float d = hash2(i + float2(1, 1));

    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float2 grad2(float2 p)
{
    float2 h = hash22(p);
    return float2(cos(h.x * 6.2831853), sin(h.y * 6.2831853));
}

float gradient_noise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);

    float2 ga = grad2(i + float2(0, 0));
    float2 gb = grad2(i + float2(1, 0));
    float2 gc = grad2(i + float2(0, 1));
    float2 gd = grad2(i + float2(1, 1));

    float va = dot(ga, f - float2(0, 0));
    float vb = dot(gb, f - float2(1, 0));
    float vc = dot(gc, f - float2(0, 1));
    float vd = dot(gd, f - float2(1, 1));

    return lerp(lerp(va, vb, u.x), lerp(vc, vd, u.x), u.y) + 0.5;
}

float fbm(float2 p, int octaves, float lacunarity, float gain)
{
    float sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float norm = 0.0;

    for (int i = 0; i < octaves; i++)
    {
        sum += gradient_noise(p * freq) * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }

    return sum / norm;
}

float ridged_noise(float2 p, int octaves)
{
    float sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float prev = 1.0;

    for (int i = 0; i < octaves; i++)
    {
        float n = gradient_noise(p * freq);
        n = 1.0 - abs(n * 2.0 - 1.0);
        n = n * n;
        sum += n * amp * prev;
        prev = n;
        freq *= 2.1;
        amp *= 0.5;
    }

    return sum;
}

float terrain_height(float2 wp)
{
    // Biome selection: large-scale noise determines terrain type
    float biome_a = fbm(wp * 0.00004 + float2(seed * 0.1, 0), 3, 2.0, 0.5);
    float biome_b = fbm(wp * 0.00006 + float2(0, seed * 0.1), 3, 2.0, 0.5);

    float desert_weight   = smoothstep(0.35, 0.50, biome_a) * smoothstep(0.55, 0.40, biome_a);
    float mountain_weight = smoothstep(0.55, 0.75, biome_a);
    float plains_weight   = smoothstep(0.45, 0.25, biome_a);

    // Desert: rolling dunes on flat base
    float desert_base = 150.0 + fbm(wp * 0.0008, 4, 2.0, 0.5) * 80.0;
    float dunes = abs(gradient_noise(wp * 0.005 + float2(biome_b * 10.0, 0))) * 40.0;
    dunes += abs(gradient_noise(wp * 0.012)) * 15.0;
    float desert = desert_base + dunes;

    // Mountains: ridged noise for peaks and valleys
    float mtn_base = 300.0 + fbm(wp * 0.0003, 5, 2.0, 0.55) * 400.0;
    float ridges = ridged_noise(wp * 0.0008, 6) * 600.0;
    float mountain = mtn_base + ridges;

    // Plains: gentle rolling terrain
    float plains = 100.0 + fbm(wp * 0.0006, 5, 2.0, 0.5) * 80.0;
    plains += fbm(wp * 0.003, 3, 2.0, 0.4) * 15.0;

    // Blend biomes
    float total = plains_weight + desert_weight + mountain_weight;
    total = max(total, 0.01);

    float h = (plains * plains_weight + desert * desert_weight + mountain * mountain_weight) / total;

    // Micro detail across all biomes
    h += (gradient_noise(wp * 0.02) - 0.5) * 8.0;

    return h;
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= tex_res || dtid.y >= tex_res)
        return;

    float2 world_pos;
    world_pos.x = origin_x + float(dtid.x) * cell_size;
    world_pos.y = origin_z + float(dtid.y) * cell_size;

    float h = terrain_height(world_pos);

    clipmap_heights[uint3(dtid.xy, level)] = h;
}

// planet_gen.cs.hlsl — Generate heightmap for one tile of the cube-sphere planet.
// Each texel maps to a point on the sphere surface; height evaluated via 3D noise.

[[vk::binding(0, 0)]] RWTexture2DArray<float> tile_pool;

[[vk::push_constant]]
cbuffer PlanetGenPC {
    float    u_min;
    float    v_min;
    float    tile_size;
    uint     face;
    uint     pool_index;
    uint     tex_res;
    uint     seed;
    float    _pad;
};

float hash31(float3 p)
{
    p = frac(p * float3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 31.32);
    return frac((p.x + p.y) * p.z);
}

float3 hash33(float3 p)
{
    p = float3(dot(p, float3(127.1, 311.7, 74.7)),
               dot(p, float3(269.5, 183.3, 246.1)),
               dot(p, float3(113.5, 271.9, 124.6)));
    return frac(sin(p) * 43758.5453123);
}

float gradient_noise_3d(float3 p)
{
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);

    float n = lerp(
        lerp(lerp(hash31(i + float3(0,0,0)), hash31(i + float3(1,0,0)), u.x),
             lerp(hash31(i + float3(0,1,0)), hash31(i + float3(1,1,0)), u.x), u.y),
        lerp(lerp(hash31(i + float3(0,0,1)), hash31(i + float3(1,0,1)), u.x),
             lerp(hash31(i + float3(0,1,1)), hash31(i + float3(1,1,1)), u.x), u.y),
        u.z);
    return n;
}

float fbm3d(float3 p, int octaves, float lacunarity, float gain)
{
    float sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float norm = 0.0;
    for (int i = 0; i < octaves; i++) {
        sum += gradient_noise_3d(p * freq + float3(seed * 0.17, 0, 0)) * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    return sum / norm;
}

float ridged3d(float3 p, int octaves)
{
    float sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float prev = 1.0;
    for (int i = 0; i < octaves; i++) {
        float n = gradient_noise_3d(p * freq + float3(seed * 0.13, 0, 0));
        n = 1.0 - abs(n * 2.0 - 1.0);
        n = n * n;
        sum += n * amp * prev;
        prev = n;
        freq *= 2.1;
        amp *= 0.5;
    }
    return sum;
}

float3 face_uv_to_cube(float2 uv, uint f)
{
    switch (f) {
        case 0: return float3( 1.0, uv.y, -uv.x);
        case 1: return float3(-1.0, uv.y,  uv.x);
        case 2: return float3(uv.x,  1.0, -uv.y);
        case 3: return float3(uv.x, -1.0,  uv.y);
        case 4: return float3(uv.x, uv.y,  1.0);
        case 5: return float3(-uv.x, uv.y, -1.0);
        default: return float3(0, 0, 0);
    }
}

float3 cube_to_sphere(float3 p)
{
    float x2 = p.x * p.x;
    float y2 = p.y * p.y;
    float z2 = p.z * p.z;
    return float3(
        p.x * sqrt(max(0.0, 1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0)),
        p.y * sqrt(max(0.0, 1.0 - x2 * 0.5 - z2 * 0.5 + x2 * z2 / 3.0)),
        p.z * sqrt(max(0.0, 1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0))
    );
}

float terrain_height(float3 sphere_dir)
{
    float3 sp = sphere_dir * 1000.0;
    float latitude = abs(sphere_dir.y);

    // Continental base — large-scale elevation variation
    float base = 200.0 + fbm3d(sp * 0.0003, 5, 2.0, 0.5) * 1500.0;

    // Biome selection
    float biome = fbm3d(sp * 0.0005 + float3(7.7, 0, 0), 3, 2.0, 0.5);

    float mtn_w    = smoothstep(0.55, 0.70, biome);
    float desert_w = smoothstep(0.35, 0.50, biome) * smoothstep(0.55, 0.40, biome)
                   * smoothstep(0.65, 0.15, latitude);
    float plains_w = smoothstep(0.45, 0.25, biome);
    float polar_w  = smoothstep(0.55, 0.80, latitude);

    // Mountains: sharp ridges + broad uplift
    float mountain = ridged3d(sp * 0.006, 7) * 4500.0;
    mountain += fbm3d(sp * 0.003, 5, 2.0, 0.55) * 2000.0;

    // Desert: mesas, dunes, sand sheets
    float desert = fbm3d(sp * 0.005, 5, 2.0, 0.5) * 500.0;
    desert += abs(gradient_noise_3d(sp * 0.03)) * 250.0;
    desert += (gradient_noise_3d(sp * 0.1) - 0.5) * 80.0;

    // Plains: gentle rolling terrain
    float plains = fbm3d(sp * 0.004, 5, 2.0, 0.5) * 400.0;
    plains += fbm3d(sp * 0.02, 3, 2.0, 0.4) * 100.0;

    // Polar: ice fields with crevasses
    float polar = fbm3d(sp * 0.003, 4, 2.0, 0.5) * 800.0;
    polar += abs(gradient_noise_3d(sp * 0.02) - 0.5) * 200.0;

    float total_w = max(mtn_w + desert_w + plains_w + polar_w, 0.01);
    float biome_h = (mountain * mtn_w + desert * desert_w + plains * plains_w + polar * polar_w) / total_w;

    float h = base + biome_h;
    h += (gradient_noise_3d(sp * 0.15) - 0.5) * 40.0;

    return h;
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= tex_res || dtid.y >= tex_res)
        return;

    float2 face_uv;
    face_uv.x = u_min + float(dtid.x) / float(tex_res - 1) * tile_size;
    face_uv.y = v_min + float(dtid.y) / float(tex_res - 1) * tile_size;

    float3 cube_pt = face_uv_to_cube(face_uv, face);
    float3 sphere_dir = cube_to_sphere(cube_pt);

    float h = terrain_height(sphere_dir);

    tile_pool[uint3(dtid.xy, pool_index)] = h;
}

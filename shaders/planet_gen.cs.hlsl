// planet_gen.cs.hlsl — Generate heightmap for one tile of the cube-sphere planet.
// Each texel maps to a point on the sphere surface; height evaluated via 3D noise.

[[vk::binding(0, 0)]] RWTexture2DArray<float> tile_pool;

struct TerrainStamp {
    float3 pos;
    float  radius;
    float  delta_h;
    float  cos_radius;
    float2 _spad;
};

[[vk::binding(1, 0)]] StructuredBuffer<TerrainStamp> stamps;

[[vk::push_constant]]
cbuffer PlanetGenPC {
    float    u_min;
    float    v_min;
    float    tile_size;
    uint     face;
    uint     pool_index;
    uint     tex_res;
    uint     seed;
    uint     stamp_count;
};

float hash31(float3 p)
{
    p = frac(p * float3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 31.32);
    return frac((p.x + p.y) * p.z);
}

float3 hash33(float3 p)
{
    float3 p3 = frac(p * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 31.32);
    return float3(
        frac((p3.x + p3.y) * p3.z),
        frac((p3.z + p3.x) * p3.y),
        frac((p3.y + p3.z) * p3.x)
    );
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

// ---------------------------------------------------------------------------
// Worley (cellular) noise — returns F1, F2 distances and the integer cell
// coords of the two nearest cell points.
// ---------------------------------------------------------------------------
struct WorleyResult {
    float  F1, F2;
    float3 cell_A, cell_B;
};

WorleyResult worley3d(float3 p, float seed_ofs)
{
    float3 ip = floor(p);
    float3 fp = frac(p);

    float d1 = 1e10, d2 = 1e10;
    float3 c1 = 0, c2 = 0;

    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        float3 cell = ip + float3(x, y, z);
        float3 pt = float3(x, y, z) + hash33(cell + seed_ofs) - fp;
        float d = dot(pt, pt);
        if (d < d1) {
            d2 = d1; c2 = c1;
            d1 = d;  c1 = cell;
        } else if (d < d2) {
            d2 = d;  c2 = cell;
        }
    }

    WorleyResult r;
    r.F1 = sqrt(d1);
    r.F2 = sqrt(d2);
    r.cell_A = c1;
    r.cell_B = c2;
    return r;
}

// ---------------------------------------------------------------------------
// Planet terrain height
//
// Rewritten to kill the old "leaf-vein" look. The previous version painted
// raised ridges directly along Worley cell boundaries at three scales
// (tectonic + two drainage-basin layers), which wraps the whole sphere in a
// polygonal vein web, and split land/sea by a per-plate binary mask, giving a
// few lobed polygonal continents. Both reads as a leaf.
//
// New approach:
//   * Continents come from smooth, DOMAIN-WARPED fBm centered on sea level,
//     so coastlines meander organically instead of tracing noise cells.
//   * Mountain belts are warped ridged-fBm confined to soft tectonic zones and
//     MASKED TO LAND -> wandering ranges, never clean cell-edge lines.
//   * No global cellular boundary ridges. Real valleys / drainage are left to
//     the hydrology + erosion pass, not faked with noise.
//
// Calibrated against sea_level = 800 m (src/ui.h): ocean floor sinks well
// below it, continental interiors rise above it, coastline crosses it.
// ---------------------------------------------------------------------------

static const float CONT_FREQ   = 1.25;   // continental scale (lower => bigger landmasses)
static const float CONT_WARP   = 0.45;   // coastline meander strength
static const float CONT_SPAN   = 4000.0; // abyssal-plain..plateau span (m)
static const float CONT_BIAS   = 0.55;   // fBm cut for shoreline (~higher => less land)

static const float PLATE_FREQ  = 2.6;    // tectonic plate count (higher => more, smaller plates)
static const float BELT_WIDTH  = 0.32;   // soft width of a mountain belt around a plate edge
static const float RANGE_H     = 3200.0; // peak height added along belts (m)
static const float RANGE_WARP  = 0.55;   // how much ranges wander off the raw plate edge

static const float HILL_H      = 280.0;  // rolling-hill amplitude on land

// 3D domain-warp offset built from fBm, recentered to roughly [-0.5, 0.5].
float3 warp3(float3 p, float seed_ofs)
{
    return float3(
        fbm3d(p + float3(seed_ofs + 11.5, 0, 0), 4, 2.0, 0.5),
        fbm3d(p + float3(seed_ofs + 31.9, 0, 0), 4, 2.0, 0.5),
        fbm3d(p + float3(seed_ofs + 57.3, 0, 0), 4, 2.0, 0.5)) - 0.5;
}

float terrain_height(float3 sphere_dir)
{
    float3 n  = sphere_dir;          // unit direction
    float3 sp = n * 1000.0;          // detail-noise sample space

    // ===== CONTINENTS (smooth, domain-warped) =====
    float3 wn   = n + CONT_WARP * warp3(n * 1.6, seed * 0.07);
    float  cont = fbm3d(wn * CONT_FREQ, 6, 2.0, 0.5);     // ~0..1
    float  land_signal = cont - CONT_BIAS;                // >0 inland, <0 ocean
    float  base = 800.0 + land_signal * CONT_SPAN;        // centered on sea level
    float  land = smoothstep(800.0, 1400.0, base);        // 0 shore/ocean .. 1 inland

    // ===== TECTONIC MOUNTAIN BELTS (warped, land only) =====
    float3 pw    = n + 0.15 * warp3(n * PLATE_FREQ, seed * 0.21);
    WorleyResult plate = worley3d(pw * PLATE_FREQ, seed * 0.13);
    float belt   = 1.0 - smoothstep(0.0, BELT_WIDTH, plate.F2 - plate.F1);
    belt        *= land;                                  // ranges ride on continents
    float3 rw    = sp * 0.02 + RANGE_WARP * warp3(sp * 0.01, seed * 0.4);
    float  ranges = ridged3d(rw, 5) * RANGE_H * belt;

    // ===== ROLLING HILLS + MULTI-OCTAVE DETAIL =====
    float hills = (fbm3d(sp * 0.05, 5, 2.0, 0.5) - 0.5) * HILL_H * land;

    float detail = 0.0;
    detail += ridged3d(sp * 0.4, 4) * 120.0 * land;       // ridged texture on land
    detail += (fbm3d(sp * 0.08, 6, 2.0, 0.5) - 0.5) * 120.0;
    detail += (gradient_noise_3d(sp * 0.15) - 0.5) * 40.0;
    detail += (fbm3d(sp * 1.6, 3, 2.0, 0.5) - 0.5) * 90.0;
    detail += (gradient_noise_3d(sp * 13.0) - 0.5) * 25.0;
    detail += (gradient_noise_3d(sp * 80.0) - 0.5) * 8.0;
    detail += (gradient_noise_3d(sp * 640.0) - 0.5) * 1.5;

    float h = base + ranges + hills + detail;
    return clamp(h, -3000.0, 8000.0);
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

    // Apply terrain edit stamps
    for (uint i = 0; i < stamp_count; i++) {
        float d = dot(sphere_dir, stamps[i].pos);
        if (d < stamps[i].cos_radius)
            continue;
        float t = (d - stamps[i].cos_radius) / max(1.0 - stamps[i].cos_radius, 1e-7);
        h += stamps[i].delta_h * exp(-4.0 * (1.0 - t) * (1.0 - t));
    }

    tile_pool[uint3(dtid.xy, pool_index)] = h;
}

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
// Tectonic + watershed terrain height
// ---------------------------------------------------------------------------

static const float PLATE_FREQ      = 1.1;
static const float BOUNDARY_WIDTH  = 0.12;

static const float BASIN_FREQ_MAJ  = 12.0;
static const float BASIN_FREQ_MIN  = 40.0;

static const float RIDGE_SHARP_MAJ = 0.08;
static const float RIDGE_SHARP_MIN = 0.06;
static const float RIDGE_H_MAJ     = 1500.0;
static const float RIDGE_H_MIN     = 600.0;

static const float BASIN_DEPTH_MAJ = 800.0;
static const float BASIN_DEPTH_MIN = 300.0;

static const float VALLEY_FLOOR_W  = 0.02;
static const float VALLEY_FLOOR_H  = 200.0;

float terrain_height(float3 sphere_dir)
{
    float3 sp = sphere_dir * 1000.0;

    // ===== LAYER 1: TECTONIC PLATES =====
    WorleyResult plate = worley3d(sphere_dir * PLATE_FREQ, seed * 0.07);

    float continental_A = step(0.45, hash31(plate.cell_A + 77.7));
    float plate_base = lerp(-200.0, 1000.0, continental_A);

    float boundary = 1.0 - smoothstep(0.0, BOUNDARY_WIDTH, plate.F2 - plate.F1);

    float3 vel_A = hash33(plate.cell_A + seed * 0.31) * 2.0 - 1.0;
    float3 vel_B = hash33(plate.cell_B + seed * 0.31) * 2.0 - 1.0;
    float3 bn = normalize(plate.cell_B - plate.cell_A + 1e-6);
    float approach = dot(vel_A - vel_B, bn);
    float convergent = smoothstep(-0.2, 0.2, approach);

    float mountain_h = boundary * convergent * 3500.0;
    float rift_h = boundary * (1.0 - convergent) * -600.0;

    float cont_swell = (fbm3d(sp * 0.0003, 4, 2.0, 0.5) - 0.5) * 800.0;

    float tectonic_h = plate_base + mountain_h + rift_h + cont_swell;

    // ===== LAYER 2: DRAINAGE BASINS =====
    WorleyResult basin_maj = worley3d(sphere_dir * BASIN_FREQ_MAJ, seed * 0.13 + 1000.0);
    WorleyResult basin_min = worley3d(sphere_dir * BASIN_FREQ_MIN, seed * 0.19 + 2000.0);

    float ridge_maj = smoothstep(RIDGE_SHARP_MAJ, 0.0, basin_maj.F2 - basin_maj.F1) * RIDGE_H_MAJ;

    float in_basin = smoothstep(0.0, 0.15, basin_maj.F2 - basin_maj.F1);
    float ridge_min = smoothstep(RIDGE_SHARP_MIN, 0.0, basin_min.F2 - basin_min.F1)
                    * RIDGE_H_MIN * in_basin;

    float slope_maj = pow(saturate(basin_maj.F1 * 3.0), 0.6) * BASIN_DEPTH_MAJ;
    float slope_min = pow(saturate(basin_min.F1 * 5.0), 0.6) * BASIN_DEPTH_MIN * in_basin;

    // ===== LAYER 3: VALLEY PROFILE =====
    float valley_flat = smoothstep(VALLEY_FLOOR_W, 0.0, basin_maj.F1) * VALLEY_FLOOR_H;
    float valley_flat_min = smoothstep(VALLEY_FLOOR_W, 0.0, basin_min.F1) * (VALLEY_FLOOR_H * 0.4) * in_basin;

    float drainage_h = ridge_maj + ridge_min + slope_maj + slope_min - valley_flat - valley_flat_min;

    // Mountain detail on convergent boundaries
    float mtn_detail = ridged3d(sp * 0.006, 5) * 1500.0 * boundary * convergent;

    // ===== LAYER 4: SURFACE DETAIL =====
    float detail = 0.0;
    detail += ridged3d(sp * 0.4, 5) * 200.0;
    detail += (fbm3d(sp * 0.08, 6, 2.0, 0.5) - 0.5) * 300.0;
    detail += (gradient_noise_3d(sp * 0.15) - 0.5) * 40.0;
    detail += (fbm3d(sp * 1.6, 3, 2.0, 0.5) - 0.5) * 150.0;
    detail += (gradient_noise_3d(sp * 13.0) - 0.5) * 30.0;
    detail += (gradient_noise_3d(sp * 80.0) - 0.5) * 8.0;
    detail += (gradient_noise_3d(sp * 640.0) - 0.5) * 1.5;

    float h = tectonic_h + drainage_h + mtn_detail + detail;
    return clamp(h, -2000.0, 8000.0);
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

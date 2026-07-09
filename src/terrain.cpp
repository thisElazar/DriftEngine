#include "terrain.h"
#include "pipeline.h"
#include <algorithm>
#include <cmath>

float cpu_hash31(glm::vec3 p) {
    p = glm::fract(p * glm::vec3(0.1031f, 0.1030f, 0.0973f));
    p += glm::dot(p, glm::vec3(p.y, p.x, p.z) + 31.32f);
    return glm::fract((p.x + p.y) * p.z);
}

glm::vec3 cpu_hash33(glm::vec3 p) {
    glm::vec3 p3 = glm::fract(p * glm::vec3(0.1031f, 0.1030f, 0.0973f));
    p3 += glm::dot(p3, glm::vec3(p3.y, p3.x, p3.z) + 31.32f);
    return glm::vec3(
        glm::fract((p3.x + p3.y) * p3.z),
        glm::fract((p3.z + p3.x) * p3.y),
        glm::fract((p3.y + p3.z) * p3.x)
    );
}

float cpu_gradient_noise_3d(glm::vec3 p) {
    glm::vec3 i = glm::floor(p);
    glm::vec3 f = glm::fract(p);
    glm::vec3 u = f * f * (3.0f - 2.0f * f);

    float n = glm::mix(
        glm::mix(glm::mix(cpu_hash31(i + glm::vec3(0,0,0)), cpu_hash31(i + glm::vec3(1,0,0)), u.x),
                 glm::mix(cpu_hash31(i + glm::vec3(0,1,0)), cpu_hash31(i + glm::vec3(1,1,0)), u.x), u.y),
        glm::mix(glm::mix(cpu_hash31(i + glm::vec3(0,0,1)), cpu_hash31(i + glm::vec3(1,0,1)), u.x),
                 glm::mix(cpu_hash31(i + glm::vec3(0,1,1)), cpu_hash31(i + glm::vec3(1,1,1)), u.x), u.y),
        u.z);
    return n;
}

float cpu_fbm3d(glm::vec3 p, int octaves, float lacunarity, float gain) {
    float sum = 0.0f, amp = 1.0f, freq = 1.0f, norm = 0.0f;
    for (int i = 0; i < octaves; i++) {
        sum += cpu_gradient_noise_3d(p * freq + glm::vec3(42 * 0.17f, 0, 0)) * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    return sum / norm;
}

float cpu_ridged3d(glm::vec3 p, int octaves) {
    float sum = 0.0f, amp = 1.0f, freq = 1.0f, prev = 1.0f;
    for (int i = 0; i < octaves; i++) {
        float n = cpu_gradient_noise_3d(p * freq + glm::vec3(42 * 0.13f, 0, 0));
        n = 1.0f - std::abs(n * 2.0f - 1.0f);
        n = n * n;
        sum += n * amp * prev;
        prev = n;
        freq *= 2.1f;
        amp *= 0.5f;
    }
    return sum;
}

struct CpuWorleyResult {
    float F1, F2;
    glm::vec3 cell_A, cell_B;
};

CpuWorleyResult cpu_worley3d(glm::vec3 p, float seed_ofs) {
    glm::vec3 ip = glm::floor(p);
    glm::vec3 fp = glm::fract(p);

    float d1 = 1e10f, d2 = 1e10f;
    glm::vec3 c1(0.0f), c2(0.0f);

    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        glm::vec3 cell = ip + glm::vec3(x, y, z);
        glm::vec3 pt = glm::vec3(x, y, z) + cpu_hash33(cell + seed_ofs) - fp;
        float d = glm::dot(pt, pt);
        if (d < d1) {
            d2 = d1; c2 = c1;
            d1 = d;  c1 = cell;
        } else if (d < d2) {
            d2 = d;  c2 = cell;
        }
    }

    CpuWorleyResult r;
    r.F1 = std::sqrt(d1);
    r.F2 = std::sqrt(d2);
    r.cell_A = c1;
    r.cell_B = c2;
    return r;
}

// NOTE: must stay in lockstep with shaders/planet_gen.cs.hlsl `terrain_height`.
// The CPU mirror is what the camera, ray-pick, brush and (future) flow
// accumulation run against; if it drifts from the GPU noise, physics happens
// on an invisible planet. Constants and math below are copied verbatim.
static constexpr float CONT_FREQ   = 1.25f;   // continental scale (lower => bigger landmasses)
static constexpr float CONT_WARP   = 0.45f;   // coastline meander strength
static constexpr float CONT_SPAN   = 4000.0f; // abyssal-plain..plateau span (m)
static constexpr float CONT_BIAS   = 0.55f;   // fBm cut for shoreline (higher => less land)
static constexpr float PLATE_FREQ  = 2.6f;    // tectonic plate count
static constexpr float BELT_WIDTH  = 0.32f;   // soft width of a mountain belt
static constexpr float RANGE_H     = 3200.0f; // peak height added along belts (m)
static constexpr float RANGE_WARP  = 0.55f;   // how much ranges wander off the plate edge
static constexpr float HILL_H      = 280.0f;  // rolling-hill amplitude on land

// ---------------------------------------------------------------------------
// CPU mirror of shaders/planet_climate.hlsli. pc_fbm3 there adds NO seed
// offset inside the loop (seed enters via the `so` offsets at call sites), so
// it gets its own helper rather than reusing cpu_fbm3d.
// ---------------------------------------------------------------------------
static float cpu_pc_fbm3(glm::vec3 p, int octaves) {
    float sum = 0.0f, amp = 1.0f, freq = 1.0f, norm = 0.0f;
    for (int i = 0; i < octaves; i++) {
        sum += cpu_gradient_noise_3d(p * freq) * amp;
        norm += amp;
        freq *= 2.0f;
        amp *= 0.5f;
    }
    return sum / norm;
}

CpuPlanetClimate cpu_planet_climate(glm::vec3 n, float seed_f) {
    glm::vec3 so(seed_f * 0.0173f, seed_f * 0.0091f, seed_f * 0.0047f);

    float cos_lat = std::sqrt(std::clamp(1.0f - n.y * n.y, 0.0f, 1.0f));
    float t = std::pow(cos_lat, 1.5f);
    t += 0.25f * (cpu_pc_fbm3(n * 2.1f + so, 4) - 0.5f);

    glm::vec3 w = glm::vec3(cpu_pc_fbm3(n * 1.3f + so + 11.7f, 3),
                            cpu_pc_fbm3(n * 1.3f + so + 31.9f, 3),
                            cpu_pc_fbm3(n * 1.3f + so + 57.3f, 3)) - 0.5f;
    float m = cpu_pc_fbm3((n + 0.55f * w) * 1.7f + so, 4);
    m = std::clamp((m - 0.30f) * 2.2f, 0.0f, 1.0f);

    CpuPlanetClimate c;
    c.temperature = std::clamp(t, 0.0f, 1.0f);
    c.moisture    = m;
    return c;
}

glm::vec4 cpu_planet_biome_weights(CpuPlanetClimate c) {
    float warm = glm::smoothstep(0.25f, 0.45f, c.temperature);
    float hot  = glm::smoothstep(0.55f, 0.75f, c.temperature);
    float wet  = glm::smoothstep(0.30f, 0.55f, c.moisture);

    glm::vec4 wgt;
    wgt.x = hot * (1.0f - wet);
    wgt.w = 1.0f - warm;
    wgt.z = warm * wet * (1.0f - wgt.x);
    wgt.y = std::max(1.0f - wgt.x - wgt.z - wgt.w, 0.0f);
    return wgt / std::max(wgt.x + wgt.y + wgt.z + wgt.w, 1e-3f);
}

// 3D domain-warp offset, recentered to roughly [-0.5, 0.5]. Mirrors warp3().
static glm::vec3 cpu_warp3(glm::vec3 p, float seed_ofs) {
    return glm::vec3(
        cpu_fbm3d(p + glm::vec3(seed_ofs + 11.5f, 0, 0), 4, 2.0f, 0.5f),
        cpu_fbm3d(p + glm::vec3(seed_ofs + 31.9f, 0, 0), 4, 2.0f, 0.5f),
        cpu_fbm3d(p + glm::vec3(seed_ofs + 57.3f, 0, 0), 4, 2.0f, 0.5f)) - 0.5f;
}

float cpu_terrain_height(glm::vec3 sphere_dir) {
    glm::vec3 n  = sphere_dir;
    glm::vec3 sp = n * 1000.0f;
    constexpr uint32_t seed = PLANET_SEED;   // shared with the GPU push constant (terrain.h)

    // ===== CONTINENTS (smooth, domain-warped) =====
    glm::vec3 wn = n + CONT_WARP * cpu_warp3(n * 1.6f, seed * 0.07f);
    float cont = cpu_fbm3d(wn * CONT_FREQ, 6, 2.0f, 0.5f);
    float land_signal = cont - CONT_BIAS;
    float base = 800.0f + land_signal * CONT_SPAN;
    float land = glm::smoothstep(800.0f, 1400.0f, base);

    // ===== TECTONIC MOUNTAIN BELTS (warped, land only) =====
    glm::vec3 pw = n + 0.15f * cpu_warp3(n * PLATE_FREQ, seed * 0.21f);
    CpuWorleyResult plate = cpu_worley3d(pw * PLATE_FREQ, seed * 0.13f);
    float belt = 1.0f - glm::smoothstep(0.0f, BELT_WIDTH, plate.F2 - plate.F1);
    belt *= land;
    glm::vec3 rw = sp * 0.02f + RANGE_WARP * cpu_warp3(sp * 0.01f, seed * 0.4f);
    float ranges = cpu_ridged3d(rw, 5) * RANGE_H * belt;

    // ===== CLIMATE -> BIOME (mirrors planet_climate.hlsli) =====
    CpuPlanetClimate clim = cpu_planet_climate(n, static_cast<float>(seed));
    glm::vec4 bw = cpu_planet_biome_weights(clim);

    // ===== ROLLING HILLS + MULTI-OCTAVE DETAIL, biome-scaled =====
    float hill_amp = HILL_H * glm::dot(bw, glm::vec4(0.45f, 1.0f, 1.1f, 0.55f));
    float hills = (cpu_fbm3d(sp * 0.05f, 5, 2.0f, 0.5f) - 0.5f) * hill_amp * land;

    float dunes = (1.0f - std::abs(2.0f * cpu_fbm3d(sp * 0.12f, 3, 2.0f, 0.5f) - 1.0f)) * 45.0f;
    dunes += (1.0f - std::abs(2.0f * cpu_fbm3d(sp * 0.45f, 2, 2.0f, 0.5f) - 1.0f)) * 12.0f;
    dunes *= bw.x * land;

    float rough = glm::dot(bw, glm::vec4(1.15f, 0.85f, 0.55f, 1.0f));

    float detail = 0.0f;
    detail += cpu_ridged3d(sp * 0.4f, 4) * 120.0f * land * rough;
    detail += (cpu_fbm3d(sp * 0.08f, 6, 2.0f, 0.5f) - 0.5f) * 120.0f;
    detail += (cpu_gradient_noise_3d(sp * 0.15f) - 0.5f) * 40.0f;
    detail += (cpu_fbm3d(sp * 1.6f, 3, 2.0f, 0.5f) - 0.5f) * 90.0f * rough;
    detail += (cpu_gradient_noise_3d(sp * 13.0f) - 0.5f) * 25.0f;
    detail += (cpu_gradient_noise_3d(sp * 80.0f) - 0.5f) * 8.0f;
    detail += (cpu_gradient_noise_3d(sp * 640.0f) - 0.5f) * 1.5f;

    float h = base + ranges + hills + dunes + detail;
    return std::clamp(h, -3000.0f, 8000.0f);
}

float cpu_terrain_height_with_stamps(glm::vec3 sphere_dir,
                                     const TerrainStamp* stamps, uint32_t count)
{
    float h = cpu_terrain_height(sphere_dir);
    for (uint32_t i = 0; i < count; i++) {
        const auto& s = stamps[i];
        glm::vec3 sp(s.pos_x, s.pos_y, s.pos_z);
        float d = glm::dot(sphere_dir, sp);
        if (d < s.cos_radius) continue;
        float t = (d - s.cos_radius) / std::max(1.0f - s.cos_radius, 1e-7f);
        h += s.delta_h * std::exp(-4.0f * (1.0f - t) * (1.0f - t));
    }
    return h;
}

static float cpu_smoothstep(float edge0, float edge1, float x)
{
    float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

std::vector<float> generate_crater_basin(const BasinParams& p)
{
    std::vector<float> data(static_cast<size_t>(p.grid_w) * p.grid_h);
    for (uint32_t y = 0; y < p.grid_h; ++y) {
        for (uint32_t x = 0; x < p.grid_w; ++x) {
            float cx = (static_cast<float>(x) - p.grid_w * 0.5f) * p.cell_spacing;
            float cy = (static_cast<float>(y) - p.grid_h * 0.5f) * p.cell_spacing;
            float r = std::sqrt(cx * cx + cy * cy);
            float h;
            if (r < p.inner_radius) {
                h = p.floor_height;
            } else if (r < p.rim_radius) {
                float t = (r - p.inner_radius) / (p.rim_radius - p.inner_radius);
                h = std::lerp(p.floor_height, p.rim_height, cpu_smoothstep(0.0f, 1.0f, t));
            } else {
                float t = std::clamp((r - p.rim_radius) / 1500.0f, 0.0f, 1.0f);
                h = std::lerp(p.rim_height, p.base_height, cpu_smoothstep(0.0f, 1.0f, t));
                h += 30.0f * std::sin(cx * 0.0003f) * std::cos(cy * 0.0004f);
            }
            data[static_cast<size_t>(y) * p.grid_w + x] = h;
        }
    }
    return data;
}

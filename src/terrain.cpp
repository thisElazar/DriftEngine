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

static constexpr float PLATE_FREQ      = 1.1f;
static constexpr float BOUNDARY_WIDTH  = 0.12f;
static constexpr float BASIN_FREQ_MAJ  = 12.0f;
static constexpr float BASIN_FREQ_MIN  = 40.0f;
static constexpr float RIDGE_SHARP_MAJ = 0.08f;
static constexpr float RIDGE_SHARP_MIN = 0.06f;
static constexpr float RIDGE_H_MAJ     = 1500.0f;
static constexpr float RIDGE_H_MIN     = 600.0f;
static constexpr float BASIN_DEPTH_MAJ = 800.0f;
static constexpr float BASIN_DEPTH_MIN = 300.0f;
static constexpr float VALLEY_FLOOR_W  = 0.02f;
static constexpr float VALLEY_FLOOR_H  = 200.0f;

float cpu_terrain_height(glm::vec3 sphere_dir) {
    glm::vec3 sp = sphere_dir * 1000.0f;
    constexpr uint32_t seed = 42;

    // ===== LAYER 1: TECTONIC PLATES =====
    CpuWorleyResult plate = cpu_worley3d(sphere_dir * PLATE_FREQ, seed * 0.07f);

    float continental_A = (cpu_hash31(plate.cell_A + 77.7f) >= 0.45f) ? 1.0f : 0.0f;
    float plate_base = glm::mix(-200.0f, 1000.0f, continental_A);

    float boundary = 1.0f - glm::smoothstep(0.0f, BOUNDARY_WIDTH, plate.F2 - plate.F1);

    glm::vec3 vel_A = cpu_hash33(plate.cell_A + seed * 0.31f) * 2.0f - 1.0f;
    glm::vec3 vel_B = cpu_hash33(plate.cell_B + seed * 0.31f) * 2.0f - 1.0f;
    glm::vec3 bn = glm::normalize(plate.cell_B - plate.cell_A + 1e-6f);
    float approach = glm::dot(vel_A - vel_B, bn);
    float convergent = glm::smoothstep(-0.2f, 0.2f, approach);

    float mountain_h = boundary * convergent * 3500.0f;
    float rift_h = boundary * (1.0f - convergent) * -600.0f;

    float cont_swell = (cpu_fbm3d(sp * 0.0003f, 4, 2.0f, 0.5f) - 0.5f) * 800.0f;

    float tectonic_h = plate_base + mountain_h + rift_h + cont_swell;

    // ===== LAYER 2: DRAINAGE BASINS =====
    CpuWorleyResult basin_maj = cpu_worley3d(sphere_dir * BASIN_FREQ_MAJ, seed * 0.13f + 1000.0f);
    CpuWorleyResult basin_min = cpu_worley3d(sphere_dir * BASIN_FREQ_MIN, seed * 0.19f + 2000.0f);

    float ridge_maj = glm::smoothstep(RIDGE_SHARP_MAJ, 0.0f, basin_maj.F2 - basin_maj.F1) * RIDGE_H_MAJ;

    float in_basin = glm::smoothstep(0.0f, 0.15f, basin_maj.F2 - basin_maj.F1);
    float ridge_min = glm::smoothstep(RIDGE_SHARP_MIN, 0.0f, basin_min.F2 - basin_min.F1)
                    * RIDGE_H_MIN * in_basin;

    float slope_maj = std::pow(std::clamp(basin_maj.F1 * 3.0f, 0.0f, 1.0f), 0.6f) * BASIN_DEPTH_MAJ;
    float slope_min = std::pow(std::clamp(basin_min.F1 * 5.0f, 0.0f, 1.0f), 0.6f) * BASIN_DEPTH_MIN * in_basin;

    // ===== LAYER 3: VALLEY PROFILE =====
    float valley_flat = glm::smoothstep(VALLEY_FLOOR_W, 0.0f, basin_maj.F1) * VALLEY_FLOOR_H;
    float valley_flat_min = glm::smoothstep(VALLEY_FLOOR_W, 0.0f, basin_min.F1)
                          * (VALLEY_FLOOR_H * 0.4f) * in_basin;

    float drainage_h = ridge_maj + ridge_min + slope_maj + slope_min - valley_flat - valley_flat_min;

    float mtn_detail = cpu_ridged3d(sp * 0.006f, 5) * 1500.0f * boundary * convergent;

    // ===== LAYER 4: SURFACE DETAIL =====
    float detail = 0.0f;
    detail += cpu_ridged3d(sp * 0.4f, 5) * 200.0f;
    detail += (cpu_fbm3d(sp * 0.08f, 6, 2.0f, 0.5f) - 0.5f) * 300.0f;
    detail += (cpu_gradient_noise_3d(sp * 0.15f) - 0.5f) * 40.0f;
    detail += (cpu_fbm3d(sp * 1.6f, 3, 2.0f, 0.5f) - 0.5f) * 150.0f;
    detail += (cpu_gradient_noise_3d(sp * 13.0f) - 0.5f) * 30.0f;
    detail += (cpu_gradient_noise_3d(sp * 80.0f) - 0.5f) * 8.0f;
    detail += (cpu_gradient_noise_3d(sp * 640.0f) - 0.5f) * 1.5f;

    float h = tectonic_h + drainage_h + mtn_detail + detail;
    return std::clamp(h, -2000.0f, 8000.0f);
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

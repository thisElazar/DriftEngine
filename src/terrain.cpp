#include "terrain.h"
#include "pipeline.h"
#include <algorithm>
#include <cmath>

float cpu_hash31(glm::vec3 p) {
    p = glm::fract(p * glm::vec3(0.1031f, 0.1030f, 0.0973f));
    p += glm::dot(p, glm::vec3(p.y, p.x, p.z) + 31.32f);
    return glm::fract((p.x + p.y) * p.z);
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

float cpu_terrain_height(glm::vec3 sphere_dir) {
    glm::vec3 sp = sphere_dir * 1000.0f;
    float latitude = std::abs(sphere_dir.y);

    float base = 200.0f + cpu_fbm3d(sp * 0.0003f, 5, 2.0f, 0.5f) * 1500.0f;

    float biome = cpu_fbm3d(sp * 0.0005f + glm::vec3(7.7f, 0, 0), 3, 2.0f, 0.5f);

    float mtn_w    = glm::smoothstep(0.55f, 0.70f, biome);
    float desert_w = glm::smoothstep(0.35f, 0.50f, biome) * glm::smoothstep(0.55f, 0.40f, biome)
                   * glm::smoothstep(0.65f, 0.15f, latitude);
    float plains_w = glm::smoothstep(0.45f, 0.25f, biome);
    float polar_w  = glm::smoothstep(0.55f, 0.80f, latitude);

    float mountain = cpu_ridged3d(sp * 0.006f, 7) * 4500.0f;
    mountain += cpu_fbm3d(sp * 0.003f, 5, 2.0f, 0.55f) * 2000.0f;

    float desert = cpu_fbm3d(sp * 0.005f, 5, 2.0f, 0.5f) * 500.0f;
    desert += std::abs(cpu_gradient_noise_3d(sp * 0.03f)) * 250.0f;
    desert += (cpu_gradient_noise_3d(sp * 0.1f) - 0.5f) * 80.0f;

    float plains = cpu_fbm3d(sp * 0.004f, 5, 2.0f, 0.5f) * 400.0f;
    plains += cpu_fbm3d(sp * 0.02f, 3, 2.0f, 0.4f) * 100.0f;

    float polar = cpu_fbm3d(sp * 0.003f, 4, 2.0f, 0.5f) * 800.0f;
    polar += std::abs(cpu_gradient_noise_3d(sp * 0.02f) - 0.5f) * 200.0f;

    float total_w = std::max(mtn_w + desert_w + plains_w + polar_w, 0.01f);
    float biome_h = (mountain * mtn_w + desert * desert_w + plains * plains_w + polar * polar_w) / total_w;

    float h = base + biome_h;
    h += (cpu_gradient_noise_3d(sp * 0.15f) - 0.5f) * 40.0f;

    return h;
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

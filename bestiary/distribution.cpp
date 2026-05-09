#include "distribution.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace bestiary {

static uint32_t dhash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

static float dhash_float(uint32_t seed, uint32_t index)
{
    return static_cast<float>(dhash(seed ^ (index * 2654435761u))) /
           static_cast<float>(0xFFFFFFFFu);
}

static float trapezoid(float v, float lo, float opt_lo, float opt_hi, float hi)
{
    if (v <= lo || v >= hi) return 0.0f;
    if (v >= opt_lo && v <= opt_hi) return 1.0f;
    if (v < opt_lo) return (v - lo) / (opt_lo - lo);
    return (hi - v) / (hi - opt_hi);
}

float compute_suitability(const SpeciesSuitability& s,
                          const EnvironmentSample& env)
{
    float m = trapezoid(env.moisture,
                        s.moisture_min, s.moisture_opt_lo,
                        s.moisture_opt_hi, s.moisture_max);
    float t = trapezoid(env.temperature,
                        s.temp_min, s.temp_opt_lo,
                        s.temp_opt_hi, s.temp_max);
    return m * t;
}

// -----------------------------------------------------------------------
// Poisson disk sampling with variable density
// -----------------------------------------------------------------------

struct DiskPoint { float x, z; };

static float density_at(float x, float z,
                        const EnvironmentField& env,
                        const EcosystemParams& eco)
{
    auto s = env(x, z);
    float max_d = 0.0f;

    if (eco.grass_enabled) {
        float suit = compute_suitability(eco.grass_suit, s);
        max_d = std::max(max_d, suit * eco.grass_suit.base_density);
    }
    if (eco.bush_enabled) {
        float suit = compute_suitability(eco.bush_suit, s);
        max_d = std::max(max_d, suit * eco.bush_suit.base_density);
    }
    if (eco.tree_enabled) {
        float suit = compute_suitability(eco.tree_suit, s);
        max_d = std::max(max_d, suit * eco.tree_suit.base_density);
    }

    float max_possible = 0.0f;
    if (eco.grass_enabled) max_possible = std::max(max_possible, eco.grass_suit.base_density);
    if (eco.bush_enabled)  max_possible = std::max(max_possible, eco.bush_suit.base_density);
    if (eco.tree_enabled)  max_possible = std::max(max_possible, eco.tree_suit.base_density);
    if (max_possible < 0.001f) return 0.0f;

    return std::clamp(max_d / max_possible * eco.density_scale, 0.0f, 1.0f);
}

static std::vector<DiskPoint> poisson_disk(float region_size,
                                           float r_min, float r_max,
                                           const EnvironmentField& env,
                                           const EcosystemParams& eco,
                                           uint32_t seed)
{
    float half = region_size * 0.5f;
    float cell = r_min / 1.41421356f;
    int grid_w = static_cast<int>(std::ceil(region_size / cell));
    if (grid_w < 1) grid_w = 1;

    std::vector<int> grid(static_cast<size_t>(grid_w * grid_w), -1);
    std::vector<DiskPoint> points;
    std::vector<int> active;

    auto to_grid = [&](float x, float z) -> std::pair<int, int> {
        int gx = static_cast<int>((x + half) / cell);
        int gz = static_cast<int>((z + half) / cell);
        return {std::clamp(gx, 0, grid_w - 1), std::clamp(gz, 0, grid_w - 1)};
    };

    auto add_point = [&](float x, float z) {
        auto [gx, gz] = to_grid(x, z);
        int idx = static_cast<int>(points.size());
        points.push_back({x, z});
        active.push_back(idx);
        grid[static_cast<size_t>(gz * grid_w + gx)] = idx;
    };

    add_point(0.0f, 0.0f);
    uint32_t rng_counter = 0;

    constexpr float pi = 3.14159265f;
    constexpr int k_candidates = 30;

    while (!active.empty()) {
        uint32_t ai = static_cast<uint32_t>(
            dhash_float(seed, rng_counter++) * static_cast<float>(active.size()));
        if (ai >= active.size()) ai = static_cast<uint32_t>(active.size()) - 1;

        int pi_idx = active[ai];
        const auto& pt = points[static_cast<size_t>(pi_idx)];

        float local_density = density_at(pt.x, pt.z, env, eco);
        float local_r = r_min + (r_max - r_min) * (1.0f - local_density);

        bool found = false;
        for (int c = 0; c < k_candidates; ++c) {
            float angle = dhash_float(seed, rng_counter++) * 2.0f * pi;
            float dist = local_r + dhash_float(seed, rng_counter++) * local_r;

            float cx = pt.x + std::cos(angle) * dist;
            float cz = pt.z + std::sin(angle) * dist;

            if (cx < -half || cx > half || cz < -half || cz > half)
                continue;

            float cand_density = density_at(cx, cz, env, eco);
            float cand_r = r_min + (r_max - r_min) * (1.0f - cand_density);
            float min_sep = std::max(local_r, cand_r);

            auto [gx, gz] = to_grid(cx, cz);
            int search = static_cast<int>(std::ceil(min_sep / cell));
            bool too_close = false;

            for (int dz = -search; dz <= search && !too_close; ++dz) {
                for (int dx = -search; dx <= search && !too_close; ++dx) {
                    int nx = gx + dx, nz_g = gz + dz;
                    if (nx < 0 || nx >= grid_w || nz_g < 0 || nz_g >= grid_w)
                        continue;
                    int ni = grid[static_cast<size_t>(nz_g * grid_w + nx)];
                    if (ni < 0) continue;
                    float ddx = points[static_cast<size_t>(ni)].x - cx;
                    float ddz = points[static_cast<size_t>(ni)].z - cz;
                    if (ddx * ddx + ddz * ddz < min_sep * min_sep)
                        too_close = true;
                }
            }

            if (!too_close) {
                add_point(cx, cz);
                found = true;
                break;
            }
        }

        if (!found) {
            active[ai] = active.back();
            active.pop_back();
        }
    }

    return points;
}

// -----------------------------------------------------------------------
// Species selection
// -----------------------------------------------------------------------

static int select_species(const EnvironmentSample& env,
                          const EcosystemParams& eco,
                          float rand_val)
{
    float weights[3] = {0.0f, 0.0f, 0.0f};
    if (eco.grass_enabled)
        weights[0] = compute_suitability(eco.grass_suit, env) * eco.grass_suit.base_density;
    if (eco.bush_enabled)
        weights[1] = compute_suitability(eco.bush_suit, env) * eco.bush_suit.base_density;
    if (eco.tree_enabled)
        weights[2] = compute_suitability(eco.tree_suit, env) * eco.tree_suit.base_density;

    float total = weights[0] + weights[1] + weights[2];
    if (total < 0.05f) return -1;

    float threshold = rand_val * total;
    float accum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        accum += weights[i];
        if (threshold <= accum) return i;
    }
    return 2;
}

// -----------------------------------------------------------------------
// Ground plane with environment-colored vertices
// -----------------------------------------------------------------------

static void emit_environment_ground(VegetationMesh& mesh,
                                    float region_size,
                                    const EnvironmentField& env)
{
    constexpr int subdivs = 32;
    float half = region_size * 0.5f;
    float step = region_size / static_cast<float>(subdivs);

    auto base_idx = static_cast<uint32_t>(mesh.vertices.size());

    for (int iz = 0; iz <= subdivs; ++iz) {
        for (int ix = 0; ix <= subdivs; ++ix) {
            float x = -half + static_cast<float>(ix) * step;
            float z = -half + static_cast<float>(iz) * step;
            auto s = env(x, z);

            float r = 0.45f + (0.20f - 0.45f) * s.moisture;
            float g = 0.35f + (0.35f - 0.35f) * s.moisture;
            float b = 0.20f + (0.15f - 0.20f) * s.moisture;

            VegetationVertex v{};
            v.position[0] = x; v.position[1] = 0.0f; v.position[2] = z;
            v.normal[0] = 0.0f; v.normal[1] = 1.0f; v.normal[2] = 0.0f;
            v.color[0] = r; v.color[1] = g; v.color[2] = b;
            mesh.vertices.push_back(v);
        }
    }

    constexpr int w = subdivs + 1;
    for (int iz = 0; iz < subdivs; ++iz) {
        for (int ix = 0; ix < subdivs; ++ix) {
            uint32_t i00 = base_idx + static_cast<uint32_t>(iz * w + ix);
            uint32_t i10 = i00 + 1;
            uint32_t i01 = i00 + static_cast<uint32_t>(w);
            uint32_t i11 = i01 + 1;
            mesh.indices.push_back(i00); mesh.indices.push_back(i10); mesh.indices.push_back(i11);
            mesh.indices.push_back(i00); mesh.indices.push_back(i11); mesh.indices.push_back(i01);
        }
    }
}

// -----------------------------------------------------------------------
// Ecosystem assembly
// -----------------------------------------------------------------------

VegetationMesh generate_ecosystem(
    const EcosystemParams& eco,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te)
{
    VegetationMesh mesh;

    auto points = poisson_disk(eco.region_size, eco.r_min, eco.r_max,
                               env, eco, eco.seed);

    for (size_t i = 0; i < points.size(); ++i) {
        float x = points[i].x;
        float z = points[i].z;

        auto sample = env(x, z);
        uint32_t pt_seed = dhash(eco.seed ^ static_cast<uint32_t>(i * 2654435761u));
        float rand_val = dhash_float(pt_seed, 0u);

        int kind = select_species(sample, eco, rand_val);
        if (kind < 0) continue;

        VegetationMesh sub;
        switch (kind) {
        case 0: {
            auto resolved = evaluate_expression(cp, ce, sample.moisture);
            sub = generate_clump(resolved, pt_seed, false, x, z);
            break;
        }
        case 1: {
            auto resolved = evaluate_bush_expression(bp, be, sample.moisture);
            sub = generate_bush(resolved, pt_seed, false, x, z);
            break;
        }
        case 2: {
            auto resolved = evaluate_tree_expression(tp, te, sample.moisture);
            sub = generate_tree(resolved, pt_seed, false, x, z);
            break;
        }
        }

        auto vert_offset = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.insert(mesh.vertices.end(),
                             sub.vertices.begin(), sub.vertices.end());
        for (uint32_t idx : sub.indices)
            mesh.indices.push_back(idx + vert_offset);
    }

    emit_environment_ground(mesh, eco.region_size, env);

    return mesh;
}

} // namespace bestiary

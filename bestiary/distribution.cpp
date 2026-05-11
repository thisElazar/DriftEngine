#include "distribution.h"
#include "creature/vegetation_field.h"
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
// Phenotype jitter — per-instance parameter variation
// -----------------------------------------------------------------------

static float jitter(float base, float variance, uint32_t seed, uint32_t idx)
{
    float r = dhash_float(seed, idx) * 2.0f - 1.0f;
    return base * (1.0f + r * variance);
}

static int jitter_int(int base, float variance, uint32_t seed, uint32_t idx)
{
    float r = dhash_float(seed, idx) * 2.0f - 1.0f;
    return std::max(1, static_cast<int>(std::round(
        static_cast<float>(base) * (1.0f + r * variance))));
}

static void jitter_clump(ClumpParams& p, float v, uint32_t seed)
{
    p.blade_count  = jitter_int(p.blade_count,  v, seed, 10u);
    p.blade_height = jitter(p.blade_height, v, seed, 11u);
    p.blade_width  = jitter(p.blade_width,  v * 0.5f, seed, 12u);
    p.splay_angle  = jitter(p.splay_angle,  v, seed, 13u);
    p.clump_radius = jitter(p.clump_radius, v, seed, 14u);
}

static void jitter_bush(BushParams& p, float v, uint32_t seed)
{
    p.leaf_count   = jitter_int(p.leaf_count,  v, seed, 20u);
    p.bush_height  = jitter(p.bush_height,  v, seed, 21u);
    p.bush_radius  = jitter(p.bush_radius,  v, seed, 22u);
    p.leaf_length  = jitter(p.leaf_length,  v * 0.5f, seed, 23u);
    p.stem_height  = jitter(p.stem_height,  v, seed, 24u);
}

static void jitter_tree(TreeParams& p, float v, uint32_t seed)
{
    p.tree_height   = jitter(p.tree_height,   v, seed, 30u);
    p.trunk_height  = jitter(p.trunk_height,  v, seed, 31u);
    p.crown_radius  = jitter(p.crown_radius,  v, seed, 32u);
    p.crown_height  = jitter(p.crown_height,  v, seed, 33u);
    p.trunk_width   = jitter(p.trunk_width,   v * 0.5f, seed, 34u);
    p.leaf_count    = jitter_int(p.leaf_count, v, seed, 35u);
}

// -----------------------------------------------------------------------
// Ecosystem assembly
// -----------------------------------------------------------------------

VegetationMesh generate_ecosystem(
    const EcosystemParams& eco,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te,
    bool include_ground,
    VegetationDensityField* veg_field)
{
    VegetationMesh mesh;

    if (veg_field) {
        size_t n = static_cast<size_t>(veg_field->grid_w) * veg_field->grid_h;
        veg_field->capacity.assign(n, 0.0f);
    }

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

        if (veg_field) {
            float weight = (kind == 0) ? 0.05f : (kind == 1) ? 0.03f : 0.01f;
            accumulate_vegetation_capacity(*veg_field, x, z, weight);
        }

        VegetationMesh sub;
        switch (kind) {
        case 0: {
            auto resolved = evaluate_expression(cp, ce, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_clump(resolved, eco.phenotype_variance, pt_seed);
            sub = generate_clump(resolved, pt_seed, false, x, z);
            break;
        }
        case 1: {
            auto resolved = evaluate_bush_expression(bp, be, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_bush(resolved, eco.phenotype_variance, pt_seed);
            sub = generate_bush(resolved, pt_seed, false, x, z);
            break;
        }
        case 2: {
            auto resolved = evaluate_tree_expression(tp, te, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_tree(resolved, eco.phenotype_variance, pt_seed);
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

    if (include_ground)
        emit_environment_ground(mesh, eco.region_size, env);

    return mesh;
}

VegetationMesh generate_ecosystem_density_modulated(
    const EcosystemParams& eco,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te,
    const VegetationDensityField& veg_field)
{
    VegetationMesh mesh;

    auto points = poisson_disk(eco.region_size, eco.r_min, eco.r_max,
                               env, eco, eco.seed);

    for (size_t i = 0; i < points.size(); ++i) {
        float x = points[i].x;
        float z = points[i].z;

        float fullness = sample_vegetation_fullness(veg_field, x, z);
        if (fullness < 0.05f) continue;

        auto sample = env(x, z);
        uint32_t pt_seed = dhash(eco.seed ^ static_cast<uint32_t>(i * 2654435761u));
        float rand_val = dhash_float(pt_seed, 0u);

        int kind = select_species(sample, eco, rand_val);
        if (kind < 0) continue;

        VegetationMesh sub;
        switch (kind) {
        case 0: {
            auto resolved = evaluate_expression(cp, ce, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_clump(resolved, eco.phenotype_variance, pt_seed);
            resolved.blade_height *= fullness;
            resolved.blade_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.blade_count) * fullness));
            sub = generate_clump(resolved, pt_seed, false, x, z);
            break;
        }
        case 1: {
            auto resolved = evaluate_bush_expression(bp, be, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_bush(resolved, eco.phenotype_variance, pt_seed);
            resolved.bush_height *= fullness;
            resolved.leaf_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.leaf_count) * fullness));
            sub = generate_bush(resolved, pt_seed, false, x, z);
            break;
        }
        case 2: {
            auto resolved = evaluate_tree_expression(tp, te, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_tree(resolved, eco.phenotype_variance, pt_seed);
            resolved.tree_height *= fullness;
            resolved.leaf_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.leaf_count) * fullness));
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

    return mesh;
}

// -----------------------------------------------------------------------
// Persistent plant population
// -----------------------------------------------------------------------

std::vector<PlantInstance> place_ecosystem(
    const EcosystemParams& eco,
    const EnvironmentField& env)
{
    std::vector<PlantInstance> plants;

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

        plants.push_back({x, z, kind, 0.01f, pt_seed});
    }

    return plants;
}

void tick_plant_population(
    std::vector<PlantInstance>& plants,
    const EnvironmentField& env,
    const EcosystemParams& eco,
    float dt,
    float growth_rate,
    float decay_rate)
{
    for (auto& p : plants) {
        if (p.health <= 0.0f) continue;

        auto sample = env(p.x, p.z);
        const SpeciesSuitability* suit = nullptr;
        switch (p.kind) {
        case 0: suit = &eco.grass_suit; break;
        case 1: suit = &eco.bush_suit;  break;
        case 2: suit = &eco.tree_suit;  break;
        }
        if (!suit) continue;

        float s = compute_suitability(*suit, sample);

        if (s > 0.1f) {
            p.health = std::min(p.health + growth_rate * s * dt, 1.0f);
        } else {
            p.health -= decay_rate * dt;
            if (p.health <= 0.0f) p.health = 0.0f;
        }
    }

    // Remove dead plants
    plants.erase(
        std::remove_if(plants.begin(), plants.end(),
                        [](const PlantInstance& p) { return p.health <= 0.0f; }),
        plants.end());
}

void sprout_plants(
    std::vector<PlantInstance>& plants,
    const EcosystemParams& eco,
    const EnvironmentField& env,
    uint32_t seed,
    int max_sprouts)
{
    float half = eco.region_size * 0.5f;
    float min_dist2 = eco.r_min * eco.r_min;
    constexpr float pi = 3.14159265f;

    // Index existing plants by kind for parent selection
    std::vector<size_t> grass_idx, bush_idx, tree_idx;
    for (size_t i = 0; i < plants.size(); ++i) {
        if (plants[i].health < 0.3f) continue; // only healthy plants reproduce
        switch (plants[i].kind) {
        case 0: grass_idx.push_back(i); break;
        case 1: bush_idx.push_back(i);  break;
        case 2: tree_idx.push_back(i);  break;
        }
    }

    int sprouted = 0;

    for (int attempt = 0; attempt < max_sprouts * 5 && sprouted < max_sprouts; ++attempt) {
        uint32_t s = dhash(seed ^ static_cast<uint32_t>(attempt * 2654435761u));

        // Weighted species selection: more grass sprouts, fewer trees
        float species_roll = dhash_float(s, 10);
        int kind;
        if (species_roll < 0.55f)       kind = 0; // grass: 55%
        else if (species_roll < 0.80f)  kind = 1; // bush: 25%
        else                            kind = 2; // tree: 20%

        if (kind == 0 && !eco.grass_enabled) continue;
        if (kind == 1 && !eco.bush_enabled)  continue;
        if (kind == 2 && !eco.tree_enabled)  continue;

        float x, z;

        if (kind == 0 && !grass_idx.empty()) {
            // Grass: spread from existing grass (runner/rhizome, 1-3m)
            size_t pi_idx = static_cast<size_t>(
                dhash_float(s, 0) * static_cast<float>(grass_idx.size()));
            if (pi_idx >= grass_idx.size()) pi_idx = grass_idx.size() - 1;
            const auto& parent = plants[grass_idx[pi_idx]];
            float angle = dhash_float(s, 1) * 2.0f * pi;
            float dist = 1.0f + dhash_float(s, 2) * 2.0f;
            x = parent.x + std::cos(angle) * dist;
            z = parent.z + std::sin(angle) * dist;
        } else if (kind == 2 && !tree_idx.empty() && dhash_float(s, 3) < 0.85f) {
            // Trees: seed shadow near parent (3-8m), 85% of the time
            size_t pi_idx = static_cast<size_t>(
                dhash_float(s, 0) * static_cast<float>(tree_idx.size()));
            if (pi_idx >= tree_idx.size()) pi_idx = tree_idx.size() - 1;
            const auto& parent = plants[tree_idx[pi_idx]];
            float angle = dhash_float(s, 1) * 2.0f * pi;
            float dist = 3.0f + dhash_float(s, 2) * 5.0f;
            x = parent.x + std::cos(angle) * dist;
            z = parent.z + std::sin(angle) * dist;
        } else {
            // Bushes + fallback: random placement (wind/bird dispersal)
            x = (dhash_float(s, 0) * 2.0f - 1.0f) * half;
            z = (dhash_float(s, 1) * 2.0f - 1.0f) * half;
        }

        // Bounds check
        if (x < -half || x > half || z < -half || z > half) continue;

        auto sample = env(x, z);
        const SpeciesSuitability* suit = nullptr;
        switch (kind) {
        case 0: suit = &eco.grass_suit; break;
        case 1: suit = &eco.bush_suit;  break;
        case 2: suit = &eco.tree_suit;  break;
        }
        if (!suit) continue;
        if (compute_suitability(*suit, sample) < 0.3f) continue;

        bool too_close = false;
        for (const auto& p : plants) {
            float dx = p.x - x;
            float dz = p.z - z;
            if (dx * dx + dz * dz < min_dist2) {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;

        uint32_t pt_seed = dhash(s ^ 0x12345678u);
        plants.push_back({x, z, kind, 0.01f, pt_seed});
        ++sprouted;
    }
}

VegetationMesh generate_mesh_from_population(
    const std::vector<PlantInstance>& plants,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te,
    float phenotype_variance)
{
    VegetationMesh mesh;

    for (const auto& p : plants) {
        if (p.health < 0.01f) continue;

        auto sample = env(p.x, p.z);

        VegetationMesh sub;
        switch (p.kind) {
        case 0: {
            auto resolved = evaluate_expression(cp, ce, sample.moisture);
            if (phenotype_variance > 0.001f)
                jitter_clump(resolved, phenotype_variance, p.seed);
            resolved.blade_height *= p.health;
            resolved.blade_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.blade_count) * p.health));
            sub = generate_clump(resolved, p.seed, false, p.x, p.z);
            break;
        }
        case 1: {
            auto resolved = evaluate_bush_expression(bp, be, sample.moisture);
            if (phenotype_variance > 0.001f)
                jitter_bush(resolved, phenotype_variance, p.seed);
            resolved.bush_height *= p.health;
            resolved.leaf_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.leaf_count) * p.health));
            sub = generate_bush(resolved, p.seed, false, p.x, p.z);
            break;
        }
        case 2: {
            auto resolved = evaluate_tree_expression(tp, te, sample.moisture);
            if (phenotype_variance > 0.001f)
                jitter_tree(resolved, phenotype_variance, p.seed);
            resolved.tree_height *= p.health;
            resolved.leaf_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.leaf_count) * p.health));
            sub = generate_tree(resolved, p.seed, false, p.x, p.z);
            break;
        }
        }

        auto vert_offset = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.insert(mesh.vertices.end(),
                             sub.vertices.begin(), sub.vertices.end());
        for (uint32_t idx : sub.indices)
            mesh.indices.push_back(idx + vert_offset);
    }

    return mesh;
}

} // namespace bestiary

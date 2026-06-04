#include "distribution.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
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

    const SpeciesSuitability* suits[] = {
        eco.grass_enabled      ? &eco.grass_suit      : nullptr,
        eco.bush_enabled       ? &eco.bush_suit       : nullptr,
        eco.tree_enabled       ? &eco.tree_suit       : nullptr,
        eco.reed_enabled       ? &eco.reed_suit       : nullptr,
        eco.wildflower_enabled ? &eco.wildflower_suit  : nullptr,
    };
    for (auto* sp : suits) {
        if (sp) max_d = std::max(max_d, compute_suitability(*sp, s) * sp->base_density);
    }

    float max_possible = 0.0f;
    for (auto* sp : suits) {
        if (sp) max_possible = std::max(max_possible, sp->base_density);
    }
    if (max_possible < 0.001f) return 0.0f;

    return std::clamp(max_d / max_possible * eco.density_scale, 0.0f, 1.0f);
}

static std::vector<DiskPoint> poisson_disk(float region_size,
                                           float r_min, float r_max,
                                           const std::function<float(float, float)>& density_fn,
                                           uint32_t seed)
{
    // Degenerate-radius guard: a tiny r_min blows up the acceleration grid
    // (region_size/cell cells per side), risking OOM/hang. Clamp before sizing.
    r_min = std::max(r_min, 0.05f);
    if (r_max < r_min) r_max = r_min;

    float half = region_size * 0.5f;
    float cell = r_min / 1.41421356f;
    int grid_w = std::clamp(static_cast<int>(std::ceil(region_size / cell)), 1, 4096);

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

        float local_density = density_fn(pt.x, pt.z);
        float local_r = r_min + (r_max - r_min) * (1.0f - local_density);

        bool found = false;
        for (int c = 0; c < k_candidates; ++c) {
            float angle = dhash_float(seed, rng_counter++) * 2.0f * pi;
            float dist = local_r + dhash_float(seed, rng_counter++) * local_r;

            float cx = pt.x + std::cos(angle) * dist;
            float cz = pt.z + std::sin(angle) * dist;

            if (cx < -half || cx > half || cz < -half || cz > half)
                continue;

            float cand_density = density_fn(cx, cz);
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
    float weights[PLANT_KIND_COUNT] = {};
    if (eco.grass_enabled)
        weights[PLANT_GRASS] = compute_suitability(eco.grass_suit, env) * eco.grass_suit.base_density;
    if (eco.bush_enabled)
        weights[PLANT_BUSH] = compute_suitability(eco.bush_suit, env) * eco.bush_suit.base_density;
    if (eco.tree_enabled)
        weights[PLANT_TREE] = compute_suitability(eco.tree_suit, env) * eco.tree_suit.base_density;
    if (eco.reed_enabled)
        weights[PLANT_REED] = compute_suitability(eco.reed_suit, env) * eco.reed_suit.base_density;
    if (eco.wildflower_enabled)
        weights[PLANT_WILDFLOWER] = compute_suitability(eco.wildflower_suit, env) * eco.wildflower_suit.base_density;

    float total = 0.0f;
    for (int i = 0; i < PLANT_KIND_COUNT; ++i) total += weights[i];
    if (total < 0.05f) return -1;

    float threshold = rand_val * total;
    float accum = 0.0f;
    for (int i = 0; i < PLANT_KIND_COUNT; ++i) {
        accum += weights[i];
        if (threshold <= accum) return i;
    }
    return PLANT_GRASS;
}

// ---------- roster-based placement (the N-species path) -----------------

// Normalized planting density [0,1] at (x,z): best suitability×base_density
// across the roster, scaled by the roster's peak base_density and eco scale.
static float density_at_roster(float x, float z,
                               const EnvironmentField& env,
                               const std::vector<PlantSpecies>& roster,
                               const EcosystemParams& eco)
{
    if (roster.empty()) return 0.0f;
    auto s = env(x, z);
    float max_d = 0.0f, max_possible = 0.0f;
    for (const auto& sp : roster) {
        max_d = std::max(max_d, compute_suitability(sp.suit, s) * sp.suit.base_density);
        max_possible = std::max(max_possible, sp.suit.base_density);
    }
    if (max_possible < 0.001f) return 0.0f;
    return std::clamp(max_d / max_possible * eco.density_scale, 0.0f, 1.0f);
}

// Pick a roster index for a point, weighted by suitability×base_density. -1 if
// nothing is viable here.
static int select_species_roster(const EnvironmentSample& env,
                                  const std::vector<PlantSpecies>& roster,
                                  float rand_val)
{
    float total = 0.0f;
    for (const auto& sp : roster)
        total += compute_suitability(sp.suit, env) * sp.suit.base_density;
    if (total < 0.05f) return -1;

    float threshold = rand_val * total;
    float accum = 0.0f;
    for (size_t i = 0; i < roster.size(); ++i) {
        accum += compute_suitability(roster[i].suit, env) * roster[i].suit.base_density;
        if (threshold <= accum) return static_cast<int>(i);
    }
    return 0;
}

// Is this kind ground-cover (spreads from a parent) vs. canopy (clusters)?
static bool kind_is_ground_cover(int kind)
{
    return kind == PLANT_GRASS || kind == PLANT_REED || kind == PLANT_WILDFLOWER;
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
    const PlantMeshParams& pmp,
    bool include_ground)
{
    VegetationMesh mesh;

    auto points = poisson_disk(eco.region_size, eco.r_min, eco.r_max,
                               [&](float x, float z) { return density_at(x, z, env, eco); },
                               eco.seed);

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
        case PLANT_GRASS: {
            auto resolved = evaluate_expression(pmp.cp, pmp.ce, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_clump(resolved, eco.phenotype_variance, pt_seed);
            sub = generate_clump(resolved, pt_seed, false, x, z);
            break;
        }
        case PLANT_BUSH: {
            auto resolved = evaluate_bush_expression(pmp.bp, pmp.be, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_bush(resolved, eco.phenotype_variance, pt_seed);
            sub = generate_bush(resolved, pt_seed, false, x, z);
            break;
        }
        case PLANT_TREE: {
            auto resolved = evaluate_tree_expression(pmp.tp, pmp.te, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_tree(resolved, eco.phenotype_variance, pt_seed);
            sub = generate_tree(resolved, pt_seed, false, x, z);
            break;
        }
        case PLANT_REED: {
            auto resolved = evaluate_expression(pmp.reed_cp, pmp.reed_ce, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_clump(resolved, eco.phenotype_variance, pt_seed);
            sub = generate_clump(resolved, pt_seed, false, x, z);
            break;
        }
        case PLANT_WILDFLOWER: {
            auto resolved = evaluate_wildflower_expression(pmp.wfp, pmp.wfe, sample.moisture);
            if (eco.phenotype_variance > 0.001f)
                jitter_wildflower(resolved, eco.phenotype_variance, pt_seed);
            sub = generate_wildflower(resolved, pt_seed, false, x, z);
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

VegetationMesh generate_ecosystem(
    const EcosystemParams& eco,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te,
    bool include_ground)
{
    PlantMeshParams pmp;
    pmp.cp = cp; pmp.ce = ce;
    pmp.bp = bp; pmp.be = be;
    pmp.tp = tp; pmp.te = te;
    pmp.reed_cp = cp; pmp.reed_ce = ce;
    pmp.wfp = {}; pmp.wfe = {};
    pmp.phenotype_variance = eco.phenotype_variance;
    return generate_ecosystem(eco, env, pmp, include_ground);
}

// -----------------------------------------------------------------------
// Persistent plant population
// -----------------------------------------------------------------------

std::vector<PlantInstance> place_ecosystem(
    const std::vector<PlantSpecies>& roster,
    const EcosystemParams& eco,
    const EnvironmentField& env)
{
    std::vector<PlantInstance> plants;
    if (roster.empty()) return plants;

    auto points = poisson_disk(eco.region_size, eco.r_min, eco.r_max,
                               [&](float x, float z) { return density_at_roster(x, z, env, roster, eco); },
                               eco.seed);

    for (size_t i = 0; i < points.size(); ++i) {
        float x = points[i].x;
        float z = points[i].z;

        auto sample = env(x, z);
        uint32_t pt_seed = dhash(eco.seed ^ static_cast<uint32_t>(i * 2654435761u));
        float rand_val = dhash_float(pt_seed, 0u);

        int si = select_species_roster(sample, roster, rand_val);
        if (si < 0) continue;

        plants.push_back({x, z, roster[static_cast<size_t>(si)].kind, 0.01f, pt_seed, si});
    }

    return plants;
}

void resolve_plant_species(std::vector<PlantInstance>& plants,
                           const std::vector<PlantSpecies>& roster)
{
    for (auto& p : plants) {
        if (p.species >= 0 && p.species < static_cast<int>(roster.size())) continue;
        int match = -1;
        for (size_t i = 0; i < roster.size(); ++i) {
            if (roster[i].kind == p.kind) { match = static_cast<int>(i); break; }
        }
        if (match < 0) { p.health = 0.0f; continue; }  // no species of this kind → dies
        p.species = match;
    }
}

void tick_plant_population(
    std::vector<PlantInstance>& plants,
    const std::vector<PlantSpecies>& roster,
    const EnvironmentField& env,
    float dt,
    float growth_rate,
    float decay_rate)
{
    for (auto& p : plants) {
        if (p.health <= 0.0f) continue;
        if (p.species < 0 || p.species >= static_cast<int>(roster.size())) continue;

        auto sample = env(p.x, p.z);
        float s = compute_suitability(roster[static_cast<size_t>(p.species)].suit, sample);

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

void collect_plant_instances(const std::vector<PlantInstance>& plants,
                              int species_index,
                              std::vector<PlantGPUInstance>& out)
{
    out.clear();
    for (const auto& p : plants)
        if (p.species == species_index && p.health > 0.0f)
            out.push_back({p.x, 0.0f, p.z, p.health, p.seed});
}

void sprout_plants(
    std::vector<PlantInstance>& plants,
    const std::vector<PlantSpecies>& roster,
    const EcosystemParams& eco,
    const EnvironmentField& env,
    uint32_t seed,
    int max_sprouts)
{
    if (roster.empty()) return;

    float half = eco.region_size * 0.5f;
    float min_dist2 = eco.r_min * eco.r_min;
    constexpr float pi = 3.14159265f;

    // Spatial grid for O(1) minimum-distance checks.
    // Cell size = r_min so only a 3x3 neighborhood needs checking. Clamp the
    // cell (a tiny r_min would blow up the grid → OOM/hang) and cap the side.
    float cell = std::max(eco.r_min, 0.05f);
    int grid_w = std::clamp(static_cast<int>(std::ceil(eco.region_size / cell)) + 2, 1, 4096);
    auto cell_of = [&](float px, float pz) -> std::pair<int,int> {
        int ix = static_cast<int>((px + half) / cell);
        int iz = static_cast<int>((pz + half) / cell);
        return {std::clamp(ix, 0, grid_w - 1), std::clamp(iz, 0, grid_w - 1)};
    };
    std::unordered_map<int, std::vector<uint32_t>> grid;
    grid.reserve(plants.size() * 2);
    for (uint32_t i = 0; i < static_cast<uint32_t>(plants.size()); ++i) {
        auto [ix, iz] = cell_of(plants[i].x, plants[i].z);
        grid[iz * grid_w + ix].push_back(i);
    }

    // Index established plants by roster species for parent selection.
    std::vector<std::vector<size_t>> species_parents(roster.size());
    for (size_t i = 0; i < plants.size(); ++i) {
        if (plants[i].health < 0.3f) continue;
        int sp = plants[i].species;
        if (sp >= 0 && sp < static_cast<int>(roster.size()))
            species_parents[static_cast<size_t>(sp)].push_back(i);
    }

    // Cumulative base-density weights for picking which species sprouts.
    float weight_total = 0.0f;
    for (const auto& sp : roster) weight_total += sp.suit.base_density;
    if (weight_total < 1e-4f) return;

    int sprouted = 0;

    for (int attempt = 0; attempt < max_sprouts * 5 && sprouted < max_sprouts; ++attempt) {
        uint32_t s = dhash(seed ^ static_cast<uint32_t>(attempt * 2654435761u));

        // Pick a roster species weighted by base_density.
        float roll = dhash_float(s, 10) * weight_total;
        int si = 0;
        float accum = 0.0f;
        for (size_t i = 0; i < roster.size(); ++i) {
            accum += roster[i].suit.base_density;
            if (roll <= accum) { si = static_cast<int>(i); break; }
        }
        int kind = roster[static_cast<size_t>(si)].kind;
        const auto& parents = species_parents[static_cast<size_t>(si)];

        float x, z;
        if (kind_is_ground_cover(kind) && !parents.empty()) {
            // Ground cover: spread from a parent of the same species (1-3m).
            size_t pi_idx = static_cast<size_t>(
                dhash_float(s, 0) * static_cast<float>(parents.size()));
            if (pi_idx >= parents.size()) pi_idx = parents.size() - 1;
            const auto& parent = plants[parents[pi_idx]];
            float angle = dhash_float(s, 1) * 2.0f * pi;
            float dist = 1.0f + dhash_float(s, 2) * 2.0f;
            x = parent.x + std::cos(angle) * dist;
            z = parent.z + std::sin(angle) * dist;
        } else if (!kind_is_ground_cover(kind) && !parents.empty() && dhash_float(s, 3) < 0.85f) {
            // Canopy/shrub: cluster near a parent of the same species (3-8m).
            size_t pi_idx = static_cast<size_t>(
                dhash_float(s, 0) * static_cast<float>(parents.size()));
            if (pi_idx >= parents.size()) pi_idx = parents.size() - 1;
            const auto& parent = plants[parents[pi_idx]];
            float angle = dhash_float(s, 1) * 2.0f * pi;
            float dist = 3.0f + dhash_float(s, 2) * 5.0f;
            x = parent.x + std::cos(angle) * dist;
            z = parent.z + std::sin(angle) * dist;
        } else {
            x = (dhash_float(s, 0) * 2.0f - 1.0f) * half;
            z = (dhash_float(s, 1) * 2.0f - 1.0f) * half;
        }

        if (x < -half || x > half || z < -half || z > half) continue;

        auto sample = env(x, z);
        if (compute_suitability(roster[static_cast<size_t>(si)].suit, sample) < 0.3f) continue;

        // Spatial grid neighborhood check (3x3 cells)
        auto [cx, cz] = cell_of(x, z);
        bool too_close = false;
        for (int dr = -1; dr <= 1 && !too_close; ++dr) {
            for (int dc = -1; dc <= 1 && !too_close; ++dc) {
                int nr = cz + dr, nc = cx + dc;
                if (nc < 0 || nc >= grid_w || nr < 0 || nr >= grid_w) continue;
                auto it = grid.find(nr * grid_w + nc);
                if (it == grid.end()) continue;
                for (uint32_t pidx : it->second) {
                    float fdx = plants[pidx].x - x;
                    float fdz = plants[pidx].z - z;
                    if (fdx * fdx + fdz * fdz < min_dist2) { too_close = true; break; }
                }
            }
        }
        if (too_close) continue;

        uint32_t pt_seed = dhash(s ^ 0x12345678u);
        plants.push_back({x, z, kind, 0.01f, pt_seed, si});
        auto [ax, az] = cell_of(x, z);
        grid[az * grid_w + ax].push_back(static_cast<uint32_t>(plants.size() - 1));
        ++sprouted;
    }
}

VegetationMesh generate_mesh_from_population(
    const std::vector<PlantInstance>& plants,
    const EnvironmentField& env,
    const PlantMeshParams& pmp)
{
    VegetationMesh mesh;

    for (const auto& p : plants) {
        if (p.health < 0.01f) continue;

        auto sample = env(p.x, p.z);
        float pv = pmp.phenotype_variance;

        VegetationMesh sub;
        switch (p.kind) {
        case PLANT_GRASS: {
            auto resolved = evaluate_expression(pmp.cp, pmp.ce, sample.moisture);
            if (pv > 0.001f) jitter_clump(resolved, pv, p.seed);
            resolved.blade_height *= p.health;
            resolved.blade_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.blade_count) * p.health));
            sub = generate_clump(resolved, p.seed, false, p.x, p.z);
            break;
        }
        case PLANT_BUSH: {
            auto resolved = evaluate_bush_expression(pmp.bp, pmp.be, sample.moisture);
            if (pv > 0.001f) jitter_bush(resolved, pv, p.seed);
            resolved.bush_height *= p.health;
            resolved.leaf_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.leaf_count) * p.health));
            sub = generate_bush(resolved, p.seed, false, p.x, p.z);
            break;
        }
        case PLANT_TREE: {
            auto resolved = evaluate_tree_expression(pmp.tp, pmp.te, sample.moisture);
            if (pv > 0.001f) jitter_tree(resolved, pv, p.seed);
            resolved.tree_height *= p.health;
            resolved.leaf_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.leaf_count) * p.health));
            sub = generate_tree(resolved, p.seed, false, p.x, p.z);
            break;
        }
        case PLANT_REED: {
            auto resolved = evaluate_expression(pmp.reed_cp, pmp.reed_ce, sample.moisture);
            if (pv > 0.001f) jitter_clump(resolved, pv, p.seed);
            resolved.blade_height *= p.health;
            resolved.blade_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.blade_count) * p.health));
            sub = generate_clump(resolved, p.seed, false, p.x, p.z);
            break;
        }
        case PLANT_WILDFLOWER: {
            auto resolved = evaluate_wildflower_expression(pmp.wfp, pmp.wfe, sample.moisture);
            if (pv > 0.001f) jitter_wildflower(resolved, pv, p.seed);
            resolved.stem_height *= p.health;
            resolved.flower_count = std::max(1,
                static_cast<int>(static_cast<float>(resolved.flower_count) * p.health));
            sub = generate_wildflower(resolved, p.seed, false, p.x, p.z);
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

VegetationMesh generate_mesh_from_population(
    const std::vector<PlantInstance>& plants,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te,
    float phenotype_variance)
{
    PlantMeshParams pmp;
    pmp.cp = cp; pmp.ce = ce;
    pmp.bp = bp; pmp.be = be;
    pmp.tp = tp; pmp.te = te;
    pmp.reed_cp = cp; pmp.reed_ce = ce;
    pmp.wfp = {}; pmp.wfe = {};
    pmp.phenotype_variance = phenotype_variance;
    return generate_mesh_from_population(plants, env, pmp);
}

} // namespace bestiary


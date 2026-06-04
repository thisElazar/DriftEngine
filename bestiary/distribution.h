#pragma once

#include "environment.h"
#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"
#include "morphology/wildflower.h"
#include "morphology/lplant.h"

#include <functional>
#include <string>
#include <vector>

namespace bestiary {

struct SpeciesSuitability {
    float moisture_min    = 0.0f;
    float moisture_opt_lo = 0.2f;
    float moisture_opt_hi = 0.8f;
    float moisture_max    = 1.0f;

    float temp_min    = 0.0f;
    float temp_opt_lo = 0.2f;
    float temp_opt_hi = 0.8f;
    float temp_max    = 1.0f;

    float base_density = 1.0f;
};

struct EcosystemParams {
    float    region_size   = 10.0f;
    float    density_scale = 1.0f;
    float    r_min         = 0.3f;
    float    r_max         = 3.0f;
    uint32_t seed          = 0;

    bool grass_enabled = true;
    bool bush_enabled  = true;
    bool tree_enabled  = true;
    bool reed_enabled  = true;
    bool wildflower_enabled = true;

    SpeciesSuitability grass_suit = {0.0f, 0.0f, 0.85f, 1.0f,
                                     0.0f, 0.1f, 0.9f, 1.0f, 4.0f};
    SpeciesSuitability bush_suit  = {0.05f, 0.15f, 0.85f, 0.95f,
                                     0.1f, 0.2f, 0.85f, 0.9f, 1.0f};
    SpeciesSuitability tree_suit  = {0.3f, 0.4f, 0.9f, 1.0f,
                                     0.15f, 0.25f, 0.8f, 0.85f, 0.15f};
    // Reeds thrive in wet areas — shallow water edges and marshland
    SpeciesSuitability reed_suit  = {0.55f, 0.70f, 0.95f, 1.0f,
                                     0.0f, 0.1f, 0.9f, 1.0f, 3.0f};
    // Wildflowers prefer moderate moisture, wide temperature tolerance
    SpeciesSuitability wildflower_suit = {0.0f, 0.05f, 0.70f, 0.90f,
                                          0.0f, 0.1f, 0.9f, 1.0f, 2.5f};

    float phenotype_variance = 0.2f;
};

float compute_suitability(const SpeciesSuitability& s,
                          const EnvironmentSample& env);

// 0=grass, 1=bush, 2=tree, 3=reed, 4=wildflower
constexpr int PLANT_GRASS      = 0;
constexpr int PLANT_BUSH       = 1;
constexpr int PLANT_TREE       = 2;
constexpr int PLANT_REED       = 3;
constexpr int PLANT_WILDFLOWER = 4;
constexpr int PLANT_KIND_COUNT = 5;   // baked preview path (generate_ecosystem) only
constexpr int PLANT_LPLANT     = 5;   // semantic tag for L-system plants in the roster

// ------------------------------------------------------------------------
// PlantSpecies roster — the data-driven replacement for the fixed-kind plant
// model. Each species carries a semantic `kind` (for trophic links + growth
// rules), a `suit` (placement/growth tolerance), and a type-erased `gen`
// closure that builds its mesh from (moisture, seed, x, z). Adding a new plant
// type means registering one generator — no switch-on-kind anywhere. The
// roster (std::vector<PlantSpecies>) is built by scanning the species library;
// see apps/world_lab and docs/WORLD_LAB_INTEGRATION.md.
// ------------------------------------------------------------------------
struct PlantSpecies {
    std::string        name;
    int                kind = PLANT_GRASS;   // PLANT_GRASS..PLANT_LPLANT
    SpeciesSuitability suit{};
    std::function<VegetationMesh(float moisture, uint32_t seed,
                                 float x, float z)> gen;
};

// Legacy overload (grass/bush/tree only)
VegetationMesh generate_ecosystem(
    const EcosystemParams& eco,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te,
    bool include_ground = true);

// -----------------------------------------------------------------------
// Persistent plant population
// -----------------------------------------------------------------------

struct PlantInstance {
    float    x, z;
    int      kind;          // PLANT_GRASS..PLANT_LPLANT (semantic; for trophic/growth)
    float    health;        // 0..1; 0 = dead
    uint32_t seed;
    int      species = -1;  // index into the PlantSpecies roster; -1 = unresolved
                            // (e.g. a creature-dispersed seed; resolve by kind).
};

// Flat per-instance data uploaded to the GPU for instanced plant rendering.
struct PlantGPUInstance {
    float    x, y, z;   // world position; y = terrain height, filled by caller
    float    health;
    uint32_t seed;
};

// Gather GPU instances for one roster entry (filters by PlantInstance::species).
void collect_plant_instances(const std::vector<PlantInstance>& plants,
                              int species_index,
                              std::vector<PlantGPUInstance>& out);

// Assign a roster species to any instance with species < 0 (e.g. a
// creature-dispersed seed, which only knows its `kind`): picks the first roster
// entry of the matching kind. Instances with no matching kind are marked dead.
void resolve_plant_species(std::vector<PlantInstance>& plants,
                           const std::vector<PlantSpecies>& roster);

// Place an initial population by sampling the roster's suitabilities across the
// region. Each instance records both its `kind` (for trophic/growth) and its
// `species` (roster index, for rendering).
std::vector<PlantInstance> place_ecosystem(
    const std::vector<PlantSpecies>& roster,
    const EcosystemParams& eco,
    const EnvironmentField& env);

void tick_plant_population(
    std::vector<PlantInstance>& plants,
    const std::vector<PlantSpecies>& roster,
    const EnvironmentField& env,
    float dt,
    float growth_rate,
    float decay_rate);

void sprout_plants(
    std::vector<PlantInstance>& plants,
    const std::vector<PlantSpecies>& roster,
    const EcosystemParams& eco,
    const EnvironmentField& env,
    uint32_t seed,
    int max_sprouts = 20);

struct PlantMeshParams {
    ClumpParams cp;
    ClumpExpression ce;
    BushParams bp;
    BushExpression be;
    TreeParams tp;
    TreeExpression te;
    ClumpParams reed_cp;
    ClumpExpression reed_ce;
    WildflowerParams wfp;
    WildflowerExpression wfe;
    float phenotype_variance = 0.2f;
};

VegetationMesh generate_ecosystem(
    const EcosystemParams& eco,
    const EnvironmentField& env,
    const PlantMeshParams& pmp,
    bool include_ground = true);

VegetationMesh generate_mesh_from_population(
    const std::vector<PlantInstance>& plants,
    const EnvironmentField& env,
    const PlantMeshParams& pmp);

// Legacy overload (grass/bush/tree only)
VegetationMesh generate_mesh_from_population(
    const std::vector<PlantInstance>& plants,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te,
    float phenotype_variance);

} // namespace bestiary

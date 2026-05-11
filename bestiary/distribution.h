#pragma once

#include "environment.h"
#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"
#include "morphology/wildflower.h"

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
constexpr int PLANT_KIND_COUNT = 5;

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
    int      kind;      // 0=grass, 1=bush, 2=tree, 3=reed, 4=wildflower
    float    health;    // 0..1, drives visual scale; 0 = dead
    uint32_t seed;
};

std::vector<PlantInstance> place_ecosystem(
    const EcosystemParams& eco,
    const EnvironmentField& env);

void tick_plant_population(
    std::vector<PlantInstance>& plants,
    const EnvironmentField& env,
    const EcosystemParams& eco,
    float dt,
    float growth_rate,
    float decay_rate);

void sprout_plants(
    std::vector<PlantInstance>& plants,
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

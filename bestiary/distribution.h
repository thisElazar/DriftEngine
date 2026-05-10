#pragma once

#include "environment.h"
#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"

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

    SpeciesSuitability grass_suit = {0.0f, 0.1f, 0.9f, 1.0f,
                                     0.1f, 0.2f, 0.8f, 0.9f, 4.0f};
    SpeciesSuitability bush_suit  = {0.15f, 0.25f, 0.85f, 0.95f,
                                     0.15f, 0.25f, 0.8f, 0.85f, 1.0f};
    SpeciesSuitability tree_suit  = {0.3f, 0.4f, 0.9f, 1.0f,
                                     0.15f, 0.25f, 0.8f, 0.85f, 0.15f};

    float phenotype_variance = 0.2f;  // 0..1, per-instance parameter jitter
};

float compute_suitability(const SpeciesSuitability& s,
                          const EnvironmentSample& env);

VegetationMesh generate_ecosystem(
    const EcosystemParams& eco,
    const EnvironmentField& env,
    const ClumpParams& cp, const ClumpExpression& ce,
    const BushParams& bp, const BushExpression& be,
    const TreeParams& tp, const TreeExpression& te,
    bool include_ground = true);

} // namespace bestiary

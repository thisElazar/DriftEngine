#pragma once

#include "clump.h"
#include <cstdint>

namespace bestiary {

struct WildflowerParams {
    int   flower_count   = 5;       // 1..20
    float stem_height    = 0.15f;   // m
    float stem_width     = 0.004f;  // m
    float petal_radius   = 0.025f;  // m
    int   petal_count    = 5;       // 3..8
    float clump_radius   = 0.06f;   // m, spread of stems from center

    float petal_color[3] = {0.85f, 0.75f, 0.25f};
    float stem_color[3]  = {0.35f, 0.50f, 0.20f};
};

struct WildflowerExpression {
    ParamRange flower_count;
    ParamRange stem_height;
    ParamRange petal_radius;
    ParamRange clump_radius;

    bool  vary_color = false;
    float dry_color[3] = {0.90f, 0.60f, 0.20f};
    float wet_color[3] = {0.70f, 0.30f, 0.80f};
};

WildflowerParams evaluate_wildflower_expression(const WildflowerParams& base,
                                                 const WildflowerExpression& expr,
                                                 float moisture);

void jitter_wildflower(WildflowerParams& params, float variance, uint32_t seed);

VegetationMesh generate_wildflower(const WildflowerParams& params, uint32_t seed = 0,
                                    bool include_ground = true,
                                    float offset_x = 0.0f, float offset_z = 0.0f);

VegetationMesh generate_wildflower_field(const WildflowerParams& base,
                                          const WildflowerExpression& expr,
                                          const FieldParams& field,
                                          uint32_t base_seed = 0);

} // namespace bestiary

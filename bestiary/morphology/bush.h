#pragma once

#include "clump.h"

namespace bestiary {

struct BushParams {
    int   leaf_count   = 30;
    float leaf_length  = 0.10f;   // m, 0.05..0.3
    float leaf_width   = 0.05f;   // m, 0.02..0.15
    float bush_height  = 0.50f;   // m, 0.1..1.5
    float bush_radius  = 0.20f;   // m, 0.05..0.5
    float stem_height  = 0.05f;   // m, 0.0..0.3

    float base_color[3] = {0.35f, 0.50f, 0.20f};

    int   n_stems         = 3;      // 1..8
    int   attractor_count = 300;    // 50..2000
    float kill_ratio      = 1.5f;   // 0.5..5.0, d_k / step distance
    float influence_ratio = 8.0f;   // 2..30, d_i / step distance
    float tropism         = 0.0f;   // -1..1, vertical growth bias
    float surface_bias    = 0.5f;   // 0..1, attractors toward envelope surface
    int   envelope_shape  = 0;      // 0=ellipsoid 1=hemisphere 2=cylinder 3=cone
    float branch_taper    = 2.5f;   // 1.5..3.5, pipe model exponent
    float branch_width_min = 0.003f; // m, 0.001..0.02
    float branch_width_max = 0.025f; // m, 0.01..0.1
    float branch_gravity  = 0.0f;   // 0..1, droop proportional to depth
    float branch_wobble   = 0.0f;   // 0..1, random growth direction jitter
    float tip_leaf_bias   = 0.7f;   // 0..1, leaf fraction at branch tips
    float leaf_droop      = 0.1f;   // 0..1, downward leaf tilt
};

struct BushExpression {
    ParamRange leaf_count;
    ParamRange leaf_length;
    ParamRange leaf_width;
    ParamRange bush_height;
    ParamRange bush_radius;
    ParamRange stem_height;

    ParamRange n_stems;
    ParamRange attractor_count;
    ParamRange kill_ratio;
    ParamRange influence_ratio;
    ParamRange tropism;
    ParamRange surface_bias;
    ParamRange branch_width_min;
    ParamRange branch_width_max;
    ParamRange branch_gravity;
    ParamRange branch_wobble;
    ParamRange tip_leaf_bias;
    ParamRange leaf_droop;

    bool  vary_color = false;
    float dry_color[3] = {0.55f, 0.45f, 0.25f};
    float wet_color[3] = {0.20f, 0.55f, 0.18f};
};

BushParams evaluate_bush_expression(const BushParams& base,
                                    const BushExpression& expr,
                                    float moisture);

VegetationMesh generate_bush(const BushParams& params, uint32_t seed = 0,
                        bool include_ground = true,
                        float offset_x = 0.0f, float offset_z = 0.0f);

VegetationMesh generate_bush_field(const BushParams& base,
                              const BushExpression& expr,
                              const FieldParams& field,
                              uint32_t base_seed = 0);

} // namespace bestiary

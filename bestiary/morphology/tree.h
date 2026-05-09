#pragma once

#include "clump.h"

namespace bestiary {

struct TreeParams {
    float tree_height    = 5.0f;    // m, 1..15
    float trunk_height   = 2.0f;    // m, 0.5..10 — bare trunk before crown
    float crown_radius   = 1.5f;    // m, 0.5..5
    float crown_height   = 3.0f;    // m, 0.5..8
    float trunk_width    = 0.08f;   // m, 0.02..0.3

    int   leaf_count     = 80;      // 20..2000
    float leaf_length    = 0.12f;   // m, 0.03..0.3
    float leaf_width     = 0.06f;   // m, 0.02..0.15

    float base_color[3]  = {0.30f, 0.50f, 0.18f};
    float trunk_color[3] = {0.35f, 0.25f, 0.15f};

    int   attractor_count = 500;    // 100..3000
    float kill_ratio      = 1.5f;   // 0.5..5.0
    float influence_ratio = 10.0f;  // 2..30
    float tropism         = 0.3f;   // -1..1, trees default slightly ascending
    float surface_bias    = 0.4f;   // 0..1
    int   crown_shape     = 0;      // 0=ellipsoid 1=hemisphere 2=cylinder 3=cone
    float branch_taper    = 2.5f;   // 1.5..3.5
    float branch_width_min = 0.004f; // m, 0.001..0.03
    float branch_width_max = 0.15f;  // m, 0.02..0.3
    float branch_gravity  = 0.1f;   // 0..1, droop proportional to depth
    float branch_wobble   = 0.05f;  // 0..1, random growth direction jitter
    float tip_leaf_bias   = 0.7f;   // 0..1
    float leaf_droop      = 0.15f;  // 0..1
};

struct TreeExpression {
    ParamRange tree_height;
    ParamRange trunk_height;
    ParamRange crown_radius;
    ParamRange crown_height;
    ParamRange trunk_width;
    ParamRange leaf_count;
    ParamRange leaf_length;
    ParamRange leaf_width;
    ParamRange attractor_count;
    ParamRange kill_ratio;
    ParamRange influence_ratio;
    ParamRange tropism;
    ParamRange surface_bias;
    ParamRange branch_taper;
    ParamRange branch_width_min;
    ParamRange branch_width_max;
    ParamRange branch_gravity;
    ParamRange branch_wobble;
    ParamRange tip_leaf_bias;
    ParamRange leaf_droop;

    bool  vary_color = false;
    float dry_color[3] = {0.50f, 0.45f, 0.20f};
    float wet_color[3] = {0.20f, 0.55f, 0.15f};
};

TreeParams evaluate_tree_expression(const TreeParams& base,
                                    const TreeExpression& expr,
                                    float moisture);

VegetationMesh generate_tree(const TreeParams& params, uint32_t seed = 0,
                             bool include_ground = true,
                             float offset_x = 0.0f, float offset_z = 0.0f);

VegetationMesh generate_tree_field(const TreeParams& base,
                                   const TreeExpression& expr,
                                   const FieldParams& field,
                                   uint32_t base_seed = 0);

} // namespace bestiary

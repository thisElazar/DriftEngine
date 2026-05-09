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
};

struct BushExpression {
    ParamRange leaf_count;
    ParamRange leaf_length;
    ParamRange leaf_width;
    ParamRange bush_height;
    ParamRange bush_radius;
    ParamRange stem_height;

    bool  vary_color = false;
    float dry_color[3] = {0.55f, 0.45f, 0.25f};
    float wet_color[3] = {0.20f, 0.55f, 0.18f};
};

BushParams evaluate_bush_expression(const BushParams& base,
                                    const BushExpression& expr,
                                    float moisture);

ClumpMesh generate_bush(const BushParams& params, uint32_t seed = 0,
                        bool include_ground = true,
                        float offset_x = 0.0f, float offset_z = 0.0f);

ClumpMesh generate_bush_field(const BushParams& base,
                              const BushExpression& expr,
                              const FieldParams& field,
                              uint32_t base_seed = 0);

} // namespace bestiary

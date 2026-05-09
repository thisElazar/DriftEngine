#pragma once

// bestiary/morphology/clump.h
//
// Grass clump primitive. A clump is one "plant" — N blades fanning out from
// a center point. Same phenotype expression machinery as bushes/trees, but
// the geometry generator differs.
//
// v0.0.1: parameter struct + stub generator. The Lab v0.0.1 wires sliders to
// these fields without rendering yet. Mesh generation lands in v0.0.2.

#include <cstdint>
#include <vector>

namespace bestiary {

struct ClumpParams {
    int   blade_count   = 12;     // 1..60
    float blade_height  = 0.35f;  // m, 0.05..2.0
    float blade_width   = 0.012f; // m, 0.005..0.05
    float splay_angle   = 25.0f;  // degrees from vertical, 0..80
    float clump_radius  = 0.05f;  // m, 0.0..0.5

    float base_color[3] = {0.42f, 0.55f, 0.22f};
};

struct VegetationVertex {
    float position[3];
    float normal[3];
    float color[3];
    float height_t;
};

struct VegetationMesh {
    std::vector<VegetationVertex> vertices;
    std::vector<uint32_t>         indices;
};

struct ParamRange {
    bool  enabled = false;
    float low     = 0.0f;
    float high    = 0.0f;
};

struct ClumpExpression {
    ParamRange blade_count;
    ParamRange blade_height;
    ParamRange blade_width;
    ParamRange splay_angle;
    ParamRange clump_radius;

    bool  vary_color = false;
    float dry_color[3] = {0.55f, 0.50f, 0.25f};
    float wet_color[3] = {0.25f, 0.60f, 0.20f};
};

struct FieldParams {
    int   grid_n  = 8;
    float spacing = 0.4f;
};

ClumpParams evaluate_expression(const ClumpParams& base,
                                const ClumpExpression& expr,
                                float moisture);

VegetationMesh generate_clump(const ClumpParams& params, uint32_t seed = 0,
                         bool include_ground = true,
                         float offset_x = 0.0f, float offset_z = 0.0f);

VegetationMesh generate_field(const ClumpParams& base,
                         const ClumpExpression& expr,
                         const FieldParams& field,
                         uint32_t base_seed = 0);

} // namespace bestiary

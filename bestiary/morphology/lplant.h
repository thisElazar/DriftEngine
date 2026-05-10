#pragma once

#include "clump.h"
#include <cstdint>

namespace bestiary {

enum class ModuleType : uint8_t {
    Apex,
    Internode,
    Leaf,
    Dormant,
    Dead,
};

struct PlantModule {
    ModuleType type = ModuleType::Apex;

    float pos[3]     = {};
    float heading[3] = {0.0f, 1.0f, 0.0f};
    float left[3]    = {1.0f, 0.0f, 0.0f};
    float up[3]      = {0.0f, 0.0f, 1.0f};

    float length = 0.0f;
    float width  = 0.0f;
    float angle  = 0.0f;

    int   order        = 0;
    float rel_position = 0.0f;
    int   age          = 0;

    float light_q   = 0.0f;
    float resource_v = 1.0f;

    int parent       = -1;
    int first_child  = -1;
    int next_sibling = -1;
    int child_count  = 0;

    float leaf_length = 0.0f;
    float leaf_width  = 0.0f;
};

enum class GrowthArchetype : int {
    Monopodial   = 0,   // single dominant axis (conifer)
    Sympodial    = 1,   // main axis forks (broadleaf)
    Dichotomous  = 2,   // equal forking
    MultistemBush = 3,  // multiple basal stems
    Whorled      = 4,   // whorled branching
};

struct LPlantParams {
    GrowthArchetype archetype = GrowthArchetype::Sympodial;

    float total_height     = 5.0f;
    float trunk_height     = 2.0f;
    float crown_radius     = 1.5f;
    float crown_height     = 3.0f;
    float trunk_width      = 0.08f;

    int   growth_steps     = 12;       // 4..25
    float branch_angle     = 0.52f;    // radians (~30 deg)
    float branch_angle_var = 0.1f;
    float phyllotaxis_angle = 137.5f;  // degrees
    int   phyllotaxis_mode  = 2;       // 0=alternate 1=opposite 2=spiral 3=whorled
    int   whorl_count       = 3;
    float internode_length  = 0.3f;
    float length_decay      = 0.85f;   // per-order multiplier

    // Borchert-Honda
    float lambda           = 0.52f;    // 0..1, apical dominance
    float resource_alpha   = 1.0f;     // vigor sensitivity (future: positional gradients)
    float v_threshold      = 0.01f;    // min vigor for bud activation

    // Space colonization
    int   attractor_count  = 400;
    float kill_ratio       = 1.5f;
    float influence_ratio  = 10.0f;
    float tropism          = 0.3f;
    float surface_bias     = 0.4f;
    int   envelope_shape   = 0;

    // Pipe model
    float branch_taper     = 2.5f;
    float branch_width_min = 0.004f;
    float branch_width_max = 0.15f;

    // Leaves
    int   leaf_count       = 120;      // 20..2000
    float leaf_length      = 0.10f;
    float leaf_width       = 0.05f;
    float leaf_droop       = 0.15f;
    float tip_leaf_bias    = 0.7f;

    // Appearance
    float base_color[3]    = {0.30f, 0.50f, 0.18f};
    float trunk_color[3]   = {0.35f, 0.25f, 0.15f};

    float branch_gravity   = 0.1f;
    float branch_wobble    = 0.05f;
};

struct LPlantExpression {
    ParamRange total_height;
    ParamRange trunk_height;
    ParamRange crown_radius;
    ParamRange crown_height;
    ParamRange trunk_width;
    ParamRange growth_steps;
    ParamRange branch_angle;
    ParamRange internode_length;
    ParamRange length_decay;
    ParamRange lambda;
    ParamRange attractor_count;
    ParamRange tropism;
    ParamRange leaf_count;
    ParamRange leaf_length;
    ParamRange leaf_width;
    ParamRange leaf_droop;
    ParamRange branch_gravity;
    ParamRange branch_wobble;

    bool  vary_color = false;
    float dry_color[3] = {0.50f, 0.45f, 0.20f};
    float wet_color[3] = {0.20f, 0.55f, 0.15f};
};

LPlantParams evaluate_lplant_expression(const LPlantParams& base,
                                        const LPlantExpression& expr,
                                        float moisture);

VegetationMesh generate_lplant(const LPlantParams& params, uint32_t seed = 0,
                               bool include_ground = true,
                               float offset_x = 0.0f, float offset_z = 0.0f);

VegetationMesh generate_lplant_field(const LPlantParams& base,
                                     const LPlantExpression& expr,
                                     const FieldParams& field,
                                     uint32_t base_seed = 0);

} // namespace bestiary

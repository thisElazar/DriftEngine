#pragma once

#include "animals/animal_mesh.h"

namespace bestiary {

struct HerbivoreParams {
    float torso_length     = 1.2f;   // m, 0.5..2.5
    float torso_girth      = 0.35f;  // m, 0.2..0.8
    float neck_length      = 0.5f;   // m, 0.1..1.0
    float head_length      = 0.25f;  // m, 0.1..0.6
    float leg_length_front = 0.8f;   // m, 0.3..1.5
    float leg_length_back  = 0.8f;   // m, 0.3..1.5
    float leg_thickness    = 0.04f;  // m, 0.02..0.15
    float hoof_size        = 0.05f;  // m, 0.02..0.12
    float coat_color[3]    = {0.55f, 0.40f, 0.25f};
    float belly_color[3]   = {0.72f, 0.62f, 0.48f};

    float walk_period_seconds = 0.9f;  // s, 0.3..3.0
    float foot_lift_height    = 0.08f; // m, 0.02..0.2
};

struct WalkCycle;

Skeleton   build_herbivore_skeleton(const HerbivoreParams& params);
AnimalMesh generate_herbivore_mesh(const HerbivoreParams& params);
WalkCycle  make_herbivore_walk(const HerbivoreParams& params);
WalkCycle  make_herbivore_trot(const HerbivoreParams& params);
WalkCycle  make_herbivore_run(const HerbivoreParams& params);
WalkCycle  make_herbivore_idle(const HerbivoreParams& params);
WalkCycle  make_herbivore_graze(const HerbivoreParams& params, float feed_height = 0.0f);

} // namespace bestiary

#pragma once

#include "animals/animal_mesh.h"

namespace bestiary {

struct PredatorParams {
    float torso_length     = 1.0f;   // m — shorter, deeper torso than herbivore
    float chest_girth      = 0.40f;  // m — deep chest
    float waist_girth      = 0.28f;  // m — narrow waist (tuck-up)
    float neck_length      = 0.35f;  // m
    float snout_length     = 0.30f;  // m — longer muzzle
    float head_width       = 0.18f;  // m — broad skull
    float leg_length_front = 0.75f;  // m
    float leg_length_back  = 0.80f;  // m — longer hind legs
    float leg_thickness    = 0.035f; // m
    float paw_size         = 0.04f;  // m
    float tail_length      = 0.45f;  // m — long bushy tail
    float coat_color[3]    = {0.45f, 0.42f, 0.38f};

    float walk_period_seconds = 0.85f;
    float foot_lift_height    = 0.07f;
};

struct WalkCycle;

Skeleton   build_predator_skeleton(const PredatorParams& params);
AnimalMesh generate_predator_mesh(const PredatorParams& params);
WalkCycle  make_predator_walk(const PredatorParams& params);
WalkCycle  make_predator_trot(const PredatorParams& params);
WalkCycle  make_predator_run(const PredatorParams& params);
WalkCycle  make_predator_idle(const PredatorParams& params);
WalkCycle  make_predator_stalk(const PredatorParams& params);

} // namespace bestiary

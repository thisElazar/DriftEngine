#pragma once

#include "animals/animal_mesh.h"

namespace bestiary {

struct BirdParams {
    float body_length      = 0.25f;  // m — compact torso
    float body_girth       = 0.14f;  // m — round body
    float neck_length      = 0.10f;  // m — flexible neck
    float head_size        = 0.04f;  // m — small round head
    float beak_length      = 0.03f;  // m
    float wing_length      = 0.22f;  // m — total from shoulder to tip
    float wing_width       = 0.06f;  // m — chord width at shoulder
    float wing_taper       = 0.3f;   // 0..1 — tip width as fraction of root
    float leg_length_upper = 0.10f;  // m — tibiotarsus
    float leg_length_lower = 0.06f;  // m — tarsometatarsus
    float leg_thickness    = 0.008f; // m — thin bird legs
    float foot_size        = 0.02f;  // m — toes
    float tail_length      = 0.06f;  // m
    float coat_color[3]    = {0.35f, 0.32f, 0.30f};
    float belly_color[3]   = {0.55f, 0.50f, 0.45f};

    float walk_period_seconds = 0.5f;
    float foot_lift_height    = 0.03f;
    float hop_height          = 0.04f;

    float flap_period         = 0.4f;   // s — wing beat cycle
    float flap_amplitude      = 40.0f;  // deg — shoulder flap range
    float flap_sweep          = 0.0f;   // 0..1 — forward/back sweep (0=up-down, 1=circular)
    float fly_height          = 0.5f;   // m — altitude above ground
};

struct WalkCycle;

Skeleton   build_bird_skeleton(const BirdParams& params);
AnimalMesh generate_bird_mesh(const BirdParams& params);
WalkCycle  make_bird_walk(const BirdParams& params);
WalkCycle  make_bird_hop(const BirdParams& params);
WalkCycle  make_bird_run(const BirdParams& params);
WalkCycle  make_bird_idle(const BirdParams& params);
WalkCycle  make_bird_fly(const BirdParams& params);
WalkCycle  make_bird_peck(const BirdParams& params);
WalkCycle  make_bird_takeoff(const BirdParams& params);
WalkCycle  make_bird_land(const BirdParams& params);
WalkCycle  make_bird_perch(const BirdParams& params);
WalkCycle  make_bird_soar(const BirdParams& params);
WalkCycle  make_bird_dive(const BirdParams& params);

} // namespace bestiary

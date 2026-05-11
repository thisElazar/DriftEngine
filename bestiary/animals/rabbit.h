#pragma once

#include "animals/animal_mesh.h"

namespace bestiary {

struct RabbitParams {
    float body_length      = 0.30f;  // m — compact torso
    float body_girth       = 0.14f;  // m
    float neck_length      = 0.06f;  // m — almost no neck
    float head_length      = 0.08f;  // m — round head
    float head_width       = 0.07f;  // m
    float ear_length       = 0.12f;  // m — signature long ears
    float leg_length_front = 0.15f;  // m — short front legs
    float leg_length_back  = 0.22f;  // m — powerful hind legs
    float leg_thickness    = 0.015f; // m
    float paw_size         = 0.015f; // m
    float tail_size        = 0.03f;  // m — puff tail
    float coat_color[3]    = {0.60f, 0.52f, 0.42f};
    float belly_color[3]   = {0.78f, 0.72f, 0.65f};

    float hop_period_seconds = 0.45f; // s
    float hop_height         = 0.06f; // m
};

struct WalkCycle;

Skeleton   build_rabbit_skeleton(const RabbitParams& params);
AnimalMesh generate_rabbit_mesh(const RabbitParams& params);
WalkCycle  make_rabbit_hop(const RabbitParams& params);
WalkCycle  make_rabbit_run(const RabbitParams& params);
WalkCycle  make_rabbit_idle(const RabbitParams& params);
WalkCycle  make_rabbit_graze(const RabbitParams& params);

} // namespace bestiary

#pragma once

#include "animals/animal_mesh.h"

namespace bestiary {

struct SnakeParams {
    float body_length     = 1.0f;   // m — total length
    float body_thickness  = 0.03f;  // m — radius at mid-body
    float head_width      = 0.035f; // m — wider head
    float head_length     = 0.04f;  // m
    float taper_tail      = 0.15f;  // tip radius as fraction of body_thickness
    float coat_color[3]   = {0.40f, 0.35f, 0.20f};
    float belly_color[3]  = {0.65f, 0.60f, 0.45f};

    float slither_period    = 1.5f;   // s
    float slither_amplitude = 15.0f;  // deg per segment
    float slither_waves     = 2.0f;   // sine waves along body
};

struct WalkCycle;

Skeleton   build_snake_skeleton(const SnakeParams& params);
AnimalMesh generate_snake_mesh(const SnakeParams& params);
WalkCycle  make_snake_slither(const SnakeParams& params);
WalkCycle  make_snake_fast(const SnakeParams& params);
WalkCycle  make_snake_idle(const SnakeParams& params);
WalkCycle  make_snake_strike(const SnakeParams& params);

} // namespace bestiary

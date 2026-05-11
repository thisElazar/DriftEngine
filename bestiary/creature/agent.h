#pragma once

#include <cstdint>

namespace bestiary {

enum class AgentState : uint8_t {
    Wander,
    Graze,
    Flee,
    SeekMate,
};

struct SteeringIntent {
    float direction[2] = {0.0f, 0.0f};
    float magnitude     = 0.0f;
};

struct Agent {
    float      pos[2]          = {0.0f, 0.0f};
    float      vel[2]          = {0.0f, 0.0f};
    float      heading         = 0.0f;
    float      energy          = 1.0f;
    float      age             = 0.0f;
    float      graze_timer     = 0.0f;
    float      mate_cooldown   = 0.0f;
    float      flee_timer      = 0.0f;
    float      flee_dir[2]     = {0.0f, 0.0f};
    AgentState state           = AgentState::Wander;
    AgentState prev_anim_state = AgentState::Wander;
    float      prev_anim_speed = 0.0f;
    float      anim_blend      = 1.0f;
    uint16_t   species_id      = 0;
    bool       alive           = true;
    bool       female          = false;
};

} // namespace bestiary

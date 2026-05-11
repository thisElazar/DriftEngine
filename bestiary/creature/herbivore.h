#pragma once

#include "agent.h"
#include "vegetation_field.h"
#include "../environment.h"
#include "../distribution.h"

#include <functional>
#include <vector>
#include <cstdint>

namespace bestiary {

struct HerbivoreProfile {
    float body_length    = 0.8f;
    float body_height    = 0.6f;
    float body_color[3]  = {0.55f, 0.40f, 0.25f};

    // Locomotion
    float move_speed     = 2.0f;
    float run_speed      = 6.0f;
    float turn_rate      = 3.0f;

    // Energy model (caloric, body-mass-based)
    float body_mass         = 30.0f;   // kg — drives basal metabolic rate
    float basal_rate        = 0.008f;  // energy/s at rest (scaled from mass)
    float locomotion_cost   = 0.003f;  // energy/s per (m/s)² of speed
    float hunger_threshold  = 0.5f;
    float starve_threshold  = 0.0f;

    // Grazing
    float graze_radius        = 3.0f;
    float graze_consume       = 0.15f;  // plant health consumed/s
    float graze_min_health    = 0.15f;
    float graze_duration      = 3.0f;
    float trophic_efficiency  = 0.10f;

    // Plant caloric values (energy gained per unit plant health consumed)
    float grass_caloric_value = 0.6f;
    float bush_caloric_value  = 1.0f;
    float tree_caloric_value  = 0.3f;   // bark/leaves less digestible

    // Wander
    float wander_radius  = 8.0f;
    float wander_jitter  = 2.0f;

    // Flee
    float flee_radius    = 15.0f;
    float flee_duration  = 4.0f;

    // Herding (boids)
    float herd_radius      = 20.0f;
    float herd_weight      = 0.3f;
    float separation_radius = 2.0f;
    float separation_weight = 0.5f;
    float alignment_weight  = 0.2f;

    // Reproduction
    float reproduce_threshold = 0.75f;
    float reproduce_cost      = 0.25f;
    float reproduce_cooldown  = 30.0f;
    float offspring_energy    = 0.40f;
    float mate_search_radius  = 20.0f;

    // Terrain awareness
    float water_avoidance     = 8.0f;   // steering weight away from water
    float max_slope           = 0.7f;   // ~35 deg — won't climb steeper
    float slope_cost_factor   = 2.0f;   // energy multiplier on slopes

    float max_age             = 300.0f;
};

struct CreatureWorldView {
    std::vector<PlantInstance>*    plant_population;
    const EnvironmentField*        env_field;
    std::function<float(float x, float z)> terrain_height;
    std::function<float(float x, float z)> water_depth;
    float tile_half_x;
    float tile_half_z;
    bool  has_threat;
    float threat_pos[2];
    float threat_radius;
};

void update_creatures(
    std::vector<Agent>& agents,
    const std::vector<HerbivoreProfile>& profiles,
    const CreatureWorldView& world,
    float dt,
    uint32_t tick_seed);

void spawn_creatures(
    std::vector<Agent>& agents,
    uint16_t species_id,
    int count,
    const EnvironmentField& env,
    float tile_half_x, float tile_half_z,
    uint32_t seed);

int count_alive(const std::vector<Agent>& agents);

int count_alive_species(const std::vector<Agent>& agents, uint16_t species_id);

float avg_energy(const std::vector<Agent>& agents);
float min_energy(const std::vector<Agent>& agents);
float max_energy(const std::vector<Agent>& agents);

} // namespace bestiary

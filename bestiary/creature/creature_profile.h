#pragma once

#include "agent.h"
#include "../environment.h"
#include "../distribution.h"

#include <functional>
#include <vector>
#include <cstdint>

namespace bestiary {

struct CreatureProfile {
    Archetype archetype  = Archetype::Herbivore;

    float body_length    = 0.8f;
    float body_height    = 0.6f;
    float body_color[3]  = {0.55f, 0.40f, 0.25f};

    // Locomotion
    float move_speed     = 2.0f;
    float run_speed      = 6.0f;
    float turn_rate      = 3.0f;

    // Energy model (caloric, body-mass-based)
    float body_mass         = 30.0f;
    float basal_rate        = 0.008f;
    float locomotion_cost   = 0.003f;
    float hunger_threshold  = 0.5f;
    float starve_threshold  = 0.0f;

    // Grazing (herbivore/rabbit)
    float graze_radius        = 3.0f;
    float graze_consume       = 0.15f;
    float graze_min_health    = 0.15f;
    float graze_duration      = 3.0f;
    float trophic_efficiency  = 0.10f;

    float grass_caloric_value      = 0.6f;
    float bush_caloric_value       = 1.0f;
    float tree_caloric_value       = 0.3f;
    float reed_caloric_value       = 0.5f;
    float wildflower_caloric_value = 0.8f;

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
    float water_avoidance     = 8.0f;
    float max_slope           = 0.7f;
    float slope_cost_factor   = 2.0f;

    float max_age             = 300.0f;

    // Drinking (passive near water)
    float drink_radius       = 2.0f;    // how close to water to sip
    float drink_rate         = 0.008f;  // energy/s gained when near water

    // Hunting (predator only)
    float hunt_radius        = 30.0f;
    float chase_speed        = 9.0f;
    float attack_range       = 1.2f;
    float kill_energy_gain   = 0.50f;
    float stalk_speed        = 1.5f;
    float consume_duration   = 3.0f;
    float max_prey_mass      = 1000.0f;

    // Flight (bird/raptor)
    bool  can_fly              = false;
    float fly_altitude         = 8.0f;
    float takeoff_speed        = 3.0f;
    float landing_speed        = 2.0f;
    float altitude_wander_amp  = 2.0f;

    // Seed dispersal (flying seed-eaters)
    bool  seed_disperser         = false;
    float seed_drop_probability  = 0.02f;
    int   seed_kind              = 0;
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
    const std::vector<CreatureProfile>& profiles,
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

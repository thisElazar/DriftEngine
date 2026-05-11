#include "herbivore.h"
#include "spatial_grid.h"

#include <algorithm>
#include <cmath>

namespace bestiary {

// -----------------------------------------------------------------------
// Deterministic hash (same as distribution.cpp)
// -----------------------------------------------------------------------

static uint32_t dhash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

static float dhash_float(uint32_t seed, uint32_t index)
{
    return static_cast<float>(dhash(seed ^ (index * 2654435761u))) /
           static_cast<float>(0xFFFFFFFFu);
}

// -----------------------------------------------------------------------
// Steering helpers
// -----------------------------------------------------------------------

static float vec2_length(float x, float z)
{
    return std::sqrt(x * x + z * z);
}

static void vec2_normalize(float& x, float& z)
{
    float len = vec2_length(x, z);
    if (len > 0.0001f) {
        x /= len;
        z /= len;
    }
}

static float angle_toward(float from_heading, float tx, float tz, float turn_rate, float dt)
{
    float target = std::atan2(tx, tz);
    float diff = target - from_heading;

    constexpr float pi = 3.14159265f;
    while (diff >  pi) diff -= 2.0f * pi;
    while (diff < -pi) diff += 2.0f * pi;

    float max_turn = turn_rate * dt;
    diff = std::clamp(diff, -max_turn, max_turn);
    return from_heading + diff;
}

// -----------------------------------------------------------------------
// Terrain queries
// -----------------------------------------------------------------------

static float sample_slope(const CreatureWorldView& world, float x, float z)
{
    if (!world.terrain_height) return 0.0f;
    constexpr float eps = 0.5f;
    float h  = world.terrain_height(x, z);
    float hx = world.terrain_height(x + eps, z);
    float hz = world.terrain_height(x, z + eps);
    float dx = (hx - h) / eps;
    float dz = (hz - h) / eps;
    return std::sqrt(dx * dx + dz * dz);
}

static float sample_water(const CreatureWorldView& world, float x, float z)
{
    if (!world.water_depth) return 0.0f;
    return std::max(0.0f, world.water_depth(x, z));
}

// -----------------------------------------------------------------------
// Terrain-aware steering: water avoidance + slope penalty
// -----------------------------------------------------------------------

static void apply_terrain_steering(const Agent& agent,
                                   const HerbivoreProfile& prof,
                                   const CreatureWorldView& world,
                                   float& dir_x, float& dir_z)
{
    constexpr float probe_dist = 3.0f;
    constexpr int   num_probes = 5;
    constexpr float pi = 3.14159265f;
    constexpr float fan_half = pi * 0.5f;

    float avoid_x = 0.0f, avoid_z = 0.0f;

    for (int p = 0; p < num_probes; ++p) {
        float angle = agent.heading + fan_half * (2.0f * p / (num_probes - 1) - 1.0f);
        float px = agent.pos[0] + std::sin(angle) * probe_dist;
        float pz = agent.pos[1] + std::cos(angle) * probe_dist;

        float water = sample_water(world, px, pz);
        if (water > 0.02f) {
            float repel = std::min(water * 10.0f, 1.0f) * prof.water_avoidance;
            avoid_x -= std::sin(angle) * repel;
            avoid_z -= std::cos(angle) * repel;
        }

        float slope = sample_slope(world, px, pz);
        if (slope > prof.max_slope * 0.7f) {
            float repel = (slope - prof.max_slope * 0.7f) / (prof.max_slope * 0.3f);
            repel = std::clamp(repel, 0.0f, 1.0f) * 3.0f;
            avoid_x -= std::sin(angle) * repel;
            avoid_z -= std::cos(angle) * repel;
        }
    }

    dir_x += avoid_x;
    dir_z += avoid_z;
}

// -----------------------------------------------------------------------
// Find nearest edible plant (using spatial grid)
// -----------------------------------------------------------------------

static int find_nearest_plant(const Agent& agent,
                              const HerbivoreProfile& profile,
                              const std::vector<PlantInstance>& plants,
                              const SpatialGrid<PlantInstance>& plant_grid)
{
    float best_d2 = profile.graze_radius * profile.graze_radius;
    int best_idx = -1;

    plant_grid.query_radius(agent.pos[0], agent.pos[1], profile.graze_radius,
        [&](uint32_t idx) {
            const auto& p = plants[idx];
            if (p.health < profile.graze_min_health) return;
            float dx = p.x - agent.pos[0];
            float dz = p.z - agent.pos[1];
            float d2 = dx * dx + dz * dz;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = static_cast<int>(idx);
            }
        });

    return best_idx;
}

static bool find_food_direction(const Agent& agent,
                                const HerbivoreProfile& profile,
                                const std::vector<PlantInstance>& plants,
                                const SpatialGrid<PlantInstance>& plant_grid,
                                float& out_dx, float& out_dz,
                                int& out_plant_idx)
{
    float search_r = profile.graze_radius * 3.0f;
    float search_r2 = search_r * search_r;
    float best_score = -1.0f;
    int best_idx = -1;

    plant_grid.query_radius(agent.pos[0], agent.pos[1], search_r,
        [&](uint32_t idx) {
            const auto& p = plants[idx];
            if (p.health < profile.graze_min_health) return;
            float dx = p.x - agent.pos[0];
            float dz = p.z - agent.pos[1];
            float d2 = dx * dx + dz * dz;
            if (d2 > search_r2) return;

            float dist = std::sqrt(d2);
            float score = p.health / (dist + 0.5f);
            if (score > best_score) {
                best_score = score;
                best_idx = static_cast<int>(idx);
            }
        });

    if (best_idx < 0) return false;

    float dx = plants[static_cast<size_t>(best_idx)].x - agent.pos[0];
    float dz = plants[static_cast<size_t>(best_idx)].z - agent.pos[1];
    float dist = std::sqrt(dx * dx + dz * dz);

    out_plant_idx = best_idx;

    if (dist < 0.5f) {
        out_dx = 0.0f;
        out_dz = 0.0f;
        return true;
    }

    out_dx = dx / dist;
    out_dz = dz / dist;
    return true;
}

// -----------------------------------------------------------------------
// Wander-circle steering
// -----------------------------------------------------------------------

static SteeringIntent steer_wander(const Agent& agent,
                                   const HerbivoreProfile& profile,
                                   uint32_t seed)
{
    float fwd_x = std::sin(agent.heading);
    float fwd_z = std::cos(agent.heading);

    float jx = (dhash_float(seed, 0) * 2.0f - 1.0f) * profile.wander_jitter;
    float jz = (dhash_float(seed, 1) * 2.0f - 1.0f) * profile.wander_jitter;

    float target_x = fwd_x * profile.wander_radius + jx;
    float target_z = fwd_z * profile.wander_radius + jz;
    vec2_normalize(target_x, target_z);

    SteeringIntent intent{};
    intent.direction[0] = target_x;
    intent.direction[1] = target_z;
    intent.magnitude = 0.5f;
    return intent;
}

// -----------------------------------------------------------------------
// Herd steering (boids: cohesion + separation + alignment)
// -----------------------------------------------------------------------

struct HerdIntent {
    float cohesion[2]    = {0.0f, 0.0f};
    float separation[2]  = {0.0f, 0.0f};
    float alignment[2]   = {0.0f, 0.0f};
    int   neighbor_count = 0;
};

static HerdIntent compute_herd(const Agent& agent,
                                const HerbivoreProfile& prof,
                                const std::vector<Agent>& agents,
                                const SpatialGrid<Agent>& agent_grid,
                                size_t self_idx)
{
    HerdIntent h{};
    float avg_x = 0.0f, avg_z = 0.0f;
    float avg_hx = 0.0f, avg_hz = 0.0f;
    float herd_r2 = prof.herd_radius * prof.herd_radius;
    float sep_r2  = prof.separation_radius * prof.separation_radius;

    agent_grid.query_radius(agent.pos[0], agent.pos[1], prof.herd_radius,
        [&](uint32_t j) {
            if (j == static_cast<uint32_t>(self_idx)) return;
            const Agent& other = agents[j];
            if (!other.alive) return;
            if (other.species_id != agent.species_id) return;

            float dx = other.pos[0] - agent.pos[0];
            float dz = other.pos[1] - agent.pos[1];
            float d2 = dx * dx + dz * dz;

            if (d2 > herd_r2 || d2 < 0.0001f) return;

            ++h.neighbor_count;
            avg_x += other.pos[0];
            avg_z += other.pos[1];
            avg_hx += std::sin(other.heading);
            avg_hz += std::cos(other.heading);

            if (d2 < sep_r2) {
                float d = std::sqrt(d2);
                float strength = 1.0f - d / prof.separation_radius;
                h.separation[0] -= (dx / d) * strength;
                h.separation[1] -= (dz / d) * strength;
            }
        });

    if (h.neighbor_count > 0) {
        float n = static_cast<float>(h.neighbor_count);
        h.cohesion[0] = (avg_x / n) - agent.pos[0];
        h.cohesion[1] = (avg_z / n) - agent.pos[1];
        vec2_normalize(h.cohesion[0], h.cohesion[1]);

        h.alignment[0] = avg_hx / n;
        h.alignment[1] = avg_hz / n;
        vec2_normalize(h.alignment[0], h.alignment[1]);

        vec2_normalize(h.separation[0], h.separation[1]);
    }

    return h;
}

// -----------------------------------------------------------------------
// Find potential mate (spatial grid accelerated)
// -----------------------------------------------------------------------

static int find_mate(const Agent& agent,
                     const HerbivoreProfile& prof,
                     const std::vector<Agent>& agents,
                     const SpatialGrid<Agent>& agent_grid,
                     size_t self_idx)
{
    float best_d2 = prof.mate_search_radius * prof.mate_search_radius;
    int best = -1;

    agent_grid.query_radius(agent.pos[0], agent.pos[1], prof.mate_search_radius,
        [&](uint32_t j) {
            if (j == static_cast<uint32_t>(self_idx)) return;
            const Agent& other = agents[j];
            if (!other.alive) return;
            if (other.species_id != agent.species_id) return;
            if (other.female == agent.female) return;
            if (other.energy < prof.reproduce_threshold) return;
            if (other.mate_cooldown > 0.0f) return;

            float dx = other.pos[0] - agent.pos[0];
            float dz = other.pos[1] - agent.pos[1];
            float d2 = dx * dx + dz * dz;
            if (d2 < best_d2) {
                best_d2 = d2;
                best = static_cast<int>(j);
            }
        });

    return best;
}

// -----------------------------------------------------------------------
// update_creatures
// -----------------------------------------------------------------------

void update_creatures(
    std::vector<Agent>& agents,
    const std::vector<HerbivoreProfile>& profiles,
    const CreatureWorldView& world,
    float dt,
    uint32_t tick_seed)
{
    if (dt <= 0.0f || !world.plant_population) return;

    auto& plants = *world.plant_population;

    // Build spatial grids
    float grid_w = world.tile_half_x * 2.0f;
    float grid_h = world.tile_half_z * 2.0f;
    constexpr float AGENT_CELL = 10.0f;
    constexpr float PLANT_CELL = 5.0f;

    SpatialGrid<Agent> agent_grid;
    agent_grid.init(-world.tile_half_x, -world.tile_half_z, grid_w, grid_h, AGENT_CELL);
    for (size_t i = 0; i < agents.size(); ++i) {
        if (agents[i].alive)
            agent_grid.insert(static_cast<uint32_t>(i), agents[i].pos[0], agents[i].pos[1]);
    }

    SpatialGrid<PlantInstance> plant_grid;
    plant_grid.init(-world.tile_half_x, -world.tile_half_z, grid_w, grid_h, PLANT_CELL);
    for (size_t i = 0; i < plants.size(); ++i) {
        if (plants[i].health > 0.0f)
            plant_grid.insert(static_cast<uint32_t>(i), plants[i].x, plants[i].z);
    }

    std::vector<Agent> offspring;

    size_t n = agents.size();
    for (size_t i = 0; i < n; ++i) {
        Agent& a = agents[i];
        if (!a.alive) continue;
        if (a.species_id >= profiles.size()) continue;

        const HerbivoreProfile& prof = profiles[a.species_id];
        uint32_t agent_seed = dhash(tick_seed ^ static_cast<uint32_t>(i * 2654435761u));

        if (a.mate_cooldown > 0.0f)
            a.mate_cooldown = std::max(0.0f, a.mate_cooldown - dt);

        // --- Threat detection: flee from brush ---
        if (world.has_threat) {
            float dx = a.pos[0] - world.threat_pos[0];
            float dz = a.pos[1] - world.threat_pos[1];
            float d2 = dx * dx + dz * dz;
            float tr2 = world.threat_radius * world.threat_radius;
            if (d2 < tr2 && d2 > 0.01f) {
                float d = std::sqrt(d2);
                a.flee_dir[0] = dx / d;
                a.flee_dir[1] = dz / d;
                a.flee_timer = prof.flee_duration;
                a.state = AgentState::Flee;
            }
        }

        SteeringIntent intent{};

        // --- State transitions + steering ---
        switch (a.state) {
        case AgentState::Wander: {
            if (a.energy >= prof.reproduce_threshold && a.mate_cooldown <= 0.0f) {
                int mate = find_mate(a, prof, agents, agent_grid, i);
                if (mate >= 0) {
                    a.state = AgentState::SeekMate;
                    Agent& m = agents[static_cast<size_t>(mate)];
                    m.state = AgentState::SeekMate;
                    break;
                }
            }

            if (a.energy < prof.hunger_threshold) {
                float food_dx, food_dz;
                int plant_idx = -1;
                if (find_food_direction(a, prof, plants, plant_grid, food_dx, food_dz, plant_idx)) {
                    float len = vec2_length(food_dx, food_dz);
                    if (len < 0.01f) {
                        a.state = AgentState::Graze;
                        a.graze_timer = 0.0f;
                        break;
                    }
                    intent.direction[0] = food_dx;
                    intent.direction[1] = food_dz;
                    intent.magnitude = 0.8f;
                    break;
                }
            }
            intent = steer_wander(a, prof, agent_seed);
            break;
        }

        case AgentState::Graze: {
            int target = find_nearest_plant(a, prof, plants, plant_grid);
            if (a.graze_timer >= prof.graze_duration || target < 0) {
                a.state = AgentState::Wander;
                break;
            }

            float bite = prof.graze_consume * dt;
            float taken = std::min(plants[static_cast<size_t>(target)].health, bite);
            plants[static_cast<size_t>(target)].health -= taken;

            // Caloric gain depends on plant species
            float caloric_value = prof.grass_caloric_value;
            int kind = plants[static_cast<size_t>(target)].kind;
            if (kind == 1) caloric_value = prof.bush_caloric_value;
            else if (kind == 2) caloric_value = prof.tree_caloric_value;

            a.energy = std::min(a.energy + taken * prof.trophic_efficiency * caloric_value, 1.0f);
            a.graze_timer += dt;

            intent.magnitude = 0.0f;
            break;
        }

        case AgentState::SeekMate: {
            int mate = find_mate(a, prof, agents, agent_grid, i);
            if (mate < 0 || a.energy < prof.reproduce_threshold) {
                a.state = AgentState::Wander;
                break;
            }

            Agent& m = agents[static_cast<size_t>(mate)];
            float dx = m.pos[0] - a.pos[0];
            float dz = m.pos[1] - a.pos[1];
            float dist = vec2_length(dx, dz);

            if (dist < 1.0f) {
                a.energy -= prof.reproduce_cost;
                m.energy -= prof.reproduce_cost;
                a.mate_cooldown = prof.reproduce_cooldown;
                m.mate_cooldown = prof.reproduce_cooldown;
                a.state = AgentState::Wander;
                m.state = AgentState::Wander;

                constexpr float pi = 3.14159265f;
                Agent child{};
                child.pos[0] = (a.pos[0] + m.pos[0]) * 0.5f;
                child.pos[1] = (a.pos[1] + m.pos[1]) * 0.5f;
                child.heading = dhash_float(agent_seed, 5) * 2.0f * pi;
                child.energy = prof.offspring_energy;
                child.species_id = a.species_id;
                child.alive = true;
                child.female = dhash_float(agent_seed, 6) > 0.5f;
                offspring.push_back(child);
            } else {
                intent.direction[0] = dx / dist;
                intent.direction[1] = dz / dist;
                intent.magnitude = 0.7f;
            }
            break;
        }

        case AgentState::Flee: {
            a.flee_timer -= dt;
            if (a.flee_timer <= 0.0f) {
                a.state = AgentState::Wander;
                break;
            }
            intent.direction[0] = a.flee_dir[0];
            intent.direction[1] = a.flee_dir[1];
            intent.magnitude = 1.0f;
            break;
        }
        }

        // --- Terrain avoidance steering ---
        if (intent.magnitude > 0.01f) {
            apply_terrain_steering(a, prof, world, intent.direction[0], intent.direction[1]);
            vec2_normalize(intent.direction[0], intent.direction[1]);
        }

        // --- Blend herd steering into movement intent ---
        if (intent.magnitude > 0.01f &&
            a.state != AgentState::Flee && a.state != AgentState::SeekMate) {
            HerdIntent herd = compute_herd(a, prof, agents, agent_grid, i);
            if (herd.neighbor_count > 0) {
                intent.direction[0] += herd.cohesion[0]   * prof.herd_weight
                                     + herd.separation[0] * prof.separation_weight
                                     + herd.alignment[0]  * prof.alignment_weight;
                intent.direction[1] += herd.cohesion[1]   * prof.herd_weight
                                     + herd.separation[1] * prof.separation_weight
                                     + herd.alignment[1]  * prof.alignment_weight;
                vec2_normalize(intent.direction[0], intent.direction[1]);
            }
        }

        // --- Integrate movement ---
        float speed = (a.state == AgentState::Flee)
                        ? prof.run_speed
                        : prof.move_speed;
        float actual_speed = speed * intent.magnitude;

        // Slope speed penalty
        float slope = sample_slope(world, a.pos[0], a.pos[1]);
        if (slope > prof.max_slope) {
            actual_speed *= 0.1f;
        } else if (slope > prof.max_slope * 0.5f) {
            float t = (slope - prof.max_slope * 0.5f) / (prof.max_slope * 0.5f);
            actual_speed *= (1.0f - t * 0.7f);
        }

        if (intent.magnitude > 0.01f) {
            a.heading = angle_toward(a.heading,
                                     intent.direction[0], intent.direction[1],
                                     prof.turn_rate, dt);

            a.vel[0] = std::sin(a.heading) * actual_speed;
            a.vel[1] = std::cos(a.heading) * actual_speed;
        } else {
            a.vel[0] = 0.0f;
            a.vel[1] = 0.0f;
        }

        a.pos[0] += a.vel[0] * dt;
        a.pos[1] += a.vel[1] * dt;

        // Push out of water (hard constraint)
        float w = sample_water(world, a.pos[0], a.pos[1]);
        if (w > 0.05f) {
            a.pos[0] -= a.vel[0] * dt;
            a.pos[1] -= a.vel[1] * dt;
            a.vel[0] = 0.0f;
            a.vel[1] = 0.0f;
        }

        a.pos[0] = std::clamp(a.pos[0],
                              -world.tile_half_x + 1.0f,
                               world.tile_half_x - 1.0f);
        a.pos[1] = std::clamp(a.pos[1],
                              -world.tile_half_z + 1.0f,
                               world.tile_half_z - 1.0f);

        // --- Energy drain (caloric model) ---
        float basal = prof.basal_rate;
        float locomotion = prof.locomotion_cost * (actual_speed * actual_speed);
        float slope_mult = 1.0f + slope * prof.slope_cost_factor;
        float total_drain = (basal + locomotion * slope_mult) * dt;
        a.energy -= total_drain;

        if (a.energy <= prof.starve_threshold || a.age >= prof.max_age) {
            a.alive = false;
        }

        a.age += dt;
    }

    for (auto& child : offspring)
        agents.push_back(child);

    // Compact dead agents periodically to prevent unbounded vector growth
    if ((tick_seed & 0x7F) == 0) {
        agents.erase(
            std::remove_if(agents.begin(), agents.end(),
                           [](const Agent& a) { return !a.alive; }),
            agents.end());
    }
}

// -----------------------------------------------------------------------
// Spawning (terrain-aware: avoid water)
// -----------------------------------------------------------------------

void spawn_creatures(
    std::vector<Agent>& agents,
    uint16_t species_id,
    int count,
    const EnvironmentField& env,
    float tile_half_x, float tile_half_z,
    uint32_t seed)
{
    constexpr float pi = 3.14159265f;

    for (int i = 0; i < count; ++i) {
        uint32_t s = dhash(seed ^ static_cast<uint32_t>(i * 2654435761u));

        float x = (dhash_float(s, 0) * 2.0f - 1.0f) * (tile_half_x - 2.0f);
        float z = (dhash_float(s, 1) * 2.0f - 1.0f) * (tile_half_z - 2.0f);

        // Skip overly wet spawn points (moisture > 0.8 implies standing water)
        auto sample = env(x, z);
        if (sample.moisture > 0.8f) {
            // Retry with offset seed
            s = dhash(s + 7u);
            x = (dhash_float(s, 0) * 2.0f - 1.0f) * (tile_half_x - 2.0f);
            z = (dhash_float(s, 1) * 2.0f - 1.0f) * (tile_half_z - 2.0f);
        }

        Agent a{};
        a.pos[0] = x;
        a.pos[1] = z;
        a.heading = dhash_float(s, 2) * 2.0f * pi;
        a.energy = 0.7f + dhash_float(s, 3) * 0.3f;
        a.species_id = species_id;
        a.alive = true;
        a.female = dhash_float(s, 4) > 0.5f;
        agents.push_back(a);
    }
}

// -----------------------------------------------------------------------
// Stats
// -----------------------------------------------------------------------

int count_alive(const std::vector<Agent>& agents)
{
    int n = 0;
    for (const auto& a : agents)
        if (a.alive) ++n;
    return n;
}

int count_alive_species(const std::vector<Agent>& agents, uint16_t species_id)
{
    int n = 0;
    for (const auto& a : agents)
        if (a.alive && a.species_id == species_id) ++n;
    return n;
}

float avg_energy(const std::vector<Agent>& agents)
{
    int n = 0;
    float sum = 0.0f;
    for (const auto& a : agents) {
        if (a.alive) {
            sum += a.energy;
            ++n;
        }
    }
    return n > 0 ? sum / static_cast<float>(n) : 0.0f;
}

float min_energy(const std::vector<Agent>& agents)
{
    float m = 1.0f;
    for (const auto& a : agents)
        if (a.alive && a.energy < m) m = a.energy;
    return m;
}

float max_energy(const std::vector<Agent>& agents)
{
    float m = 0.0f;
    for (const auto& a : agents)
        if (a.alive && a.energy > m) m = a.energy;
    return m;
}

} // namespace bestiary

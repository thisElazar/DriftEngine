#include "creature_mesh.h"
#include "creature_profile.h"
#include "animals/herbivore.h"
#include "animals/predator.h"
#include "animals/rabbit.h"
#include "animals/bird.h"
#include "animals/snake.h"
#include "skeleton/animation.h"
#include "skeleton/skinning.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace bestiary {

enum CycleSlot : int {
    CYCLE_WALK  = 0,
    CYCLE_TROT  = 1,
    CYCLE_RUN   = 2,
    CYCLE_IDLE  = 3,
    CYCLE_GRAZE = 4,
    CYCLE_STALK = 5,
    CYCLE_COUNT = 6,
};

struct CachedAnimalModel {
    AnimalMesh  mesh;
    WalkCycle   cycles[CYCLE_COUNT];
    Archetype   archetype = Archetype::Herbivore;
    bool        valid = false;
};

static std::vector<CachedAnimalModel> s_models;

static HerbivoreParams make_herbivore_params(const CreatureProfile& profile)
{
    HerbivoreParams p{};
    p.torso_length     = profile.body_length;
    p.torso_girth      = profile.body_length * 0.35f;
    p.neck_length      = profile.body_length * 0.4f;
    p.head_length      = profile.body_length * 0.2f;
    p.leg_length_front = profile.body_height * 0.6f;
    p.leg_length_back  = profile.body_height * 0.6f;
    p.leg_thickness    = profile.body_length * 0.035f;
    p.hoof_size        = profile.body_length * 0.04f;
    p.coat_color[0]    = profile.body_color[0];
    p.coat_color[1]    = profile.body_color[1];
    p.coat_color[2]    = profile.body_color[2];
    p.walk_period_seconds = 0.9f;
    p.foot_lift_height    = 0.08f;
    return p;
}

static PredatorParams make_predator_params(const CreatureProfile& profile)
{
    PredatorParams p{};
    p.torso_length     = profile.body_length;
    p.chest_girth      = profile.body_length * 0.40f;
    p.waist_girth      = profile.body_length * 0.28f;
    p.neck_length      = profile.body_length * 0.35f;
    p.snout_length     = profile.body_length * 0.30f;
    p.head_width       = profile.body_length * 0.18f;
    p.leg_length_front = profile.body_height * 0.55f;
    p.leg_length_back  = profile.body_height * 0.60f;
    p.leg_thickness    = profile.body_length * 0.035f;
    p.paw_size         = profile.body_length * 0.04f;
    p.tail_length      = profile.body_length * 0.45f;
    p.coat_color[0]    = profile.body_color[0];
    p.coat_color[1]    = profile.body_color[1];
    p.coat_color[2]    = profile.body_color[2];
    p.walk_period_seconds = 0.85f;
    p.foot_lift_height    = 0.07f;
    return p;
}

static RabbitParams make_rabbit_params(const CreatureProfile& profile)
{
    RabbitParams p{};
    p.body_length      = profile.body_length;
    p.body_girth       = profile.body_length * 0.47f;
    p.neck_length      = profile.body_length * 0.2f;
    p.head_length      = profile.body_length * 0.27f;
    p.head_width       = profile.body_length * 0.23f;
    p.ear_length       = profile.body_length * 0.4f;
    p.leg_length_front = profile.body_height * 0.45f;
    p.leg_length_back  = profile.body_height * 0.65f;
    p.leg_thickness    = profile.body_length * 0.05f;
    p.paw_size         = profile.body_length * 0.05f;
    p.tail_size        = profile.body_length * 0.1f;
    p.coat_color[0]    = profile.body_color[0];
    p.coat_color[1]    = profile.body_color[1];
    p.coat_color[2]    = profile.body_color[2];
    p.hop_period_seconds = 0.45f;
    p.hop_height         = 0.06f;
    return p;
}

static BirdParams make_bird_params(const CreatureProfile& profile)
{
    BirdParams p{};
    p.body_length      = profile.body_length;
    p.body_girth       = profile.body_length * 0.47f;
    p.neck_length      = profile.body_length * 0.4f;
    p.head_size        = profile.body_length * 0.16f;
    p.beak_length      = profile.body_length * 0.12f;
    p.wing_length      = profile.body_length * 0.88f;
    p.leg_length_upper = profile.body_height * 0.5f;
    p.leg_length_lower = profile.body_height * 0.3f;
    p.leg_thickness    = profile.body_length * 0.032f;
    p.foot_size        = profile.body_length * 0.08f;
    p.tail_length      = profile.body_length * 0.24f;
    p.coat_color[0]    = profile.body_color[0];
    p.coat_color[1]    = profile.body_color[1];
    p.coat_color[2]    = profile.body_color[2];
    p.walk_period_seconds = 0.5f;
    p.foot_lift_height    = 0.03f;
    p.hop_height          = 0.04f;
    return p;
}

static CachedAnimalModel& ensure_model(uint16_t species_id,
                                        const CreatureProfile& profile)
{
    if (s_models.size() <= species_id)
        s_models.resize(static_cast<size_t>(species_id) + 1);

    auto& model = s_models[species_id];
    if (model.valid) return model;

    model.archetype = profile.archetype;

    switch (profile.archetype) {
    case Archetype::Herbivore: {
        auto p = make_herbivore_params(profile);
        model.mesh = generate_herbivore_mesh(p);
        model.cycles[CYCLE_WALK]  = make_herbivore_walk(p);
        model.cycles[CYCLE_TROT]  = make_herbivore_trot(p);
        model.cycles[CYCLE_RUN]   = make_herbivore_run(p);
        model.cycles[CYCLE_IDLE]  = make_herbivore_idle(p);
        model.cycles[CYCLE_GRAZE] = make_herbivore_graze(p);
        model.cycles[CYCLE_STALK] = model.cycles[CYCLE_WALK];
        break;
    }
    case Archetype::Predator: {
        auto p = make_predator_params(profile);
        model.mesh = generate_predator_mesh(p);
        model.cycles[CYCLE_WALK]  = make_predator_walk(p);
        model.cycles[CYCLE_TROT]  = make_predator_trot(p);
        model.cycles[CYCLE_RUN]   = make_predator_run(p);
        model.cycles[CYCLE_IDLE]  = make_predator_idle(p);
        model.cycles[CYCLE_GRAZE] = model.cycles[CYCLE_IDLE];
        model.cycles[CYCLE_STALK] = make_predator_stalk(p);
        break;
    }
    case Archetype::Rabbit: {
        auto p = make_rabbit_params(profile);
        model.mesh = generate_rabbit_mesh(p);
        model.cycles[CYCLE_WALK]  = make_rabbit_hop(p);
        model.cycles[CYCLE_TROT]  = make_rabbit_hop(p);
        model.cycles[CYCLE_RUN]   = make_rabbit_run(p);
        model.cycles[CYCLE_IDLE]  = make_rabbit_idle(p);
        model.cycles[CYCLE_GRAZE] = make_rabbit_graze(p);
        model.cycles[CYCLE_STALK] = model.cycles[CYCLE_WALK];
        break;
    }
    case Archetype::Bird: {
        BirdParams p = make_bird_params(profile);
        model.mesh = generate_bird_mesh(p);
        model.cycles[CYCLE_WALK]  = make_bird_walk(p);
        model.cycles[CYCLE_TROT]  = make_bird_hop(p);
        model.cycles[CYCLE_RUN]   = make_bird_run(p);
        model.cycles[CYCLE_IDLE]  = make_bird_idle(p);
        model.cycles[CYCLE_GRAZE] = make_bird_peck(p);
        model.cycles[CYCLE_STALK] = model.cycles[CYCLE_WALK];
        break;
    }
    case Archetype::Raptor: {
        BirdParams p = make_bird_params(profile);
        model.mesh = generate_bird_mesh(p);
        model.cycles[CYCLE_WALK]  = make_bird_walk(p);
        model.cycles[CYCLE_TROT]  = make_bird_hop(p);
        model.cycles[CYCLE_RUN]   = make_bird_soar(p);
        model.cycles[CYCLE_IDLE]  = make_bird_idle(p);
        model.cycles[CYCLE_GRAZE] = make_bird_dive(p);
        model.cycles[CYCLE_STALK] = make_bird_soar(p);
        break;
    }
    case Archetype::Snake: {
        SnakeParams p;
        p.body_length     = profile.body_length;
        p.body_thickness  = profile.body_height * 0.05f;
        p.head_width      = profile.body_height * 0.06f;
        p.coat_color[0] = profile.body_color[0];
        p.coat_color[1] = profile.body_color[1];
        p.coat_color[2] = profile.body_color[2];
        model.mesh = generate_snake_mesh(p);
        model.cycles[CYCLE_WALK]  = make_snake_slither(p);
        model.cycles[CYCLE_TROT]  = make_snake_slither(p);
        model.cycles[CYCLE_RUN]   = make_snake_fast(p);
        model.cycles[CYCLE_IDLE]  = make_snake_idle(p);
        model.cycles[CYCLE_GRAZE] = make_snake_idle(p);
        model.cycles[CYCLE_STALK] = make_snake_strike(p);
        break;
    }
    }

    model.valid = true;
    return model;
}

static std::vector<glm::mat4> compute_world_posed(const Skeleton& skel,
                                                   const std::vector<glm::quat>& pose)
{
    size_t n = skel.joints.size();
    std::vector<glm::mat4> world(n);

    for (size_t i = 0; i < n; ++i) {
        glm::mat4 local = skel.joints[i].local_bind;
        if (i < pose.size()) {
            local = local * glm::mat4_cast(pose[i]);
        }
        world[i] = (skel.joints[i].parent == -1)
            ? local
            : world[skel.joints[i].parent] * local;
    }
    return world;
}

static const WalkCycle* select_cycle(const CachedAnimalModel& model,
                                     AgentState state, float speed)
{
    switch (state) {
    case AgentState::Graze:    return &model.cycles[CYCLE_GRAZE];
    case AgentState::Consume:  return &model.cycles[CYCLE_IDLE];
    case AgentState::Hunt:     return &model.cycles[CYCLE_STALK];
    case AgentState::Flee:
    case AgentState::Chase:    return &model.cycles[CYCLE_RUN];
    case AgentState::Fly:      return &model.cycles[CYCLE_RUN];
    case AgentState::Perch:    return &model.cycles[CYCLE_IDLE];
    case AgentState::Dive:     return &model.cycles[CYCLE_GRAZE];
    case AgentState::Ambush:   return &model.cycles[CYCLE_IDLE];
    case AgentState::Strike:   return &model.cycles[CYCLE_STALK];
    case AgentState::SeekMate:
    case AgentState::Wander:
        if (speed > 4.0f)      return &model.cycles[CYCLE_RUN];
        if (speed > 2.0f)      return &model.cycles[CYCLE_TROT];
        if (speed > 0.1f)      return &model.cycles[CYCLE_WALK];
        return &model.cycles[CYCLE_IDLE];
    }
    return &model.cycles[CYCLE_IDLE];
}

static void append_skinned_agent(VegetationMesh& mesh,
                                 Agent& agent,
                                 const CachedAnimalModel& model,
                                 float terrain_y,
                                 float dt)
{
    const auto& skel = model.mesh.skeleton;
    int joint_count = static_cast<int>(skel.joints.size());

    float speed = std::sqrt(agent.vel[0] * agent.vel[0] + agent.vel[1] * agent.vel[1]);
    const WalkCycle* cycle = select_cycle(model, agent.state, speed);

    const WalkCycle* prev_cycle = select_cycle(model, agent.prev_anim_state, agent.prev_anim_speed);
    if (cycle != prev_cycle) {
        agent.anim_blend = 0.0f;
        agent.prev_anim_state = agent.state;
        agent.prev_anim_speed = speed;
    }

    constexpr float BLEND_DURATION = 0.2f;
    agent.anim_blend = std::min(agent.anim_blend + dt / BLEND_DURATION, 1.0f);

    float walk_phase = 0.0f;
    if (cycle->period_seconds > 0.01f) {
        walk_phase = std::fmod(agent.age / cycle->period_seconds, 1.0f);
    }

    auto pose = sample_walk(*cycle, walk_phase, joint_count);

    if (agent.anim_blend < 1.0f && prev_cycle != cycle) {
        float prev_phase = 0.0f;
        if (prev_cycle->period_seconds > 0.01f)
            prev_phase = std::fmod(agent.age / prev_cycle->period_seconds, 1.0f);

        auto prev_pose = sample_walk(*prev_cycle, prev_phase, joint_count);
        for (int j = 0; j < joint_count; ++j)
            pose[static_cast<size_t>(j)] = glm::slerp(
                prev_pose[static_cast<size_t>(j)],
                pose[static_cast<size_t>(j)],
                agent.anim_blend);
    }

    agent.prev_anim_state = agent.state;
    agent.prev_anim_speed = speed;
    auto world_mats = compute_world_posed(skel, pose);

    std::vector<glm::mat4> palette(skel.joints.size());
    for (size_t i = 0; i < skel.joints.size(); ++i) {
        palette[i] = world_mats[i] * skel.inverse_bind[i];
    }

    std::vector<SkinnedVertex> skinned;
    cpu_skin(model.mesh.vertices, palette, skinned);

    float cos_h = std::cos(agent.heading);
    float sin_h = std::sin(agent.heading);

    auto base = static_cast<uint32_t>(mesh.vertices.size());

    for (const auto& sv : skinned) {
        float lx = sv.position[0];
        float ly = sv.position[1];
        float lz = sv.position[2];

        VegetationVertex v{};
        v.position[0] = agent.pos[0] + lx * cos_h + lz * sin_h;
        v.position[1] = terrain_y + agent.altitude + ly;
        v.position[2] = agent.pos[1] - lx * sin_h + lz * cos_h;

        v.normal[0] = sv.normal[0] * cos_h + sv.normal[2] * sin_h;
        v.normal[1] = sv.normal[1];
        v.normal[2] = -sv.normal[0] * sin_h + sv.normal[2] * cos_h;

        v.color[0] = sv.color[0];
        v.color[1] = sv.color[1];
        v.color[2] = sv.color[2];
        v.height_t = 0.0f;

        mesh.vertices.push_back(v);
    }

    for (uint32_t idx : model.mesh.indices) {
        mesh.indices.push_back(base + idx);
    }
}

VegetationMesh generate_creature_meshes(
    std::vector<Agent>& agents,
    const std::vector<CreatureProfile>& profiles,
    std::function<float(float x, float z)> terrain_height,
    float dt)
{
    VegetationMesh mesh;
    if (profiles.empty()) return mesh;

    for (auto& agent : agents) {
        if (!agent.alive) continue;
        if (agent.species_id >= profiles.size()) continue;

        auto& model = ensure_model(agent.species_id, profiles[agent.species_id]);

        float y = terrain_height ? terrain_height(agent.pos[0], agent.pos[1]) : 0.0f;
        append_skinned_agent(mesh, agent, model, y, dt);
    }

    return mesh;
}

void clear_creature_mesh_cache()
{
    s_models.clear();
}

} // namespace bestiary

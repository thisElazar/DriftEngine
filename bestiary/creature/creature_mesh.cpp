#include "creature_mesh.h"
#include "herbivore.h"
#include "animals/herbivore.h"
#include "skeleton/animation.h"
#include "skeleton/skinning.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace bestiary {

struct CachedAnimalModel {
    AnimalMesh       mesh;
    WalkCycle        walk;
    WalkCycle        trot;
    WalkCycle        run;
    WalkCycle        idle;
    WalkCycle        graze;
    HerbivoreParams  params;
    bool             valid = false;
};

static std::vector<CachedAnimalModel> s_models;

static CachedAnimalModel& ensure_model(uint16_t species_id,
                                        const HerbivoreProfile& profile)
{
    if (s_models.size() <= species_id)
        s_models.resize(static_cast<size_t>(species_id) + 1);

    auto& model = s_models[species_id];
    if (model.valid) return model;

    HerbivoreParams params{};
    params.torso_length     = profile.body_length;
    params.torso_girth      = profile.body_length * 0.35f;
    params.neck_length      = profile.body_length * 0.4f;
    params.head_length      = profile.body_length * 0.2f;
    params.leg_length_front = profile.body_height * 0.6f;
    params.leg_length_back  = profile.body_height * 0.6f;
    params.leg_thickness    = profile.body_length * 0.035f;
    params.hoof_size        = profile.body_length * 0.04f;
    params.coat_color[0]    = profile.body_color[0];
    params.coat_color[1]    = profile.body_color[1];
    params.coat_color[2]    = profile.body_color[2];
    params.walk_period_seconds = 0.9f;
    params.foot_lift_height    = 0.08f;

    model.params = params;
    model.mesh   = generate_herbivore_mesh(params);
    model.walk   = make_herbivore_walk(params);
    model.trot   = make_herbivore_trot(params);
    model.run    = make_herbivore_run(params);
    model.idle   = make_herbivore_idle(params);
    model.graze  = make_herbivore_graze(params);
    model.valid  = true;
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
    case AgentState::Graze:    return &model.graze;
    case AgentState::Flee:     return &model.run;
    case AgentState::SeekMate:
    case AgentState::Wander:
        if (speed > 4.0f)      return &model.run;
        if (speed > 2.0f)      return &model.trot;
        if (speed > 0.1f)      return &model.walk;
        return &model.idle;
    }
    return &model.idle;
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
        v.position[1] = terrain_y + ly;
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
    const std::vector<HerbivoreProfile>& profiles,
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

} // namespace bestiary

#include "animals/predator.h"
#include "skeleton/animation.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace bestiary {

static glm::quat pitch(float degrees)
{
    return glm::angleAxis(glm::radians(degrees), glm::vec3(1, 0, 0));
}

static glm::quat roll(float degrees)
{
    return glm::angleAxis(glm::radians(degrees), glm::vec3(0, 0, 1));
}

static glm::quat yaw(float degrees)
{
    return glm::angleAxis(glm::radians(degrees), glm::vec3(0, 1, 0));
}

static float wrap(float p)
{
    p = std::fmod(p, 1.0f);
    return (p < 0.0f) ? p + 1.0f : p;
}

static void add_key(std::vector<JointKeyframe>& track, float phase, glm::quat rot)
{
    track.push_back({phase, rot});
}

static void sort_track(std::vector<JointKeyframe>& track)
{
    std::sort(track.begin(), track.end(),
              [](const JointKeyframe& a, const JointKeyframe& b) { return a.phase < b.phase; });
}

struct LegKey { float p; float hip_deg; float knee_deg; };

static void add_leg_keys(WalkCycle& cycle,
                         int hip_joint, int knee_joint,
                         float phase_offset, float knee_sign, float lift,
                         const LegKey* keys, int count)
{
    auto& hip  = cycle.tracks[hip_joint];
    auto& knee = cycle.tracks[knee_joint];

    for (int i = 0; i < count; ++i) {
        add_key(hip,  wrap(keys[i].p + phase_offset), pitch(keys[i].hip_deg));
        add_key(knee, wrap(keys[i].p + phase_offset), pitch(keys[i].knee_deg * knee_sign * lift));
    }

    sort_track(hip);
    sort_track(knee);
}

// ---------------------------------------------------------------------------
// Walk — deliberate, purposeful stride
// ---------------------------------------------------------------------------
static const LegKey pred_walk_leg[] = {
    {0.00f, -16.0f,    0.0f},
    {0.10f, -12.0f,    0.0f},
    {0.25f,  -4.0f,    0.0f},
    {0.45f,   6.0f,    0.0f},
    {0.65f,  13.0f,    0.0f},
    {0.78f,  16.0f,    0.0f},
    {0.82f,  13.0f,  -35.0f},
    {0.88f,   0.0f,  -35.0f},
    {0.94f, -16.0f,   -8.0f},
    {0.97f, -16.0f,    0.0f},
};

WalkCycle make_predator_walk(const PredatorParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds;
    cycle.hip_swing_deg   = 16.0f;
    cycle.stance_fraction = 0.80f;
    cycle.tracks.resize(26);

    float lift = p.foot_lift_height / 0.07f;

    add_leg_keys(cycle, 15, 16, 0.00f,  1.0f, lift, pred_walk_leg, 10);
    add_leg_keys(cycle,  7,  8, 0.25f, -1.0f, lift, pred_walk_leg, 10);
    add_leg_keys(cycle, 19, 20, 0.50f,  1.0f, lift, pred_walk_leg, 10);
    add_leg_keys(cycle, 11, 12, 0.75f, -1.0f, lift, pred_walk_leg, 10);

    auto& spine1 = cycle.tracks[1];
    auto& spine2 = cycle.tracks[2];
    add_key(spine1, 0.00f, roll( 2.0f));
    add_key(spine1, 0.25f, roll( 0.0f));
    add_key(spine1, 0.50f, roll(-2.0f));
    add_key(spine1, 0.75f, roll( 0.0f));
    add_key(spine2, 0.00f, roll( 1.0f));
    add_key(spine2, 0.50f, roll(-1.0f));

    auto& neck = cycle.tracks[4];
    add_key(neck, 0.00f, pitch( 2.0f));
    add_key(neck, 0.25f, pitch(-1.0f));
    add_key(neck, 0.50f, pitch( 2.0f));
    add_key(neck, 0.75f, pitch(-1.0f));

    auto& tail_base = cycle.tracks[23];
    auto& tail_mid  = cycle.tracks[24];
    add_key(tail_base, 0.00f, roll(-3.0f));
    add_key(tail_base, 0.50f, roll( 3.0f));
    add_key(tail_mid,  0.00f, roll(-2.0f));
    add_key(tail_mid,  0.50f, roll( 2.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Trot — diagonal pairs, primary cruising gait for canids
// ---------------------------------------------------------------------------
static const LegKey pred_trot_leg[] = {
    {0.00f, -20.0f,    0.0f},
    {0.08f, -16.0f,    0.0f},
    {0.20f,  -6.0f,    0.0f},
    {0.40f,   6.0f,    0.0f},
    {0.55f,  16.0f,    0.0f},
    {0.66f,  20.0f,    0.0f},
    {0.70f,  16.0f,  -42.0f},
    {0.78f,   0.0f,  -42.0f},
    {0.90f, -20.0f,  -10.0f},
    {0.95f, -20.0f,    0.0f},
};

WalkCycle make_predator_trot(const PredatorParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds * 0.60f;
    cycle.hip_swing_deg   = 20.0f;
    cycle.stance_fraction = 0.68f;
    cycle.tracks.resize(26);

    float lift = p.foot_lift_height / 0.07f;

    add_leg_keys(cycle, 15, 16, 0.00f,  1.0f, lift, pred_trot_leg, 10);
    add_leg_keys(cycle, 11, 12, 0.00f, -1.0f, lift, pred_trot_leg, 10);
    add_leg_keys(cycle, 19, 20, 0.50f,  1.0f, lift, pred_trot_leg, 10);
    add_leg_keys(cycle,  7,  8, 0.50f, -1.0f, lift, pred_trot_leg, 10);

    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, roll( 0.8f));
    add_key(spine1, 0.25f, roll( 0.0f));
    add_key(spine1, 0.50f, roll(-0.8f));
    add_key(spine1, 0.75f, roll( 0.0f));

    auto& neck = cycle.tracks[4];
    add_key(neck, 0.00f, pitch( 1.5f));
    add_key(neck, 0.50f, pitch( 1.5f));

    auto& tail_base = cycle.tracks[23];
    auto& tail_mid  = cycle.tracks[24];
    add_key(tail_base, 0.00f, pitch( 5.0f));
    add_key(tail_base, 0.50f, pitch( 5.0f));
    add_key(tail_mid,  0.00f, roll(-1.5f));
    add_key(tail_mid,  0.50f, roll( 1.5f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Run (gallop) — rotary or transverse gallop, pronounced spinal flexion
// ---------------------------------------------------------------------------
static const LegKey pred_run_leg[] = {
    {0.00f, -25.0f,    0.0f},
    {0.06f, -20.0f,    0.0f},
    {0.16f,  -8.0f,    0.0f},
    {0.30f,  10.0f,    0.0f},
    {0.45f,  22.0f,    0.0f},
    {0.50f,  25.0f,    0.0f},
    {0.55f,  18.0f,  -58.0f},
    {0.65f,   0.0f,  -58.0f},
    {0.80f, -25.0f,  -15.0f},
    {0.90f, -25.0f,    0.0f},
};

WalkCycle make_predator_run(const PredatorParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds * 0.40f;
    cycle.hip_swing_deg   = 25.0f;
    cycle.stance_fraction = 0.50f;
    cycle.tracks.resize(26);

    float lift = p.foot_lift_height / 0.07f;

    // Rotary gallop: hind pair slightly offset, front pair slightly offset
    add_leg_keys(cycle, 15, 16, 0.00f,  1.0f, lift, pred_run_leg, 10);
    add_leg_keys(cycle, 19, 20, 0.08f,  1.0f, lift, pred_run_leg, 10);
    add_leg_keys(cycle,  7,  8, 0.50f, -1.0f, lift, pred_run_leg, 10);
    add_leg_keys(cycle, 11, 12, 0.58f, -1.0f, lift, pred_run_leg, 10);

    // Pronounced spinal flexion-extension
    auto& spine1 = cycle.tracks[1];
    auto& spine2 = cycle.tracks[2];
    add_key(spine1, 0.00f, pitch(-6.0f));
    add_key(spine1, 0.25f, pitch( 7.0f));
    add_key(spine1, 0.50f, pitch(-6.0f));
    add_key(spine1, 0.75f, pitch( 7.0f));
    add_key(spine2, 0.00f, pitch(-4.0f));
    add_key(spine2, 0.25f, pitch( 5.0f));
    add_key(spine2, 0.50f, pitch(-4.0f));
    add_key(spine2, 0.75f, pitch( 5.0f));

    auto& neck = cycle.tracks[4];
    add_key(neck, 0.00f, pitch( 4.0f));
    add_key(neck, 0.25f, pitch(-2.0f));
    add_key(neck, 0.50f, pitch( 4.0f));
    add_key(neck, 0.75f, pitch(-2.0f));

    // Tail streams behind, bounces with stride
    auto& tail_base = cycle.tracks[23];
    auto& tail_mid  = cycle.tracks[24];
    auto& tail_tip  = cycle.tracks[25];
    add_key(tail_base, 0.00f, pitch( 10.0f));
    add_key(tail_base, 0.25f, pitch(  5.0f));
    add_key(tail_base, 0.50f, pitch( 10.0f));
    add_key(tail_base, 0.75f, pitch(  5.0f));
    add_key(tail_mid,  0.00f, pitch(  5.0f));
    add_key(tail_mid,  0.25f, pitch(-3.0f));
    add_key(tail_mid,  0.50f, pitch(  5.0f));
    add_key(tail_mid,  0.75f, pitch(-3.0f));
    add_key(tail_tip,  0.00f, pitch(  3.0f));
    add_key(tail_tip,  0.50f, pitch( -2.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Idle — alert standing, ears up, scanning
// ---------------------------------------------------------------------------
WalkCycle make_predator_idle(const PredatorParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 4.0f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.tracks.resize(26);

    // Weight shift
    auto& hip_l = cycle.tracks[15];
    auto& hip_r = cycle.tracks[19];
    add_key(hip_l, 0.00f, pitch(-1.0f));
    add_key(hip_l, 0.50f, pitch( 0.5f));
    add_key(hip_r, 0.00f, pitch( 0.5f));
    add_key(hip_r, 0.50f, pitch(-1.0f));

    // Spine sway
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, roll( 0.8f));
    add_key(spine1, 0.50f, roll(-0.8f));

    // Head scanning — alert, looking around
    auto& neck1 = cycle.tracks[4];
    add_key(neck1, 0.00f, pitch( 3.0f));
    add_key(neck1, 0.30f, pitch(-1.0f));
    add_key(neck1, 0.60f, pitch( 2.0f));
    add_key(neck1, 0.85f, pitch( 0.0f));

    auto& neck2 = cycle.tracks[5];
    add_key(neck2, 0.00f, yaw( 8.0f));
    add_key(neck2, 0.25f, yaw(-5.0f));
    add_key(neck2, 0.55f, yaw(10.0f));
    add_key(neck2, 0.80f, yaw(-8.0f));

    auto& head = cycle.tracks[6];
    add_key(head, 0.00f, yaw(-3.0f));
    add_key(head, 0.40f, yaw( 4.0f));
    add_key(head, 0.75f, yaw(-2.0f));

    // Tail: relaxed gentle sway
    auto& tail_base = cycle.tracks[23];
    auto& tail_mid  = cycle.tracks[24];
    add_key(tail_base, 0.00f, roll(-2.0f) * pitch(5.0f));
    add_key(tail_base, 0.50f, roll( 2.0f) * pitch(5.0f));
    add_key(tail_mid,  0.00f, roll(-3.0f));
    add_key(tail_mid,  0.50f, roll( 3.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Stalk — low, slow, predatory approach
// ---------------------------------------------------------------------------
static const LegKey pred_stalk_leg[] = {
    {0.00f, -10.0f,    0.0f},
    {0.15f,  -6.0f,    0.0f},
    {0.35f,   0.0f,    0.0f},
    {0.55f,   6.0f,    0.0f},
    {0.72f,  10.0f,    0.0f},
    {0.78f,   8.0f,  -20.0f},
    {0.85f,   0.0f,  -20.0f},
    {0.93f, -10.0f,   -5.0f},
    {0.97f, -10.0f,    0.0f},
};

WalkCycle make_predator_stalk(const PredatorParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds * 1.8f;
    cycle.hip_swing_deg   = 10.0f;
    cycle.stance_fraction = 0.78f;
    cycle.pelvis_drop     = 0.08f;
    cycle.tracks.resize(26);

    float lift = p.foot_lift_height / 0.07f * 0.5f;

    // Slow lateral sequence, careful placement
    add_leg_keys(cycle, 15, 16, 0.00f,  1.0f, lift, pred_stalk_leg, 9);
    add_leg_keys(cycle,  7,  8, 0.25f, -1.0f, lift, pred_stalk_leg, 9);
    add_leg_keys(cycle, 19, 20, 0.50f,  1.0f, lift, pred_stalk_leg, 9);
    add_leg_keys(cycle, 11, 12, 0.75f, -1.0f, lift, pred_stalk_leg, 9);

    // Spine low and level
    auto& spine1 = cycle.tracks[1];
    auto& spine2 = cycle.tracks[2];
    add_key(spine1, 0.00f, pitch(3.0f) * roll( 0.5f));
    add_key(spine1, 0.50f, pitch(3.0f) * roll(-0.5f));
    add_key(spine2, 0.00f, pitch(2.0f));
    add_key(spine2, 0.50f, pitch(2.0f));

    // Neck extended forward, head low and locked on target
    auto& neck1 = cycle.tracks[4];
    auto& neck2 = cycle.tracks[5];
    add_key(neck1, 0.00f, pitch(8.0f));
    add_key(neck1, 0.50f, pitch(8.0f));
    add_key(neck2, 0.00f, pitch(4.0f));
    add_key(neck2, 0.50f, pitch(4.0f));

    // Head steady, focused
    auto& head = cycle.tracks[6];
    add_key(head, 0.00f, pitch(-5.0f));
    add_key(head, 0.50f, pitch(-5.0f));

    // Tail low and still
    auto& tail_base = cycle.tracks[23];
    auto& tail_mid  = cycle.tracks[24];
    auto& tail_tip  = cycle.tracks[25];
    add_key(tail_base, 0.00f, pitch(15.0f));
    add_key(tail_base, 0.50f, pitch(15.0f));
    add_key(tail_mid,  0.00f, pitch(5.0f));
    add_key(tail_mid,  0.50f, pitch(5.0f));
    add_key(tail_tip,  0.00f, pitch(3.0f));
    add_key(tail_tip,  0.50f, pitch(3.0f));

    return cycle;
}

} // namespace bestiary

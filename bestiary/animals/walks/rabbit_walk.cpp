#include "animals/rabbit.h"
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
// Hop — the primary rabbit gait
// Both hind legs push together, both front legs land together.
// Phase: hind push (0.0) → flight (0.2) → front land (0.4) → gather (0.6) → hind reach (0.8)
// ---------------------------------------------------------------------------
static const LegKey rabbit_hop_hind[] = {
    {0.00f, -30.0f,   0.0f},   // hind legs fully extended back (push-off)
    {0.10f, -15.0f,   0.0f},   // leaving ground
    {0.20f,   0.0f, -50.0f},   // tucked under during flight
    {0.40f,  25.0f, -60.0f},   // swinging forward
    {0.60f,  35.0f, -40.0f},   // reaching forward
    {0.75f,  30.0f, -10.0f},   // about to plant
    {0.85f,  15.0f,   0.0f},   // planted, weight loading
    {0.95f, -15.0f,   0.0f},   // pushing back
};

static const LegKey rabbit_hop_fore[] = {
    {0.00f,  15.0f,   0.0f},   // front legs ahead, absorbing
    {0.15f,   5.0f,   0.0f},   // pushing off
    {0.25f, -10.0f,   0.0f},   // leaving ground
    {0.35f,   0.0f, -35.0f},   // tucking during flight
    {0.50f,  10.0f, -40.0f},   // reaching forward
    {0.65f,  20.0f, -15.0f},   // extending to land
    {0.75f,  20.0f,   0.0f},   // landing
    {0.90f,  15.0f,   0.0f},   // absorbing
};

WalkCycle make_rabbit_hop(const RabbitParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.hop_period_seconds;
    cycle.hip_swing_deg   = 30.0f;
    cycle.stance_fraction = 0.45f;
    cycle.tracks.resize(24);

    float lift = p.hop_height / 0.06f;

    // Both hind legs together (phase 0)
    add_leg_keys(cycle, 15, 16, 0.0f,  1.0f, lift, rabbit_hop_hind, 8);
    add_leg_keys(cycle, 19, 20, 0.0f,  1.0f, lift, rabbit_hop_hind, 8);

    // Both front legs together (offset ~0.4 cycles behind hind)
    add_leg_keys(cycle,  7,  8, 0.0f, -1.0f, lift, rabbit_hop_fore, 8);
    add_leg_keys(cycle, 11, 12, 0.0f, -1.0f, lift, rabbit_hop_fore, 8);

    // Spine flexion-extension — pronounced, drives the hop
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(-8.0f));    // extended at push-off
    add_key(spine1, 0.20f, pitch( 12.0f));   // flexed during flight (tucked)
    add_key(spine1, 0.50f, pitch(-5.0f));    // extending at front landing
    add_key(spine1, 0.80f, pitch( 8.0f));    // flexing as hind reach forward

    // Head stays relatively level (counter-bobs)
    auto& neck = cycle.tracks[3];
    add_key(neck, 0.00f, pitch( 5.0f));
    add_key(neck, 0.20f, pitch(-8.0f));
    add_key(neck, 0.50f, pitch( 3.0f));
    add_key(neck, 0.80f, pitch(-5.0f));

    // Ears bounce — slight lag behind body
    auto& ear_l = cycle.tracks[5];
    auto& ear_r = cycle.tracks[6];
    add_key(ear_l, 0.00f, pitch(-5.0f));
    add_key(ear_l, 0.25f, pitch( 10.0f));
    add_key(ear_l, 0.50f, pitch(-3.0f));
    add_key(ear_l, 0.75f, pitch( 8.0f));
    add_key(ear_r, 0.00f, pitch(-5.0f));
    add_key(ear_r, 0.25f, pitch( 10.0f));
    add_key(ear_r, 0.50f, pitch(-3.0f));
    add_key(ear_r, 0.75f, pitch( 8.0f));

    // Tail bobs
    auto& tail = cycle.tracks[23];
    add_key(tail, 0.00f, pitch(-5.0f));
    add_key(tail, 0.25f, pitch( 8.0f));
    add_key(tail, 0.50f, pitch(-3.0f));
    add_key(tail, 0.75f, pitch( 6.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Run — fast bounding, more airborne time
// ---------------------------------------------------------------------------
WalkCycle make_rabbit_run(const RabbitParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.hop_period_seconds * 0.6f;
    cycle.hip_swing_deg   = 35.0f;
    cycle.stance_fraction = 0.35f;
    cycle.tracks.resize(24);

    float lift = p.hop_height / 0.06f * 1.3f;

    add_leg_keys(cycle, 15, 16, 0.0f,  1.0f, lift, rabbit_hop_hind, 8);
    add_leg_keys(cycle, 19, 20, 0.0f,  1.0f, lift, rabbit_hop_hind, 8);
    add_leg_keys(cycle,  7,  8, 0.0f, -1.0f, lift, rabbit_hop_fore, 8);
    add_leg_keys(cycle, 11, 12, 0.0f, -1.0f, lift, rabbit_hop_fore, 8);

    // More pronounced spinal flexion at speed
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(-12.0f));
    add_key(spine1, 0.20f, pitch( 15.0f));
    add_key(spine1, 0.50f, pitch(-8.0f));
    add_key(spine1, 0.80f, pitch( 12.0f));

    auto& neck = cycle.tracks[3];
    add_key(neck, 0.00f, pitch( 8.0f));
    add_key(neck, 0.20f, pitch(-10.0f));
    add_key(neck, 0.50f, pitch( 5.0f));
    add_key(neck, 0.80f, pitch(-8.0f));

    auto& ear_l = cycle.tracks[5];
    auto& ear_r = cycle.tracks[6];
    add_key(ear_l, 0.00f, pitch(-8.0f));
    add_key(ear_l, 0.30f, pitch( 15.0f));
    add_key(ear_l, 0.60f, pitch(-5.0f));
    add_key(ear_r, 0.00f, pitch(-8.0f));
    add_key(ear_r, 0.30f, pitch( 15.0f));
    add_key(ear_r, 0.60f, pitch(-5.0f));

    auto& tail = cycle.tracks[23];
    add_key(tail, 0.00f, pitch(-8.0f));
    add_key(tail, 0.25f, pitch( 12.0f));
    add_key(tail, 0.50f, pitch(-5.0f));
    add_key(tail, 0.75f, pitch( 10.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Idle — sitting alert, nose twitching, ears swiveling
// ---------------------------------------------------------------------------
WalkCycle make_rabbit_idle(const RabbitParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 3.0f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.tracks.resize(24);

    // Slight weight shift
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(1.0f));
    add_key(spine1, 0.50f, pitch(-0.5f));

    // Head looking around
    auto& neck = cycle.tracks[3];
    add_key(neck, 0.00f, pitch( 2.0f) * yaw( 5.0f));
    add_key(neck, 0.20f, pitch(-1.0f) * yaw(-8.0f));
    add_key(neck, 0.45f, pitch( 3.0f) * yaw( 3.0f));
    add_key(neck, 0.70f, pitch( 0.0f) * yaw(-6.0f));
    add_key(neck, 0.90f, pitch( 1.0f) * yaw( 7.0f));

    // Ears — independent swiveling
    auto& ear_l = cycle.tracks[5];
    auto& ear_r = cycle.tracks[6];
    add_key(ear_l, 0.00f, roll( 5.0f));
    add_key(ear_l, 0.15f, roll(-8.0f));
    add_key(ear_l, 0.40f, roll(10.0f));
    add_key(ear_l, 0.65f, roll(-3.0f));
    add_key(ear_l, 0.85f, roll( 7.0f));

    add_key(ear_r, 0.00f, roll(-3.0f));
    add_key(ear_r, 0.20f, roll( 8.0f));
    add_key(ear_r, 0.50f, roll(-6.0f));
    add_key(ear_r, 0.75f, roll(10.0f));
    add_key(ear_r, 0.90f, roll(-5.0f));

    // Nose twitch — fast, subtle head movement
    auto& head = cycle.tracks[4];
    add_key(head, 0.00f, pitch( 1.5f));
    add_key(head, 0.08f, pitch(-1.0f));
    add_key(head, 0.16f, pitch( 1.5f));
    add_key(head, 0.24f, pitch(-0.5f));
    add_key(head, 0.50f, pitch( 1.0f));
    add_key(head, 0.58f, pitch(-1.0f));
    add_key(head, 0.66f, pitch( 1.5f));
    add_key(head, 0.74f, pitch(-0.5f));

    // Tail — tiny wobble
    auto& tail = cycle.tracks[23];
    add_key(tail, 0.00f, roll(-1.0f));
    add_key(tail, 0.50f, roll( 1.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Graze — nose down, nibbling
// ---------------------------------------------------------------------------
WalkCycle make_rabbit_graze(const RabbitParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 2.0f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.pelvis_drop     = 0.0f;
    cycle.tracks.resize(24);

    // Spine tilts forward
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(5.0f));
    add_key(spine1, 0.50f, pitch(5.0f));

    // Neck angles down
    auto& neck = cycle.tracks[3];
    add_key(neck, 0.00f, pitch(25.0f) * yaw( 3.0f));
    add_key(neck, 0.25f, pitch(28.0f));
    add_key(neck, 0.50f, pitch(25.0f) * yaw(-3.0f));
    add_key(neck, 0.75f, pitch(28.0f));

    // Head — nibbling motion
    auto& head = cycle.tracks[4];
    add_key(head, 0.00f, pitch( 15.0f));
    add_key(head, 0.10f, pitch( 18.0f));
    add_key(head, 0.20f, pitch( 15.0f));
    add_key(head, 0.30f, pitch( 17.0f));
    add_key(head, 0.50f, pitch( 15.0f));
    add_key(head, 0.60f, pitch( 18.0f));
    add_key(head, 0.70f, pitch( 15.0f));
    add_key(head, 0.80f, pitch( 17.0f));

    // Ears relaxed, gentle sway
    auto& ear_l = cycle.tracks[5];
    auto& ear_r = cycle.tracks[6];
    add_key(ear_l, 0.00f, pitch(5.0f) * roll( 3.0f));
    add_key(ear_l, 0.50f, pitch(5.0f) * roll(-2.0f));
    add_key(ear_r, 0.00f, pitch(5.0f) * roll(-2.0f));
    add_key(ear_r, 0.50f, pitch(5.0f) * roll( 3.0f));

    // Front shoulders counter-rotate slightly to keep legs vertical with spine tilt
    auto& shoulder_l = cycle.tracks[7];
    auto& shoulder_r = cycle.tracks[11];
    add_key(shoulder_l, 0.00f, pitch(-5.0f));
    add_key(shoulder_l, 0.50f, pitch(-5.0f));
    add_key(shoulder_r, 0.00f, pitch(-5.0f));
    add_key(shoulder_r, 0.50f, pitch(-5.0f));

    return cycle;
}

} // namespace bestiary

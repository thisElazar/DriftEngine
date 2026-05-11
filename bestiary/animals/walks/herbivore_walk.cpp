#include "animals/herbivore.h"
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

// ---------------------------------------------------------------------------
// Generic leg keyframe applicator
// ---------------------------------------------------------------------------
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
// Walk — lateral sequence, LH → LF → RH → RF
// ---------------------------------------------------------------------------
static const LegKey walk_leg[] = {
    // Hip pitch: negative = foot ahead, positive = foot behind
    // contact (foot ahead) → long stance → lift-off (foot behind)
    {0.00f, -14.0f,    0.0f},
    {0.10f, -10.0f,    0.0f},
    {0.25f,  -3.0f,    0.0f},
    {0.45f,   5.0f,    0.0f},
    {0.65f,  11.0f,    0.0f},
    {0.80f,  14.0f,    0.0f},
    // swing: lift → carry → place
    {0.83f,  12.0f,  -38.0f},
    {0.88f,   0.0f,  -38.0f},
    {0.94f, -14.0f,  -10.0f},
    {0.97f, -14.0f,    0.0f},
};

WalkCycle make_herbivore_walk(const HerbivoreParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds;
    cycle.hip_swing_deg   = 14.0f;
    cycle.stance_fraction = 0.83f;
    cycle.tracks.resize(24);

    float lift = p.foot_lift_height / 0.08f;

    // Lateral-sequence walk: same-side hind leads fore
    add_leg_keys(cycle, 15, 16, 0.00f,  1.0f, lift, walk_leg, 10);  // left hind
    add_leg_keys(cycle,  7,  8, 0.25f, -1.0f, lift, walk_leg, 10);  // left fore
    add_leg_keys(cycle, 19, 20, 0.50f,  1.0f, lift, walk_leg, 10);  // right hind
    add_leg_keys(cycle, 11, 12, 0.75f, -1.0f, lift, walk_leg, 10);  // right fore

    // Spine sway — lateral bend following the gait rhythm
    auto& spine1 = cycle.tracks[1];
    auto& spine2 = cycle.tracks[2];
    struct { float p; float deg; } sway[] = {
        {0.00f,  2.5f}, {0.25f, 0.0f}, {0.50f, -2.5f}, {0.75f, 0.0f},
    };
    for (auto& s : sway) {
        add_key(spine1, s.p, roll(s.deg));
        add_key(spine2, s.p, roll(s.deg * 0.6f));
    }

    // Head bob
    auto& neck = cycle.tracks[4];
    add_key(neck, 0.00f, pitch( 3.0f));
    add_key(neck, 0.25f, pitch(-1.5f));
    add_key(neck, 0.50f, pitch( 3.0f));
    add_key(neck, 0.75f, pitch(-1.5f));

    // Tail sway
    auto& tail = cycle.tracks[23];
    add_key(tail, 0.00f, roll(-4.0f));
    add_key(tail, 0.25f, roll(-1.0f));
    add_key(tail, 0.50f, roll( 4.0f));
    add_key(tail, 0.75f, roll( 1.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Trot — diagonal pairs: LH+RF together, RH+LF together
// ---------------------------------------------------------------------------
static const LegKey trot_leg[] = {
    // contact → stance → lift-off (70% stance, 30% swing)
    {0.00f, -18.0f,    0.0f},
    {0.08f, -14.0f,    0.0f},
    {0.20f,  -5.0f,    0.0f},
    {0.40f,   5.0f,    0.0f},
    {0.55f,  14.0f,    0.0f},
    {0.68f,  18.0f,    0.0f},
    // swing: lift → carry → place
    {0.72f,  14.0f,  -45.0f},
    {0.80f,   0.0f,  -45.0f},
    {0.90f, -18.0f,  -12.0f},
    {0.95f, -18.0f,    0.0f},
};

WalkCycle make_herbivore_trot(const HerbivoreParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds * 0.65f;
    cycle.hip_swing_deg   = 18.0f;
    cycle.stance_fraction = 0.70f;
    cycle.tracks.resize(24);

    float lift = p.foot_lift_height / 0.08f;

    // Diagonal pairs — LH+RF at phase 0, RH+LF at phase 0.5
    add_leg_keys(cycle, 15, 16, 0.00f,  1.0f, lift, trot_leg, 10);  // left hind
    add_leg_keys(cycle, 11, 12, 0.00f, -1.0f, lift, trot_leg, 10);  // right fore
    add_leg_keys(cycle, 19, 20, 0.50f,  1.0f, lift, trot_leg, 10);  // right hind
    add_leg_keys(cycle,  7,  8, 0.50f, -1.0f, lift, trot_leg, 10);  // left fore

    // Minimal spine sway — diagonal pairs cancel lateral forces
    auto& spine1 = cycle.tracks[1];
    auto& spine2 = cycle.tracks[2];
    add_key(spine1, 0.00f, roll( 1.0f));
    add_key(spine1, 0.25f, roll( 0.0f));
    add_key(spine1, 0.50f, roll(-1.0f));
    add_key(spine1, 0.75f, roll( 0.0f));
    add_key(spine2, 0.00f, roll( 0.5f));
    add_key(spine2, 0.50f, roll(-0.5f));

    // Head stays relatively steady in trot
    auto& neck = cycle.tracks[4];
    add_key(neck, 0.00f, pitch( 1.5f));
    add_key(neck, 0.25f, pitch(-0.5f));
    add_key(neck, 0.50f, pitch( 1.5f));
    add_key(neck, 0.75f, pitch(-0.5f));

    // Tail — gentle sway
    auto& tail = cycle.tracks[23];
    add_key(tail, 0.00f, roll(-2.0f));
    add_key(tail, 0.50f, roll( 2.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Run (bound) — front pair + back pair, spine flexion-extension
// ---------------------------------------------------------------------------
static const LegKey run_leg[] = {
    // contact → stance → lift-off (55% stance, 45% swing)
    {0.00f, -22.0f,    0.0f},
    {0.06f, -18.0f,    0.0f},
    {0.18f,  -6.0f,    0.0f},
    {0.32f,   8.0f,    0.0f},
    {0.48f,  20.0f,    0.0f},
    {0.53f,  22.0f,    0.0f},
    // swing: lift → carry → place
    {0.58f,  16.0f,  -55.0f},
    {0.68f,   0.0f,  -55.0f},
    {0.82f, -22.0f,  -15.0f},
    {0.92f, -22.0f,    0.0f},
};

WalkCycle make_herbivore_run(const HerbivoreParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds * 0.45f;
    cycle.hip_swing_deg   = 22.0f;
    cycle.stance_fraction = 0.55f;
    cycle.tracks.resize(24);

    float lift = p.foot_lift_height / 0.08f;

    // Front/back pairs — back legs at phase 0, front legs at phase 0.5
    add_leg_keys(cycle, 15, 16, 0.00f,  1.0f, lift, run_leg, 10);  // left hind
    add_leg_keys(cycle, 19, 20, 0.00f,  1.0f, lift, run_leg, 10);  // right hind
    add_leg_keys(cycle,  7,  8, 0.50f, -1.0f, lift, run_leg, 10);  // left fore
    add_leg_keys(cycle, 11, 12, 0.50f, -1.0f, lift, run_leg, 10);  // right fore

    // Spine flexion-extension — the signature of a gallop/bound
    auto& spine1 = cycle.tracks[1];
    auto& spine2 = cycle.tracks[2];
    // Back-leg contact → spine flexes; back-leg push-off → spine extends
    add_key(spine1, 0.00f, pitch(-4.0f));   // flexed at back-leg contact
    add_key(spine1, 0.25f, pitch( 5.0f));   // extended, gathered flight
    add_key(spine1, 0.50f, pitch(-4.0f));   // flexed at front-leg contact
    add_key(spine1, 0.75f, pitch( 5.0f));   // extended, extended flight
    add_key(spine2, 0.00f, pitch(-3.0f));
    add_key(spine2, 0.25f, pitch( 4.0f));
    add_key(spine2, 0.50f, pitch(-3.0f));
    add_key(spine2, 0.75f, pitch( 4.0f));

    // Head — bobs with the stride
    auto& neck = cycle.tracks[4];
    add_key(neck, 0.00f, pitch( 5.0f));
    add_key(neck, 0.25f, pitch(-3.0f));
    add_key(neck, 0.50f, pitch( 5.0f));
    add_key(neck, 0.75f, pitch(-3.0f));

    // Tail — bounces vertically, slight sway
    auto& tail = cycle.tracks[23];
    add_key(tail, 0.00f, pitch( 8.0f));
    add_key(tail, 0.25f, pitch(-5.0f));
    add_key(tail, 0.50f, pitch( 8.0f));
    add_key(tail, 0.75f, pitch(-5.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Idle — standing, looking around, weight shifting
// ---------------------------------------------------------------------------
WalkCycle make_herbivore_idle(const HerbivoreParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 5.0f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.tracks.resize(24);

    // Weight shift — hips rock gently, alternating sides
    auto& hip_l = cycle.tracks[15];
    auto& hip_r = cycle.tracks[19];
    add_key(hip_l, 0.00f, pitch(-1.5f));
    add_key(hip_l, 0.50f, pitch( 1.0f));
    add_key(hip_r, 0.00f, pitch( 1.0f));
    add_key(hip_r, 0.50f, pitch(-1.5f));

    // Spine: lateral sway follows weight shift
    auto& spine1 = cycle.tracks[1];
    auto& spine2 = cycle.tracks[2];
    add_key(spine1, 0.00f, roll( 1.2f));
    add_key(spine1, 0.50f, roll(-1.2f));
    add_key(spine2, 0.00f, roll( 0.6f));
    add_key(spine2, 0.50f, roll(-0.6f));

    // Neck: looking around — asymmetric keyframes for non-repetitive feel
    auto& neck1 = cycle.tracks[4];
    add_key(neck1, 0.00f, pitch( 4.0f));
    add_key(neck1, 0.20f, pitch(-2.0f));
    add_key(neck1, 0.55f, pitch( 1.0f));
    add_key(neck1, 0.80f, pitch( 3.0f));

    auto& neck2 = cycle.tracks[5];
    add_key(neck2, 0.00f, roll( 5.0f));
    add_key(neck2, 0.30f, roll(-3.0f));
    add_key(neck2, 0.60f, roll( 6.0f));
    add_key(neck2, 0.85f, roll(-4.0f));

    // Head: extra turn detail
    auto& head = cycle.tracks[6];
    add_key(head, 0.00f, roll(-2.0f));
    add_key(head, 0.35f, roll( 3.0f));
    add_key(head, 0.70f, roll(-1.0f));

    // Tail: slow lazy sway
    auto& tail = cycle.tracks[23];
    add_key(tail, 0.00f, roll(-3.0f));
    add_key(tail, 0.33f, roll( 2.0f));
    add_key(tail, 0.66f, roll(-1.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Graze — feeding at a given height, nibbling, weight shifts
// ---------------------------------------------------------------------------
WalkCycle make_herbivore_graze(const HerbivoreParams& p, float feed_height)
{
    WalkCycle cycle;
    cycle.period_seconds  = 3.0f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.tracks.resize(24);

    // t=0 → ground, t=1 → max reach (upward)
    float max_reach = p.leg_length_front + p.neck_length + p.head_length;
    float t = glm::clamp(feed_height / max_reach, 0.0f, 1.0f);
    float down = 1.0f - t;

    cycle.pelvis_drop = 0.0f;

    // THE HINGE: rotate spine_3 (joint 3) — the ball between the front legs
    float hinge = glm::mix(95.0f, 0.0f, t);
    auto& spine3 = cycle.tracks[3];
    add_key(spine3, 0.00f, pitch(hinge));
    add_key(spine3, 0.50f, pitch(hinge));

    // Counter-rotate front shoulders so legs stay vertical despite spine_3 rotation
    auto& shoulder_l = cycle.tracks[7];
    auto& shoulder_r = cycle.tracks[11];
    add_key(shoulder_l, 0.00f, pitch(-hinge - 1.0f * down));
    add_key(shoulder_l, 0.50f, pitch(-hinge + 1.0f * down));
    add_key(shoulder_r, 0.00f, pitch(-hinge + 1.0f * down));
    add_key(shoulder_r, 0.50f, pitch(-hinge - 1.0f * down));

    // Spine: subtle forward tilt + sway
    auto& spine1 = cycle.tracks[1];
    auto& spine2 = cycle.tracks[2];
    float spine_tilt = 3.0f * down;
    add_key(spine1, 0.00f, pitch(spine_tilt) * roll( 0.5f));
    add_key(spine1, 0.50f, pitch(spine_tilt) * roll(-0.5f));
    add_key(spine2, 0.00f, pitch(spine_tilt * 0.5f));
    add_key(spine2, 0.50f, pitch(spine_tilt * 0.5f));

    // Neck stays straight — spine_3 already did the heavy lifting
    float n1_base = glm::mix(  5.0f,  -8.0f, t);
    float n2_base = glm::mix(  3.0f,  -3.0f, t);
    float hd_base = glm::mix( 30.0f,  -2.0f, t);
    float vary    = glm::mix(  2.0f,   1.5f, t);

    auto& neck1 = cycle.tracks[4];
    add_key(neck1, 0.00f, pitch(n1_base - vary));
    add_key(neck1, 0.50f, pitch(n1_base + vary));

    auto& neck2 = cycle.tracks[5];
    add_key(neck2, 0.00f, pitch(n2_base - vary * 0.7f));
    add_key(neck2, 0.50f, pitch(n2_base + vary * 0.7f));

    // Head: searching side to side, nose pitched toward food
    auto& head = cycle.tracks[6];
    add_key(head, 0.00f,  pitch(hd_base) * roll( 4.0f));
    add_key(head, 0.25f,  pitch(hd_base + 3.0f));
    add_key(head, 0.50f,  pitch(hd_base) * roll(-4.0f));
    add_key(head, 0.75f,  pitch(hd_base + 3.0f));

    // Tail: lazy sway
    auto& tail = cycle.tracks[23];
    add_key(tail, 0.00f, roll(-2.0f));
    add_key(tail, 0.50f, roll( 2.0f));

    return cycle;
}

} // namespace bestiary

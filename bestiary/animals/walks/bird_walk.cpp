#include "animals/bird.h"
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

static constexpr int N_JOINTS = 23;

// ---------------------------------------------------------------------------
// Walk — alternating bipedal walk with head bob
// ---------------------------------------------------------------------------
static const LegKey bird_walk_leg[] = {
    {0.00f,  15.0f,   0.0f},   // foot planted forward
    {0.15f,   8.0f,   0.0f},   // mid-stance
    {0.30f,  -5.0f,   0.0f},   // foot passing under
    {0.42f, -15.0f,   0.0f},   // foot behind, pushing off
    {0.50f, -10.0f, -30.0f},   // toe-off, knee bends
    {0.62f,   0.0f, -40.0f},   // swing, high lift
    {0.75f,  12.0f, -20.0f},   // extending forward
    {0.88f,  15.0f,  -5.0f},   // placing foot
};

WalkCycle make_bird_walk(const BirdParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds;
    cycle.hip_swing_deg   = 15.0f;
    cycle.stance_fraction = 0.65f;
    cycle.tracks.resize(N_JOINTS);

    float lift = p.foot_lift_height / 0.03f;

    add_leg_keys(cycle, 14, 15, 0.0f, 1.0f, lift, bird_walk_leg, 8);
    add_leg_keys(cycle, 18, 19, 0.5f, 1.0f, lift, bird_walk_leg, 8);

    // Head bob — thrust forward, hold, retract (twice per stride)
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch(-6.0f));
    add_key(neck1, 0.08f, pitch(-10.0f));
    add_key(neck1, 0.30f, pitch(-10.0f));
    add_key(neck1, 0.42f, pitch( 0.0f));
    add_key(neck1, 0.50f, pitch(-6.0f));
    add_key(neck1, 0.58f, pitch(-10.0f));
    add_key(neck1, 0.80f, pitch(-10.0f));
    add_key(neck1, 0.92f, pitch( 0.0f));

    // Body sway — slight lateral rocking
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, roll(-2.0f));
    add_key(spine1, 0.25f, roll( 0.0f));
    add_key(spine1, 0.50f, roll( 2.0f));
    add_key(spine1, 0.75f, roll( 0.0f));

    // Tail bobs opposite to body
    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch(-3.0f));
    add_key(tail, 0.25f, pitch( 2.0f));
    add_key(tail, 0.50f, pitch(-3.0f));
    add_key(tail, 0.75f, pitch( 2.0f));

    // Wings held tight against body (slight fold via animation)
    auto& wing_l = cycle.tracks[6];
    auto& wing_r = cycle.tracks[10];
    add_key(wing_l, 0.0f, pitch(0.0f));
    add_key(wing_r, 0.0f, pitch(0.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Hop — sparrow-like, both feet together
// ---------------------------------------------------------------------------
static const LegKey bird_hop_leg[] = {
    {0.00f, -20.0f,   0.0f},   // legs back, push-off
    {0.10f, -10.0f,   0.0f},   // leaving ground
    {0.20f,   0.0f, -40.0f},   // tucked during flight
    {0.40f,  15.0f, -45.0f},   // swinging forward
    {0.55f,  20.0f, -20.0f},   // extending to land
    {0.70f,  15.0f,   0.0f},   // landing, absorbing
    {0.85f,   0.0f,   0.0f},   // crouching to push
    {0.95f, -15.0f,   0.0f},   // pushing back
};

WalkCycle make_bird_hop(const BirdParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds * 0.7f;
    cycle.hip_swing_deg   = 20.0f;
    cycle.stance_fraction = 0.50f;
    cycle.tracks.resize(N_JOINTS);

    float lift = p.hop_height / 0.04f;

    // Both legs together
    add_leg_keys(cycle, 14, 15, 0.0f, 1.0f, lift, bird_hop_leg, 8);
    add_leg_keys(cycle, 18, 19, 0.0f, 1.0f, lift, bird_hop_leg, 8);

    // Spine crunch and extend
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(-5.0f));
    add_key(spine1, 0.20f, pitch( 3.0f));
    add_key(spine1, 0.50f, pitch(-3.0f));
    add_key(spine1, 0.80f, pitch( 5.0f));

    // Head stays level (counter-bob)
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch( 3.0f));
    add_key(neck1, 0.20f, pitch(-5.0f));
    add_key(neck1, 0.50f, pitch( 2.0f));
    add_key(neck1, 0.80f, pitch(-3.0f));

    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch(-4.0f));
    add_key(tail, 0.25f, pitch( 5.0f));
    add_key(tail, 0.50f, pitch(-3.0f));
    add_key(tail, 0.75f, pitch( 4.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Run — fast alternating walk, longer stride
// ---------------------------------------------------------------------------
WalkCycle make_bird_run(const BirdParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.walk_period_seconds * 0.55f;
    cycle.hip_swing_deg   = 22.0f;
    cycle.stance_fraction = 0.50f;
    cycle.tracks.resize(N_JOINTS);

    float lift = p.foot_lift_height / 0.03f * 1.3f;

    add_leg_keys(cycle, 14, 15, 0.0f, 1.0f, lift, bird_walk_leg, 8);
    add_leg_keys(cycle, 18, 19, 0.5f, 1.0f, lift, bird_walk_leg, 8);

    // More body lean forward at speed
    auto& chest = cycle.tracks[2];
    add_key(chest, 0.0f, pitch(-4.0f));
    add_key(chest, 0.5f, pitch(-4.0f));

    // Faster head bob
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch(-8.0f));
    add_key(neck1, 0.06f, pitch(-12.0f));
    add_key(neck1, 0.25f, pitch(-12.0f));
    add_key(neck1, 0.40f, pitch( 0.0f));
    add_key(neck1, 0.50f, pitch(-8.0f));
    add_key(neck1, 0.56f, pitch(-12.0f));
    add_key(neck1, 0.75f, pitch(-12.0f));
    add_key(neck1, 0.90f, pitch( 0.0f));

    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, roll(-3.0f));
    add_key(spine1, 0.50f, roll( 3.0f));

    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch(-4.0f));
    add_key(tail, 0.25f, pitch( 3.0f));
    add_key(tail, 0.50f, pitch(-4.0f));
    add_key(tail, 0.75f, pitch( 3.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Fly — wing flapping, legs tucked, airborne
// ---------------------------------------------------------------------------
WalkCycle make_bird_fly(const BirdParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.flap_period;
    cycle.hip_swing_deg   = 25.0f;
    cycle.stance_fraction = 0.5f;
    cycle.tracks.resize(N_JOINTS);

    float flap = p.flap_amplitude;
    float sweep = flap * p.flap_sweep * 0.5f;

    // Wing flap — left: roll = up/down, yaw = forward/back (90° out of phase for elliptical path)
    auto& wl = cycle.tracks[6];
    add_key(wl, 0.00f, roll( flap)        * yaw(-sweep));
    add_key(wl, 0.15f, roll( flap * 0.3f) * yaw(-sweep * 0.8f));
    add_key(wl, 0.25f, roll( 0.0f)        * yaw( 0.0f));
    add_key(wl, 0.40f, roll(-flap * 0.7f) * yaw( sweep));
    add_key(wl, 0.55f, roll(-flap * 0.3f) * yaw( sweep * 0.6f));
    add_key(wl, 0.70f, roll( 0.0f)        * yaw( 0.0f));
    add_key(wl, 0.85f, roll( flap * 0.7f) * yaw(-sweep * 0.8f));

    // Elbow — folds on upstroke, extends on downstroke
    auto& el = cycle.tracks[7];
    add_key(el, 0.00f, roll(-12.0f));
    add_key(el, 0.35f, roll(  0.0f));
    add_key(el, 0.60f, roll(-18.0f));
    add_key(el, 0.85f, roll(-10.0f));

    // Wrist trails
    auto& wrl = cycle.tracks[8];
    add_key(wrl, 0.00f, roll(-5.0f));
    add_key(wrl, 0.40f, roll( 5.0f));
    add_key(wrl, 0.70f, roll(-8.0f));

    // Wing flap — right: mirrored roll, same yaw direction
    auto& wr = cycle.tracks[10];
    add_key(wr, 0.00f, roll(-flap)        * yaw( sweep));
    add_key(wr, 0.15f, roll(-flap * 0.3f) * yaw( sweep * 0.8f));
    add_key(wr, 0.25f, roll( 0.0f)        * yaw( 0.0f));
    add_key(wr, 0.40f, roll( flap * 0.7f) * yaw(-sweep));
    add_key(wr, 0.55f, roll( flap * 0.3f) * yaw(-sweep * 0.6f));
    add_key(wr, 0.70f, roll( 0.0f)        * yaw( 0.0f));
    add_key(wr, 0.85f, roll(-flap * 0.7f) * yaw( sweep * 0.8f));

    auto& er = cycle.tracks[11];
    add_key(er, 0.00f, roll( 12.0f));
    add_key(er, 0.35f, roll(  0.0f));
    add_key(er, 0.60f, roll( 18.0f));
    add_key(er, 0.85f, roll( 10.0f));

    auto& wrr = cycle.tracks[12];
    add_key(wrr, 0.00f, roll( 5.0f));
    add_key(wrr, 0.40f, roll(-5.0f));
    add_key(wrr, 0.70f, roll( 8.0f));

    // Legs tucked against body
    auto& hl = cycle.tracks[14];
    auto& hr = cycle.tracks[18];
    add_key(hl, 0.0f, pitch(35.0f));
    add_key(hr, 0.0f, pitch(35.0f));

    auto& ul = cycle.tracks[15];
    auto& ur = cycle.tracks[19];
    add_key(ul, 0.0f, pitch(-55.0f));
    add_key(ur, 0.0f, pitch(-55.0f));

    // Body pitched slightly forward for flight
    auto& chest = cycle.tracks[2];
    add_key(chest, 0.0f, pitch(-6.0f));

    // Spine bob from wing beats
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(-2.0f));
    add_key(spine1, 0.35f, pitch( 2.0f));
    add_key(spine1, 0.70f, pitch(-1.0f));

    // Head stays level
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.0f, pitch(6.0f));

    // Tail streams, oscillates with wingbeat
    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch( 5.0f));
    add_key(tail, 0.35f, pitch( 10.0f));
    add_key(tail, 0.70f, pitch( 6.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Idle — standing, looking around, occasional wing ruffle
// ---------------------------------------------------------------------------
WalkCycle make_bird_idle(const BirdParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 4.0f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.tracks.resize(N_JOINTS);

    // Weight shift
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, roll(-1.0f));
    add_key(spine1, 0.50f, roll( 1.0f));

    // Head looking around — asymmetric for natural feel
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch(-2.0f) * yaw( 8.0f));
    add_key(neck1, 0.15f, pitch( 1.0f) * yaw(-5.0f));
    add_key(neck1, 0.35f, pitch(-3.0f) * yaw(-10.0f));
    add_key(neck1, 0.55f, pitch( 0.0f) * yaw( 12.0f));
    add_key(neck1, 0.75f, pitch(-1.0f) * yaw( 3.0f));
    add_key(neck1, 0.90f, pitch( 2.0f) * yaw(-7.0f));

    auto& neck2 = cycle.tracks[4];
    add_key(neck2, 0.00f, yaw( 3.0f));
    add_key(neck2, 0.30f, yaw(-5.0f));
    add_key(neck2, 0.60f, yaw( 6.0f));
    add_key(neck2, 0.85f, yaw(-4.0f));

    // Head tilt
    auto& head = cycle.tracks[5];
    add_key(head, 0.00f, roll( 3.0f));
    add_key(head, 0.20f, roll(-5.0f));
    add_key(head, 0.50f, roll( 4.0f));
    add_key(head, 0.80f, roll(-3.0f));

    // Occasional wing ruffle — left wing lifts slightly
    auto& wing_l = cycle.tracks[6];
    add_key(wing_l, 0.00f, roll( 0.0f));
    add_key(wing_l, 0.40f, roll( 0.0f));
    add_key(wing_l, 0.45f, roll( 8.0f));
    add_key(wing_l, 0.52f, roll( 0.0f));

    // Tail — gentle sway
    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch(-1.0f) * roll(-1.0f));
    add_key(tail, 0.50f, pitch( 1.0f) * roll( 1.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Peck — head down, pecking at ground
// ---------------------------------------------------------------------------
WalkCycle make_bird_peck(const BirdParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 1.2f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.pelvis_drop     = 0.01f;
    cycle.tracks.resize(N_JOINTS);

    // Body tips forward
    auto& chest = cycle.tracks[2];
    add_key(chest, 0.00f, pitch(-8.0f));
    add_key(chest, 0.50f, pitch(-8.0f));

    // Neck — dramatic down-peck motion
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch( 20.0f));
    add_key(neck1, 0.08f, pitch( 35.0f));
    add_key(neck1, 0.15f, pitch( 20.0f));
    add_key(neck1, 0.30f, pitch( 15.0f));
    add_key(neck1, 0.50f, pitch( 20.0f));
    add_key(neck1, 0.58f, pitch( 35.0f));
    add_key(neck1, 0.65f, pitch( 20.0f));
    add_key(neck1, 0.80f, pitch( 15.0f));

    auto& neck2 = cycle.tracks[4];
    add_key(neck2, 0.00f, pitch( 10.0f));
    add_key(neck2, 0.08f, pitch( 15.0f));
    add_key(neck2, 0.15f, pitch( 10.0f));
    add_key(neck2, 0.50f, pitch( 10.0f));
    add_key(neck2, 0.58f, pitch( 15.0f));
    add_key(neck2, 0.65f, pitch( 10.0f));

    // Head — quick strike motion
    auto& head = cycle.tracks[5];
    add_key(head, 0.00f, pitch( 5.0f));
    add_key(head, 0.08f, pitch( 15.0f));
    add_key(head, 0.12f, pitch( 5.0f));
    add_key(head, 0.50f, pitch( 5.0f));
    add_key(head, 0.58f, pitch( 15.0f));
    add_key(head, 0.62f, pitch( 5.0f));

    // Slight side-to-side look between pecks
    add_key(neck1, 0.25f, pitch(15.0f) * yaw( 8.0f));
    add_key(neck1, 0.75f, pitch(15.0f) * yaw(-6.0f));
    sort_track(neck1);

    // Shoulders counter-rotate to keep legs vertical
    auto& hip_l = cycle.tracks[14];
    auto& hip_r = cycle.tracks[18];
    add_key(hip_l, 0.0f, pitch(8.0f));
    add_key(hip_l, 0.5f, pitch(8.0f));
    add_key(hip_r, 0.0f, pitch(8.0f));
    add_key(hip_r, 0.5f, pitch(8.0f));

    // Tail lifts to counterbalance
    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch( 5.0f));
    add_key(tail, 0.08f, pitch( 8.0f));
    add_key(tail, 0.15f, pitch( 5.0f));
    add_key(tail, 0.50f, pitch( 5.0f));
    add_key(tail, 0.58f, pitch( 8.0f));
    add_key(tail, 0.65f, pitch( 5.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Takeoff — crouch, jump, transition into flight
// ---------------------------------------------------------------------------
WalkCycle make_bird_takeoff(const BirdParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = 1.2f;
    cycle.hip_swing_deg   = 12.0f;
    cycle.stance_fraction = 0.6f;
    cycle.tracks.resize(N_JOINTS);

    float flap = p.flap_amplitude;

    // Legs: crouch → jump → tuck
    auto& hl = cycle.tracks[14];
    auto& hr = cycle.tracks[18];
    add_key(hl, 0.00f, pitch(  0.0f));
    add_key(hl, 0.15f, pitch( 15.0f));
    add_key(hl, 0.30f, pitch(-20.0f));
    add_key(hl, 0.50f, pitch( 25.0f));
    add_key(hl, 0.80f, pitch( 35.0f));
    add_key(hr, 0.00f, pitch(  0.0f));
    add_key(hr, 0.15f, pitch( 15.0f));
    add_key(hr, 0.30f, pitch(-20.0f));
    add_key(hr, 0.50f, pitch( 25.0f));
    add_key(hr, 0.80f, pitch( 35.0f));

    auto& ul = cycle.tracks[15];
    auto& ur = cycle.tracks[19];
    add_key(ul, 0.00f, pitch(   0.0f));
    add_key(ul, 0.15f, pitch( -25.0f));
    add_key(ul, 0.30f, pitch(  10.0f));
    add_key(ul, 0.50f, pitch( -40.0f));
    add_key(ul, 0.80f, pitch( -55.0f));
    add_key(ur, 0.00f, pitch(   0.0f));
    add_key(ur, 0.15f, pitch( -25.0f));
    add_key(ur, 0.30f, pitch(  10.0f));
    add_key(ur, 0.50f, pitch( -40.0f));
    add_key(ur, 0.80f, pitch( -55.0f));

    // Wings: explosive flapping from the start — max effort near ground
    auto& wl = cycle.tracks[6];
    add_key(wl, 0.00f, roll(  flap * 0.3f));
    add_key(wl, 0.08f, roll( -flap));
    add_key(wl, 0.18f, roll(  flap));
    add_key(wl, 0.30f, roll( -flap * 0.9f));
    add_key(wl, 0.42f, roll(  flap * 0.9f));
    add_key(wl, 0.55f, roll( -flap * 0.8f));
    add_key(wl, 0.68f, roll(  flap * 0.8f));
    add_key(wl, 0.82f, roll( -flap * 0.7f));
    add_key(wl, 0.95f, roll(  flap * 0.7f));

    auto& wr = cycle.tracks[10];
    add_key(wr, 0.00f, roll( -flap * 0.3f));
    add_key(wr, 0.08f, roll(  flap));
    add_key(wr, 0.18f, roll( -flap));
    add_key(wr, 0.30f, roll(  flap * 0.9f));
    add_key(wr, 0.42f, roll( -flap * 0.9f));
    add_key(wr, 0.55f, roll(  flap * 0.8f));
    add_key(wr, 0.68f, roll( -flap * 0.8f));
    add_key(wr, 0.82f, roll(  flap * 0.7f));
    add_key(wr, 0.95f, roll( -flap * 0.7f));

    // Elbows pump with the flaps
    auto& el = cycle.tracks[7];
    add_key(el, 0.00f, roll( -5.0f));
    add_key(el, 0.08f, roll(  0.0f));
    add_key(el, 0.18f, roll(-15.0f));
    add_key(el, 0.30f, roll(  0.0f));
    add_key(el, 0.42f, roll(-12.0f));
    add_key(el, 0.68f, roll(-10.0f));

    auto& er = cycle.tracks[11];
    add_key(er, 0.00f, roll(  5.0f));
    add_key(er, 0.08f, roll(  0.0f));
    add_key(er, 0.18f, roll( 15.0f));
    add_key(er, 0.30f, roll(  0.0f));
    add_key(er, 0.42f, roll( 12.0f));
    add_key(er, 0.68f, roll( 10.0f));

    // Body: dip into crouch → launch upward
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(  0.0f));
    add_key(spine1, 0.15f, pitch(  5.0f));
    add_key(spine1, 0.35f, pitch( -5.0f));
    add_key(spine1, 0.60f, pitch( -3.0f));

    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch(  0.0f));
    add_key(neck1, 0.15f, pitch(  3.0f));
    add_key(neck1, 0.35f, pitch( -8.0f));
    add_key(neck1, 0.60f, pitch(  5.0f));

    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch(  0.0f));
    add_key(tail, 0.30f, pitch( -5.0f));
    add_key(tail, 0.50f, pitch(  8.0f));
    add_key(tail, 0.80f, pitch(  6.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Landing — brake with wings, extend legs, settle
// ---------------------------------------------------------------------------
WalkCycle make_bird_land(const BirdParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = 1.5f;
    cycle.hip_swing_deg   = 8.0f;
    cycle.stance_fraction = 0.8f;
    cycle.tracks.resize(N_JOINTS);

    float flap = p.flap_amplitude;

    // Wings: hard braking flaps → spread to stall → fold on touchdown
    auto& wl = cycle.tracks[6];
    add_key(wl, 0.00f, roll(  flap * 0.8f));
    add_key(wl, 0.08f, roll( -flap * 0.9f));
    add_key(wl, 0.18f, roll(  flap * 0.9f));
    add_key(wl, 0.28f, roll( -flap * 0.8f));
    add_key(wl, 0.38f, roll(  flap * 0.7f));
    add_key(wl, 0.48f, roll( -flap * 0.5f));
    add_key(wl, 0.58f, roll(  flap * 0.6f));
    add_key(wl, 0.70f, roll(  flap * 0.4f));
    add_key(wl, 0.82f, roll(  flap * 0.15f));
    add_key(wl, 0.92f, roll(  0.0f));

    auto& wr = cycle.tracks[10];
    add_key(wr, 0.00f, roll( -flap * 0.8f));
    add_key(wr, 0.08f, roll(  flap * 0.9f));
    add_key(wr, 0.18f, roll( -flap * 0.9f));
    add_key(wr, 0.28f, roll(  flap * 0.8f));
    add_key(wr, 0.38f, roll( -flap * 0.7f));
    add_key(wr, 0.48f, roll(  flap * 0.5f));
    add_key(wr, 0.58f, roll( -flap * 0.6f));
    add_key(wr, 0.70f, roll( -flap * 0.4f));
    add_key(wr, 0.82f, roll( -flap * 0.15f));
    add_key(wr, 0.92f, roll(  0.0f));

    // Elbows flex with braking flaps
    auto& el = cycle.tracks[7];
    add_key(el, 0.00f, roll(-10.0f));
    add_key(el, 0.18f, roll(-15.0f));
    add_key(el, 0.38f, roll(-10.0f));
    add_key(el, 0.70f, roll( -5.0f));
    add_key(el, 0.92f, roll(  0.0f));

    auto& er = cycle.tracks[11];
    add_key(er, 0.00f, roll( 10.0f));
    add_key(er, 0.18f, roll( 15.0f));
    add_key(er, 0.38f, roll( 10.0f));
    add_key(er, 0.70f, roll(  5.0f));
    add_key(er, 0.92f, roll(  0.0f));

    // Legs: tucked → extend → absorb impact → standing
    auto& hl = cycle.tracks[14];
    auto& hr = cycle.tracks[18];
    add_key(hl, 0.00f, pitch( 30.0f));
    add_key(hl, 0.30f, pitch( 20.0f));
    add_key(hl, 0.50f, pitch(  5.0f));
    add_key(hl, 0.65f, pitch(  0.0f));
    add_key(hl, 0.75f, pitch(  8.0f));
    add_key(hl, 0.90f, pitch(  0.0f));
    add_key(hr, 0.00f, pitch( 30.0f));
    add_key(hr, 0.30f, pitch( 20.0f));
    add_key(hr, 0.50f, pitch(  5.0f));
    add_key(hr, 0.65f, pitch(  0.0f));
    add_key(hr, 0.75f, pitch(  8.0f));
    add_key(hr, 0.90f, pitch(  0.0f));

    auto& ul = cycle.tracks[15];
    auto& ur = cycle.tracks[19];
    add_key(ul, 0.00f, pitch(-45.0f));
    add_key(ul, 0.30f, pitch(-30.0f));
    add_key(ul, 0.50f, pitch(-10.0f));
    add_key(ul, 0.65f, pitch(-15.0f));
    add_key(ul, 0.90f, pitch(  0.0f));
    add_key(ur, 0.00f, pitch(-45.0f));
    add_key(ur, 0.30f, pitch(-30.0f));
    add_key(ur, 0.50f, pitch(-10.0f));
    add_key(ur, 0.65f, pitch(-15.0f));
    add_key(ur, 0.90f, pitch(  0.0f));

    // Body pitches back to brake, then settles upright
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(-3.0f));
    add_key(spine1, 0.40f, pitch( 3.0f));
    add_key(spine1, 0.65f, pitch( 5.0f));
    add_key(spine1, 0.85f, pitch( 0.0f));

    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch( 5.0f));
    add_key(neck1, 0.40f, pitch(-3.0f));
    add_key(neck1, 0.70f, pitch( 0.0f));

    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch(  6.0f));
    add_key(tail, 0.30f, pitch( -8.0f));
    add_key(tail, 0.60f, pitch( -5.0f));
    add_key(tail, 0.90f, pitch(  0.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Perch — sitting on branch, upright body, looking around
// ---------------------------------------------------------------------------
WalkCycle make_bird_perch(const BirdParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 5.0f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.tracks.resize(N_JOINTS);

    // Body more upright
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(8.0f));
    add_key(spine1, 0.50f, pitch(6.0f));

    // Legs slightly bent — gripping
    auto& hl = cycle.tracks[14];
    auto& hr = cycle.tracks[18];
    add_key(hl, 0.0f, pitch(5.0f));
    add_key(hr, 0.0f, pitch(5.0f));

    auto& ul = cycle.tracks[15];
    auto& ur = cycle.tracks[19];
    add_key(ul, 0.0f, pitch(-12.0f));
    add_key(ur, 0.0f, pitch(-12.0f));

    // Head looking around — calm, slow
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch(-5.0f) * yaw( 5.0f));
    add_key(neck1, 0.20f, pitch(-3.0f) * yaw(-8.0f));
    add_key(neck1, 0.45f, pitch(-6.0f) * yaw( 10.0f));
    add_key(neck1, 0.65f, pitch(-2.0f) * yaw(-5.0f));
    add_key(neck1, 0.85f, pitch(-4.0f) * yaw( 3.0f));

    auto& neck2 = cycle.tracks[4];
    add_key(neck2, 0.00f, yaw( 2.0f));
    add_key(neck2, 0.35f, yaw(-4.0f));
    add_key(neck2, 0.70f, yaw( 5.0f));

    auto& head = cycle.tracks[5];
    add_key(head, 0.00f, roll( 2.0f));
    add_key(head, 0.30f, roll(-4.0f));
    add_key(head, 0.60f, roll( 3.0f));
    add_key(head, 0.90f, roll(-2.0f));

    // Tail hangs down
    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch(10.0f));
    add_key(tail, 0.50f, pitch(12.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Soar — slow gliding with occasional lazy flaps
// ---------------------------------------------------------------------------
WalkCycle make_bird_soar(const BirdParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.flap_period * 3.0f;
    cycle.hip_swing_deg   = 20.0f;
    cycle.stance_fraction = 0.5f;
    cycle.tracks.resize(N_JOINTS);

    float flap = p.flap_amplitude * 0.4f;
    float sweep = flap * p.flap_sweep * 0.5f;

    // Wings mostly spread, gentle flaps with sweep for circular motion
    auto& wl = cycle.tracks[6];
    add_key(wl, 0.00f, roll(  flap * 0.2f) * yaw(-sweep * 0.3f));
    add_key(wl, 0.15f, roll( -flap * 0.3f) * yaw( sweep * 0.4f));
    add_key(wl, 0.30f, roll(  flap * 0.5f) * yaw(-sweep * 0.5f));
    add_key(wl, 0.45f, roll(  flap * 0.3f) * yaw( 0.0f));
    add_key(wl, 0.70f, roll(  flap * 0.1f) * yaw( sweep * 0.2f));
    add_key(wl, 0.85f, roll(  flap * 0.4f) * yaw(-sweep * 0.4f));

    auto& wr = cycle.tracks[10];
    add_key(wr, 0.00f, roll( -flap * 0.2f) * yaw( sweep * 0.3f));
    add_key(wr, 0.15f, roll(  flap * 0.3f) * yaw(-sweep * 0.4f));
    add_key(wr, 0.30f, roll( -flap * 0.5f) * yaw( sweep * 0.5f));
    add_key(wr, 0.45f, roll( -flap * 0.3f) * yaw( 0.0f));
    add_key(wr, 0.70f, roll( -flap * 0.1f) * yaw(-sweep * 0.2f));
    add_key(wr, 0.85f, roll( -flap * 0.4f) * yaw( sweep * 0.4f));

    // Wingtip dihedral — tips tilt up slightly while soaring
    auto& wrl = cycle.tracks[8];
    auto& wrr = cycle.tracks[12];
    add_key(wrl, 0.00f, roll( 5.0f));
    add_key(wrl, 0.50f, roll( 8.0f));
    add_key(wrr, 0.00f, roll(-5.0f));
    add_key(wrr, 0.50f, roll(-8.0f));

    // Legs tucked
    auto& hl = cycle.tracks[14];
    auto& hr = cycle.tracks[18];
    add_key(hl, 0.0f, pitch(35.0f));
    add_key(hr, 0.0f, pitch(35.0f));
    auto& ul = cycle.tracks[15];
    auto& ur = cycle.tracks[19];
    add_key(ul, 0.0f, pitch(-55.0f));
    add_key(ur, 0.0f, pitch(-55.0f));

    // Head scanning — looking for prey
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch( 8.0f) * yaw( 10.0f));
    add_key(neck1, 0.25f, pitch(10.0f) * yaw(-15.0f));
    add_key(neck1, 0.50f, pitch( 6.0f) * yaw(  5.0f));
    add_key(neck1, 0.75f, pitch(12.0f) * yaw(-10.0f));

    auto& neck2 = cycle.tracks[4];
    add_key(neck2, 0.00f, yaw( 5.0f));
    add_key(neck2, 0.30f, yaw(-8.0f));
    add_key(neck2, 0.60f, yaw( 6.0f));
    add_key(neck2, 0.90f, yaw(-3.0f));

    // Subtle body bank into turns
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, roll(-3.0f));
    add_key(spine1, 0.25f, roll( 2.0f));
    add_key(spine1, 0.50f, roll(-2.0f));
    add_key(spine1, 0.75f, roll( 3.0f));

    // Tail steers — slight pitch adjustments
    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch( 3.0f));
    add_key(tail, 0.25f, pitch( 6.0f));
    add_key(tail, 0.50f, pitch( 4.0f));
    add_key(tail, 0.75f, pitch( 5.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Dive — wings tucked, steep descent, talons forward
// ---------------------------------------------------------------------------
WalkCycle make_bird_dive(const BirdParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 1.0f;
    cycle.hip_swing_deg   = 15.0f;
    cycle.stance_fraction = 0.5f;
    cycle.tracks.resize(N_JOINTS);

    // Wings pulled tight against body
    auto& wl = cycle.tracks[6];
    add_key(wl, 0.00f, roll(-5.0f));
    add_key(wl, 0.50f, roll(-8.0f));
    add_key(wl, 0.85f, roll( 15.0f));

    auto& wr = cycle.tracks[10];
    add_key(wr, 0.00f, roll( 5.0f));
    add_key(wr, 0.50f, roll( 8.0f));
    add_key(wr, 0.85f, roll(-15.0f));

    // Elbows folded tight
    auto& el = cycle.tracks[7];
    add_key(el, 0.00f, roll(-25.0f));
    add_key(el, 0.80f, roll(-25.0f));
    add_key(el, 0.90f, roll( 0.0f));

    auto& er = cycle.tracks[11];
    add_key(er, 0.00f, roll( 25.0f));
    add_key(er, 0.80f, roll( 25.0f));
    add_key(er, 0.90f, roll( 0.0f));

    // Body pitched steeply down
    auto& spine1 = cycle.tracks[1];
    add_key(spine1, 0.00f, pitch(-15.0f));
    add_key(spine1, 0.40f, pitch(-20.0f));
    add_key(spine1, 0.80f, pitch(-10.0f));
    add_key(spine1, 0.95f, pitch(  0.0f));

    auto& chest = cycle.tracks[2];
    add_key(chest, 0.00f, pitch(-10.0f));
    add_key(chest, 0.40f, pitch(-15.0f));
    add_key(chest, 0.80f, pitch( -5.0f));
    add_key(chest, 0.95f, pitch(  0.0f));

    // Legs extend forward — talons reaching for prey
    auto& hl = cycle.tracks[14];
    auto& hr = cycle.tracks[18];
    add_key(hl, 0.00f, pitch( 30.0f));
    add_key(hl, 0.50f, pitch( 15.0f));
    add_key(hl, 0.75f, pitch(-10.0f));
    add_key(hl, 0.90f, pitch(-20.0f));
    add_key(hr, 0.00f, pitch( 30.0f));
    add_key(hr, 0.50f, pitch( 15.0f));
    add_key(hr, 0.75f, pitch(-10.0f));
    add_key(hr, 0.90f, pitch(-20.0f));

    auto& ul = cycle.tracks[15];
    auto& ur = cycle.tracks[19];
    add_key(ul, 0.00f, pitch(-45.0f));
    add_key(ul, 0.50f, pitch(-30.0f));
    add_key(ul, 0.75f, pitch(-10.0f));
    add_key(ul, 0.90f, pitch( 10.0f));
    add_key(ur, 0.00f, pitch(-45.0f));
    add_key(ur, 0.50f, pitch(-30.0f));
    add_key(ur, 0.75f, pitch(-10.0f));
    add_key(ur, 0.90f, pitch( 10.0f));

    // Head locked forward and down — focused on target
    auto& neck1 = cycle.tracks[3];
    add_key(neck1, 0.00f, pitch(15.0f));
    add_key(neck1, 0.40f, pitch(20.0f));
    add_key(neck1, 0.80f, pitch(10.0f));
    add_key(neck1, 0.95f, pitch( 0.0f));

    // Tail tucked for streamlining, fans at pull-up
    auto& tail = cycle.tracks[22];
    add_key(tail, 0.00f, pitch( 2.0f));
    add_key(tail, 0.50f, pitch( 0.0f));
    add_key(tail, 0.80f, pitch(-10.0f));
    add_key(tail, 0.95f, pitch(  0.0f));

    return cycle;
}

} // namespace bestiary

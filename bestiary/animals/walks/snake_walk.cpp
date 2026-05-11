#include "animals/snake.h"
#include "skeleton/animation.h"
#include <glm/gtc/quaternion.hpp>
#include <cmath>

namespace bestiary {

static glm::quat pitch(float degrees)
{
    return glm::angleAxis(glm::radians(degrees), glm::vec3(1, 0, 0));
}

static glm::quat yaw(float degrees)
{
    return glm::angleAxis(glm::radians(degrees), glm::vec3(0, 1, 0));
}

static void add_key(std::vector<JointKeyframe>& track, float phase, glm::quat rot)
{
    track.push_back({phase, rot});
}

static constexpr int N_JOINTS  = 15;
static constexpr int N_KEYS    = 8;
static constexpr float PI2     = 6.28318530f;

// Body position (0=head, 1=tail) for each joint.
// Spatial order: head(5)=0, fwd(4..1)=1..4, anchor(0)=5, back(6..13)=6..13, tail(14)=14
// Normalized: joint_body_pos[j] = spatial_index / 14
static float joint_body_pos(int j)
{
    int si;
    if (j == 0)       si = 5;           // anchor
    else if (j <= 5)  si = 5 - j;       // fwd chain: j=1→si=4, j=5→si=0 (head)
    else              si = j;            // back chain: j=6→si=6, j=14→si=14
    return static_cast<float>(si) / 14.0f;
}

// Forward chain joints get normal yaw; backward chain joints get inverted yaw
// (because the chain extends in -Z, so yaw has opposite lateral effect)
static float joint_yaw_sign(int j)
{
    if (j == 0)      return 0.0f;  // anchor: no direct undulation
    if (j <= 5)      return 1.0f;  // forward chain
    return -1.0f;                  // backward chain
}

static void apply_sine_wave(WalkCycle& cycle, float amp, float waves)
{
    for (int j = 0; j < N_JOINTS; ++j) {
        float sign = joint_yaw_sign(j);
        if (sign == 0.0f) continue;

        float bpos = joint_body_pos(j);
        auto& track = cycle.tracks[j];

        float extra = 1.0f;
        if (j == 5)  extra = 0.4f;  // head: reduced amplitude
        if (j == 14) extra = 1.4f;  // tail tip: extra whip

        for (int k = 0; k < N_KEYS; ++k) {
            float ph = static_cast<float>(k) / static_cast<float>(N_KEYS);
            float angle = amp * extra * sign * std::sin(PI2 * (ph - bpos * waves));
            add_key(track, ph, yaw(angle));
        }
    }
}

// ---------------------------------------------------------------------------
// Slither — lateral undulation
// ---------------------------------------------------------------------------
WalkCycle make_snake_slither(const SnakeParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.slither_period;
    cycle.hip_swing_deg   = 25.0f;
    cycle.stance_fraction = 0.5f;
    cycle.tracks.resize(N_JOINTS);

    apply_sine_wave(cycle, p.slither_amplitude, p.slither_waves);

    return cycle;
}

// ---------------------------------------------------------------------------
// Fast — quicker, tighter undulations
// ---------------------------------------------------------------------------
WalkCycle make_snake_fast(const SnakeParams& p)
{
    WalkCycle cycle;
    cycle.period_seconds  = p.slither_period * 0.5f;
    cycle.hip_swing_deg   = 30.0f;
    cycle.stance_fraction = 0.45f;
    cycle.tracks.resize(N_JOINTS);

    apply_sine_wave(cycle, p.slither_amplitude * 1.3f, p.slither_waves + 0.5f);

    return cycle;
}

// ---------------------------------------------------------------------------
// Idle — loose S-curve, head scanning, tongue flick
// ---------------------------------------------------------------------------
WalkCycle make_snake_idle(const SnakeParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 5.0f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.tracks.resize(N_JOINTS);

    // Resting S-curve — each segment holds a gentle static bend
    for (int j = 1; j <= 5; ++j) {
        float bpos = joint_body_pos(j);
        float base = 15.0f * std::sin(PI2 * bpos * 1.2f);
        auto& track = cycle.tracks[j];
        add_key(track, 0.00f, yaw(base));
        add_key(track, 0.50f, yaw(base * 0.9f));
    }
    for (int j = 6; j <= 14; ++j) {
        float bpos = joint_body_pos(j);
        float base = -15.0f * std::sin(PI2 * bpos * 1.2f);
        auto& track = cycle.tracks[j];
        add_key(track, 0.00f, yaw(base));
        add_key(track, 0.50f, yaw(base * 0.9f));
    }

    // Head (joint 5) — slow scanning
    auto& head = cycle.tracks[5];
    head.clear();
    add_key(head, 0.00f, yaw(  8.0f) * pitch(-3.0f));
    add_key(head, 0.15f, yaw( -5.0f) * pitch(-1.0f));
    add_key(head, 0.30f, yaw(-12.0f) * pitch(-4.0f));
    add_key(head, 0.36f, yaw(-12.0f) * pitch( 5.0f));
    add_key(head, 0.39f, yaw(-12.0f) * pitch(-4.0f));
    add_key(head, 0.50f, yaw(  5.0f) * pitch(-2.0f));
    add_key(head, 0.70f, yaw( 15.0f) * pitch(-3.0f));
    add_key(head, 0.85f, yaw(  0.0f) * pitch(-1.0f));

    return cycle;
}

// ---------------------------------------------------------------------------
// Strike — S-coil, explosive lunge, retract
// ---------------------------------------------------------------------------
WalkCycle make_snake_strike(const SnakeParams& /*p*/)
{
    WalkCycle cycle;
    cycle.period_seconds  = 0.8f;
    cycle.hip_swing_deg   = 0.0f;
    cycle.stance_fraction = 1.0f;
    cycle.tracks.resize(N_JOINTS);

    // Head (5) — raise, coil, STRIKE, hold, retract
    auto& head = cycle.tracks[5];
    add_key(head, 0.00f, pitch(-15.0f));
    add_key(head, 0.30f, pitch(-25.0f));
    add_key(head, 0.42f, pitch( 15.0f));
    add_key(head, 0.55f, pitch( 10.0f));
    add_key(head, 0.80f, pitch(  5.0f));
    add_key(head, 0.95f, pitch(-10.0f));

    // Neck (4) — curves up for the S-coil
    auto& neck = cycle.tracks[4];
    add_key(neck, 0.00f, pitch( 25.0f) * yaw( 5.0f));
    add_key(neck, 0.30f, pitch( 35.0f) * yaw( 8.0f));
    add_key(neck, 0.42f, pitch(-10.0f) * yaw( 0.0f));
    add_key(neck, 0.55f, pitch( -5.0f));
    add_key(neck, 0.95f, pitch( 18.0f) * yaw( 3.0f));

    // fwd_1 (3) — counter-curve of the S
    auto& f1 = cycle.tracks[3];
    add_key(f1, 0.00f, pitch(-12.0f) * yaw(-5.0f));
    add_key(f1, 0.30f, pitch(-18.0f) * yaw(-8.0f));
    add_key(f1, 0.42f, pitch(  5.0f) * yaw( 0.0f));
    add_key(f1, 0.55f, pitch(  0.0f));
    add_key(f1, 0.95f, pitch(-10.0f) * yaw(-3.0f));

    // fwd_2 (2) — anchor transition
    auto& f2 = cycle.tracks[2];
    add_key(f2, 0.00f, pitch( 5.0f) * yaw( 3.0f));
    add_key(f2, 0.30f, pitch( 8.0f) * yaw( 4.0f));
    add_key(f2, 0.42f, pitch( 0.0f));
    add_key(f2, 0.95f, pitch( 3.0f) * yaw( 2.0f));

    // Back body — tension ripple on strike
    for (int j = 6; j <= 10; ++j) {
        auto& track = cycle.tracks[j];
        float lag = static_cast<float>(j - 6) * 0.03f;
        add_key(track, 0.00f, yaw(-3.0f));
        add_key(track, std::min(0.42f + lag, 0.55f), yaw( 2.0f));
        add_key(track, 0.95f, yaw(-2.0f));
    }

    return cycle;
}

} // namespace bestiary

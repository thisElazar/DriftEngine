// pilot_main.cpp — scripted-flight harness: "play the game" without a human.
//
// Runs the REAL globe module (same globe_tick/globe_render as drift_engine)
// but synthesizes the per-frame InputFrame from a flight plan instead of
// polling the mouse/keyboard, so performance and behavior can be measured
// reproducibly. The window still opens (Vulkan swapchain needs a surface);
// nobody has to touch it. Perf telemetry comes from the [perf] stderr lines
// that globe_tick emits (5 s summaries + per-spike attribution), plus the
// pilot's own per-phase stats and an end-of-run verdict block.
//
// Flight plan (phases print as they start):
//   spawn    4 s   baseline at the spawn arm (near surface)
//   climb  <=8 s   scroll out until the orbit arm passes 3000 km
//   idle     4 s   baseline at far orbit
//   orbit    8 s   steady yaw sweep (tile streaming at constant LOD)
//   dive  <=14 s   zoom until the orbit arm is under ~60 km (deep LOD churn)
//   cruise  10 s   WASD movement low over the surface (max tile gen pressure)
//   brush    6 s   water brush held at screen center (SWE ring + one pulse)
//   settle  42 s   hands off — quiesce, bake-backs, steady state
//
// Exit code = number of failed verdicts (0 = all green), so CI and I can gate
// on it. Synthetic mouse deltas are dt-scaled (px/s, not px/frame) so the
// flight is the same at any frame rate.
//
// Usage: drift_pilot [--fp]   (--fp: toggle into first-person for the cruise)

#include "globe/globe.h"

#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

enum Phase { SPAWN, CLIMB, IDLE, ORBIT, DIVE, CRUISE, BRUSH, SETTLE, DONE };
const char* PHASE_NAMES[] = {"spawn", "climb", "idle", "orbit", "dive",
                             "cruise", "brush", "settle", "done"};

constexpr double CLIMB_TARGET_ARM = 3000000.0;  // m — far-orbit baseline
constexpr double DIVE_TARGET_ARM  = 60000.0;    // m — deep-LOD altitude
constexpr float  SPIKE_DT         = 0.025f;     // frame counts as a spike above this

// Per-phase stats the pilot accumulates itself (independent of the engine's
// 5 s perf windows, which straddle phase boundaries).
struct PhaseStats {
    int    frames = 0;
    int    spikes = 0;
    double dt_sum = 0.0;
    float  dt_max = 0.0f;
    // Worst single-frame cost of each pipeline section (ms) — attributes
    // stalls that the engine's own spike lines can't see (fence waits,
    // swapchain acquire, present).
    double worst_tick = 0.0, worst_begin = 0.0, worst_render = 0.0, worst_end = 0.0;
};

struct Verdict { const char* name; bool pass; char detail[160]; };

}  // namespace

int main(int argc, char** argv)
{
    bool use_fp = (argc > 1 && std::strcmp(argv[1], "--fp") == 0);

    Renderer r{};
    renderer_init(r, 1280, 720, "Drift Pilot — scripted flight");

    GlobeState state{};
    globe_init(state, r);

    std::fprintf(stderr, "[pilot] takeoff (fp=%d)\n", use_fp ? 1 : 0);

    double start = glfwGetTime();
    double last  = start;
    double phase_t0 = 0.0;       // time the current phase began
    int    phase = SPAWN;
    bool   fp_toggled   = false;
    bool   pulse_fired  = false; // latch: the brush space-pulse fires exactly once
    bool   pool_exhausted = false;

    PhaseStats stats[DONE + 1]{};
    size_t tiles_prebrush = 0;
    double spike_dt = SPIKE_DT;   // re-based to 3x the spawn average once known
    int    total_bakebacks = 0;   // accumulated across engine perf windows
    int    prev_window_bakebacks = 0;

    while (!glfwWindowShouldClose(r.window)) {
        double now = glfwGetTime();
        float  dt  = static_cast<float>(now - last);
        last = now;
        double t  = now - start;
        double pt = t - phase_t0;  // time within the current phase

        glfwPollEvents();

        InputFrame in{};
        glfwGetWindowSize(r.window, &in.win_w, &in.win_h);
        glfwGetFramebufferSize(r.window, &in.fb_w, &in.fb_h);
        in.mouse_x = in.win_w * 0.5;
        in.mouse_y = in.win_h * 0.5;

        // Pilot-side stats for the current phase (dt describes the previous
        // frame, which ran in this phase for all but the boundary frame —
        // close enough for phase-level aggregates).
        {
            PhaseStats& ps = stats[phase];
            ps.frames++;
            ps.dt_sum += dt;
            ps.dt_max = std::max(ps.dt_max, dt);
            if (dt > spike_dt) ps.spikes++;
        }
        if (state.free_slots.empty() && !state.tile_slot_map.empty())
            pool_exhausted = true;
        // perf_bakebacks is a per-5s-window counter that resets; integrate the
        // positive deltas to get a flight total.
        if (state.perf_bakebacks > prev_window_bakebacks)
            total_bakebacks += state.perf_bakebacks - prev_window_bakebacks;
        prev_window_bakebacks = state.perf_bakebacks;

        auto advance = [&](int next) {
            const PhaseStats& ps = stats[phase];
            std::fprintf(stderr,
                "[pilot] %-6s done: %5.1f fps, dt max %5.1f ms, spikes %d/%d | "
                "tiles %zu, disturbed %zu, free %zu | worst ms: tick %.1f "
                "begin %.1f render %.1f end %.1f\n",
                PHASE_NAMES[phase],
                ps.frames / std::max(ps.dt_sum, 1e-6),
                ps.dt_max * 1000.0, ps.spikes, ps.frames,
                state.tile_slot_map.size(), state.disturbed_tiles.size(),
                state.free_slots.size(),
                ps.worst_tick, ps.worst_begin, ps.worst_render, ps.worst_end);
            if (phase == SPAWN) {
                // Re-base the spike threshold to this machine's demonstrated
                // baseline so a thermally-throttled run judges itself fairly.
                double avg = ps.dt_sum / std::max(ps.frames, 1);
                spike_dt = std::max(static_cast<double>(SPIKE_DT), 3.0 * avg);
                std::fprintf(stderr, "[pilot] spike threshold: %.1f ms (3x spawn avg)\n",
                             spike_dt * 1000.0);
            }
            phase = next;
            phase_t0 = t;
            std::fprintf(stderr, "[pilot] t=%5.1fs phase -> %s (arm %.0f km)\n",
                         t, PHASE_NAMES[next], state.camera.orbit.arm_length / 1000.0);
        };

        switch (phase) {
        case SPAWN:  // baseline at the spawn arm
            if (pt >= 4.0) advance(CLIMB);
            break;
        case CLIMB:  // closed-loop zoom OUT to far orbit
            if (state.camera.orbit.arm_length < CLIMB_TARGET_ARM)
                in.scroll = -0.5f;
            if (state.camera.orbit.arm_length >= CLIMB_TARGET_ARM || pt >= 8.0)
                advance(IDLE);
            break;
        case IDLE:   // far-orbit baseline
            if (pt >= 4.0) advance(ORBIT);
            break;
        case ORBIT:  // steady yaw sweep
            in.rmb = true;
            in.mouse_dx = 720.0f * dt;                                    // ~6 px/frame @120fps
            in.mouse_dy = static_cast<float>(std::sin(pt * 0.7)) * 180.0f * dt;
            if (pt >= 8.0) advance(DIVE);
            break;
        case DIVE:   // closed-loop zoom to deep LOD
            if (state.camera.orbit.arm_length > DIVE_TARGET_ARM)
                in.scroll = 0.35f;
            in.rmb = true;
            in.mouse_dx = 180.0f * dt;
            if (state.camera.orbit.arm_length <= DIVE_TARGET_ARM || pt >= 14.0)
                advance(CRUISE);
            break;
        case CRUISE: // movement keys low over the surface
            if (use_fp && !fp_toggled) {
                in.key_f_pressed = true;   // toggle into first-person once
                fp_toggled = true;
            }
            in.key_w = true;
            in.key_shift = (pt > 4.0);                    // speed up halfway in
            in.key_a = (std::fmod(pt, 4.0) < 2.0);        // gentle weave
            in.key_d = !in.key_a;
            in.rmb = true;
            in.mouse_dy = static_cast<float>(std::sin(pt * 0.5)) * 120.0f * dt;
            if (pt >= 10.0) {
                if (use_fp && fp_toggled) in.key_f_pressed = true;  // back to orbital
                tiles_prebrush = state.tile_slot_map.size();
                advance(BRUSH);
            }
            break;
        case BRUSH:  // water brush at screen center
            in.brush_digit = 3;            // water mode
            in.lmb = true;
            if (pt > 2.0 && !pulse_fired) {
                in.key_space_pressed = true;   // exactly one flood pulse
                pulse_fired = true;
            }
            if (pt >= 6.0) advance(SETTLE);
            break;
        case SETTLE: // hands off, let quiesce + bake-back run. Must outlast
                     // GLOBE_SWE_DISTURB_MAX_S so the flow ceiling provably
                     // returns every tile to the field.
            if (pt >= 42.0) advance(DONE);
            break;
        case DONE:
            glfwSetWindowShouldClose(r.window, GLFW_TRUE);
            break;
        }

        double s0 = glfwGetTime();
        if (!globe_tick(state, r, in, dt))
            break;
        double s1 = glfwGetTime();

        FrameData* frame = nullptr;
        uint32_t   image_index = 0;
        VkExtent2D extent{};
        double s2 = s1, s3 = s1, s4 = s1;
        if (renderer_begin_frame(r, frame, image_index, extent)) {
            s2 = glfwGetTime();
            globe_render(state, r, *frame, image_index, extent);
            s3 = glfwGetTime();
            renderer_end_frame(r, *frame, image_index);
            s4 = glfwGetTime();
        }
        {
            PhaseStats& ps = stats[phase];
            ps.worst_tick   = std::max(ps.worst_tick,   (s1 - s0) * 1000.0);
            ps.worst_begin  = std::max(ps.worst_begin,  (s2 - s1) * 1000.0);
            ps.worst_render = std::max(ps.worst_render, (s3 - s2) * 1000.0);
            ps.worst_end    = std::max(ps.worst_end,    (s4 - s3) * 1000.0);
            // Attribute monster stalls immediately — the engine's spike line
            // only sees what tick instrumented, not fence/acquire/present.
            if (s4 - s0 > 0.25)
                std::fprintf(stderr,
                    "[pilot] STALL %.0f ms at t=%.1fs (%s): tick %.0f, begin %.0f, "
                    "render %.0f, end %.0f\n",
                    (s4 - s0) * 1000.0, t, PHASE_NAMES[phase],
                    (s1 - s0) * 1000.0, (s2 - s1) * 1000.0,
                    (s3 - s2) * 1000.0, (s4 - s3) * 1000.0);
        }
    }

    std::fprintf(stderr, "[pilot] landed after %.1f s\n", glfwGetTime() - start);

    // ---- Verdicts: the playtest judges itself --------------------------------
    std::vector<Verdict> verdicts;
    auto check = [&](const char* name, bool pass, const char* fmt, auto... args) {
        Verdict v{name, pass, {}};
        std::snprintf(v.detail, sizeof(v.detail), fmt, args...);
        verdicts.push_back(v);
    };

    const PhaseStats& spawn  = stats[SPAWN];
    const PhaseStats& settle = stats[SETTLE];
    double spawn_fps  = spawn.frames  / std::max(spawn.dt_sum, 1e-6);
    double settle_fps = settle.frames / std::max(settle.dt_sum, 1e-6);
    double settle_spike_frac =
        settle.frames ? double(settle.spikes) / settle.frames : 1.0;

    check("settle-quiesces", settle_spike_frac < 0.02,
          "spike frames in settle: %d/%d (%.1f%%, want <2%%)",
          settle.spikes, settle.frames, settle_spike_frac * 100.0);
    // The field flood keeps routing downstream well past the flight (that is
    // the live-water design), so full drainage is not assertable here. Assert
    // the recovery MECHANISM instead: bake-backs actually fire, and the
    // disturbed/resident sets stay bounded instead of cascading.
    check("bakebacks-fire", total_bakebacks > 0,
          "bake-backs during flight: %d (want >0); disturbed at end: %zu",
          total_bakebacks, state.disturbed_tiles.size());
    check("disturbed-bounded", state.disturbed_tiles.size() < 128,
          "disturbed tiles at end: %zu (want <128; cascade era was 500+)",
          state.disturbed_tiles.size());
    check("tiles-bounded", state.tile_slot_map.size() <= tiles_prebrush + 128,
          "resident at end: %zu vs pre-brush %zu (want <=%zu)",
          state.tile_slot_map.size(), tiles_prebrush, tiles_prebrush + 128);
    check("pool-never-exhausted", !pool_exhausted,
          "free slots hit zero during flight: %s", pool_exhausted ? "yes" : "no");
    check("settle-fps-holds", settle_fps >= 0.8 * spawn_fps,
          "settle %.1f fps vs spawn %.1f fps (want >=80%%)", settle_fps, spawn_fps);
    {
        double arm = state.camera.orbit.arm_length;
        check("camera-arm-sane", std::isfinite(arm) && arm >= 10.0,
              "final orbital arm: %.1f m (want finite, >=10)", arm);
    }

    int failures = 0;
    for (const Verdict& v : verdicts) {
        if (!v.pass) failures++;
        std::fprintf(stderr, "[pilot] %s %-20s %s\n",
                     v.pass ? "PASS" : "FAIL", v.name, v.detail);
    }
    std::fprintf(stderr, "[pilot] verdict: %d/%zu passed\n",
                 static_cast<int>(verdicts.size()) - failures, verdicts.size());

    vkDeviceWaitIdle(r.device);
    globe_shutdown(state, r);
    renderer_shutdown(r);
    return failures;
}

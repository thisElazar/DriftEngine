// input_frame.h — one immutable snapshot of user input for a single frame.
//
// Stage 2 of the launcher convergence: instead of each lab reading input its
// own way (Globe via GLFW callbacks, the orbit labs via direct glfwGet* polling
// + a global scroll accumulator), the launcher polls once per frame into an
// InputFrame and passes it (const&) to every xxx_tick. No lab installs GLFW
// input callbacks; the per-frame poll is the single source.
//
// This header is intentionally POD with no GLFW/ImGui dependency so it can live
// in drift_engine_core and be included by every lab and by camera.h. The
// poll_input_frame() helper that fills it lives in input_poll.{h,cpp} (it needs
// GLFW + ImGui), keeping this struct free of those dependencies.
//
// Edge fields (*_pressed) are "went down this frame" — derived by the poll by
// diffing the matching held field against the previous frame's value, so the
// caller MUST feed last frame's InputFrame back in as `prev`.
//
// See docs/INPUT_UNIFICATION.md for the migration plan.
#pragma once

struct InputFrame {
    // --- Pointer (window coords, matching glfwGetCursorPos) ---
    double mouse_x = 0.0, mouse_y = 0.0;
    float  mouse_dx = 0.0f, mouse_dy = 0.0f;  // delta since previous frame
    float  scroll   = 0.0f;                    // accumulated wheel this frame

    // --- Buttons: held state + this-frame press edge ---
    bool lmb = false, rmb = false, mmb = false;
    bool lmb_pressed = false, rmb_pressed = false;

    // --- Movement / modifiers (held) — globe camera WASD/QE + speed mods ---
    bool key_w = false, key_a = false, key_s = false, key_d = false;
    bool key_q = false, key_e = false;
    bool key_shift = false, key_alt = false;

    // --- Other held keys (consumed directly and/or used for edge detection) ---
    bool key_r     = false;   // rain (world lab) — held
    bool key_space = false, key_f = false, key_c = false, key_f5 = false;
    bool key_grave = false;   // ` — toggle in-lab menu
    bool key_lbracket = false, key_rbracket = false;  // [ ] brush radius
    bool key_minus = false, key_equal = false;        // - = brush strength
    bool key_esc   = false;

    // --- Edge-triggered (went down this frame) ---
    bool key_space_pressed = false;  // pulse (globe)
    bool key_f_pressed     = false;  // toggle camera mode (globe)
    bool key_c_pressed     = false;  // fly-to-cursor (globe)
    bool key_f5_pressed    = false;  // hot-reload shaders
    bool key_grave_pressed = false;  // toggle menu
    bool key_lbracket_pressed = false, key_rbracket_pressed = false;
    bool key_minus_pressed = false, key_equal_pressed = false;
    bool esc_pressed       = false;  // request back-to-menu / quit
    int  brush_digit = 0;            // 1..4 held (globe brush mode), 0 = none

    // --- ImGui capture flags — labs must gate world interaction on these ---
    bool ui_wants_mouse    = false;
    bool ui_wants_keyboard = false;

    // --- Sizes (saves every lab re-querying GLFW) ---
    int win_w = 0, win_h = 0;  // window coords (for cursor→NDC)
    int fb_w  = 0, fb_h  = 0;  // framebuffer pixels
};

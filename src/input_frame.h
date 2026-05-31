// input_frame.h — one immutable snapshot of user input for a single frame.
//
// Stage 2 of the launcher convergence: instead of each lab reading input its
// own way (Globe via GLFW callbacks, the orbit labs via direct glfwGet* polling
// + the global g_scroll_accum), the launcher polls once per frame into an
// InputFrame and passes it (const&) to every xxx_tick. No lab touches GLFW
// input directly; cursor capture becomes a request the launcher honors.
//
// This header is intentionally POD with no GLFW/ImGui dependency so it can live
// in drift_engine_core and be included by every lab. The poll_input() helper
// that fills it lives in the launcher (it needs GLFW + ImGui + the scroll
// accumulator), keeping core free of an apps/ dependency.
//
// See docs/INPUT_UNIFICATION.md for the migration plan.
#pragma once

struct InputFrame {
    // --- Pointer (window/screen coords, matching glfwGetCursorPos) ---
    double mouse_x = 0.0, mouse_y = 0.0;
    float  mouse_dx = 0.0f, mouse_dy = 0.0f;  // delta since previous frame
    float  scroll   = 0.0f;                    // accumulated wheel this frame

    // --- Buttons: held state + this-frame press edge ---
    bool lmb = false, rmb = false, mmb = false;
    bool lmb_pressed = false, rmb_pressed = false;

    // --- Keys the labs actually consume (extend as needed) ---
    bool key_shift = false;
    bool key_r     = false;   // rain (world lab)
    bool key_space = false;   // pulse (globe)
    bool key_f     = false;   // toggle camera mode (globe)
    bool key_c     = false;   // fly-to-cursor (globe)
    bool key_f5    = false;   // hot-reload shaders
    bool esc_pressed = false; // request back-to-menu
    int  brush_digit = 0;     // 1..4 pressed this frame (globe brush mode), 0 = none

    // --- ImGui capture flags — labs must gate world interaction on these ---
    bool ui_wants_mouse    = false;
    bool ui_wants_keyboard = false;

    // --- Sizes (saves every lab re-querying GLFW) ---
    int win_w = 0, win_h = 0;  // window coords (for cursor→NDC)
    int fb_w  = 0, fb_h  = 0;  // framebuffer pixels

    // Labs set this to request cursor capture; the launcher applies the actual
    // glfwSetInputMode after tick. (Globe's RMB-look capture.)
    // NOTE: this is an out-field — labs write it, launcher reads it.
    bool want_cursor_capture = false;
};

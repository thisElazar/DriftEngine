#pragma once

#include <cstdint>

struct Camera;
struct UIState;
struct GLFWwindow;

enum class BrushMode : uint32_t { Raise, Lower, Water, Sand };

struct InputState {
    bool   lmb_held      = false;
    bool   rmb_held       = false;
    bool   first_mouse    = true;
    double cursor_x       = 0.0;
    double cursor_y       = 0.0;
    double last_cursor_x  = 0.0;
    double last_cursor_y  = 0.0;

    bool framebuffer_resized = false;
    bool reload_shaders      = false;
    bool pulse_pending       = false;
    bool toggle_camera_mode  = false;
    bool warp_to_cursor      = false;

    BrushMode brush_mode = BrushMode::Water;
};

struct CallbackContext {
    InputState* input;
    Camera*     camera;
    UIState*    ui;
};

void input_install_callbacks(GLFWwindow* window, CallbackContext* ctx);

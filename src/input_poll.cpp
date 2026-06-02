#include "input_poll.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

// Wheel deltas only come through a callback; accumulate here and drain per frame.
static float g_scroll_accum = 0.0f;

// Last GLFW_CURSOR mode we observed, so we can detect a capture transition and
// drop the (garbage) mouse delta on the frame the cursor mode flips. -1 = unset.
static int g_prev_cursor_mode = -1;

static void scroll_cb(GLFWwindow*, double /*xoffset*/, double yoffset)
{
    g_scroll_accum += static_cast<float>(yoffset);
}

void input_install_scroll_callback(GLFWwindow* window)
{
    glfwSetScrollCallback(window, scroll_cb);
}

InputFrame poll_input_frame(GLFWwindow* window, const InputFrame& prev)
{
    InputFrame in;

    glfwGetCursorPos(window, &in.mouse_x, &in.mouse_y);
    in.mouse_dx = static_cast<float>(in.mouse_x - prev.mouse_x);
    in.mouse_dy = static_cast<float>(in.mouse_y - prev.mouse_y);

    // When the cursor mode changes (e.g. RMB capture switches it to
    // GLFW_CURSOR_DISABLED), GLFW resets the virtual cursor position, so the
    // first reported delta is meaningless. Zero it for that one frame.
    int cursor_mode = glfwGetInputMode(window, GLFW_CURSOR);
    if (cursor_mode != g_prev_cursor_mode) {
        in.mouse_dx = 0.0f;
        in.mouse_dy = 0.0f;
        g_prev_cursor_mode = cursor_mode;
    }

    in.scroll = g_scroll_accum;
    g_scroll_accum = 0.0f;

    in.lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
    in.rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
    in.mmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    in.lmb_pressed = in.lmb && !prev.lmb;
    in.rmb_pressed = in.rmb && !prev.rmb;

    in.key_w     = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
    in.key_a     = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    in.key_s     = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    in.key_d     = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
    in.key_q     = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
    in.key_e     = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;
    in.key_shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS
                || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    in.key_alt   = glfwGetKey(window, GLFW_KEY_LEFT_ALT)  == GLFW_PRESS
                || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;

    in.key_r        = glfwGetKey(window, GLFW_KEY_R)             == GLFW_PRESS;
    in.key_space    = glfwGetKey(window, GLFW_KEY_SPACE)         == GLFW_PRESS;
    in.key_f        = glfwGetKey(window, GLFW_KEY_F)             == GLFW_PRESS;
    in.key_c        = glfwGetKey(window, GLFW_KEY_C)             == GLFW_PRESS;
    in.key_f5       = glfwGetKey(window, GLFW_KEY_F5)            == GLFW_PRESS;
    in.key_grave    = glfwGetKey(window, GLFW_KEY_GRAVE_ACCENT)  == GLFW_PRESS;
    in.key_lbracket = glfwGetKey(window, GLFW_KEY_LEFT_BRACKET)  == GLFW_PRESS;
    in.key_rbracket = glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS;
    in.key_minus    = glfwGetKey(window, GLFW_KEY_MINUS)         == GLFW_PRESS;
    in.key_equal    = glfwGetKey(window, GLFW_KEY_EQUAL)         == GLFW_PRESS;
    in.key_esc      = glfwGetKey(window, GLFW_KEY_ESCAPE)        == GLFW_PRESS;

    // Edge fields: down this frame, up last frame.
    in.key_space_pressed    = in.key_space    && !prev.key_space;
    in.key_f_pressed        = in.key_f        && !prev.key_f;
    in.key_c_pressed        = in.key_c        && !prev.key_c;
    in.key_f5_pressed       = in.key_f5       && !prev.key_f5;
    in.key_grave_pressed    = in.key_grave    && !prev.key_grave;
    in.key_lbracket_pressed = in.key_lbracket && !prev.key_lbracket;
    in.key_rbracket_pressed = in.key_rbracket && !prev.key_rbracket;
    in.key_minus_pressed    = in.key_minus    && !prev.key_minus;
    in.key_equal_pressed    = in.key_equal    && !prev.key_equal;
    in.esc_pressed          = in.key_esc      && !prev.key_esc;

    if      (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) in.brush_digit = 1;
    else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) in.brush_digit = 2;
    else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) in.brush_digit = 3;
    else if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) in.brush_digit = 4;

    if (ImGui::GetCurrentContext()) {
        in.ui_wants_mouse    = ImGui::GetIO().WantCaptureMouse;
        in.ui_wants_keyboard = ImGui::GetIO().WantCaptureKeyboard;
    }

    glfwGetWindowSize(window, &in.win_w, &in.win_h);
    glfwGetFramebufferSize(window, &in.fb_w, &in.fb_h);
    return in;
}

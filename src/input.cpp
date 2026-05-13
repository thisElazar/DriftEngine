#include "input.h"
#include "camera.h"
#include "ui.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>

static void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    auto* ctx = static_cast<CallbackContext*>(glfwGetWindowUserPointer(window));
    if (!ctx) return;
    auto& input = *ctx->input;
    auto& ui = *ctx->ui;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_GRAVE_ACCENT && action == GLFW_PRESS)
        ui.show_menu = !ui.show_menu;
    if (key == GLFW_KEY_F5 && action == GLFW_PRESS)
        input.reload_shaders = true;
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard)
        return;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        input.pulse_pending = true;
    if (key == GLFW_KEY_F && action == GLFW_PRESS)
        input.toggle_camera_mode = true;
    if (key == GLFW_KEY_C && action == GLFW_PRESS)
        input.warp_to_cursor = true;
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_1) input.brush_mode = BrushMode::Raise;
        if (key == GLFW_KEY_2) input.brush_mode = BrushMode::Lower;
        if (key == GLFW_KEY_3) input.brush_mode = BrushMode::Water;
        if (key == GLFW_KEY_4) input.brush_mode = BrushMode::Sand;
    }
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_LEFT_BRACKET)
            ui.brush_radius_grid = std::max(2.0f, ui.brush_radius_grid * 0.85f);
        if (key == GLFW_KEY_RIGHT_BRACKET)
            ui.brush_radius_grid = std::min(300.0f, ui.brush_radius_grid * 1.18f);
        if (key == GLFW_KEY_MINUS) {
            if (input.brush_mode == BrushMode::Water)
                ui.brush_strength = std::max(0.05f, ui.brush_strength * 0.85f);
            else
                ui.terrain_strength = std::max(2.0f, ui.terrain_strength * 0.85f);
        }
        if (key == GLFW_KEY_EQUAL) {
            if (input.brush_mode == BrushMode::Water)
                ui.brush_strength = std::min(20.0f, ui.brush_strength * 1.18f);
            else
                ui.terrain_strength = std::min(500.0f, ui.terrain_strength * 1.18f);
        }
    }
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    auto* ctx = static_cast<CallbackContext*>(glfwGetWindowUserPointer(window));
    if (!ctx) return;
    auto& input = *ctx->input;

    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse)
        return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        input.lmb_held = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        input.rmb_held = (action == GLFW_PRESS);
        // Cursor mode is driven from main loop (per-frame state machine).
        // We just signal first_mouse on RMB press so the look delta starts
        // fresh after the cursor jumps from visible-pos to disabled-virtual.
        if (input.rmb_held) input.first_mouse = true;
    }
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    auto* ctx = static_cast<CallbackContext*>(glfwGetWindowUserPointer(window));
    if (!ctx) return;
    auto& input = *ctx->input;

    input.cursor_x = xpos;
    input.cursor_y = ypos;

    if (input.rmb_held) {
        if (input.first_mouse) {
            input.last_cursor_x = xpos;
            input.last_cursor_y = ypos;
            input.first_mouse = false;
            return;
        }
        float dx = static_cast<float>(xpos - input.last_cursor_x);
        float dy = static_cast<float>(ypos - input.last_cursor_y);
        camera_apply_mouse_look(*ctx->camera, dx, dy);
    }
    input.last_cursor_x = xpos;
    input.last_cursor_y = ypos;
}

static void scroll_callback(GLFWwindow* window, double /*xoffset*/, double yoffset)
{
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse)
        return;
    auto* ctx = static_cast<CallbackContext*>(glfwGetWindowUserPointer(window));
    if (!ctx) return;
    camera_zoom(*ctx->camera, yoffset);
}

static void framebuffer_resize_callback(GLFWwindow* window, int /*width*/, int /*height*/)
{
    auto* ctx = static_cast<CallbackContext*>(glfwGetWindowUserPointer(window));
    if (!ctx) return;
    ctx->input->framebuffer_resized = true;
}

void input_install_callbacks(GLFWwindow* window, CallbackContext* ctx)
{
    glfwSetWindowUserPointer(window, ctx);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
}

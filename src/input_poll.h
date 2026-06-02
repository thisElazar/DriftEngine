// input_poll.h — the single per-frame input poll for the unified input system.
//
// GLFW scroll deltas only arrive via a callback, so we accumulate them into a
// translation-unit-local total that poll_input_frame() drains once per frame.
// Install the scroll callback BEFORE ImGui's callbacks so ImGui chains to it
// (ImGui_ImplGlfw_InstallCallbacks preserves the previously-set callback).
//
// poll_input_frame() reads every other input (cursor, buttons, keys, ImGui
// capture flags, sizes) directly with glfwGet*, derives edge (*_pressed) fields
// by diffing held state against `prev`, and zeroes the mouse delta on any
// GLFW_CURSOR mode change so RMB-capture (cursor→disabled) doesn't produce a
// look jump. This replaces both the launcher's old local poll_input() and the
// per-app g_scroll_accum/lab_scroll_cb that used to live in lab_common.h.
#pragma once

#include "input_frame.h"

struct GLFWwindow;

// Install the wheel accumulator. Call once at startup, before
// ImGui_ImplGlfw_InstallCallbacks so ImGui chains to it.
void input_install_scroll_callback(GLFWwindow* window);

// Poll all input into one immutable frame. Pass last frame's result as `prev`
// (edge fields and mouse delta are computed relative to it).
InputFrame poll_input_frame(GLFWwindow* window, const InputFrame& prev);

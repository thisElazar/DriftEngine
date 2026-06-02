// Drift Engine v0.5.0 — standalone entry point.
//
// Thin wrapper around the globe module. All logic lives in
// globe.cpp so the same code can be embedded in the unified launcher.

#include "globe/globe.h"
#include "input_poll.h"

#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>

int main()
{
    Renderer r{};
    renderer_init(r, 1280, 720, "Drift Engine v0.5.0");

    // Install the wheel accumulator first, then ImGui's GLFW callbacks (which
    // chain to it). All gameplay input is polled into an InputFrame each frame;
    // Globe installs no callbacks of its own. See docs/INPUT_UNIFICATION.md.
    input_install_scroll_callback(r.window);
    ImGui_ImplGlfw_InstallCallbacks(r.window);

    GlobeState state{};
    globe_init(state, r);

    InputFrame input_prev{};
    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(r.window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        glfwPollEvents();

        InputFrame in = poll_input_frame(r.window, input_prev);
        input_prev = in;

        if (in.esc_pressed)
            glfwSetWindowShouldClose(r.window, GLFW_TRUE);

        globe_tick(state, r, in, dt);

        FrameData* frame = nullptr;
        uint32_t   image_index = 0;
        VkExtent2D extent{};
        if (renderer_begin_frame(r, frame, image_index, extent)) {
            globe_render(state, r, *frame, image_index, extent);
            renderer_end_frame(r, *frame, image_index);
        }
    }

    vkDeviceWaitIdle(r.device);
    globe_shutdown(state, r);
    renderer_shutdown(r);
    return 0;
}

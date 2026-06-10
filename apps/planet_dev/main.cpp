// PLANET (DEV) — standalone entry point.
//
// Thin wrapper around the planet_dev module. All logic lives in
// planet_dev.cpp so the same code can be embedded in the unified launcher.

#include "planet_dev.h"
#include "input_poll.h"
#include "shared/lab_common.h"

#include <imgui_impl_glfw.h>

int main()
{
    Renderer r{};
    renderer_init(r, 1280, 800, "Planet (Dev)");

    input_install_scroll_callback(r.window);
    ImGui_ImplGlfw_InstallCallbacks(r.window);

    PlanetDevState state{};
    planet_dev_init(state, r);

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

        planet_dev_tick(state, r, in, dt);

        FrameData* frame = nullptr;
        uint32_t   image_index = 0;
        VkExtent2D extent{};
        if (renderer_begin_frame(r, frame, image_index, extent)) {
            planet_dev_render(state, r, *frame, image_index, extent);
            renderer_end_frame(r, *frame, image_index);
        }
    }

    planet_dev_shutdown(state, r);
    renderer_shutdown(r);
    return 0;
}

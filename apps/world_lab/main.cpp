// World Lab — standalone entry point (v0.0.2)
//
// Thin wrapper around the world_lab module.  All logic lives in
// world_lab.cpp so the same code can be embedded in the unified launcher.

#include "world_lab.h"
#include "shared/lab_common.h"

#include <imgui_impl_glfw.h>

int main()
{
    Renderer r{};
    renderer_init(r, 1280, 800, "World Lab");

    glfwSetScrollCallback(r.window, lab_scroll_cb);
    ImGui_ImplGlfw_InstallCallbacks(r.window);

    WorldLabState state{};
    world_lab_init(state, r);

    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(r.window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        glfwPollEvents();

        if (glfwGetKey(r.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(r.window, GLFW_TRUE);

        world_lab_tick(state, r, dt);

        FrameData* frame = nullptr;
        uint32_t   image_index = 0;
        VkExtent2D extent{};
        if (renderer_begin_frame(r, frame, image_index, extent)) {
            world_lab_render(state, r, *frame, image_index, extent);
            renderer_end_frame(r, *frame, image_index);
        }
    }

    world_lab_shutdown(state, r);
    renderer_shutdown(r);
    return 0;
}

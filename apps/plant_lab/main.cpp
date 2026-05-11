// Plant Lab — standalone entry point (v0.0.5)
//
// Thin wrapper around the plant_lab module.  All logic lives in
// plant_lab.cpp so the same code can be embedded in the unified launcher.

#include "plant_lab.h"
#include "shared/lab_common.h"

#include <imgui_impl_glfw.h>

int main()
{
    Renderer r{};
    renderer_init(r, 1280, 800, "Plant Lab");

    glfwSetScrollCallback(r.window, lab_scroll_cb);
    ImGui_ImplGlfw_InstallCallbacks(r.window);

    PlantLabState state{};
    plant_lab_init(state, r);

    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(r.window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        glfwPollEvents();

        plant_lab_tick(state, r, dt);

        FrameData* frame = nullptr;
        uint32_t   image_index = 0;
        VkExtent2D extent{};
        if (renderer_begin_frame(r, frame, image_index, extent)) {
            plant_lab_render(state, r, *frame, image_index, extent);
            renderer_end_frame(r, *frame, image_index);
        }
    }

    plant_lab_shutdown(state, r);
    renderer_shutdown(r);
    return 0;
}

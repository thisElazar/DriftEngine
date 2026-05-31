// Drift Engine v0.5.0 — standalone entry point.
//
// Thin wrapper around the globe module. All logic lives in
// globe.cpp so the same code can be embedded in the unified launcher.

#include "globe/globe.h"

#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>

int main()
{
    Renderer r{};
    renderer_init(r, 1280, 720, "Drift Engine v0.5.0");

    GlobeState state{};
    globe_init(state, r);

    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(r.window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        glfwPollEvents();

        if (glfwGetKey(r.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(r.window, GLFW_TRUE);

        // Standalone Globe still reads input via its own GLFW callbacks, so it
        // ignores the InputFrame; pass a default one for signature parity.
        globe_tick(state, r, InputFrame{}, dt);

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

// Drift Engine — v0.1.1 bring-up
//
// Opens a GLFW window, creates a Vulkan instance with the extensions GLFW
// requires, enumerates physical devices, then runs an event loop until ESC
// or window close.

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

#define VK_CHECK(expr)                                                                  \
    do {                                                                                 \
        VkResult _r = (expr);                                                            \
        if (_r != VK_SUCCESS) {                                                          \
            std::fprintf(stderr, "Vulkan error %d at %s:%d: %s\n",                       \
                         static_cast<int>(_r), __FILE__, __LINE__, #expr);               \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

const char* device_type_name(VkPhysicalDeviceType t)
{
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "CPU";
        default:                                     return "other";
    }
}

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

} // namespace

int main()
{
    // ---- GLFW init -----------------------------------------------------------
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "drift_engine", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return 1;
    }

    glfwSetKeyCallback(window, key_callback);

    // ---- Gather instance extensions -----------------------------------------
    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);

#ifdef __APPLE__
    extensions.push_back("VK_KHR_portability_enumeration");
#endif

    // ---- Vulkan instance ----------------------------------------------------
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "drift_engine";
    app.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    app.pEngineName        = "drift_engine";
    app.engineVersion      = VK_MAKE_VERSION(0, 0, 1);
    app.apiVersion         = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo        = &app;
    ici.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ici.ppEnabledExtensionNames = extensions.data();

#ifdef __APPLE__
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkInstance instance = VK_NULL_HANDLE;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    // ---- Enumerate physical devices -----------------------------------------
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));

    if (count == 0) {
        std::fprintf(stderr, "No Vulkan-capable physical devices found.\n");
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

    std::printf("drift_engine v0.1.1 — Vulkan up.\n");
    std::printf("Found %u physical device(s):\n", count);

    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[i], &props);

        std::printf("  [%u] %s (%s, API %u.%u.%u)\n",
                    i,
                    props.deviceName,
                    device_type_name(props.deviceType),
                    VK_VERSION_MAJOR(props.apiVersion),
                    VK_VERSION_MINOR(props.apiVersion),
                    VK_VERSION_PATCH(props.apiVersion));
    }

    // ---- Event loop ---------------------------------------------------------
    std::printf("Window open. Press ESC or close the window to exit.\n");

    while (!glfwWindowShouldClose(window))
        glfwPollEvents();

    // ---- Cleanup ------------------------------------------------------------
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

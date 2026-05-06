// Drift Engine — v0.0 bring-up
//
// Goal of this file: prove the toolchain is wired correctly end-to-end.
// We create a Vulkan instance, enumerate physical devices, print them, and exit.
// No window, no rendering, no compute. Those come next.

#include <vulkan/vulkan.h>

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

} // namespace

int main()
{
    // ---- Instance ---------------------------------------------------------
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "drift_engine";
    app.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
    app.pEngineName        = "drift_engine";
    app.engineVersion      = VK_MAKE_VERSION(0, 0, 1);
    app.apiVersion         = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

#ifdef __APPLE__
    // MoltenVK requires the portability enumeration extension on recent SDKs.
    const char* exts[] = { "VK_KHR_portability_enumeration" };
    ici.enabledExtensionCount   = 1;
    ici.ppEnabledExtensionNames = exts;
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkInstance instance = VK_NULL_HANDLE;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    // ---- Enumerate physical devices --------------------------------------
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));

    if (count == 0) {
        std::fprintf(stderr, "No Vulkan-capable physical devices found.\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

    std::printf("drift_engine v0.0 — Vulkan up.\n");
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

    // ---- Cleanup ----------------------------------------------------------
    vkDestroyInstance(instance, nullptr);
    return 0;
}

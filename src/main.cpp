// Drift Engine — v0.1.4
//
// VMA integration: creates a swapchain-sized RGBA16F storage image via VMA.
// The image is allocated but unused — v0.1.5 will write to it from compute.
// Frame loop still clears the swapchain directly (unchanged from v0.1.3).

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr uint32_t FRAMES_IN_FLIGHT = 2;

struct FrameData {
    VkCommandPool   pool;
    VkCommandBuffer cmd;
    VkSemaphore     image_available;
    VkSemaphore     render_finished;
    VkFence         in_flight;
};

struct RenderTarget {
    VkImage       image;
    VmaAllocation allocation;
    VkImageView   view;
};

bool g_framebuffer_resized = false;

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void framebuffer_resize_callback(GLFWwindow* /*window*/, int /*width*/, int /*height*/)
{
    g_framebuffer_resized = true;
}

#define VK_CHECK(expr)                                                         \
    do {                                                                        \
        VkResult _r = (expr);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            std::fprintf(stderr, "Vulkan error %d at %s:%d: %s\n",              \
                         static_cast<int>(_r), __FILE__, __LINE__, #expr);      \
            std::abort();                                                        \
        }                                                                       \
    } while (0)

RenderTarget create_render_target(VkDevice device, VmaAllocator allocator, VkExtent2D extent)
{
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    img_ci.extent = {extent.width, extent.height, 1};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT
                 | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                 | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    alloc_ci.priority = 1.0f;

    RenderTarget rt{};
    VK_CHECK(vmaCreateImage(allocator, &img_ci, &alloc_ci, &rt.image, &rt.allocation, nullptr));

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = rt.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &view_ci, nullptr, &rt.view));

    return rt;
}

void destroy_render_target(VkDevice device, VmaAllocator allocator, RenderTarget& rt)
{
    vkDestroyImageView(device, rt.view, nullptr);
    vmaDestroyImage(allocator, rt.image, rt.allocation);
    rt = {};
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
    glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);

    // ---- Gather GLFW instance extensions ------------------------------------
    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    // ---- Vulkan instance (via vk-bootstrap) ---------------------------------
    vkb::InstanceBuilder instance_builder(vkGetInstanceProcAddr);
    instance_builder
        .set_app_name("drift_engine")
        .set_engine_name("drift_engine")
        .require_api_version(1, 3, 0)
        .enable_extensions(glfw_ext_count, glfw_exts)
        .request_validation_layers(true)
        .use_default_debug_messenger();

    auto inst_ret = instance_builder.build();
    if (!inst_ret) {
        std::fprintf(stderr, "Failed to create Vulkan instance: %s\n",
                     inst_ret.error().message().c_str());
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    vkb::Instance vkb_inst = inst_ret.value();

    // ---- Surface ------------------------------------------------------------
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult surf_result = glfwCreateWindowSurface(vkb_inst.instance, window, nullptr, &surface);
    if (surf_result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create window surface (VkResult %d).\n",
                     static_cast<int>(surf_result));
        vkb::destroy_instance(vkb_inst);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // ---- Physical device selection ------------------------------------------
    VkPhysicalDeviceVulkan13Features features_13{};
    features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features_13.dynamicRendering = VK_TRUE;
    features_13.synchronization2 = VK_TRUE;

    vkb::PhysicalDeviceSelector selector(vkb_inst);
    selector
        .set_surface(surface)
        .set_minimum_version(1, 3)
        .set_required_features_13(features_13)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete);

    auto phys_ret = selector.select();
    if (!phys_ret) {
        std::fprintf(stderr, "Failed to select physical device: %s\n",
                     phys_ret.error().message().c_str());
        vkDestroySurfaceKHR(vkb_inst.instance, surface, nullptr);
        vkb::destroy_instance(vkb_inst);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    vkb::PhysicalDevice vkb_phys = phys_ret.value();

    // ---- Logical device -----------------------------------------------------
    vkb::DeviceBuilder device_builder(vkb_phys);
    auto dev_ret = device_builder.build();
    if (!dev_ret) {
        std::fprintf(stderr, "Failed to create logical device: %s\n",
                     dev_ret.error().message().c_str());
        vkDestroySurfaceKHR(vkb_inst.instance, surface, nullptr);
        vkb::destroy_instance(vkb_inst);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    vkb::Device vkb_device = dev_ret.value();
    VkDevice device = vkb_device.device;

    // ---- Queue handles ------------------------------------------------------
    VkQueue graphics_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
    VkQueue present_queue = vkb_device.get_queue(vkb::QueueType::present).value();
    uint32_t gfx_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    // ---- VMA allocator ------------------------------------------------------
    VmaVulkanFunctions vk_funcs{};
    vk_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vk_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_info{};
    alloc_info.physicalDevice = vkb_phys.physical_device;
    alloc_info.device = device;
    alloc_info.instance = vkb_inst.instance;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_3;
    alloc_info.pVulkanFunctions = &vk_funcs;

    VmaAllocator allocator = VK_NULL_HANDLE;
    VK_CHECK(vmaCreateAllocator(&alloc_info, &allocator));

    // ---- Swapchain ----------------------------------------------------------
    auto build_swapchain = [&]() -> vkb::Swapchain {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);

        vkb::SwapchainBuilder sc_builder(vkb_device);
        sc_builder
            .set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
            .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR)
            .set_desired_extent(static_cast<uint32_t>(w), static_cast<uint32_t>(h))
            .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        auto sc_ret = sc_builder.build();
        if (!sc_ret) {
            std::fprintf(stderr, "Failed to create swapchain: %s\n",
                         sc_ret.error().message().c_str());
            std::abort();
        }
        return sc_ret.value();
    };

    vkb::Swapchain vkb_swapchain = build_swapchain();
    std::vector<VkImage> swapchain_images = vkb_swapchain.get_images().value();
    std::vector<VkImageView> swapchain_views = vkb_swapchain.get_image_views().value();

    // ---- Storage image (render target) --------------------------------------
    RenderTarget render_target = create_render_target(device, allocator, vkb_swapchain.extent);

    // ---- Per-frame resources ------------------------------------------------
    std::array<FrameData, FRAMES_IN_FLIGHT> frames{};

    for (auto& f : frames) {
        VkCommandPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_ci.queueFamilyIndex = gfx_family;
        VK_CHECK(vkCreateCommandPool(device, &pool_ci, nullptr, &f.pool));

        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = f.pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmd_alloc, &f.cmd));

        VkSemaphoreCreateInfo sem_ci{};
        sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(device, &sem_ci, nullptr, &f.image_available));
        VK_CHECK(vkCreateSemaphore(device, &sem_ci, nullptr, &f.render_finished));

        VkFenceCreateInfo fence_ci{};
        fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(device, &fence_ci, nullptr, &f.in_flight));
    }

    // ---- Swapchain rebuild helper -------------------------------------------
    auto rebuild_swapchain = [&]() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        while (w == 0 || h == 0) {
            glfwWaitEvents();
            glfwGetFramebufferSize(window, &w, &h);
        }

        vkDeviceWaitIdle(device);

        for (auto view : swapchain_views)
            vkDestroyImageView(device, view, nullptr);
        vkb::destroy_swapchain(vkb_swapchain);

        vkb_swapchain = build_swapchain();
        swapchain_images = vkb_swapchain.get_images().value();
        swapchain_views = vkb_swapchain.get_image_views().value();

        destroy_render_target(device, allocator, render_target);
        render_target = create_render_target(device, allocator, vkb_swapchain.extent);

        g_framebuffer_resized = false;
    };

    // ---- Startup printout ---------------------------------------------------
    std::printf("drift_engine v0.1.4 — Vulkan up.\n");
    std::printf("Device:   %s\n", vkb_phys.name.c_str());
    std::printf("Queues:   graphics=%u  present=%u\n",
                gfx_family,
                vkb_device.get_queue_index(vkb::QueueType::present).value());
    std::printf("Swapchain: %ux%u, %u images\n",
                vkb_swapchain.extent.width, vkb_swapchain.extent.height,
                static_cast<uint32_t>(swapchain_images.size()));
    std::printf("Storage image: %ux%u R16G16B16A16_SFLOAT (%.1f MiB)\n",
                vkb_swapchain.extent.width, vkb_swapchain.extent.height,
                (vkb_swapchain.extent.width * vkb_swapchain.extent.height * 8)
                    / (1024.0 * 1024.0));
    std::printf("Window open. Press ESC or close the window to exit.\n");
    std::fflush(stdout);

    // ---- Frame loop ---------------------------------------------------------
    uint32_t current_frame = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        if (fb_w == 0 || fb_h == 0)
            continue;

        FrameData& frame = frames[current_frame];

        VK_CHECK(vkWaitForFences(device, 1, &frame.in_flight, VK_TRUE, UINT64_MAX));

        uint32_t image_index = 0;
        VkResult acquire_result = vkAcquireNextImageKHR(
            device, vkb_swapchain.swapchain, UINT64_MAX,
            frame.image_available, VK_NULL_HANDLE, &image_index);

        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
            rebuild_swapchain();
            continue;
        }
        if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
            std::fprintf(stderr, "Failed to acquire swapchain image (VkResult %d).\n",
                         static_cast<int>(acquire_result));
            break;
        }

        VK_CHECK(vkResetFences(device, 1, &frame.in_flight));
        VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(frame.cmd, &begin_info));

        // Barrier: UNDEFINED → TRANSFER_DST
        VkImageMemoryBarrier2 barrier_to_clear{};
        barrier_to_clear.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier_to_clear.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier_to_clear.srcAccessMask = VK_ACCESS_2_NONE;
        barrier_to_clear.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        barrier_to_clear.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier_to_clear.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier_to_clear.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_to_clear.image = swapchain_images[image_index];
        barrier_to_clear.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep_to_clear{};
        dep_to_clear.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_to_clear.imageMemoryBarrierCount = 1;
        dep_to_clear.pImageMemoryBarriers = &barrier_to_clear;

        vkCmdPipelineBarrier2(frame.cmd, &dep_to_clear);

        // Clear with cycling color
        float t = static_cast<float>(glfwGetTime());
        VkClearColorValue clear_color = {{
            0.5f + 0.5f * std::sin(t),
            0.5f + 0.5f * std::sin(t + 2.094f),
            0.5f + 0.5f * std::sin(t + 4.188f),
            1.0f
        }};

        VkImageSubresourceRange clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(frame.cmd, swapchain_images[image_index],
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clear_color, 1, &clear_range);

        // Barrier: TRANSFER_DST → PRESENT_SRC
        VkImageMemoryBarrier2 barrier_to_present{};
        barrier_to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier_to_present.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        barrier_to_present.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier_to_present.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier_to_present.dstAccessMask = VK_ACCESS_2_NONE;
        barrier_to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier_to_present.image = swapchain_images[image_index];
        barrier_to_present.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep_to_present{};
        dep_to_present.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_to_present.imageMemoryBarrierCount = 1;
        dep_to_present.pImageMemoryBarriers = &barrier_to_present;

        vkCmdPipelineBarrier2(frame.cmd, &dep_to_present);

        VK_CHECK(vkEndCommandBuffer(frame.cmd));

        // Submit
        VkSemaphoreSubmitInfo wait_sem{};
        wait_sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        wait_sem.semaphore = frame.image_available;
        wait_sem.stageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;

        VkSemaphoreSubmitInfo signal_sem{};
        signal_sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signal_sem.semaphore = frame.render_finished;
        signal_sem.stageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

        VkCommandBufferSubmitInfo cmd_info{};
        cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cmd_info.commandBuffer = frame.cmd;

        VkSubmitInfo2 submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submit.waitSemaphoreInfoCount = 1;
        submit.pWaitSemaphoreInfos = &wait_sem;
        submit.signalSemaphoreInfoCount = 1;
        submit.pSignalSemaphoreInfos = &signal_sem;
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &cmd_info;

        VK_CHECK(vkQueueSubmit2(graphics_queue, 1, &submit, frame.in_flight));

        // Present
        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &frame.render_finished;
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &vkb_swapchain.swapchain;
        present_info.pImageIndices = &image_index;

        VkResult present_result = vkQueuePresentKHR(present_queue, &present_info);

        if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
            present_result == VK_SUBOPTIMAL_KHR || g_framebuffer_resized) {
            rebuild_swapchain();
        } else if (present_result != VK_SUCCESS) {
            std::fprintf(stderr, "Failed to present (VkResult %d).\n",
                         static_cast<int>(present_result));
            break;
        }

        current_frame = (current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    // ---- Cleanup (LIFO) -----------------------------------------------------
    vkDeviceWaitIdle(device);

    for (auto& f : frames) {
        vkDestroyFence(device, f.in_flight, nullptr);
        vkDestroySemaphore(device, f.render_finished, nullptr);
        vkDestroySemaphore(device, f.image_available, nullptr);
        vkDestroyCommandPool(device, f.pool, nullptr);
    }

    for (auto view : swapchain_views)
        vkDestroyImageView(device, view, nullptr);
    vkb::destroy_swapchain(vkb_swapchain);

    destroy_render_target(device, allocator, render_target);
    vmaDestroyAllocator(allocator);

    vkb::destroy_device(vkb_device);
    vkDestroySurfaceKHR(vkb_inst.instance, surface, nullptr);
    vkb::destroy_instance(vkb_inst);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

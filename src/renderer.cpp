#include "renderer.h"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <cstdio>

static vkb::Swapchain build_swapchain(Renderer& r)
{
    int w, h;
    glfwGetFramebufferSize(r.window, &w, &h);

    vkb::SwapchainBuilder sc_builder(r.vkb_device);
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
}

void renderer_init(Renderer& r, int width, int height, const char* title)
{
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        std::abort();
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    r.window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!r.window) {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        std::abort();
    }

    uint32_t glfw_ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

    vkb::InstanceBuilder instance_builder(vkGetInstanceProcAddr);
    instance_builder
        .set_app_name(title)
        .set_engine_name(title)
        .require_api_version(1, 3, 0)
        .enable_extensions(glfw_ext_count, glfw_exts)
        .request_validation_layers(true)
        .use_default_debug_messenger();

    auto inst_ret = instance_builder.build();
    if (!inst_ret) {
        std::fprintf(stderr, "Failed to create Vulkan instance: %s\n",
                     inst_ret.error().message().c_str());
        std::abort();
    }
    r.vkb_inst = inst_ret.value();

    VkResult surf_result = glfwCreateWindowSurface(r.vkb_inst.instance, r.window, nullptr, &r.surface);
    if (surf_result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to create window surface (VkResult %d).\n",
                     static_cast<int>(surf_result));
        std::abort();
    }

    VkPhysicalDeviceVulkan13Features features_13{};
    features_13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features_13.dynamicRendering = VK_TRUE;
    features_13.synchronization2 = VK_TRUE;

    vkb::PhysicalDeviceSelector selector(r.vkb_inst);
    selector
        .set_surface(r.surface)
        .set_minimum_version(1, 3)
        .set_required_features_13(features_13)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete);

    auto phys_ret = selector.select();
    if (!phys_ret) {
        std::fprintf(stderr, "Failed to select physical device: %s\n",
                     phys_ret.error().message().c_str());
        std::abort();
    }
    r.vkb_phys = phys_ret.value();

    vkb::DeviceBuilder device_builder(r.vkb_phys);
    auto dev_ret = device_builder.build();
    if (!dev_ret) {
        std::fprintf(stderr, "Failed to create logical device: %s\n",
                     dev_ret.error().message().c_str());
        std::abort();
    }
    r.vkb_device = dev_ret.value();
    r.device = r.vkb_device.device;

    r.graphics_queue = r.vkb_device.get_queue(vkb::QueueType::graphics).value();
    r.present_queue = r.vkb_device.get_queue(vkb::QueueType::present).value();
    r.gfx_family = r.vkb_device.get_queue_index(vkb::QueueType::graphics).value();

    VmaVulkanFunctions vk_funcs{};
    vk_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vk_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_info{};
    alloc_info.physicalDevice = r.vkb_phys.physical_device;
    alloc_info.device = r.device;
    alloc_info.instance = r.vkb_inst.instance;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_3;
    alloc_info.pVulkanFunctions = &vk_funcs;

    VK_CHECK(vmaCreateAllocator(&alloc_info, &r.allocator));

    r.vkb_swapchain = build_swapchain(r);
    r.swapchain_images = r.vkb_swapchain.get_images().value();
    r.swapchain_views = r.vkb_swapchain.get_image_views().value();

    r.depth_buffer = create_depth_buffer(r.device, r.allocator, r.vkb_swapchain.extent);

    for (auto& f : r.frames) {
        VkCommandPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_ci.queueFamilyIndex = r.gfx_family;
        VK_CHECK(vkCreateCommandPool(r.device, &pool_ci, nullptr, &f.pool));

        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = f.pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(r.device, &cmd_alloc, &f.cmd));

        VkSemaphoreCreateInfo sem_ci{};
        sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(r.device, &sem_ci, nullptr, &f.image_available));
        VK_CHECK(vkCreateSemaphore(r.device, &sem_ci, nullptr, &f.render_finished));

        VkFenceCreateInfo fence_ci{};
        fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(r.device, &fence_ci, nullptr, &f.in_flight));
    }

    VkQueryPoolCreateInfo qp_ci{};
    qp_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qp_ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qp_ci.queryCount = 2 * FRAMES_IN_FLIGHT;
    VK_CHECK(vkCreateQueryPool(r.device, &qp_ci, nullptr, &r.query_pool));

    ImGui::CreateContext();
    // We manage GLFW_CURSOR mode ourselves (per-frame state machine in main).
    // Without this flag, ImGui_ImplGlfw_NewFrame forces CURSOR_NORMAL each
    // frame and the OS cursor leaks back over the rendered world.
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    ImGui_ImplGlfw_InitForVulkan(r.window, false);

    VkDescriptorPoolSize imgui_pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000}
    };
    VkDescriptorPoolCreateInfo imgui_dp_ci{};
    imgui_dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    imgui_dp_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    imgui_dp_ci.maxSets = 1000;
    imgui_dp_ci.poolSizeCount = 1;
    imgui_dp_ci.pPoolSizes = imgui_pool_sizes;
    VK_CHECK(vkCreateDescriptorPool(r.device, &imgui_dp_ci, nullptr, &r.imgui_pool));

    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkPipelineRenderingCreateInfo imgui_rendering_ci{};
    imgui_rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    imgui_rendering_ci.colorAttachmentCount = 1;
    imgui_rendering_ci.pColorAttachmentFormats = &swapchain_format;
    imgui_rendering_ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    ImGui_ImplVulkan_InitInfo imgui_init{};
    imgui_init.Instance = r.vkb_inst.instance;
    imgui_init.PhysicalDevice = r.vkb_phys.physical_device;
    imgui_init.Device = r.device;
    imgui_init.QueueFamily = r.gfx_family;
    imgui_init.Queue = r.graphics_queue;
    imgui_init.DescriptorPool = r.imgui_pool;
    imgui_init.MinImageCount = 2;
    imgui_init.ImageCount = static_cast<uint32_t>(r.swapchain_images.size());
    imgui_init.UseDynamicRendering = true;
    imgui_init.PipelineRenderingCreateInfo = imgui_rendering_ci;
    ImGui_ImplVulkan_Init(&imgui_init);
}

void renderer_rebuild_swapchain(Renderer& r)
{
    int w = 0, h = 0;
    glfwGetFramebufferSize(r.window, &w, &h);
    while (w == 0 || h == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(r.window, &w, &h);
    }

    vkDeviceWaitIdle(r.device);

    for (auto view : r.swapchain_views)
        vkDestroyImageView(r.device, view, nullptr);
    vkb::destroy_swapchain(r.vkb_swapchain);

    r.vkb_swapchain = build_swapchain(r);
    r.swapchain_images = r.vkb_swapchain.get_images().value();
    r.swapchain_views = r.vkb_swapchain.get_image_views().value();

    destroy_depth_buffer(r.device, r.allocator, r.depth_buffer);
    r.depth_buffer = create_depth_buffer(r.device, r.allocator, r.vkb_swapchain.extent);
}

bool renderer_begin_frame(Renderer& r, FrameData*& frame, uint32_t& image_index, VkExtent2D& extent)
{
    int fb_w, fb_h;
    glfwGetFramebufferSize(r.window, &fb_w, &fb_h);
    if (fb_w == 0 || fb_h == 0)
        return false;

    frame = &r.frames[r.current_frame];

    VK_CHECK(vkWaitForFences(r.device, 1, &frame->in_flight, VK_TRUE, UINT64_MAX));

    VkResult acquire_result = vkAcquireNextImageKHR(
        r.device, r.vkb_swapchain.swapchain, UINT64_MAX,
        frame->image_available, VK_NULL_HANDLE, &image_index);

    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
        renderer_rebuild_swapchain(r);
        return false;
    }
    if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
        std::fprintf(stderr, "Failed to acquire swapchain image (VkResult %d).\n",
                     static_cast<int>(acquire_result));
        return false;
    }

    VK_CHECK(vkResetFences(r.device, 1, &frame->in_flight));
    VK_CHECK(vkResetCommandBuffer(frame->cmd, 0));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(frame->cmd, &begin_info));

    extent = r.vkb_swapchain.extent;
    return true;
}

void renderer_end_frame(Renderer& r, FrameData& frame, uint32_t image_index)
{
    VK_CHECK(vkEndCommandBuffer(frame.cmd));

    VkSemaphoreSubmitInfo wait_sem{};
    wait_sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait_sem.semaphore = frame.image_available;
    wait_sem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

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

    VK_CHECK(vkQueueSubmit2(r.graphics_queue, 1, &submit, frame.in_flight));

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &frame.render_finished;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &r.vkb_swapchain.swapchain;
    present_info.pImageIndices = &image_index;

    VkResult present_result = vkQueuePresentKHR(r.present_queue, &present_info);

    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR) {
        renderer_rebuild_swapchain(r);
    } else if (present_result != VK_SUCCESS) {
        std::fprintf(stderr, "Failed to present (VkResult %d).\n",
                     static_cast<int>(present_result));
    }

    r.current_frame = (r.current_frame + 1) % FRAMES_IN_FLIGHT;
}

void renderer_shutdown(Renderer& r)
{
    vkDeviceWaitIdle(r.device);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(r.device, r.imgui_pool, nullptr);

    for (auto& f : r.frames) {
        vkDestroyFence(r.device, f.in_flight, nullptr);
        vkDestroySemaphore(r.device, f.render_finished, nullptr);
        vkDestroySemaphore(r.device, f.image_available, nullptr);
        vkDestroyCommandPool(r.device, f.pool, nullptr);
    }

    vkDestroyQueryPool(r.device, r.query_pool, nullptr);

    for (auto view : r.swapchain_views)
        vkDestroyImageView(r.device, view, nullptr);
    vkb::destroy_swapchain(r.vkb_swapchain);

    destroy_depth_buffer(r.device, r.allocator, r.depth_buffer);

    vmaDestroyAllocator(r.allocator);

    vkb::destroy_device(r.vkb_device);
    vkDestroySurfaceKHR(r.vkb_inst.instance, r.surface, nullptr);
    vkb::destroy_instance(r.vkb_inst);

    glfwDestroyWindow(r.window);
    glfwTerminate();
}

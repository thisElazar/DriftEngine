// Drift Engine — v0.2.1b
//
// Procedural Crater Lake basin (R32_SFLOAT) + SWE solver running every frame.
// Visualization: elevation color ramp with "is wet" debug overlay.

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr uint32_t FRAMES_IN_FLIGHT = 2;
constexpr uint32_t SWE_GRID_W = 1024;
constexpr uint32_t SWE_GRID_H = 1024;

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

struct PushConstants {
    float    time;
    uint32_t width;
    uint32_t height;
    float    max_elevation;
};

struct SweInitPC {
    uint32_t grid_w;
    uint32_t grid_h;
    float    initial_water_level;
    float    _pad;
};

struct SweStepPC {
    float    time;
    float    dt;
    float    gravity;
    float    friction;
    float    dx;
    float    sea_level;
    float    damping;
    float    _pad0;
    uint32_t grid_w;
    uint32_t grid_h;
    float    pulse_x;
    float    pulse_y;
    float    pulse_radius;
    float    pulse_amount;
};

struct HeightmapData {
    std::vector<float> values;
    uint32_t width;
    uint32_t height;
};

struct HeightmapGPU {
    VkImage       image;
    VmaAllocation allocation;
    VkImageView   view;
};

struct SweImage {
    VkImage       image;
    VmaAllocation allocation;
    VkImageView   view;
};

bool g_framebuffer_resized = false;
bool g_pulse_pending = false;

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        g_pulse_pending = true;
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

std::vector<uint32_t> load_spirv(const char* path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "Failed to open SPIR-V file: %s\n", path);
        std::abort();
    }
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));
    return buffer;
}

struct BasinParams {
    uint32_t grid_w        = 1024;
    uint32_t grid_h        = 1024;
    float    cell_spacing  = 10.0f;
    float    floor_height  = 100.0f;
    float    rim_height    = 1500.0f;
    float    base_height   = 800.0f;
    float    inner_radius  = 2000.0f;
    float    rim_radius    = 2800.0f;
    float    initial_water = 700.0f;
};

float cpu_smoothstep(float edge0, float edge1, float x)
{
    float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

std::vector<float> generate_crater_basin(const BasinParams& p)
{
    std::vector<float> data(static_cast<size_t>(p.grid_w) * p.grid_h);
    for (uint32_t y = 0; y < p.grid_h; ++y) {
        for (uint32_t x = 0; x < p.grid_w; ++x) {
            float cx = (static_cast<float>(x) - p.grid_w * 0.5f) * p.cell_spacing;
            float cy = (static_cast<float>(y) - p.grid_h * 0.5f) * p.cell_spacing;
            float r = std::sqrt(cx * cx + cy * cy);
            float h;
            if (r < p.inner_radius) {
                h = p.floor_height;
            } else if (r < p.rim_radius) {
                float t = (r - p.inner_radius) / (p.rim_radius - p.inner_radius);
                h = std::lerp(p.floor_height, p.rim_height, cpu_smoothstep(0.0f, 1.0f, t));
            } else {
                float t = std::clamp((r - p.rim_radius) / 1500.0f, 0.0f, 1.0f);
                h = std::lerp(p.rim_height, p.base_height, cpu_smoothstep(0.0f, 1.0f, t));
                h += 30.0f * std::sin(cx * 0.0003f) * std::cos(cy * 0.0004f);
            }
            data[static_cast<size_t>(y) * p.grid_w + x] = h;
        }
    }
    return data;
}

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

SweImage create_swe_image(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h)
{
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    img_ci.extent = {w, h, 1};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    SweImage img{};
    VK_CHECK(vmaCreateImage(allocator, &img_ci, &alloc_ci, &img.image, &img.allocation, nullptr));

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = img.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &view_ci, nullptr, &img.view));

    return img;
}

void destroy_swe_image(VkDevice device, VmaAllocator allocator, SweImage& img)
{
    vkDestroyImageView(device, img.view, nullptr);
    vmaDestroyImage(allocator, img.image, img.allocation);
    img = {};
}

HeightmapGPU upload_heightmap(VkDevice device, VmaAllocator allocator,
                              VkQueue queue, uint32_t queue_family,
                              const HeightmapData& hm)
{
    VkDeviceSize buf_size = static_cast<VkDeviceSize>(hm.width) * hm.height * sizeof(float);

    // Staging buffer
    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = buf_size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo staging_ai{};
    staging_ai.usage = VMA_MEMORY_USAGE_AUTO;
    staging_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                     | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer staging_buf = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo staging_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &buf_ci, &staging_ai,
                             &staging_buf, &staging_alloc, &staging_info));

    std::memcpy(staging_info.pMappedData, hm.values.data(), buf_size);
    vmaFlushAllocation(allocator, staging_alloc, 0, VK_WHOLE_SIZE);

    // GPU image
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = VK_FORMAT_R32_SFLOAT;
    img_ci.extent = {hm.width, hm.height, 1};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo gpu_ai{};
    gpu_ai.usage = VMA_MEMORY_USAGE_AUTO;
    gpu_ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    HeightmapGPU gpu{};
    VK_CHECK(vmaCreateImage(allocator, &img_ci, &gpu_ai, &gpu.image, &gpu.allocation, nullptr));

    // One-shot command buffer
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_ci.queueFamilyIndex = queue_family;

    VkCommandPool tmp_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(device, &pool_ci, nullptr, &tmp_pool));

    VkCommandBufferAllocateInfo cmd_ai{};
    cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ai.commandPool = tmp_pool;
    cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmd_ai, &cmd));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    // Barrier: UNDEFINED → TRANSFER_DST
    VkImageMemoryBarrier2 barrier_to_dst{};
    barrier_to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier_to_dst.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    barrier_to_dst.srcAccessMask = VK_ACCESS_2_NONE;
    barrier_to_dst.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier_to_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier_to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_to_dst.image = gpu.image;
    barrier_to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep_to_dst{};
    dep_to_dst.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_to_dst.imageMemoryBarrierCount = 1;
    dep_to_dst.pImageMemoryBarriers = &barrier_to_dst;
    vkCmdPipelineBarrier2(cmd, &dep_to_dst);

    // Copy buffer → image
    VkBufferImageCopy2 copy_region{};
    copy_region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
    copy_region.bufferOffset = 0;
    copy_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.imageExtent = {hm.width, hm.height, 1};

    VkCopyBufferToImageInfo2 copy_info{};
    copy_info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
    copy_info.srcBuffer = staging_buf;
    copy_info.dstImage = gpu.image;
    copy_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copy_info.regionCount = 1;
    copy_info.pRegions = &copy_region;
    vkCmdCopyBufferToImage2(cmd, &copy_info);

    // Barrier: TRANSFER_DST → SHADER_READ_ONLY
    VkImageMemoryBarrier2 barrier_to_read{};
    barrier_to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier_to_read.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier_to_read.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier_to_read.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier_to_read.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    barrier_to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_to_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_to_read.image = gpu.image;
    barrier_to_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep_to_read{};
    dep_to_read.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_to_read.imageMemoryBarrierCount = 1;
    dep_to_read.pImageMemoryBarriers = &barrier_to_read;
    vkCmdPipelineBarrier2(cmd, &dep_to_read);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // Submit and wait
    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device, &fence_ci, nullptr, &fence));

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, tmp_pool, nullptr);
    vmaDestroyBuffer(allocator, staging_buf, staging_alloc);

    // Image view
    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = gpu.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_R32_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &view_ci, nullptr, &gpu.view));

    return gpu;
}

} // namespace

int main()
{
#ifdef __APPLE__
    setenv("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "0", 0);
#endif

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

    // ---- Procedural basin ---------------------------------------------------
    BasinParams bp;
    auto basin = generate_crater_basin(bp);
    HeightmapData hm{std::move(basin), bp.grid_w, bp.grid_h};

    auto [hm_min, hm_max] = std::minmax_element(hm.values.begin(), hm.values.end());

    HeightmapGPU heightmap_gpu = upload_heightmap(device, allocator, graphics_queue, gfx_family, hm);

    // ---- SWE images (ping-pong state + output) ------------------------------
    SweImage swe_state_a = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    SweImage swe_state_b = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    SweImage swe_output  = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

    // ---- Sampler ------------------------------------------------------------
    VkSamplerCreateInfo sampler_ci{};
    sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_ci.magFilter = VK_FILTER_LINEAR;
    sampler_ci.minFilter = VK_FILTER_LINEAR;
    sampler_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    VkSampler sampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(device, &sampler_ci, nullptr, &sampler));

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

    // ---- Visualize pipeline -------------------------------------------------
    auto viz_spirv = load_spirv("shaders/visualize.spv");

    VkShaderModuleCreateInfo viz_sm_ci{};
    viz_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    viz_sm_ci.codeSize = viz_spirv.size() * sizeof(uint32_t);
    viz_sm_ci.pCode = viz_spirv.data();

    VkShaderModule viz_shader = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &viz_sm_ci, nullptr, &viz_shader));

    VkDescriptorSetLayoutBinding viz_bindings[3]{};
    viz_bindings[0].binding = 0;
    viz_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    viz_bindings[0].descriptorCount = 1;
    viz_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    viz_bindings[1].binding = 1;
    viz_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    viz_bindings[1].descriptorCount = 1;
    viz_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    viz_bindings[2].binding = 2;
    viz_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    viz_bindings[2].descriptorCount = 1;
    viz_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo viz_dsl_ci{};
    viz_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    viz_dsl_ci.bindingCount = 3;
    viz_dsl_ci.pBindings = viz_bindings;

    VkDescriptorSetLayout viz_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &viz_dsl_ci, nullptr, &viz_desc_layout));

    VkPushConstantRange viz_push_range{};
    viz_push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    viz_push_range.offset = 0;
    viz_push_range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo viz_pl_ci{};
    viz_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    viz_pl_ci.setLayoutCount = 1;
    viz_pl_ci.pSetLayouts = &viz_desc_layout;
    viz_pl_ci.pushConstantRangeCount = 1;
    viz_pl_ci.pPushConstantRanges = &viz_push_range;

    VkPipelineLayout viz_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &viz_pl_ci, nullptr, &viz_pipeline_layout));

    VkPipelineShaderStageCreateInfo viz_stage{};
    viz_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    viz_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    viz_stage.module = viz_shader;
    viz_stage.pName = "main";

    VkComputePipelineCreateInfo viz_cp_ci{};
    viz_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    viz_cp_ci.stage = viz_stage;
    viz_cp_ci.layout = viz_pipeline_layout;

    VkPipeline viz_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &viz_cp_ci, nullptr, &viz_pipeline));

    // ---- SWE init pipeline --------------------------------------------------
    auto swe_init_spirv = load_spirv("shaders/swe_init.spv");

    VkShaderModuleCreateInfo swe_init_sm_ci{};
    swe_init_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    swe_init_sm_ci.codeSize = swe_init_spirv.size() * sizeof(uint32_t);
    swe_init_sm_ci.pCode = swe_init_spirv.data();

    VkShaderModule swe_init_shader = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &swe_init_sm_ci, nullptr, &swe_init_shader));

    VkDescriptorSetLayoutBinding swe_init_bindings[2]{};
    swe_init_bindings[0].binding = 0;
    swe_init_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    swe_init_bindings[0].descriptorCount = 1;
    swe_init_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    swe_init_bindings[1].binding = 1;
    swe_init_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swe_init_bindings[1].descriptorCount = 1;
    swe_init_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo swe_init_dsl_ci{};
    swe_init_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    swe_init_dsl_ci.bindingCount = 2;
    swe_init_dsl_ci.pBindings = swe_init_bindings;

    VkDescriptorSetLayout swe_init_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &swe_init_dsl_ci, nullptr, &swe_init_desc_layout));

    VkPushConstantRange swe_init_push{};
    swe_init_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    swe_init_push.offset = 0;
    swe_init_push.size = sizeof(SweInitPC);

    VkPipelineLayoutCreateInfo swe_init_pl_ci{};
    swe_init_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    swe_init_pl_ci.setLayoutCount = 1;
    swe_init_pl_ci.pSetLayouts = &swe_init_desc_layout;
    swe_init_pl_ci.pushConstantRangeCount = 1;
    swe_init_pl_ci.pPushConstantRanges = &swe_init_push;

    VkPipelineLayout swe_init_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &swe_init_pl_ci, nullptr, &swe_init_pipeline_layout));

    VkPipelineShaderStageCreateInfo swe_init_stage{};
    swe_init_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    swe_init_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    swe_init_stage.module = swe_init_shader;
    swe_init_stage.pName = "main";

    VkComputePipelineCreateInfo swe_init_cp_ci{};
    swe_init_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    swe_init_cp_ci.stage = swe_init_stage;
    swe_init_cp_ci.layout = swe_init_pipeline_layout;

    VkPipeline swe_init_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &swe_init_cp_ci, nullptr, &swe_init_pipeline));

    // ---- SWE step pipeline --------------------------------------------------
    auto swe_step_spirv = load_spirv("shaders/swe_step.spv");

    VkShaderModuleCreateInfo swe_step_sm_ci{};
    swe_step_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    swe_step_sm_ci.codeSize = swe_step_spirv.size() * sizeof(uint32_t);
    swe_step_sm_ci.pCode = swe_step_spirv.data();

    VkShaderModule swe_step_shader = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &swe_step_sm_ci, nullptr, &swe_step_shader));

    VkDescriptorSetLayoutBinding swe_step_bindings[4]{};
    swe_step_bindings[0].binding = 0;
    swe_step_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    swe_step_bindings[0].descriptorCount = 1;
    swe_step_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    swe_step_bindings[1].binding = 1;
    swe_step_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    swe_step_bindings[1].descriptorCount = 1;
    swe_step_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    swe_step_bindings[2].binding = 2;
    swe_step_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swe_step_bindings[2].descriptorCount = 1;
    swe_step_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    swe_step_bindings[3].binding = 3;
    swe_step_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swe_step_bindings[3].descriptorCount = 1;
    swe_step_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo swe_step_dsl_ci{};
    swe_step_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    swe_step_dsl_ci.bindingCount = 4;
    swe_step_dsl_ci.pBindings = swe_step_bindings;

    VkDescriptorSetLayout swe_step_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &swe_step_dsl_ci, nullptr, &swe_step_desc_layout));

    VkPushConstantRange swe_step_push{};
    swe_step_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    swe_step_push.offset = 0;
    swe_step_push.size = sizeof(SweStepPC);

    VkPipelineLayoutCreateInfo swe_step_pl_ci{};
    swe_step_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    swe_step_pl_ci.setLayoutCount = 1;
    swe_step_pl_ci.pSetLayouts = &swe_step_desc_layout;
    swe_step_pl_ci.pushConstantRangeCount = 1;
    swe_step_pl_ci.pPushConstantRanges = &swe_step_push;

    VkPipelineLayout swe_step_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &swe_step_pl_ci, nullptr, &swe_step_pipeline_layout));

    VkPipelineShaderStageCreateInfo swe_step_stage{};
    swe_step_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    swe_step_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    swe_step_stage.module = swe_step_shader;
    swe_step_stage.pName = "main";

    VkComputePipelineCreateInfo swe_step_cp_ci{};
    swe_step_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    swe_step_cp_ci.stage = swe_step_stage;
    swe_step_cp_ci.layout = swe_step_pipeline_layout;

    VkPipeline swe_step_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &swe_step_cp_ci, nullptr, &swe_step_pipeline));

    // ---- Descriptor pool + sets ---------------------------------------------
    VkDescriptorPoolSize pool_sizes[3]{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 10;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = 4;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[2].descriptorCount = 6;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 4;
    dp_ci.poolSizeCount = 3;
    dp_ci.pPoolSizes = pool_sizes;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(device, &dp_ci, nullptr, &desc_pool));

    // Visualize descriptor set
    VkDescriptorSetAllocateInfo viz_ds_ai{};
    viz_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    viz_ds_ai.descriptorPool = desc_pool;
    viz_ds_ai.descriptorSetCount = 1;
    viz_ds_ai.pSetLayouts = &viz_desc_layout;

    VkDescriptorSet viz_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &viz_ds_ai, &viz_desc_set));

    // SWE init descriptor set
    VkDescriptorSetAllocateInfo swe_init_ds_ai{};
    swe_init_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    swe_init_ds_ai.descriptorPool = desc_pool;
    swe_init_ds_ai.descriptorSetCount = 1;
    swe_init_ds_ai.pSetLayouts = &swe_init_desc_layout;

    VkDescriptorSet swe_init_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &swe_init_ds_ai, &swe_init_desc_set));

    // SWE step descriptor sets (2 for ping-pong)
    VkDescriptorSetLayout swe_step_layouts[2] = {swe_step_desc_layout, swe_step_desc_layout};
    VkDescriptorSetAllocateInfo swe_step_ds_ai{};
    swe_step_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    swe_step_ds_ai.descriptorPool = desc_pool;
    swe_step_ds_ai.descriptorSetCount = 2;
    swe_step_ds_ai.pSetLayouts = swe_step_layouts;

    VkDescriptorSet swe_step_desc_sets[2] = {};
    VK_CHECK(vkAllocateDescriptorSets(device, &swe_step_ds_ai, swe_step_desc_sets));

    // ---- Write descriptor sets ----------------------------------------------
    auto write_viz_descriptors = [&](VkImageView storage_view) {
        VkDescriptorImageInfo storage_info{};
        storage_info.imageView = storage_view;
        storage_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo heightmap_info{};
        heightmap_info.imageView = heightmap_gpu.view;
        heightmap_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        heightmap_info.sampler = sampler;

        VkDescriptorImageInfo water_info{};
        water_info.imageView = swe_state_a.view;
        water_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        water_info.sampler = sampler;

        VkWriteDescriptorSet writes[3]{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = viz_desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &storage_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = viz_desc_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &heightmap_info;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = viz_desc_set;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &water_info;

        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    };

    write_viz_descriptors(render_target.view);

    // SWE init descriptors: terrain(sampled) + state_a(storage)
    {
        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = heightmap_gpu.view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo state_info{};
        state_info.imageView = swe_state_a.view;
        state_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = swe_init_desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].pImageInfo = &terrain_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = swe_init_desc_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &state_info;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }

    // SWE step descriptors: set[0] reads A writes B, set[1] reads B writes A
    {
        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = heightmap_gpu.view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo state_a_read{};
        state_a_read.imageView = swe_state_a.view;
        state_a_read.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_b_read{};
        state_b_read.imageView = swe_state_b.view;
        state_b_read.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_a_write{};
        state_a_write.imageView = swe_state_a.view;
        state_a_write.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_b_write{};
        state_b_write.imageView = swe_state_b.view;
        state_b_write.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo output_info{};
        output_info.imageView = swe_output.view;
        output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Set 0: read A → write B
        VkWriteDescriptorSet writes_0[4]{};
        writes_0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[0].dstSet = swe_step_desc_sets[0];
        writes_0[0].dstBinding = 0;
        writes_0[0].descriptorCount = 1;
        writes_0[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_0[0].pImageInfo = &terrain_info;

        writes_0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[1].dstSet = swe_step_desc_sets[0];
        writes_0[1].dstBinding = 1;
        writes_0[1].descriptorCount = 1;
        writes_0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_0[1].pImageInfo = &state_a_read;

        writes_0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[2].dstSet = swe_step_desc_sets[0];
        writes_0[2].dstBinding = 2;
        writes_0[2].descriptorCount = 1;
        writes_0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_0[2].pImageInfo = &state_b_write;

        writes_0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[3].dstSet = swe_step_desc_sets[0];
        writes_0[3].dstBinding = 3;
        writes_0[3].descriptorCount = 1;
        writes_0[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_0[3].pImageInfo = &output_info;

        vkUpdateDescriptorSets(device, 4, writes_0, 0, nullptr);

        // Set 1: read B → write A
        VkWriteDescriptorSet writes_1[4]{};
        writes_1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[0].dstSet = swe_step_desc_sets[1];
        writes_1[0].dstBinding = 0;
        writes_1[0].descriptorCount = 1;
        writes_1[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_1[0].pImageInfo = &terrain_info;

        writes_1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[1].dstSet = swe_step_desc_sets[1];
        writes_1[1].dstBinding = 1;
        writes_1[1].descriptorCount = 1;
        writes_1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_1[1].pImageInfo = &state_b_read;

        writes_1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[2].dstSet = swe_step_desc_sets[1];
        writes_1[2].dstBinding = 2;
        writes_1[2].descriptorCount = 1;
        writes_1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_1[2].pImageInfo = &state_a_write;

        writes_1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[3].dstSet = swe_step_desc_sets[1];
        writes_1[3].dstBinding = 3;
        writes_1[3].descriptorCount = 1;
        writes_1[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_1[3].pImageInfo = &output_info;

        vkUpdateDescriptorSets(device, 4, writes_1, 0, nullptr);
    }

    // ---- SWE init dispatch (one-shot) ---------------------------------------
    {
        VkCommandPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_ci.queueFamilyIndex = gfx_family;

        VkCommandPool tmp_pool = VK_NULL_HANDLE;
        VK_CHECK(vkCreateCommandPool(device, &pool_ci, nullptr, &tmp_pool));

        VkCommandBufferAllocateInfo cmd_ai{};
        cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_ai.commandPool = tmp_pool;
        cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_ai.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmd_ai, &cmd));

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

        // Transition SWE images: UNDEFINED → GENERAL
        VkImageMemoryBarrier2 init_barriers[3]{};
        for (int i = 0; i < 3; ++i) {
            init_barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            init_barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            init_barriers[i].srcAccessMask = VK_ACCESS_2_NONE;
            init_barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            init_barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            init_barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            init_barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            init_barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        init_barriers[0].image = swe_state_a.image;
        init_barriers[1].image = swe_state_b.image;
        init_barriers[2].image = swe_output.image;

        VkDependencyInfo dep_init{};
        dep_init.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_init.imageMemoryBarrierCount = 3;
        dep_init.pImageMemoryBarriers = init_barriers;
        vkCmdPipelineBarrier2(cmd, &dep_init);

        // Dispatch swe_init
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swe_init_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                swe_init_pipeline_layout, 0, 1, &swe_init_desc_set, 0, nullptr);

        SweInitPC init_pc{};
        init_pc.grid_w = SWE_GRID_W;
        init_pc.grid_h = SWE_GRID_H;
        init_pc.initial_water_level = bp.floor_height + bp.initial_water;
        init_pc._pad = 0.0f;
        vkCmdPushConstants(cmd, swe_init_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(init_pc), &init_pc);

        vkCmdDispatch(cmd, (SWE_GRID_W + 7) / 8, (SWE_GRID_H + 7) / 8, 1);

        // Barrier: swe_init write → swe_step read
        VkMemoryBarrier2 mem_barrier{};
        mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mem_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mem_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mem_barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo dep_after_init{};
        dep_after_init.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_after_init.memoryBarrierCount = 1;
        dep_after_init.pMemoryBarriers = &mem_barrier;
        vkCmdPipelineBarrier2(cmd, &dep_after_init);

        VK_CHECK(vkEndCommandBuffer(cmd));

        VkFenceCreateInfo fence_ci{};
        fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        VK_CHECK(vkCreateFence(device, &fence_ci, nullptr, &fence));

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(graphics_queue, 1, &submit, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, tmp_pool, nullptr);
    }

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
        write_viz_descriptors(render_target.view);

        g_framebuffer_resized = false;
    };

    // ---- Startup printout ---------------------------------------------------
    std::printf("drift_engine v0.2.1b — Vulkan up.\n");
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
    std::printf("Terrain: %ux%u R32_SFLOAT, min=%.1f max=%.1f center=%.1f corner=%.1f m\n",
                hm.width, hm.height, *hm_min, *hm_max,
                hm.values[static_cast<size_t>(hm.height / 2) * hm.width + hm.width / 2],
                hm.values[0]);
    std::printf("SWE: %ux%u RGBA16F, dx=%.1f m, init water=%.0f m\n",
                SWE_GRID_W, SWE_GRID_H, bp.cell_spacing, bp.initial_water);
    std::printf("Compute pipelines ready. Window open.\n");
    std::fflush(stdout);

    // ---- Frame loop ---------------------------------------------------------
    uint32_t current_frame = 0;
    uint32_t swe_ping_pong = 0;
    double last_time = glfwGetTime();
    const VkImageSubresourceRange color_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

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

        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        VkExtent2D extent = vkb_swapchain.extent;

        // ---- SWE step dispatch ----
        {
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swe_step_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    swe_step_pipeline_layout, 0, 1,
                                    &swe_step_desc_sets[swe_ping_pong], 0, nullptr);

            SweStepPC swe_pc{};
            swe_pc.time = static_cast<float>(now);
            swe_pc.dt = std::min(dt, 0.033f);
            swe_pc.gravity = 9.81f;
            swe_pc.friction = 0.01f;
            swe_pc.dx = bp.cell_spacing;
            swe_pc.sea_level = bp.floor_height + bp.initial_water;
            swe_pc.damping = 0.001f;
            swe_pc._pad0 = 0.0f;
            swe_pc.grid_w = SWE_GRID_W;
            swe_pc.grid_h = SWE_GRID_H;
            swe_pc.pulse_x = SWE_GRID_W * 0.5f;
            swe_pc.pulse_y = SWE_GRID_H * 0.5f;
            swe_pc.pulse_radius = 30.0f;
            swe_pc.pulse_amount = g_pulse_pending ? 50.0f : 0.0f;
            g_pulse_pending = false;
            vkCmdPushConstants(frame.cmd, swe_step_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(swe_pc), &swe_pc);

            vkCmdDispatch(frame.cmd, (SWE_GRID_W + 7) / 8, (SWE_GRID_H + 7) / 8, 1);

            swe_ping_pong ^= 1;
        }

        // Barrier: SWE write → visualize read
        {
            VkMemoryBarrier2 mem_bar{};
            mem_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mem_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mem_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mem_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mem_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                  | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                  | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &mem_bar;
            vkCmdPipelineBarrier2(frame.cmd, &dep);
        }

        // ---- Visualize dispatch ----
        VkImageMemoryBarrier2 barrier_to_general{};
        barrier_to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier_to_general.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier_to_general.srcAccessMask = VK_ACCESS_2_NONE;
        barrier_to_general.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier_to_general.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier_to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier_to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier_to_general.image = render_target.image;
        barrier_to_general.subresourceRange = color_range;

        VkDependencyInfo dep_to_general{};
        dep_to_general.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_to_general.imageMemoryBarrierCount = 1;
        dep_to_general.pImageMemoryBarriers = &barrier_to_general;
        vkCmdPipelineBarrier2(frame.cmd, &dep_to_general);

        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, viz_pipeline);
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                viz_pipeline_layout, 0, 1, &viz_desc_set, 0, nullptr);

        PushConstants pc{};
        pc.time = static_cast<float>(now);
        pc.width = extent.width;
        pc.height = extent.height;
        pc.max_elevation = 2000.0f;
        vkCmdPushConstants(frame.cmd, viz_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        vkCmdDispatch(frame.cmd,
                      (extent.width + 15) / 16,
                      (extent.height + 15) / 16,
                      1);

        // ---- Blit to swapchain ----
        VkImageMemoryBarrier2 barrier_rt_to_src{};
        barrier_rt_to_src.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier_rt_to_src.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier_rt_to_src.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier_rt_to_src.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        barrier_rt_to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier_rt_to_src.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier_rt_to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier_rt_to_src.image = render_target.image;
        barrier_rt_to_src.subresourceRange = color_range;

        VkImageMemoryBarrier2 barrier_sc_to_dst{};
        barrier_sc_to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier_sc_to_dst.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier_sc_to_dst.srcAccessMask = VK_ACCESS_2_NONE;
        barrier_sc_to_dst.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        barrier_sc_to_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier_sc_to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier_sc_to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_sc_to_dst.image = swapchain_images[image_index];
        barrier_sc_to_dst.subresourceRange = color_range;

        VkImageMemoryBarrier2 blit_barriers[] = {barrier_rt_to_src, barrier_sc_to_dst};
        VkDependencyInfo dep_blit{};
        dep_blit.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_blit.imageMemoryBarrierCount = 2;
        dep_blit.pImageMemoryBarriers = blit_barriers;
        vkCmdPipelineBarrier2(frame.cmd, &dep_blit);

        VkImageBlit2 region{};
        region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[0] = {0, 0, 0};
        region.srcOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstOffsets[0] = {0, 0, 0};
        region.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};

        VkBlitImageInfo2 blit_info{};
        blit_info.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
        blit_info.srcImage = render_target.image;
        blit_info.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blit_info.dstImage = swapchain_images[image_index];
        blit_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blit_info.regionCount = 1;
        blit_info.pRegions = &region;
        blit_info.filter = VK_FILTER_NEAREST;

        vkCmdBlitImage2(frame.cmd, &blit_info);

        VkImageMemoryBarrier2 barrier_to_present{};
        barrier_to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier_to_present.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        barrier_to_present.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier_to_present.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier_to_present.dstAccessMask = VK_ACCESS_2_NONE;
        barrier_to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier_to_present.image = swapchain_images[image_index];
        barrier_to_present.subresourceRange = color_range;

        VkDependencyInfo dep_to_present{};
        dep_to_present.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_to_present.imageMemoryBarrierCount = 1;
        dep_to_present.pImageMemoryBarriers = &barrier_to_present;
        vkCmdPipelineBarrier2(frame.cmd, &dep_to_present);

        VK_CHECK(vkEndCommandBuffer(frame.cmd));

        VkSemaphoreSubmitInfo wait_sem{};
        wait_sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        wait_sem.semaphore = frame.image_available;
        wait_sem.stageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;

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

    vkDestroyPipeline(device, swe_step_pipeline, nullptr);
    vkDestroyPipelineLayout(device, swe_step_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, swe_step_desc_layout, nullptr);
    vkDestroyShaderModule(device, swe_step_shader, nullptr);

    vkDestroyPipeline(device, swe_init_pipeline, nullptr);
    vkDestroyPipelineLayout(device, swe_init_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, swe_init_desc_layout, nullptr);
    vkDestroyShaderModule(device, swe_init_shader, nullptr);

    vkDestroyPipeline(device, viz_pipeline, nullptr);
    vkDestroyPipelineLayout(device, viz_pipeline_layout, nullptr);
    vkDestroyDescriptorPool(device, desc_pool, nullptr);
    vkDestroyDescriptorSetLayout(device, viz_desc_layout, nullptr);
    vkDestroyShaderModule(device, viz_shader, nullptr);

    destroy_render_target(device, allocator, render_target);

    destroy_swe_image(device, allocator, swe_output);
    destroy_swe_image(device, allocator, swe_state_b);
    destroy_swe_image(device, allocator, swe_state_a);

    vkDestroySampler(device, sampler, nullptr);
    vkDestroyImageView(device, heightmap_gpu.view, nullptr);
    vmaDestroyImage(allocator, heightmap_gpu.image, heightmap_gpu.allocation);

    vmaDestroyAllocator(allocator);

    vkb::destroy_device(vkb_device);
    vkDestroySurfaceKHR(vkb_inst.instance, surface, nullptr);
    vkb::destroy_instance(vkb_inst);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

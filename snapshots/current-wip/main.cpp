// Drift Engine — v0.4.1
//
// Capacity-based hydraulic erosion with sediment transport.

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

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
constexpr uint32_t MESH_RES = 512;

struct FrameData {
    VkCommandPool   pool;
    VkCommandBuffer cmd;
    VkSemaphore     image_available;
    VkSemaphore     render_finished;
    VkFence         in_flight;
};

struct DepthBuffer {
    VkImage       image;
    VmaAllocation allocation;
    VkImageView   view;
};

struct CameraData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 sun_dir;   float _pad0;
    glm::vec3 sun_color; float _pad1;
    glm::vec3 cam_pos;   float _pad2;
    glm::vec4 brush_world;
    glm::vec4 brush_color;
};

struct GfxPC {
    float terrain_size;
    float heightmap_texel;
    float max_elevation;
    float _pad;
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

struct TerrainBrushPC {
    float    brush_x;
    float    brush_y;
    float    brush_radius;
    float    brush_amount;
    uint32_t grid_w;
    uint32_t grid_h;
    uint32_t _pad0;
    uint32_t _pad1;
};

struct ErosionPC {
    float    dt;
    float    dx;
    uint32_t grid_w;
    uint32_t grid_h;
    float    k_erosion;
    float    k_deposit;
    float    k_capacity;
    float    min_slope;
    float    min_depth;
    float    max_change;
    float    max_sediment;
    uint32_t _pad1;
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

struct Camera {
    glm::vec3 position = glm::vec3(3000.0f, 2500.0f, 3000.0f);
    float yaw   = -2.356f;
    float pitch = -0.5f;
    float fov_y = glm::radians(60.0f);
    float near_plane = 1.0f;
    float far_plane  = 50000.0f;
};

enum class BrushMode { Raise, Lower, Water };

bool g_framebuffer_resized = false;
bool g_pulse_pending = false;
bool g_lmb_held = false;
bool g_rmb_held = false;
bool g_first_mouse = true;
double g_cursor_x = 0.0, g_cursor_y = 0.0;
double g_last_cursor_x = 0.0, g_last_cursor_y = 0.0;
float g_brush_radius_grid = 30.0f;
float g_brush_strength = 1.5f;
float g_terrain_strength = 50.0f;
bool g_cursor_on_world = false;
float g_cursor_world_x = 0.0f;
float g_cursor_world_z = 0.0f;
Camera g_camera;
BrushMode g_brush_mode = BrushMode::Water;

bool g_show_menu = true;
float g_gravity = 9.81f;
float g_friction = 0.01f;
float g_damping = 0.001f;
float g_time_scale = 1.0f;
float g_pulse_amount = 50.0f;
float g_pulse_radius_cells = 30.0f;
bool g_request_water_reset = false;
bool g_request_basin_reset = false;
bool g_erosion_enabled = false;
float g_k_erosion = 0.0005f;
float g_k_deposit = 0.001f;
float g_k_capacity = 0.01f;
float g_min_slope = 0.005f;
float g_max_change_m = 5.0f;
float g_min_erosion_depth = 0.01f;
float g_max_sediment = 0.5f;
float g_mud_visibility = 5.0f;
bool g_request_sediment_reset = false;

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_GRAVE_ACCENT && action == GLFW_PRESS)
        g_show_menu = !g_show_menu;
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard)
        return;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        g_pulse_pending = true;
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_1) g_brush_mode = BrushMode::Raise;
        if (key == GLFW_KEY_2) g_brush_mode = BrushMode::Lower;
        if (key == GLFW_KEY_3) g_brush_mode = BrushMode::Water;
    }
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_LEFT_BRACKET)
            g_brush_radius_grid = std::max(2.0f, g_brush_radius_grid * 0.85f);
        if (key == GLFW_KEY_RIGHT_BRACKET)
            g_brush_radius_grid = std::min(300.0f, g_brush_radius_grid * 1.18f);
        if (key == GLFW_KEY_MINUS) {
            if (g_brush_mode == BrushMode::Water)
                g_brush_strength = std::max(0.05f, g_brush_strength * 0.85f);
            else
                g_terrain_strength = std::max(2.0f, g_terrain_strength * 0.85f);
        }
        if (key == GLFW_KEY_EQUAL) {
            if (g_brush_mode == BrushMode::Water)
                g_brush_strength = std::min(20.0f, g_brush_strength * 1.18f);
            else
                g_terrain_strength = std::min(500.0f, g_terrain_strength * 1.18f);
        }
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse)
        return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_lmb_held = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_rmb_held = (action == GLFW_PRESS);
        if (g_rmb_held) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            g_first_mouse = true;
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

void cursor_pos_callback(GLFWwindow* /*window*/, double xpos, double ypos)
{
    g_cursor_x = xpos;
    g_cursor_y = ypos;

    if (g_rmb_held) {
        if (g_first_mouse) {
            g_last_cursor_x = xpos;
            g_last_cursor_y = ypos;
            g_first_mouse = false;
            return;
        }
        float sensitivity = 0.002f;
        float dx = static_cast<float>(xpos - g_last_cursor_x);
        float dy = static_cast<float>(ypos - g_last_cursor_y);
        g_camera.yaw   += dx * sensitivity;
        g_camera.pitch -= dy * sensitivity;
        g_camera.pitch = std::clamp(g_camera.pitch, -1.5f, 1.5f);
    }
    g_last_cursor_x = xpos;
    g_last_cursor_y = ypos;
}

void scroll_callback(GLFWwindow* /*window*/, double /*xoffset*/, double /*yoffset*/)
{
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
    float    initial_water = 50.0f;
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

DepthBuffer create_depth_buffer(VkDevice device, VmaAllocator allocator, VkExtent2D extent)
{
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = VK_FORMAT_D32_SFLOAT;
    img_ci.extent = {extent.width, extent.height, 1};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    DepthBuffer db{};
    VK_CHECK(vmaCreateImage(allocator, &img_ci, &alloc_ci, &db.image, &db.allocation, nullptr));

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = db.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_D32_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &view_ci, nullptr, &db.view));

    return db;
}

void destroy_depth_buffer(VkDevice device, VmaAllocator allocator, DepthBuffer& db)
{
    vkDestroyImageView(device, db.view, nullptr);
    vmaDestroyImage(allocator, db.image, db.allocation);
    db = {};
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

SweImage create_sediment_image(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h)
{
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = VK_FORMAT_R16_SFLOAT;
    img_ci.extent = {w, h, 1};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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
    view_ci.format = VK_FORMAT_R16_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &view_ci, nullptr, &img.view));

    return img;
}

HeightmapGPU upload_heightmap(VkDevice device, VmaAllocator allocator,
                              VkQueue queue, uint32_t queue_family,
                              const HeightmapData& hm)
{
    VkDeviceSize buf_size = static_cast<VkDeviceSize>(hm.width) * hm.height * sizeof(float);

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

    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = VK_FORMAT_R32_SFLOAT;
    img_ci.extent = {hm.width, hm.height, 1};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo gpu_ai{};
    gpu_ai.usage = VMA_MEMORY_USAGE_AUTO;
    gpu_ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    HeightmapGPU gpu{};
    VK_CHECK(vmaCreateImage(allocator, &img_ci, &gpu_ai, &gpu.image, &gpu.allocation, nullptr));

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

    VkImageMemoryBarrier2 barrier_to_read{};
    barrier_to_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier_to_read.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier_to_read.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier_to_read.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier_to_read.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barrier_to_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_to_read.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier_to_read.image = gpu.image;
    barrier_to_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep_to_read{};
    dep_to_read.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_to_read.imageMemoryBarrierCount = 1;
    dep_to_read.pImageMemoryBarriers = &barrier_to_read;
    vkCmdPipelineBarrier2(cmd, &dep_to_read);

    VK_CHECK(vkEndCommandBuffer(cmd));

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
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

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
    float terrain_size = static_cast<float>(bp.grid_w) * bp.cell_spacing;

    HeightmapGPU heightmap_gpu = upload_heightmap(device, allocator, graphics_queue, gfx_family, hm);

    // ---- Terrain mesh (VBO + IBO) -------------------------------------------
    std::vector<float> mesh_vertices;
    mesh_vertices.reserve(MESH_RES * MESH_RES * 2);
    for (uint32_t y = 0; y < MESH_RES; ++y) {
        for (uint32_t x = 0; x < MESH_RES; ++x) {
            mesh_vertices.push_back(static_cast<float>(x));
            mesh_vertices.push_back(static_cast<float>(y));
        }
    }

    std::vector<uint32_t> mesh_indices;
    mesh_indices.reserve((MESH_RES - 1) * (MESH_RES - 1) * 6);
    for (uint32_t y = 0; y < MESH_RES - 1; ++y) {
        for (uint32_t x = 0; x < MESH_RES - 1; ++x) {
            uint32_t i00 = y * MESH_RES + x;
            uint32_t i10 = y * MESH_RES + (x + 1);
            uint32_t i01 = (y + 1) * MESH_RES + x;
            uint32_t i11 = (y + 1) * MESH_RES + (x + 1);
            mesh_indices.push_back(i00);
            mesh_indices.push_back(i01);
            mesh_indices.push_back(i10);
            mesh_indices.push_back(i10);
            mesh_indices.push_back(i01);
            mesh_indices.push_back(i11);
        }
    }
    uint32_t index_count = static_cast<uint32_t>(mesh_indices.size());

    VkBufferCreateInfo vbo_ci{};
    vbo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vbo_ci.size = mesh_vertices.size() * sizeof(float);
    vbo_ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VmaAllocationCreateInfo vbo_ai{};
    vbo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    vbo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VmaAllocation vertex_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo vbo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &vbo_ci, &vbo_ai, &vertex_buffer, &vertex_alloc, &vbo_info));
    std::memcpy(vbo_info.pMappedData, mesh_vertices.data(), vbo_ci.size);
    vmaFlushAllocation(allocator, vertex_alloc, 0, VK_WHOLE_SIZE);

    VkBufferCreateInfo ibo_ci{};
    ibo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ibo_ci.size = mesh_indices.size() * sizeof(uint32_t);
    ibo_ci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    VmaAllocationCreateInfo ibo_ai{};
    ibo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    ibo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer index_buffer = VK_NULL_HANDLE;
    VmaAllocation index_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo ibo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &ibo_ci, &ibo_ai, &index_buffer, &index_alloc, &ibo_info));
    std::memcpy(ibo_info.pMappedData, mesh_indices.data(), ibo_ci.size);
    vmaFlushAllocation(allocator, index_alloc, 0, VK_WHOLE_SIZE);

    // ---- Camera UBO ---------------------------------------------------------
    VkBufferCreateInfo ubo_ci{};
    ubo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ubo_ci.size = sizeof(CameraData);
    ubo_ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo ubo_ai{};
    ubo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    ubo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer camera_ubo = VK_NULL_HANDLE;
    VmaAllocation camera_ubo_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo camera_ubo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &ubo_ci, &ubo_ai, &camera_ubo, &camera_ubo_alloc, &camera_ubo_info));

    // ---- SWE images (ping-pong state + output) ------------------------------
    SweImage swe_state_a = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    SweImage swe_state_b = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    SweImage swe_output  = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

    // ---- Sediment images (ping-pong) ----------------------------------------
    SweImage sediment_a = create_sediment_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    SweImage sediment_b = create_sediment_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

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

    // ---- Depth buffer -------------------------------------------------------
    DepthBuffer depth_buffer = create_depth_buffer(device, allocator, vkb_swapchain.extent);

    // Camera UBO is written per-frame now (perspective + free camera).

    // ---- ImGui setup --------------------------------------------------------
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(window, true);

    VkDescriptorPoolSize imgui_pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000}
    };
    VkDescriptorPoolCreateInfo imgui_dp_ci{};
    imgui_dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    imgui_dp_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    imgui_dp_ci.maxSets = 1000;
    imgui_dp_ci.poolSizeCount = 1;
    imgui_dp_ci.pPoolSizes = imgui_pool_sizes;

    VkDescriptorPool imgui_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(device, &imgui_dp_ci, nullptr, &imgui_pool));

    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkPipelineRenderingCreateInfo imgui_rendering_ci{};
    imgui_rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    imgui_rendering_ci.colorAttachmentCount = 1;
    imgui_rendering_ci.pColorAttachmentFormats = &swapchain_format;
    imgui_rendering_ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    ImGui_ImplVulkan_InitInfo imgui_init{};
    imgui_init.Instance = vkb_inst.instance;
    imgui_init.PhysicalDevice = vkb_phys.physical_device;
    imgui_init.Device = device;
    imgui_init.QueueFamily = gfx_family;
    imgui_init.Queue = graphics_queue;
    imgui_init.DescriptorPool = imgui_pool;
    imgui_init.MinImageCount = 2;
    imgui_init.ImageCount = static_cast<uint32_t>(swapchain_images.size());
    imgui_init.UseDynamicRendering = true;
    imgui_init.PipelineRenderingCreateInfo = imgui_rendering_ci;
    ImGui_ImplVulkan_Init(&imgui_init);

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

    // ---- Graphics pipeline (terrain) ----------------------------------------
    auto terrain_vs_spirv = load_spirv("shaders/terrain_vs.spv");
    auto terrain_fs_spirv = load_spirv("shaders/terrain_fs.spv");

    VkShaderModuleCreateInfo vs_sm_ci{};
    vs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vs_sm_ci.codeSize = terrain_vs_spirv.size() * sizeof(uint32_t);
    vs_sm_ci.pCode = terrain_vs_spirv.data();

    VkShaderModule terrain_vs = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &vs_sm_ci, nullptr, &terrain_vs));

    VkShaderModuleCreateInfo fs_sm_ci{};
    fs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fs_sm_ci.codeSize = terrain_fs_spirv.size() * sizeof(uint32_t);
    fs_sm_ci.pCode = terrain_fs_spirv.data();

    VkShaderModule terrain_fs = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &fs_sm_ci, nullptr, &terrain_fs));

    // Graphics descriptor set layout: UBO + heightmap sampler + swe_output sampler
    VkDescriptorSetLayoutBinding gfx_bindings[4]{};
    gfx_bindings[0].binding = 0;
    gfx_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    gfx_bindings[0].descriptorCount = 1;
    gfx_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[1].binding = 1;
    gfx_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[1].descriptorCount = 1;
    gfx_bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[2].binding = 2;
    gfx_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[2].descriptorCount = 1;
    gfx_bindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[3].binding = 3;
    gfx_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[3].descriptorCount = 1;
    gfx_bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo gfx_dsl_ci{};
    gfx_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    gfx_dsl_ci.bindingCount = 4;
    gfx_dsl_ci.pBindings = gfx_bindings;

    VkDescriptorSetLayout gfx_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &gfx_dsl_ci, nullptr, &gfx_desc_layout));

    VkPushConstantRange gfx_push_range{};
    gfx_push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    gfx_push_range.offset = 0;
    gfx_push_range.size = sizeof(GfxPC);

    VkPipelineLayoutCreateInfo gfx_pl_ci{};
    gfx_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_pl_ci.setLayoutCount = 1;
    gfx_pl_ci.pSetLayouts = &gfx_desc_layout;
    gfx_pl_ci.pushConstantRangeCount = 1;
    gfx_pl_ci.pPushConstantRanges = &gfx_push_range;

    VkPipelineLayout gfx_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &gfx_pl_ci, nullptr, &gfx_pipeline_layout));

    VkPipelineShaderStageCreateInfo gfx_stages[2]{};
    gfx_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    gfx_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    gfx_stages[0].module = terrain_vs;
    gfx_stages[0].pName = "main";

    gfx_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    gfx_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    gfx_stages[1].module = terrain_fs;
    gfx_stages[1].pName = "main";

    VkVertexInputBindingDescription vertex_binding{};
    vertex_binding.binding = 0;
    vertex_binding.stride = sizeof(float) * 2;
    vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vertex_attr{};
    vertex_attr.binding = 0;
    vertex_attr.location = 0;
    vertex_attr.format = VK_FORMAT_R32G32_SFLOAT;
    vertex_attr.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &vertex_binding;
    vertex_input.vertexAttributeDescriptionCount = 1;
    vertex_input.pVertexAttributeDescriptions = &vertex_attr;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                    | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &blend_attachment;

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkFormat color_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    VkPipelineRenderingCreateInfo rendering_ci{};
    rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_ci.colorAttachmentCount = 1;
    rendering_ci.pColorAttachmentFormats = &color_format;
    rendering_ci.depthAttachmentFormat = depth_format;

    VkGraphicsPipelineCreateInfo gfx_pipe_ci{};
    gfx_pipe_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_pipe_ci.pNext = &rendering_ci;
    gfx_pipe_ci.stageCount = 2;
    gfx_pipe_ci.pStages = gfx_stages;
    gfx_pipe_ci.pVertexInputState = &vertex_input;
    gfx_pipe_ci.pInputAssemblyState = &input_assembly;
    gfx_pipe_ci.pViewportState = &viewport_state;
    gfx_pipe_ci.pRasterizationState = &rasterizer;
    gfx_pipe_ci.pMultisampleState = &multisample;
    gfx_pipe_ci.pDepthStencilState = &depth_stencil;
    gfx_pipe_ci.pColorBlendState = &color_blend;
    gfx_pipe_ci.pDynamicState = &dynamic_state;
    gfx_pipe_ci.layout = gfx_pipeline_layout;

    VkPipeline gfx_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gfx_pipe_ci, nullptr, &gfx_pipeline));

    // ---- Water pipeline (transparent, same layout) --------------------------
    auto water_vs_spirv = load_spirv("shaders/water_vs.spv");
    auto water_fs_spirv = load_spirv("shaders/water_fs.spv");

    VkShaderModuleCreateInfo water_vs_sm_ci{};
    water_vs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    water_vs_sm_ci.codeSize = water_vs_spirv.size() * sizeof(uint32_t);
    water_vs_sm_ci.pCode = water_vs_spirv.data();

    VkShaderModule water_vs = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &water_vs_sm_ci, nullptr, &water_vs));

    VkShaderModuleCreateInfo water_fs_sm_ci{};
    water_fs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    water_fs_sm_ci.codeSize = water_fs_spirv.size() * sizeof(uint32_t);
    water_fs_sm_ci.pCode = water_fs_spirv.data();

    VkShaderModule water_fs = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &water_fs_sm_ci, nullptr, &water_fs));

    VkPipelineShaderStageCreateInfo water_stages[2]{};
    water_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    water_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    water_stages[0].module = water_vs;
    water_stages[0].pName = "main";

    water_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    water_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    water_stages[1].module = water_fs;
    water_stages[1].pName = "main";

    VkPipelineDepthStencilStateCreateInfo water_depth_stencil{};
    water_depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    water_depth_stencil.depthTestEnable = VK_TRUE;
    water_depth_stencil.depthWriteEnable = VK_FALSE;
    water_depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState water_blend_attachment{};
    water_blend_attachment.blendEnable = VK_TRUE;
    water_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    water_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    water_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    water_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    water_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    water_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    water_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo water_color_blend{};
    water_color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    water_color_blend.attachmentCount = 1;
    water_color_blend.pAttachments = &water_blend_attachment;

    VkGraphicsPipelineCreateInfo water_pipe_ci = gfx_pipe_ci;
    water_pipe_ci.pStages = water_stages;
    water_pipe_ci.pDepthStencilState = &water_depth_stencil;
    water_pipe_ci.pColorBlendState = &water_color_blend;

    VkPipeline water_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &water_pipe_ci, nullptr, &water_pipeline));

    // ---- Terrain brush compute pipeline -------------------------------------
    auto terrain_brush_spirv = load_spirv("shaders/terrain_brush.spv");

    VkShaderModuleCreateInfo tb_sm_ci{};
    tb_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    tb_sm_ci.codeSize = terrain_brush_spirv.size() * sizeof(uint32_t);
    tb_sm_ci.pCode = terrain_brush_spirv.data();

    VkShaderModule terrain_brush_shader = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &tb_sm_ci, nullptr, &terrain_brush_shader));

    VkDescriptorSetLayoutBinding tb_binding{};
    tb_binding.binding = 0;
    tb_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    tb_binding.descriptorCount = 1;
    tb_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo tb_dsl_ci{};
    tb_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    tb_dsl_ci.bindingCount = 1;
    tb_dsl_ci.pBindings = &tb_binding;

    VkDescriptorSetLayout tb_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &tb_dsl_ci, nullptr, &tb_desc_layout));

    VkPushConstantRange tb_push{};
    tb_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tb_push.offset = 0;
    tb_push.size = sizeof(TerrainBrushPC);

    VkPipelineLayoutCreateInfo tb_pl_ci{};
    tb_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tb_pl_ci.setLayoutCount = 1;
    tb_pl_ci.pSetLayouts = &tb_desc_layout;
    tb_pl_ci.pushConstantRangeCount = 1;
    tb_pl_ci.pPushConstantRanges = &tb_push;

    VkPipelineLayout tb_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &tb_pl_ci, nullptr, &tb_pipeline_layout));

    VkPipelineShaderStageCreateInfo tb_stage{};
    tb_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    tb_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    tb_stage.module = terrain_brush_shader;
    tb_stage.pName = "main";

    VkComputePipelineCreateInfo tb_cp_ci{};
    tb_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    tb_cp_ci.stage = tb_stage;
    tb_cp_ci.layout = tb_pipeline_layout;

    VkPipeline terrain_brush_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &tb_cp_ci, nullptr, &terrain_brush_pipeline));

    // ---- Erosion compute pipeline -------------------------------------------
    auto erosion_spirv = load_spirv("shaders/erosion.spv");

    VkShaderModuleCreateInfo ero_sm_ci{};
    ero_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ero_sm_ci.codeSize = erosion_spirv.size() * sizeof(uint32_t);
    ero_sm_ci.pCode = erosion_spirv.data();

    VkShaderModule erosion_shader = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &ero_sm_ci, nullptr, &erosion_shader));

    VkDescriptorSetLayoutBinding ero_bindings[4]{};
    ero_bindings[0].binding = 0;
    ero_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ero_bindings[0].descriptorCount = 1;
    ero_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    ero_bindings[1].binding = 1;
    ero_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    ero_bindings[1].descriptorCount = 1;
    ero_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    ero_bindings[2].binding = 2;
    ero_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    ero_bindings[2].descriptorCount = 1;
    ero_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    ero_bindings[3].binding = 3;
    ero_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ero_bindings[3].descriptorCount = 1;
    ero_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo ero_dsl_ci{};
    ero_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ero_dsl_ci.bindingCount = 4;
    ero_dsl_ci.pBindings = ero_bindings;

    VkDescriptorSetLayout ero_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &ero_dsl_ci, nullptr, &ero_desc_layout));

    VkPushConstantRange ero_push{};
    ero_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ero_push.offset = 0;
    ero_push.size = sizeof(ErosionPC);

    VkPipelineLayoutCreateInfo ero_pl_ci{};
    ero_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ero_pl_ci.setLayoutCount = 1;
    ero_pl_ci.pSetLayouts = &ero_desc_layout;
    ero_pl_ci.pushConstantRangeCount = 1;
    ero_pl_ci.pPushConstantRanges = &ero_push;

    VkPipelineLayout ero_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &ero_pl_ci, nullptr, &ero_pipeline_layout));

    VkPipelineShaderStageCreateInfo ero_stage{};
    ero_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ero_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ero_stage.module = erosion_shader;
    ero_stage.pName = "main";

    VkComputePipelineCreateInfo ero_cp_ci{};
    ero_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ero_cp_ci.stage = ero_stage;
    ero_cp_ci.layout = ero_pipeline_layout;

    VkPipeline erosion_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ero_cp_ci, nullptr, &erosion_pipeline));

    // ---- Descriptor pool + sets ---------------------------------------------
    // Counts: SWE init(2) + SWE step×2(8) + terrain brush(1) + gfx(4 incl sediment) + erosion×2(8) = needs ~23 descriptors
    VkDescriptorPoolSize pool_sizes[4]{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 12;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[1].descriptorCount = 10;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[2].descriptorCount = 1;
    pool_sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[3].descriptorCount = 3;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 8;
    dp_ci.poolSizeCount = 4;
    dp_ci.pPoolSizes = pool_sizes;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(device, &dp_ci, nullptr, &desc_pool));

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

    // Graphics descriptor set
    VkDescriptorSetAllocateInfo gfx_ds_ai{};
    gfx_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    gfx_ds_ai.descriptorPool = desc_pool;
    gfx_ds_ai.descriptorSetCount = 1;
    gfx_ds_ai.pSetLayouts = &gfx_desc_layout;

    VkDescriptorSet gfx_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &gfx_ds_ai, &gfx_desc_set));

    // Terrain brush descriptor set
    VkDescriptorSetAllocateInfo tb_ds_ai{};
    tb_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    tb_ds_ai.descriptorPool = desc_pool;
    tb_ds_ai.descriptorSetCount = 1;
    tb_ds_ai.pSetLayouts = &tb_desc_layout;

    VkDescriptorSet tb_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &tb_ds_ai, &tb_desc_set));

    // Erosion descriptor sets (2 for ping-pong)
    VkDescriptorSetLayout ero_layouts[2] = {ero_desc_layout, ero_desc_layout};
    VkDescriptorSetAllocateInfo ero_ds_ai{};
    ero_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ero_ds_ai.descriptorPool = desc_pool;
    ero_ds_ai.descriptorSetCount = 2;
    ero_ds_ai.pSetLayouts = ero_layouts;

    VkDescriptorSet ero_desc_sets[2] = {};
    VK_CHECK(vkAllocateDescriptorSets(device, &ero_ds_ai, ero_desc_sets));

    // ---- Write descriptor sets ----------------------------------------------
    // SWE init descriptors
    {
        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = heightmap_gpu.view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

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

    // SWE step descriptors
    {
        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = heightmap_gpu.view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_a_info{};
        state_a_info.imageView = swe_state_a.view;
        state_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_b_info{};
        state_b_info.imageView = swe_state_b.view;
        state_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo output_info{};
        output_info.imageView = swe_output.view;
        output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Set 0: read A -> write B
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
        writes_0[1].pImageInfo = &state_a_info;

        writes_0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[2].dstSet = swe_step_desc_sets[0];
        writes_0[2].dstBinding = 2;
        writes_0[2].descriptorCount = 1;
        writes_0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_0[2].pImageInfo = &state_b_info;

        writes_0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[3].dstSet = swe_step_desc_sets[0];
        writes_0[3].dstBinding = 3;
        writes_0[3].descriptorCount = 1;
        writes_0[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_0[3].pImageInfo = &output_info;

        vkUpdateDescriptorSets(device, 4, writes_0, 0, nullptr);

        // Set 1: read B -> write A
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
        writes_1[1].pImageInfo = &state_b_info;

        writes_1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[2].dstSet = swe_step_desc_sets[1];
        writes_1[2].dstBinding = 2;
        writes_1[2].descriptorCount = 1;
        writes_1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_1[2].pImageInfo = &state_a_info;

        writes_1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[3].dstSet = swe_step_desc_sets[1];
        writes_1[3].dstBinding = 3;
        writes_1[3].descriptorCount = 1;
        writes_1[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_1[3].pImageInfo = &output_info;

        vkUpdateDescriptorSets(device, 4, writes_1, 0, nullptr);
    }

    // Graphics descriptors
    {
        VkDescriptorBufferInfo ubo_buf_info{};
        ubo_buf_info.buffer = camera_ubo;
        ubo_buf_info.offset = 0;
        ubo_buf_info.range = sizeof(CameraData);

        VkDescriptorImageInfo heightmap_info{};
        heightmap_info.imageView = heightmap_gpu.view;
        heightmap_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        heightmap_info.sampler = sampler;

        VkDescriptorImageInfo swe_out_info{};
        swe_out_info.imageView = swe_output.view;
        swe_out_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        swe_out_info.sampler = sampler;

        VkDescriptorImageInfo sediment_info{};
        sediment_info.imageView = sediment_a.view;
        sediment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sediment_info.sampler = sampler;

        VkWriteDescriptorSet writes[4]{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = gfx_desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &ubo_buf_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = gfx_desc_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &heightmap_info;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = gfx_desc_set;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &swe_out_info;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = gfx_desc_set;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &sediment_info;

        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    }

    // Terrain brush descriptor
    {
        VkDescriptorImageInfo tb_img_info{};
        tb_img_info.imageView = heightmap_gpu.view;
        tb_img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet tb_write{};
        tb_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        tb_write.dstSet = tb_desc_set;
        tb_write.dstBinding = 0;
        tb_write.descriptorCount = 1;
        tb_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        tb_write.pImageInfo = &tb_img_info;

        vkUpdateDescriptorSets(device, 1, &tb_write, 0, nullptr);
    }

    // Erosion descriptors (2 sets for sediment ping-pong, always read swe_state_a)
    {
        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = heightmap_gpu.view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo swe_info{};
        swe_info.imageView = swe_state_a.view;
        swe_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo sed_a_info{};
        sed_a_info.imageView = sediment_a.view;
        sed_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo sed_b_info{};
        sed_b_info.imageView = sediment_b.view;
        sed_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Set 0: read sediment_a -> write sediment_b
        VkWriteDescriptorSet w0[4]{};
        w0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[0].dstSet = ero_desc_sets[0];
        w0[0].dstBinding = 0;
        w0[0].descriptorCount = 1;
        w0[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0[0].pImageInfo = &terrain_info;

        w0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[1].dstSet = ero_desc_sets[0];
        w0[1].dstBinding = 1;
        w0[1].descriptorCount = 1;
        w0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[1].pImageInfo = &swe_info;

        w0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[2].dstSet = ero_desc_sets[0];
        w0[2].dstBinding = 2;
        w0[2].descriptorCount = 1;
        w0[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[2].pImageInfo = &sed_a_info;

        w0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[3].dstSet = ero_desc_sets[0];
        w0[3].dstBinding = 3;
        w0[3].descriptorCount = 1;
        w0[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0[3].pImageInfo = &sed_b_info;

        vkUpdateDescriptorSets(device, 4, w0, 0, nullptr);

        // Set 1: read sediment_b -> write sediment_a
        VkWriteDescriptorSet w1[4]{};
        w1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[0].dstSet = ero_desc_sets[1];
        w1[0].dstBinding = 0;
        w1[0].descriptorCount = 1;
        w1[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1[0].pImageInfo = &terrain_info;

        w1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[1].dstSet = ero_desc_sets[1];
        w1[1].dstBinding = 1;
        w1[1].descriptorCount = 1;
        w1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[1].pImageInfo = &swe_info;

        w1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[2].dstSet = ero_desc_sets[1];
        w1[2].dstBinding = 2;
        w1[2].descriptorCount = 1;
        w1[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[2].pImageInfo = &sed_b_info;

        w1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[3].dstSet = ero_desc_sets[1];
        w1[3].dstBinding = 3;
        w1[3].descriptorCount = 1;
        w1[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1[3].pImageInfo = &sed_a_info;

        vkUpdateDescriptorSets(device, 4, w1, 0, nullptr);
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

        VkImageMemoryBarrier2 init_barriers[5]{};
        for (int i = 0; i < 5; ++i) {
            init_barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            init_barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            init_barriers[i].srcAccessMask = VK_ACCESS_2_NONE;
            init_barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            init_barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            init_barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        for (int i = 0; i < 3; ++i) {
            init_barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            init_barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        }
        init_barriers[0].image = swe_state_a.image;
        init_barriers[1].image = swe_state_b.image;
        init_barriers[2].image = swe_output.image;

        for (int i = 3; i < 5; ++i) {
            init_barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            init_barriers[i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        }
        init_barriers[3].image = sediment_a.image;
        init_barriers[4].image = sediment_b.image;

        VkDependencyInfo dep_init{};
        dep_init.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_init.imageMemoryBarrierCount = 5;
        dep_init.pImageMemoryBarriers = init_barriers;
        vkCmdPipelineBarrier2(cmd, &dep_init);

        VkClearColorValue clear_zero{};
        clear_zero.float32[0] = 0.0f;
        VkImageSubresourceRange clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, sediment_a.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
        vkCmdClearColorImage(cmd, sediment_b.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);

        VkImageMemoryBarrier2 sed_after_clear[2]{};
        for (int i = 0; i < 2; ++i) {
            sed_after_clear[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            sed_after_clear[i].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            sed_after_clear[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            sed_after_clear[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                            | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            sed_after_clear[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            sed_after_clear[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            sed_after_clear[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            sed_after_clear[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        sed_after_clear[0].image = sediment_a.image;
        sed_after_clear[1].image = sediment_b.image;

        VkDependencyInfo dep_sed_clear{};
        dep_sed_clear.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_sed_clear.imageMemoryBarrierCount = 2;
        dep_sed_clear.pImageMemoryBarriers = sed_after_clear;
        vkCmdPipelineBarrier2(cmd, &dep_sed_clear);

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

    // ---- Timestamp query pool -----------------------------------------------
    VkQueryPoolCreateInfo qp_ci{};
    qp_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qp_ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qp_ci.queryCount = 2 * FRAMES_IN_FLIGHT;

    VkQueryPool query_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateQueryPool(device, &qp_ci, nullptr, &query_pool));

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

        destroy_depth_buffer(device, allocator, depth_buffer);
        depth_buffer = create_depth_buffer(device, allocator, vkb_swapchain.extent);

        g_framebuffer_resized = false;
    };

    // ---- Startup printout ---------------------------------------------------
    std::printf("drift_engine v0.4.1 — Vulkan up.\n");
    std::printf("Device:   %s\n", vkb_phys.name.c_str());
    std::printf("Queues:   graphics=%u  present=%u\n",
                gfx_family,
                vkb_device.get_queue_index(vkb::QueueType::present).value());
    std::printf("Swapchain: %ux%u, %u images\n",
                vkb_swapchain.extent.width, vkb_swapchain.extent.height,
                static_cast<uint32_t>(swapchain_images.size()));
    std::printf("Terrain: %ux%u R32_SFLOAT, min=%.1f max=%.1f\n",
                hm.width, hm.height, *hm_min, *hm_max);
    std::printf("Mesh: %ux%u vertices, %u triangles\n",
                MESH_RES, MESH_RES, index_count / 3);
    std::printf("SWE: %ux%u RGBA16F, dx=%.1f m, init water=%.0f m\n",
                SWE_GRID_W, SWE_GRID_H, bp.cell_spacing, bp.initial_water);
    std::printf("Graphics: perspective free-cam (RMB look, WASD move, Shift fast, Alt slow)\n");
    std::fflush(stdout);

    // ---- Frame loop ---------------------------------------------------------
    uint32_t current_frame = 0;
    uint32_t swe_ping_pong = 0;
    uint32_t sediment_ping_pong = 0;
    double last_time = glfwGetTime();

    constexpr int AVG_FRAMES = 30;
    double cpu_times[AVG_FRAMES] = {};
    double gpu_times[AVG_FRAMES] = {};
    int timing_index = 0;
    int timing_count = 0;
    double last_title_update = 0.0;
    double cpu_avg_ms = 0.0, gpu_avg_ms = 0.0;
    double ns_per_tick = static_cast<double>(vkb_phys.properties.limits.timestampPeriod);
    bool queries_valid = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        if (fb_w == 0 || fb_h == 0)
            continue;

        FrameData& frame = frames[current_frame];

        VK_CHECK(vkWaitForFences(device, 1, &frame.in_flight, VK_TRUE, UINT64_MAX));

        if (queries_valid) {
            uint64_t timestamps[2] = {};
            VkResult qr = vkGetQueryPoolResults(device, query_pool,
                current_frame * 2, 2,
                sizeof(timestamps), timestamps, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT);
            if (qr == VK_SUCCESS) {
                double swe_ms = static_cast<double>(timestamps[1] - timestamps[0]) * ns_per_tick / 1e6;
                gpu_times[timing_index] = swe_ms;
            }
        }

        double frame_start = glfwGetTime();

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

        // ---- ImGui frame ----
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (g_show_menu) {
            ImGui::Begin("drift_engine", &g_show_menu);

            if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("CPU: %.2f ms (%.1f fps)", cpu_avg_ms, 1000.0 / std::max(cpu_avg_ms, 0.001));
                ImGui::Text("GPU SWE: %.2f ms", gpu_avg_ms);
            }

            if (ImGui::CollapsingHeader("Water Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Gravity", &g_gravity, 0.5f, 30.0f);
                ImGui::SliderFloat("Friction", &g_friction, 0.0f, 0.05f, "%.4f");
                ImGui::SliderFloat("Damping", &g_damping, 0.0f, 0.05f, "%.4f");
                ImGui::SliderFloat("Time scale", &g_time_scale, 0.0f, 5.0f);
                if (ImGui::Button("Reset water")) g_request_water_reset = true;
            }

            if (ImGui::CollapsingHeader("Brushes", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Brush radius (cells)", &g_brush_radius_grid, 2.0f, 300.0f);
                ImGui::SliderFloat("Water strength", &g_brush_strength, 0.0f, 20.0f);
                ImGui::SliderFloat("Terrain strength m/s", &g_terrain_strength, 1.0f, 500.0f);
                ImGui::SliderFloat("Pulse amount (m)", &g_pulse_amount, 1.0f, 200.0f);
                ImGui::SliderFloat("Pulse radius (cells)", &g_pulse_radius_cells, 5.0f, 200.0f);
            }

            if (ImGui::CollapsingHeader("Erosion", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Erosion enabled", &g_erosion_enabled);
                if (g_erosion_enabled) {
                    ImGui::SliderFloat("k_erosion", &g_k_erosion, 0.0f, 0.01f, "%.5f");
                    ImGui::SliderFloat("k_deposit", &g_k_deposit, 0.0f, 0.01f, "%.5f");
                    ImGui::SliderFloat("k_capacity", &g_k_capacity, 0.0f, 0.1f, "%.4f");
                    ImGui::SliderFloat("min_slope", &g_min_slope, 0.0001f, 0.05f, "%.4f");
                    ImGui::SliderFloat("max_change m/s", &g_max_change_m, 0.1f, 20.0f, "%.1f");
                    ImGui::SliderFloat("min depth", &g_min_erosion_depth, 0.001f, 0.5f, "%.3f");
                    ImGui::SliderFloat("max sediment", &g_max_sediment, 0.05f, 5.0f, "%.2f");
                }
                ImGui::SliderFloat("Mud visibility", &g_mud_visibility, 0.0f, 20.0f);
                if (ImGui::Button("Reset sediment")) g_request_sediment_reset = true;
            }

            if (ImGui::CollapsingHeader("Scene")) {
                if (ImGui::Button("Regenerate basin")) g_request_basin_reset = true;
                ImGui::TextDisabled("Pulse: SPACE | Modes: 1=raise 2=lower 3=water");
                ImGui::TextDisabled("Look: RMB+drag | Move: WASD+QE | Menu: `");
            }
            ImGui::End();
        }

        ImGui::Render();

        // ---- Reset handlers (heavy, stall GPU) ----
        if (g_request_basin_reset || g_request_water_reset) {
            vkDeviceWaitIdle(device);

            if (g_request_basin_reset) {
                auto new_basin = generate_crater_basin(bp);
                VkDeviceSize buf_size = static_cast<VkDeviceSize>(bp.grid_w) * bp.grid_h * sizeof(float);

                VkBufferCreateInfo stg_ci{};
                stg_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                stg_ci.size = buf_size;
                stg_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                VmaAllocationCreateInfo stg_ai{};
                stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
                stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                             | VMA_ALLOCATION_CREATE_MAPPED_BIT;
                VkBuffer stg_buf = VK_NULL_HANDLE;
                VmaAllocation stg_alloc = VK_NULL_HANDLE;
                VmaAllocationInfo stg_info{};
                VK_CHECK(vmaCreateBuffer(allocator, &stg_ci, &stg_ai, &stg_buf, &stg_alloc, &stg_info));
                std::memcpy(stg_info.pMappedData, new_basin.data(), buf_size);
                vmaFlushAllocation(allocator, stg_alloc, 0, VK_WHOLE_SIZE);

                VkCommandPoolCreateInfo tmp_pool_ci{};
                tmp_pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                tmp_pool_ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
                tmp_pool_ci.queueFamilyIndex = gfx_family;
                VkCommandPool tmp_pool = VK_NULL_HANDLE;
                VK_CHECK(vkCreateCommandPool(device, &tmp_pool_ci, nullptr, &tmp_pool));

                VkCommandBufferAllocateInfo tmp_cmd_ai{};
                tmp_cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                tmp_cmd_ai.commandPool = tmp_pool;
                tmp_cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                tmp_cmd_ai.commandBufferCount = 1;
                VkCommandBuffer tmp_cmd = VK_NULL_HANDLE;
                VK_CHECK(vkAllocateCommandBuffers(device, &tmp_cmd_ai, &tmp_cmd));

                VkCommandBufferBeginInfo tmp_begin{};
                tmp_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                tmp_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkBeginCommandBuffer(tmp_cmd, &tmp_begin));

                VkImageMemoryBarrier2 to_dst{};
                to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                to_dst.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                to_dst.srcAccessMask = VK_ACCESS_2_NONE;
                to_dst.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                to_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                to_dst.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                to_dst.image = heightmap_gpu.image;
                to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo dep_dst{};
                dep_dst.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_dst.imageMemoryBarrierCount = 1;
                dep_dst.pImageMemoryBarriers = &to_dst;
                vkCmdPipelineBarrier2(tmp_cmd, &dep_dst);

                VkBufferImageCopy2 copy_region{};
                copy_region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
                copy_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy_region.imageExtent = {bp.grid_w, bp.grid_h, 1};
                VkCopyBufferToImageInfo2 copy_info{};
                copy_info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
                copy_info.srcBuffer = stg_buf;
                copy_info.dstImage = heightmap_gpu.image;
                copy_info.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                copy_info.regionCount = 1;
                copy_info.pRegions = &copy_region;
                vkCmdCopyBufferToImage2(tmp_cmd, &copy_info);

                VkImageMemoryBarrier2 to_gen{};
                to_gen.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                to_gen.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                to_gen.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                to_gen.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                to_gen.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                to_gen.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                to_gen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                to_gen.image = heightmap_gpu.image;
                to_gen.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo dep_gen{};
                dep_gen.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_gen.imageMemoryBarrierCount = 1;
                dep_gen.pImageMemoryBarriers = &to_gen;
                vkCmdPipelineBarrier2(tmp_cmd, &dep_gen);

                VK_CHECK(vkEndCommandBuffer(tmp_cmd));
                VkFenceCreateInfo tmp_fence_ci{};
                tmp_fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                VkFence tmp_fence = VK_NULL_HANDLE;
                VK_CHECK(vkCreateFence(device, &tmp_fence_ci, nullptr, &tmp_fence));
                VkSubmitInfo tmp_submit{};
                tmp_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                tmp_submit.commandBufferCount = 1;
                tmp_submit.pCommandBuffers = &tmp_cmd;
                VK_CHECK(vkQueueSubmit(graphics_queue, 1, &tmp_submit, tmp_fence));
                VK_CHECK(vkWaitForFences(device, 1, &tmp_fence, VK_TRUE, UINT64_MAX));
                vkDestroyFence(device, tmp_fence, nullptr);
                vkDestroyCommandPool(device, tmp_pool, nullptr);
                vmaDestroyBuffer(allocator, stg_buf, stg_alloc);
                g_request_basin_reset = false;
                g_request_water_reset = true;
            }

            if (g_request_water_reset) {
                VkCommandPoolCreateInfo tmp_pool_ci{};
                tmp_pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                tmp_pool_ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
                tmp_pool_ci.queueFamilyIndex = gfx_family;
                VkCommandPool tmp_pool = VK_NULL_HANDLE;
                VK_CHECK(vkCreateCommandPool(device, &tmp_pool_ci, nullptr, &tmp_pool));

                VkCommandBufferAllocateInfo tmp_cmd_ai{};
                tmp_cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                tmp_cmd_ai.commandPool = tmp_pool;
                tmp_cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                tmp_cmd_ai.commandBufferCount = 1;
                VkCommandBuffer tmp_cmd = VK_NULL_HANDLE;
                VK_CHECK(vkAllocateCommandBuffers(device, &tmp_cmd_ai, &tmp_cmd));

                VkCommandBufferBeginInfo tmp_begin{};
                tmp_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                tmp_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkBeginCommandBuffer(tmp_cmd, &tmp_begin));

                vkCmdBindPipeline(tmp_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swe_init_pipeline);
                vkCmdBindDescriptorSets(tmp_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        swe_init_pipeline_layout, 0, 1, &swe_init_desc_set, 0, nullptr);
                SweInitPC init_pc{};
                init_pc.grid_w = SWE_GRID_W;
                init_pc.grid_h = SWE_GRID_H;
                init_pc.initial_water_level = bp.floor_height + bp.initial_water;
                vkCmdPushConstants(tmp_cmd, swe_init_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(init_pc), &init_pc);
                vkCmdDispatch(tmp_cmd, (SWE_GRID_W + 7) / 8, (SWE_GRID_H + 7) / 8, 1);

                VkMemoryBarrier2 mem_bar{};
                mem_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                mem_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mem_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                mem_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mem_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo dep_bar{};
                dep_bar.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_bar.memoryBarrierCount = 1;
                dep_bar.pMemoryBarriers = &mem_bar;
                vkCmdPipelineBarrier2(tmp_cmd, &dep_bar);

                VK_CHECK(vkEndCommandBuffer(tmp_cmd));
                VkFenceCreateInfo tmp_fence_ci{};
                tmp_fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                VkFence tmp_fence = VK_NULL_HANDLE;
                VK_CHECK(vkCreateFence(device, &tmp_fence_ci, nullptr, &tmp_fence));
                VkSubmitInfo tmp_submit{};
                tmp_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                tmp_submit.commandBufferCount = 1;
                tmp_submit.pCommandBuffers = &tmp_cmd;
                VK_CHECK(vkQueueSubmit(graphics_queue, 1, &tmp_submit, tmp_fence));
                VK_CHECK(vkWaitForFences(device, 1, &tmp_fence, VK_TRUE, UINT64_MAX));
                vkDestroyFence(device, tmp_fence, nullptr);
                vkDestroyCommandPool(device, tmp_pool, nullptr);
                swe_ping_pong = 0;
                g_request_water_reset = false;
                g_request_sediment_reset = true;
            }
        }

        if (g_request_sediment_reset) {
            vkDeviceWaitIdle(device);

            VkCommandPoolCreateInfo tmp_pool_ci{};
            tmp_pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            tmp_pool_ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            tmp_pool_ci.queueFamilyIndex = gfx_family;
            VkCommandPool tmp_pool = VK_NULL_HANDLE;
            VK_CHECK(vkCreateCommandPool(device, &tmp_pool_ci, nullptr, &tmp_pool));

            VkCommandBufferAllocateInfo tmp_cmd_ai{};
            tmp_cmd_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            tmp_cmd_ai.commandPool = tmp_pool;
            tmp_cmd_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            tmp_cmd_ai.commandBufferCount = 1;
            VkCommandBuffer tmp_cmd = VK_NULL_HANDLE;
            VK_CHECK(vkAllocateCommandBuffers(device, &tmp_cmd_ai, &tmp_cmd));

            VkCommandBufferBeginInfo tmp_begin{};
            tmp_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            tmp_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(tmp_cmd, &tmp_begin));

            VkClearColorValue clear_zero{};
            clear_zero.float32[0] = 0.0f;
            VkImageSubresourceRange clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(tmp_cmd, sediment_a.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
            vkCmdClearColorImage(tmp_cmd, sediment_b.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);

            VK_CHECK(vkEndCommandBuffer(tmp_cmd));
            VkFenceCreateInfo tmp_fence_ci{};
            tmp_fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkFence tmp_fence = VK_NULL_HANDLE;
            VK_CHECK(vkCreateFence(device, &tmp_fence_ci, nullptr, &tmp_fence));
            VkSubmitInfo tmp_submit{};
            tmp_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            tmp_submit.commandBufferCount = 1;
            tmp_submit.pCommandBuffers = &tmp_cmd;
            VK_CHECK(vkQueueSubmit(graphics_queue, 1, &tmp_submit, tmp_fence));
            VK_CHECK(vkWaitForFences(device, 1, &tmp_fence, VK_TRUE, UINT64_MAX));
            vkDestroyFence(device, tmp_fence, nullptr);
            vkDestroyCommandPool(device, tmp_pool, nullptr);
            sediment_ping_pong = 0;
            g_request_sediment_reset = false;
        }

        // ---- Per-frame camera movement ----
        {
            float altitude = g_camera.position.y;
            float base_speed = altitude * 0.5f;
            float speed = base_speed;
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                speed *= 3.0f;
            if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
                speed *= 0.2f;

            float cy = std::cos(g_camera.yaw);
            float sy = std::sin(g_camera.yaw);
            float cp = std::cos(g_camera.pitch);
            float sp = std::sin(g_camera.pitch);

            glm::vec3 forward(cy * cp, sp, sy * cp);
            glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
            glm::vec3 up(0, 1, 0);

            glm::vec3 move(0.0f);
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += forward;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= forward;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += right;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= right;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move += up;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move -= up;

            if (glm::dot(move, move) > 0.0f)
                g_camera.position += glm::normalize(move) * speed * dt;
        }

        // ---- Camera UBO update (perspective) ----
        glm::mat4 cam_view, cam_proj;
        {
            float aspect = static_cast<float>(extent.width) / extent.height;
            cam_proj = glm::perspectiveRH_ZO(g_camera.fov_y, aspect, g_camera.near_plane, g_camera.far_plane);
            cam_proj[1][1] *= -1.0f;

            float cy = std::cos(g_camera.yaw);
            float sy = std::sin(g_camera.yaw);
            float cp = std::cos(g_camera.pitch);
            float sp = std::sin(g_camera.pitch);
            glm::vec3 forward(cy * cp, sp, sy * cp);

            cam_view = glm::lookAtRH(g_camera.position, g_camera.position + forward, glm::vec3(0, 1, 0));
        }

        // ---- Ray-pick cursor → grid coords (every frame for ring) ----
        float grid_x = 0.0f, grid_y = 0.0f;
        {
            int win_w, win_h;
            glfwGetWindowSize(window, &win_w, &win_h);
            double scale_x = static_cast<double>(fb_w) / win_w;
            double scale_y = static_cast<double>(fb_h) / win_h;
            float ndc_x = static_cast<float>((g_cursor_x * scale_x) / fb_w) * 2.0f - 1.0f;
            float ndc_y = static_cast<float>((g_cursor_y * scale_y) / fb_h) * 2.0f - 1.0f;

            glm::mat4 inv_vp = glm::inverse(cam_proj * cam_view);
            glm::vec4 near_ndc(ndc_x, ndc_y, 0.0f, 1.0f);
            glm::vec4 far_ndc(ndc_x, ndc_y, 1.0f, 1.0f);
            glm::vec4 near_world = inv_vp * near_ndc;
            glm::vec4 far_world  = inv_vp * far_ndc;
            near_world /= near_world.w;
            far_world  /= far_world.w;

            glm::vec3 ray_origin(near_world);
            glm::vec3 ray_dir = glm::normalize(glm::vec3(far_world) - ray_origin);

            float sea_level = bp.floor_height + bp.initial_water;
            float t_hit = (sea_level - ray_origin.y) / ray_dir.y;
            g_cursor_on_world = (t_hit > 0.0f);
            if (g_cursor_on_world) {
                glm::vec3 hit = ray_origin + ray_dir * t_hit;
                g_cursor_world_x = hit.x;
                g_cursor_world_z = hit.z;
                grid_x = (hit.x / terrain_size + 0.5f) * SWE_GRID_W;
                grid_y = (hit.z / terrain_size + 0.5f) * SWE_GRID_H;
            }
        }
        bool brush_active = g_lmb_held && !g_rmb_held;
        bool brush_hit = brush_active && g_cursor_on_world;

        // ---- Camera UBO update (perspective + brush ring) ----
        {
            float brush_radius_world = g_brush_radius_grid * bp.cell_spacing;
            glm::vec4 brush_color;
            if (g_brush_mode == BrushMode::Raise)
                brush_color = glm::vec4(0.30f, 0.95f, 0.40f, 1.0f);
            else if (g_brush_mode == BrushMode::Lower)
                brush_color = glm::vec4(0.95f, 0.45f, 0.20f, 1.0f);
            else
                brush_color = glm::vec4(0.30f, 0.85f, 0.95f, 1.0f);

            CameraData cam{};
            cam.view = cam_view;
            cam.proj = cam_proj;
            cam.sun_dir = glm::normalize(glm::vec3(0.4f, 0.7f, -0.3f));
            cam._pad0 = 0.0f;
            cam.sun_color = glm::vec3(1.0f, 0.95f, 0.85f);
            cam._pad1 = 0.0f;
            cam.cam_pos = g_camera.position;
            cam._pad2 = g_mud_visibility;
            cam.brush_world = glm::vec4(
                g_cursor_world_x, g_cursor_world_z,
                brush_radius_world,
                g_cursor_on_world ? 1.0f : 0.0f);
            cam.brush_color = brush_color;
            std::memcpy(camera_ubo_info.pMappedData, &cam, sizeof(cam));
            vmaFlushAllocation(allocator, camera_ubo_alloc, 0, VK_WHOLE_SIZE);
        }

        // ---- Terrain brush dispatch (before SWE) ----
        if (brush_hit && (g_brush_mode == BrushMode::Raise || g_brush_mode == BrushMode::Lower)) {
            float sign = (g_brush_mode == BrushMode::Raise) ? 1.0f : -1.0f;

            TerrainBrushPC tb_pc{};
            tb_pc.brush_x = grid_x;
            tb_pc.brush_y = grid_y;
            tb_pc.brush_radius = g_brush_radius_grid;
            tb_pc.brush_amount = sign * g_terrain_strength * std::min(dt, 0.033f);
            tb_pc.grid_w = bp.grid_w;
            tb_pc.grid_h = bp.grid_h;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, terrain_brush_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    tb_pipeline_layout, 0, 1, &tb_desc_set, 0, nullptr);
            vkCmdPushConstants(frame.cmd, tb_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(tb_pc), &tb_pc);
            vkCmdDispatch(frame.cmd, (bp.grid_w + 15) / 16, (bp.grid_h + 15) / 16, 1);

            VkMemoryBarrier2 tb_bar{};
            tb_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            tb_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            tb_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            tb_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                                | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            tb_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

            VkDependencyInfo tb_dep{};
            tb_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            tb_dep.memoryBarrierCount = 1;
            tb_dep.pMemoryBarriers = &tb_bar;
            vkCmdPipelineBarrier2(frame.cmd, &tb_dep);
        }

        // ---- SWE step dispatch (CFL sub-stepping) ----
        {
            vkCmdResetQueryPool(frame.cmd, query_pool, current_frame * 2, 2);

            float swe_total_dt = std::min(dt, 0.033f) * g_time_scale;
            float c_max = std::sqrt(g_gravity * 200.0f);
            float cfl_dt = bp.cell_spacing / (c_max * 2.5f);
            int substeps = std::max(1, static_cast<int>(std::ceil(swe_total_dt / cfl_dt)));
            substeps = std::min(substeps, 16);
            float sub_dt = swe_total_dt / substeps;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swe_step_pipeline);

            vkCmdWriteTimestamp2(frame.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                query_pool, current_frame * 2 + 0);

            for (int step = 0; step < substeps; ++step) {
                if (step > 0) {
                    VkMemoryBarrier2 sub_bar{};
                    sub_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    sub_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    sub_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    sub_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    sub_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                         | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    VkDependencyInfo sub_dep{};
                    sub_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    sub_dep.memoryBarrierCount = 1;
                    sub_dep.pMemoryBarriers = &sub_bar;
                    vkCmdPipelineBarrier2(frame.cmd, &sub_dep);
                }

                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        swe_step_pipeline_layout, 0, 1,
                                        &swe_step_desc_sets[swe_ping_pong], 0, nullptr);

                SweStepPC swe_pc{};
                swe_pc.time = static_cast<float>(now);
                swe_pc.dt = sub_dt;
                swe_pc.gravity = g_gravity;
                swe_pc.friction = g_friction;
                swe_pc.dx = bp.cell_spacing;
                swe_pc.sea_level = bp.floor_height + bp.initial_water;
                swe_pc.damping = g_damping;
                swe_pc._pad0 = 0.0f;
                swe_pc.grid_w = SWE_GRID_W;
                swe_pc.grid_h = SWE_GRID_H;

                if (step == 0) {
                    bool water_brush_active = brush_hit && g_brush_mode == BrushMode::Water;
                    if (g_pulse_pending) {
                        swe_pc.pulse_x = SWE_GRID_W * 0.5f;
                        swe_pc.pulse_y = SWE_GRID_H * 0.5f;
                        swe_pc.pulse_radius = g_pulse_radius_cells;
                        swe_pc.pulse_amount = g_pulse_amount;
                        g_pulse_pending = false;
                    } else if (water_brush_active) {
                        swe_pc.pulse_x = grid_x;
                        swe_pc.pulse_y = grid_y;
                        swe_pc.pulse_radius = g_brush_radius_grid;
                        swe_pc.pulse_amount = g_brush_strength;
                    } else {
                        swe_pc.pulse_amount = 0.0f;
                    }
                } else {
                    swe_pc.pulse_amount = 0.0f;
                }

                vkCmdPushConstants(frame.cmd, swe_step_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(swe_pc), &swe_pc);
                vkCmdDispatch(frame.cmd, (SWE_GRID_W + 7) / 8, (SWE_GRID_H + 7) / 8, 1);

                swe_ping_pong ^= 1;
            }

            vkCmdWriteTimestamp2(frame.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                query_pool, current_frame * 2 + 1);
        }

        // ---- Erosion dispatch (after SWE, before graphics) ----
        if (g_erosion_enabled) {
            VkMemoryBarrier2 swe_to_ero{};
            swe_to_ero.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            swe_to_ero.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            swe_to_ero.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            swe_to_ero.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            swe_to_ero.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                     | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                     | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

            VkDependencyInfo dep_swe_ero{};
            dep_swe_ero.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep_swe_ero.memoryBarrierCount = 1;
            dep_swe_ero.pMemoryBarriers = &swe_to_ero;
            vkCmdPipelineBarrier2(frame.cmd, &dep_swe_ero);

            float ero_dt = std::min(dt, 0.033f) * g_time_scale;

            ErosionPC ero_pc{};
            ero_pc.dt = ero_dt;
            ero_pc.dx = bp.cell_spacing;
            ero_pc.grid_w = SWE_GRID_W;
            ero_pc.grid_h = SWE_GRID_H;
            ero_pc.k_erosion = g_k_erosion;
            ero_pc.k_deposit = g_k_deposit;
            ero_pc.k_capacity = g_k_capacity;
            ero_pc.min_slope = g_min_slope;
            ero_pc.min_depth = g_min_erosion_depth;
            ero_pc.max_change = g_max_change_m;
            ero_pc.max_sediment = g_max_sediment;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, erosion_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    ero_pipeline_layout, 0, 1,
                                    &ero_desc_sets[sediment_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, ero_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(ero_pc), &ero_pc);
            vkCmdDispatch(frame.cmd, (SWE_GRID_W + 15) / 16, (SWE_GRID_H + 15) / 16, 1);

            sediment_ping_pong ^= 1;
        }

        // ---- Barrier: compute -> graphics rendering ----
        {
            VkMemoryBarrier2 mem_bar{};
            mem_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mem_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mem_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mem_bar.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
                                 | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            mem_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                  | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            VkImageMemoryBarrier2 sc_barrier{};
            sc_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            sc_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            sc_barrier.srcAccessMask = VK_ACCESS_2_NONE;
            sc_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            sc_barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            sc_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            sc_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            sc_barrier.image = swapchain_images[image_index];
            sc_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkImageMemoryBarrier2 depth_barrier{};
            depth_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            depth_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            depth_barrier.srcAccessMask = VK_ACCESS_2_NONE;
            depth_barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            depth_barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depth_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depth_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_barrier.image = depth_buffer.image;
            depth_barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

            VkImageMemoryBarrier2 img_barriers[] = {sc_barrier, depth_barrier};

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &mem_bar;
            dep.imageMemoryBarrierCount = 2;
            dep.pImageMemoryBarriers = img_barriers;
            vkCmdPipelineBarrier2(frame.cmd, &dep);
        }

        // ---- Dynamic rendering ----
        {
            VkRenderingAttachmentInfo color_attachment{};
            color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            color_attachment.imageView = swapchain_views[image_index];
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attachment.clearValue.color = {{0.02f, 0.02f, 0.05f, 1.0f}};

            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = depth_buffer.view;
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depth_attachment.clearValue.depthStencil = {1.0f, 0};

            VkRenderingInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering_info.renderArea = {{0, 0}, extent};
            rendering_info.layerCount = 1;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_attachment;
            rendering_info.pDepthAttachment = &depth_attachment;

            vkCmdBeginRendering(frame.cmd, &rendering_info);

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gfx_pipeline);

            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    gfx_pipeline_layout, 0, 1, &gfx_desc_set, 0, nullptr);

            GfxPC tpc{};
            tpc.terrain_size = terrain_size;
            tpc.heightmap_texel = 1.0f / static_cast<float>(bp.grid_w);
            tpc.max_elevation = 2000.0f;
            tpc._pad = 0.0f;
            vkCmdPushConstants(frame.cmd, gfx_pipeline_layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(tpc), &tpc);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(extent.width);
            viewport.height = static_cast<float>(extent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(frame.cmd, 0, 1, &viewport);

            VkRect2D scissor{{0, 0}, extent};
            vkCmdSetScissor(frame.cmd, 0, 1, &scissor);

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer, &offset);
            vkCmdBindIndexBuffer(frame.cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(frame.cmd, index_count, 1, 0, 0, 0);

            // Water pass (same VBO/IBO, same descriptor set, different pipeline)
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, water_pipeline);
            vkCmdDrawIndexed(frame.cmd, index_count, 1, 0, 0, 0);

            vkCmdEndRendering(frame.cmd);
        }

        // ---- ImGui render pass ----
        {
            VkRenderingAttachmentInfo imgui_color{};
            imgui_color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            imgui_color.imageView = swapchain_views[image_index];
            imgui_color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            imgui_color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            imgui_color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo imgui_ri{};
            imgui_ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            imgui_ri.renderArea = {{0, 0}, extent};
            imgui_ri.layerCount = 1;
            imgui_ri.colorAttachmentCount = 1;
            imgui_ri.pColorAttachments = &imgui_color;

            vkCmdBeginRendering(frame.cmd, &imgui_ri);
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame.cmd);
            vkCmdEndRendering(frame.cmd);
        }

        // ---- Transition swapchain to present ----
        {
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_NONE;
            barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.image = swapchain_images[image_index];
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(frame.cmd, &dep);
        }

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

        double frame_end = glfwGetTime();
        cpu_times[timing_index] = (frame_end - frame_start) * 1000.0;
        timing_index = (timing_index + 1) % AVG_FRAMES;
        if (timing_count < AVG_FRAMES) timing_count++;
        queries_valid = true;

        if (frame_end - last_title_update >= 1.0) {
            double cpu_sum = 0.0, gpu_sum = 0.0;
            for (int i = 0; i < timing_count; ++i) {
                cpu_sum += cpu_times[i];
                gpu_sum += gpu_times[i];
            }
            cpu_avg_ms = cpu_sum / timing_count;
            gpu_avg_ms = gpu_sum / timing_count;
            const char* mode_str = "water";
            if (g_brush_mode == BrushMode::Raise) mode_str = "raise";
            else if (g_brush_mode == BrushMode::Lower) mode_str = "lower";
            float strength_display = (g_brush_mode == BrushMode::Water) ? g_brush_strength : g_terrain_strength;
            const char* unit = (g_brush_mode == BrushMode::Water) ? "m/frame" : "m/s";
            char title[256];
            std::snprintf(title, sizeof(title),
                "drift_engine — CPU %.1f ms | GPU %.1f ms | %.0f fps | %s | size %.0f | strength %.1f %s",
                cpu_avg_ms, gpu_avg_ms, 1000.0 / cpu_avg_ms, mode_str, g_brush_radius_grid, strength_display, unit);
            glfwSetWindowTitle(window, title);
            last_title_update = frame_end;
        }

        current_frame = (current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    // ---- Cleanup (LIFO) -----------------------------------------------------
    vkDeviceWaitIdle(device);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device, imgui_pool, nullptr);

    for (auto& f : frames) {
        vkDestroyFence(device, f.in_flight, nullptr);
        vkDestroySemaphore(device, f.render_finished, nullptr);
        vkDestroySemaphore(device, f.image_available, nullptr);
        vkDestroyCommandPool(device, f.pool, nullptr);
    }

    vkDestroyQueryPool(device, query_pool, nullptr);

    for (auto view : swapchain_views)
        vkDestroyImageView(device, view, nullptr);
    vkb::destroy_swapchain(vkb_swapchain);

    vkDestroyPipeline(device, water_pipeline, nullptr);
    vkDestroyShaderModule(device, water_fs, nullptr);
    vkDestroyShaderModule(device, water_vs, nullptr);

    vkDestroyPipeline(device, erosion_pipeline, nullptr);
    vkDestroyPipelineLayout(device, ero_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, ero_desc_layout, nullptr);
    vkDestroyShaderModule(device, erosion_shader, nullptr);

    vkDestroyPipeline(device, terrain_brush_pipeline, nullptr);
    vkDestroyPipelineLayout(device, tb_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, tb_desc_layout, nullptr);
    vkDestroyShaderModule(device, terrain_brush_shader, nullptr);

    vkDestroyPipeline(device, gfx_pipeline, nullptr);
    vkDestroyPipelineLayout(device, gfx_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, gfx_desc_layout, nullptr);
    vkDestroyShaderModule(device, terrain_fs, nullptr);
    vkDestroyShaderModule(device, terrain_vs, nullptr);

    vkDestroyPipeline(device, swe_step_pipeline, nullptr);
    vkDestroyPipelineLayout(device, swe_step_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, swe_step_desc_layout, nullptr);
    vkDestroyShaderModule(device, swe_step_shader, nullptr);

    vkDestroyPipeline(device, swe_init_pipeline, nullptr);
    vkDestroyPipelineLayout(device, swe_init_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, swe_init_desc_layout, nullptr);
    vkDestroyShaderModule(device, swe_init_shader, nullptr);

    vkDestroyDescriptorPool(device, desc_pool, nullptr);

    destroy_depth_buffer(device, allocator, depth_buffer);

    destroy_swe_image(device, allocator, swe_output);
    destroy_swe_image(device, allocator, swe_state_b);
    destroy_swe_image(device, allocator, swe_state_a);

    destroy_swe_image(device, allocator, sediment_b);
    destroy_swe_image(device, allocator, sediment_a);

    vmaDestroyBuffer(allocator, index_buffer, index_alloc);
    vmaDestroyBuffer(allocator, vertex_buffer, vertex_alloc);
    vmaDestroyBuffer(allocator, camera_ubo, camera_ubo_alloc);

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

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
#include <functional>
#include <queue>
#include <vector>

namespace {

constexpr uint32_t FRAMES_IN_FLIGHT = 2;
constexpr uint32_t SWE_GRID_W = 1024;
constexpr uint32_t SWE_GRID_H = 1024;
constexpr uint32_t ATMO_W = 128;
constexpr uint32_t ATMO_H = 128;
constexpr uint32_t ATMO_D = 32;
constexpr float ATMO_LAYER_HEIGHT = 100.0f;
constexpr uint32_t SAND_MAX_PARTICLES = 131072;
constexpr uint32_t SAND_EMIT_PER_FRAME = 2048;
constexpr uint32_t MESH_RES = 512;
constexpr uint32_t PLANET_TILE_RES      = 64;
constexpr uint32_t PLANET_TILE_POOL     = 2048;
constexpr uint32_t PLANET_MAX_LEVEL     = 14;
constexpr float    PLANET_RADIUS        = 6371000.0f;
constexpr float    PLANET_MAX_ELEVATION = 8000.0f;
constexpr float    TILE_SUBDIVIDE_PX    = 512.0f;

// ---- CPU-side terrain height (mirrors planet_gen.cs.hlsl) ------------------

float cpu_hash31(glm::vec3 p) {
    p = glm::fract(p * glm::vec3(0.1031f, 0.1030f, 0.0973f));
    p += glm::dot(p, glm::vec3(p.y, p.x, p.z) + 31.32f);
    return glm::fract((p.x + p.y) * p.z);
}

float cpu_gradient_noise_3d(glm::vec3 p) {
    glm::vec3 i = glm::floor(p);
    glm::vec3 f = glm::fract(p);
    glm::vec3 u = f * f * (3.0f - 2.0f * f);

    float n = glm::mix(
        glm::mix(glm::mix(cpu_hash31(i + glm::vec3(0,0,0)), cpu_hash31(i + glm::vec3(1,0,0)), u.x),
                 glm::mix(cpu_hash31(i + glm::vec3(0,1,0)), cpu_hash31(i + glm::vec3(1,1,0)), u.x), u.y),
        glm::mix(glm::mix(cpu_hash31(i + glm::vec3(0,0,1)), cpu_hash31(i + glm::vec3(1,0,1)), u.x),
                 glm::mix(cpu_hash31(i + glm::vec3(0,1,1)), cpu_hash31(i + glm::vec3(1,1,1)), u.x), u.y),
        u.z);
    return n;
}

float cpu_fbm3d(glm::vec3 p, int octaves, float lacunarity, float gain) {
    float sum = 0.0f, amp = 1.0f, freq = 1.0f, norm = 0.0f;
    for (int i = 0; i < octaves; i++) {
        sum += cpu_gradient_noise_3d(p * freq + glm::vec3(42 * 0.17f, 0, 0)) * amp;
        norm += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    return sum / norm;
}

float cpu_ridged3d(glm::vec3 p, int octaves) {
    float sum = 0.0f, amp = 1.0f, freq = 1.0f, prev = 1.0f;
    for (int i = 0; i < octaves; i++) {
        float n = cpu_gradient_noise_3d(p * freq + glm::vec3(42 * 0.13f, 0, 0));
        n = 1.0f - std::abs(n * 2.0f - 1.0f);
        n = n * n;
        sum += n * amp * prev;
        prev = n;
        freq *= 2.1f;
        amp *= 0.5f;
    }
    return sum;
}

float cpu_terrain_height(glm::vec3 sphere_dir) {
    glm::vec3 sp = sphere_dir * 1000.0f;
    float latitude = std::abs(sphere_dir.y);

    float base = 200.0f + cpu_fbm3d(sp * 0.0003f, 5, 2.0f, 0.5f) * 1500.0f;

    float biome = cpu_fbm3d(sp * 0.0005f + glm::vec3(7.7f, 0, 0), 3, 2.0f, 0.5f);

    float mtn_w    = glm::smoothstep(0.55f, 0.70f, biome);
    float desert_w = glm::smoothstep(0.35f, 0.50f, biome) * glm::smoothstep(0.55f, 0.40f, biome)
                   * glm::smoothstep(0.65f, 0.15f, latitude);
    float plains_w = glm::smoothstep(0.45f, 0.25f, biome);
    float polar_w  = glm::smoothstep(0.55f, 0.80f, latitude);

    float mountain = cpu_ridged3d(sp * 0.006f, 7) * 4500.0f;
    mountain += cpu_fbm3d(sp * 0.003f, 5, 2.0f, 0.55f) * 2000.0f;

    float desert = cpu_fbm3d(sp * 0.005f, 5, 2.0f, 0.5f) * 500.0f;
    desert += std::abs(cpu_gradient_noise_3d(sp * 0.03f)) * 250.0f;
    desert += (cpu_gradient_noise_3d(sp * 0.1f) - 0.5f) * 80.0f;

    float plains = cpu_fbm3d(sp * 0.004f, 5, 2.0f, 0.5f) * 400.0f;
    plains += cpu_fbm3d(sp * 0.02f, 3, 2.0f, 0.4f) * 100.0f;

    float polar = cpu_fbm3d(sp * 0.003f, 4, 2.0f, 0.5f) * 800.0f;
    polar += std::abs(cpu_gradient_noise_3d(sp * 0.02f) - 0.5f) * 200.0f;

    float total_w = std::max(mtn_w + desert_w + plains_w + polar_w, 0.01f);
    float biome_h = (mountain * mtn_w + desert * desert_w + plains * plains_w + polar * polar_w) / total_w;

    float h = base + biome_h;
    h += (cpu_gradient_noise_3d(sp * 0.15f) - 0.5f) * 40.0f;

    return h;
}

// ---- Cube-sphere mapping helpers (CPU, float) ------------------------------

glm::vec3 cpu_face_uv_to_cube(float u, float v, uint32_t face) {
    switch (face) {
        case 0: return {1.0f, v, -u};
        case 1: return {-1.0f, v, u};
        case 2: return {u, 1.0f, -v};
        case 3: return {u, -1.0f, v};
        case 4: return {u, v, 1.0f};
        case 5: return {-u, v, -1.0f};
        default: return {0, 0, 0};
    }
}

glm::vec3 cpu_cube_to_sphere(glm::vec3 p) {
    float x2 = p.x*p.x, y2 = p.y*p.y, z2 = p.z*p.z;
    return {
        p.x * std::sqrt(std::max(0.0f, 1.0f - y2*0.5f - z2*0.5f + y2*z2/3.0f)),
        p.y * std::sqrt(std::max(0.0f, 1.0f - x2*0.5f - z2*0.5f + x2*z2/3.0f)),
        p.z * std::sqrt(std::max(0.0f, 1.0f - x2*0.5f - y2*0.5f + x2*y2/3.0f))
    };
}

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
    glm::mat4 inv_view_proj;
};

struct RaymarchPC {
    float terrain_size;
    float max_elevation;
    float cloud_opacity;
    float cloud_base;
    uint32_t vol_w;
    uint32_t vol_h;
    uint32_t vol_d;
    float layer_height;
};

struct GfxPC {
    float terrain_size;
    float heightmap_texel;
    float max_elevation;
    float cloud_opacity;
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

struct Atmo3DPC {
    float    dt;
    float    accumulated_time;
    uint32_t grid_w;
    uint32_t grid_h;
    uint32_t grid_d;
    float    terrain_scale;
    float    layer_height;
    float    max_elevation;
    float    orographic_lift_coeff;
    float    adiabatic_cooling_rate;
    float    rain_shadow_intensity;
    uint32_t force_init;
    uint32_t sand_enabled;
    float    sand_loft_threshold;
    float    sand_loft_rate;
    float    sand_settling;
};

struct SandSimPC {
    float    dt;
    float    terrain_size;
    float    loft_threshold;
    float    loft_rate;
    float    gravity;
    float    accumulated_time;
    uint32_t max_particles;
    uint32_t emit_offset;
    uint32_t emit_count;
    uint32_t grid_d;
    float    layer_height;
    float    bounce_energy;
};

struct SandRenderPC {
    float    streak_length;
    float    particle_alpha;
    uint32_t max_particles;
    float    _pad;
};

struct PlanetTilePC {
    float    rel_x, rel_y, rel_z;
    float    u_min, v_min, tile_size;
    uint32_t face;
    uint32_t pool_index;
    float    planet_radius;
    float    max_elevation;
    float    heightmap_texel;
    float    cloud_opacity;
};

struct PlanetGenPC {
    float    u_min;
    float    v_min;
    float    tile_size;
    uint32_t face;
    uint32_t pool_index;
    uint32_t tex_res;
    uint32_t seed;
    uint32_t stamp_count;
};

struct TerrainStamp {
    float    pos_x, pos_y, pos_z;
    float    radius;
    float    delta_h;
    float    cos_radius;
    float    _pad0, _pad1;
};

constexpr uint32_t MAX_STAMPS = 4096;

struct QuadNode {
    uint32_t face;
    uint32_t level;
    uint32_t x, y;
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
    // Spawn at 45° N latitude, 20km above sphere surface
    double pos_x = (6371000.0 + 20000.0) * 0.7071;
    double pos_y = (6371000.0 + 20000.0) * 0.7071;
    double pos_z = 0.0;
    float yaw   = 0.0f;
    float pitch = -0.3f;
    float fov_y = glm::radians(60.0f);
    float near_plane = 0.5f;
    float far_plane  = 100000000.0f;  // not used with infinite reversed-Z
};

enum class BrushMode { Raise, Lower, Water, Sand };

bool g_framebuffer_resized = false;
bool g_reload_shaders = false;
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
double g_terrain_height_at_cam = 0.0;
double g_altitude_above_terrain = 100000.0;
std::vector<TerrainStamp> g_stamps;
bool g_stamps_dirty = false;
float g_stamp_angular_scale = 0.0001f;

float cpu_terrain_height_with_stamps(glm::vec3 sphere_dir) {
    float h = cpu_terrain_height(sphere_dir);
    for (const auto& s : g_stamps) {
        glm::vec3 sp(s.pos_x, s.pos_y, s.pos_z);
        float d = glm::dot(sphere_dir, sp);
        if (d < s.cos_radius) continue;
        float t = (d - s.cos_radius) / std::max(1.0f - s.cos_radius, 1e-7f);
        h += s.delta_h * std::exp(-4.0f * (1.0f - t) * (1.0f - t));
    }
    return h;
}

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
bool  g_atmosphere_enabled = false;
float g_orographic_lift    = 0.5f;
float g_adiabatic_cooling  = 0.0065f;
float g_rain_shadow        = 0.7f;
float g_cloud_opacity      = 1.0f;
float g_cloud_altitude     = 0.0f;
bool  g_sand_enabled       = false;
float g_sand_loft_threshold = 1.5f;
float g_sand_loft_rate     = 0.5f;
float g_sand_settling      = 2.0f;
float g_sand_streak        = 0.05f;
float g_sand_alpha         = 0.8f;
float g_sand_bounce        = 0.3f;
float g_sand_gravity       = 9.81f;
uint32_t g_visible_tile_count = 0;
bool  g_request_atmo_reset = false;
float g_accumulated_atmo_time = 0.0f;

void key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_GRAVE_ACCENT && action == GLFW_PRESS)
        g_show_menu = !g_show_menu;
    if (key == GLFW_KEY_F5 && action == GLFW_PRESS)
        g_reload_shaders = true;
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard)
        return;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        g_pulse_pending = true;
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_1) g_brush_mode = BrushMode::Raise;
        if (key == GLFW_KEY_2) g_brush_mode = BrushMode::Lower;
        if (key == GLFW_KEY_3) g_brush_mode = BrushMode::Water;
        if (key == GLFW_KEY_4) g_brush_mode = BrushMode::Sand;
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

SweImage create_wind_image(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h)
{
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = VK_FORMAT_R16G16_SFLOAT;
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
    view_ci.format = VK_FORMAT_R16G16_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &view_ci, nullptr, &img.view));

    return img;
}

SweImage create_volume_image(VkDevice device, VmaAllocator allocator,
                             uint32_t w, uint32_t h, uint32_t d, VkFormat format)
{
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_3D;
    img_ci.format = format;
    img_ci.extent = {w, h, d};
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
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view_ci.format = format;
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

    // ---- Planet tile grid mesh (PLANET_TILE_RES x PLANET_TILE_RES + skirts) ----
    constexpr uint32_t R = PLANET_TILE_RES;
    constexpr uint32_t GRID_VERTS = R * R;

    std::vector<float> clip_verts;
    clip_verts.reserve((GRID_VERTS + 4 * R) * 2);

    // Main grid: vertices (x, y) for x,y in [0, R-1]
    for (uint32_t y = 0; y < R; ++y)
        for (uint32_t x = 0; x < R; ++x) {
            clip_verts.push_back(static_cast<float>(x));
            clip_verts.push_back(static_cast<float>(y));
        }
    // Skirt vertices: encode as out-of-range so VS can detect and drop them
    // Bottom edge (y = -1): indices [GRID_VERTS .. GRID_VERTS + R - 1]
    for (uint32_t x = 0; x < R; ++x) { clip_verts.push_back(static_cast<float>(x)); clip_verts.push_back(-1.0f); }
    // Top edge (y = R): indices [GRID_VERTS + R .. GRID_VERTS + 2R - 1]
    for (uint32_t x = 0; x < R; ++x) { clip_verts.push_back(static_cast<float>(x)); clip_verts.push_back(static_cast<float>(R)); }
    // Left edge (x = -1): indices [GRID_VERTS + 2R .. GRID_VERTS + 3R - 1]
    for (uint32_t y = 0; y < R; ++y) { clip_verts.push_back(-1.0f); clip_verts.push_back(static_cast<float>(y)); }
    // Right edge (x = R): indices [GRID_VERTS + 3R .. GRID_VERTS + 4R - 1]
    for (uint32_t y = 0; y < R; ++y) { clip_verts.push_back(static_cast<float>(R)); clip_verts.push_back(static_cast<float>(y)); }

    std::vector<uint32_t> clip_indices;
    clip_indices.reserve((R - 1) * (R - 1) * 6 + 4 * (R - 1) * 6);

    // Main grid triangles
    for (uint32_t y = 0; y < R - 1; ++y)
        for (uint32_t x = 0; x < R - 1; ++x) {
            uint32_t tl = y * R + x;
            uint32_t tr = tl + 1;
            uint32_t bl = tl + R;
            uint32_t br = bl + 1;
            clip_indices.push_back(tl); clip_indices.push_back(bl); clip_indices.push_back(tr);
            clip_indices.push_back(tr); clip_indices.push_back(bl); clip_indices.push_back(br);
        }

    // Skirt triangles — connect each edge to its skirt strip
    auto add_skirt_strip = [&](auto edge_idx, auto skirt_idx, bool flip) {
        for (uint32_t i = 0; i < R - 1; ++i) {
            uint32_t e0 = edge_idx(i), e1 = edge_idx(i + 1);
            uint32_t s0 = skirt_idx(i), s1 = skirt_idx(i + 1);
            if (flip) {
                clip_indices.push_back(e0); clip_indices.push_back(e1); clip_indices.push_back(s0);
                clip_indices.push_back(s0); clip_indices.push_back(e1); clip_indices.push_back(s1);
            } else {
                clip_indices.push_back(e0); clip_indices.push_back(s0); clip_indices.push_back(e1);
                clip_indices.push_back(e1); clip_indices.push_back(s0); clip_indices.push_back(s1);
            }
        }
    };
    // Bottom: grid row 0 → skirt bottom
    add_skirt_strip([](uint32_t i) { return i; },
                    [](uint32_t i) { return GRID_VERTS + i; }, true);
    // Top: grid row R-1 → skirt top
    add_skirt_strip([](uint32_t i) { return (R - 1) * R + i; },
                    [](uint32_t i) { return GRID_VERTS + R + i; }, false);
    // Left: grid col 0 → skirt left
    add_skirt_strip([](uint32_t i) { return i * R; },
                    [](uint32_t i) { return GRID_VERTS + 2 * R + i; }, false);
    // Right: grid col R-1 → skirt right
    add_skirt_strip([](uint32_t i) { return i * R + (R - 1); },
                    [](uint32_t i) { return GRID_VERTS + 3 * R + i; }, true);

    uint32_t clipmap_index_count = static_cast<uint32_t>(clip_indices.size());

    VkBufferCreateInfo clip_vbo_ci{};
    clip_vbo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    clip_vbo_ci.size = clip_verts.size() * sizeof(float);
    clip_vbo_ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VmaAllocationCreateInfo clip_vbo_ai{};
    clip_vbo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    clip_vbo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer clipmap_vbo = VK_NULL_HANDLE;
    VmaAllocation clipmap_vbo_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo clip_vbo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &clip_vbo_ci, &clip_vbo_ai, &clipmap_vbo, &clipmap_vbo_alloc, &clip_vbo_info));
    std::memcpy(clip_vbo_info.pMappedData, clip_verts.data(), clip_vbo_ci.size);
    vmaFlushAllocation(allocator, clipmap_vbo_alloc, 0, VK_WHOLE_SIZE);

    VkBufferCreateInfo clip_ibo_ci{};
    clip_ibo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    clip_ibo_ci.size = clip_indices.size() * sizeof(uint32_t);
    clip_ibo_ci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    VmaAllocationCreateInfo clip_ibo_ai{};
    clip_ibo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    clip_ibo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer clipmap_ibo = VK_NULL_HANDLE;
    VmaAllocation clipmap_ibo_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo clip_ibo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &clip_ibo_ci, &clip_ibo_ai, &clipmap_ibo, &clipmap_ibo_alloc, &clip_ibo_info));
    std::memcpy(clip_ibo_info.pMappedData, clip_indices.data(), clip_ibo_ci.size);
    vmaFlushAllocation(allocator, clipmap_ibo_alloc, 0, VK_WHOLE_SIZE);

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

    // ---- Stamp buffer (SSBO for terrain edit stamps) ------------------------
    VkBufferCreateInfo stamp_buf_ci{};
    stamp_buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stamp_buf_ci.size = MAX_STAMPS * sizeof(TerrainStamp);
    stamp_buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo stamp_buf_ai{};
    stamp_buf_ai.usage = VMA_MEMORY_USAGE_AUTO;
    stamp_buf_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                       | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stamp_buf = VK_NULL_HANDLE;
    VmaAllocation stamp_buf_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo stamp_buf_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &stamp_buf_ci, &stamp_buf_ai,
        &stamp_buf, &stamp_buf_alloc, &stamp_buf_info));

    // ---- SWE images (ping-pong state + output) ------------------------------
    SweImage swe_state_a = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    SweImage swe_state_b = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    SweImage swe_output  = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

    // ---- Sediment images (ping-pong) ----------------------------------------
    SweImage sediment_a = create_sediment_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    SweImage sediment_b = create_sediment_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

    // ---- Atmosphere images -----------------------------------------------------
    SweImage atmo_state_a = create_volume_image(device, allocator, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    SweImage atmo_state_b = create_volume_image(device, allocator, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    SweImage wind_field_a = create_volume_image(device, allocator, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    SweImage wind_field_b = create_volume_image(device, allocator, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    SweImage atmo_shadow  = create_sediment_image(device, allocator, ATMO_W, ATMO_H);
    SweImage sand_deposit = create_sediment_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

    // ---- Tile pool (replaces clipmap heightmap array) ----------------------------
    VkImageCreateInfo clip_hm_ci{};
    clip_hm_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    clip_hm_ci.imageType = VK_IMAGE_TYPE_2D;
    clip_hm_ci.format = VK_FORMAT_R32_SFLOAT;
    clip_hm_ci.extent = {PLANET_TILE_RES, PLANET_TILE_RES, 1};
    clip_hm_ci.mipLevels = 1;
    clip_hm_ci.arrayLayers = PLANET_TILE_POOL;
    clip_hm_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    clip_hm_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    clip_hm_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    clip_hm_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo clip_hm_ai{};
    clip_hm_ai.usage = VMA_MEMORY_USAGE_AUTO;
    clip_hm_ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkImage clipmap_hm_image = VK_NULL_HANDLE;
    VmaAllocation clipmap_hm_alloc = VK_NULL_HANDLE;
    VK_CHECK(vmaCreateImage(allocator, &clip_hm_ci, &clip_hm_ai, &clipmap_hm_image, &clipmap_hm_alloc, nullptr));

    VkImageViewCreateInfo clip_hm_view_ci{};
    clip_hm_view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    clip_hm_view_ci.image = clipmap_hm_image;
    clip_hm_view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    clip_hm_view_ci.format = VK_FORMAT_R32_SFLOAT;
    clip_hm_view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};

    VkImageView clipmap_hm_view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(device, &clip_hm_view_ci, nullptr, &clipmap_hm_view));

    // ---- Sand particle buffer --------------------------------------------------
    VkBufferCreateInfo sand_buf_ci{};
    sand_buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    sand_buf_ci.size = SAND_MAX_PARTICLES * 32; // 8 floats per particle
    sand_buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo sand_buf_ai{};
    sand_buf_ai.usage = VMA_MEMORY_USAGE_AUTO;
    sand_buf_ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuffer sand_particle_buf = VK_NULL_HANDLE;
    VmaAllocation sand_particle_alloc = VK_NULL_HANDLE;
    VK_CHECK(vmaCreateBuffer(allocator, &sand_buf_ci, &sand_buf_ai, &sand_particle_buf, &sand_particle_alloc, nullptr));

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

    // ---- Terrain linear sampler (for atmosphere terrain reads) ----
    VkSamplerCreateInfo terrain_samp_ci{};
    terrain_samp_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    terrain_samp_ci.magFilter = VK_FILTER_LINEAR;
    terrain_samp_ci.minFilter = VK_FILTER_LINEAR;
    terrain_samp_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    terrain_samp_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    terrain_samp_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler terrain_linear_sampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(device, &terrain_samp_ci, nullptr, &terrain_linear_sampler));

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

    // Graphics descriptor set layout: UBO + heightmap + swe_output + sediment + atmo_shadow + cloud_vol_3d + wind_vol_3d + sand_deposit
    VkDescriptorSetLayoutBinding gfx_bindings[8]{};
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

    gfx_bindings[4].binding = 4;
    gfx_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[4].descriptorCount = 1;
    gfx_bindings[4].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[5].binding = 5;
    gfx_bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[5].descriptorCount = 1;
    gfx_bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[6].binding = 6;
    gfx_bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[6].descriptorCount = 1;
    gfx_bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[7].binding = 7;
    gfx_bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[7].descriptorCount = 1;
    gfx_bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo gfx_dsl_ci{};
    gfx_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    gfx_dsl_ci.bindingCount = 8;
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

    VkPushConstantRange rm_push_range{};
    rm_push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    rm_push_range.offset = 0;
    rm_push_range.size = sizeof(RaymarchPC);

    VkPipelineLayoutCreateInfo rm_pl_ci{};
    rm_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    rm_pl_ci.setLayoutCount = 1;
    rm_pl_ci.pSetLayouts = &gfx_desc_layout;
    rm_pl_ci.pushConstantRangeCount = 1;
    rm_pl_ci.pPushConstantRanges = &rm_push_range;

    VkPipelineLayout raymarch_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &rm_pl_ci, nullptr, &raymarch_pipeline_layout));

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
    depth_stencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

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

    // ---- Planet terrain graphics pipeline layout + pipeline -------------------
    VkPushConstantRange clip_gfx_push{};
    clip_gfx_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    clip_gfx_push.offset = 0;
    clip_gfx_push.size = sizeof(PlanetTilePC);

    VkPipelineLayoutCreateInfo clip_gfx_pl_ci{};
    clip_gfx_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    clip_gfx_pl_ci.setLayoutCount = 1;
    clip_gfx_pl_ci.pSetLayouts = &gfx_desc_layout;
    clip_gfx_pl_ci.pushConstantRangeCount = 1;
    clip_gfx_pl_ci.pPushConstantRanges = &clip_gfx_push;

    VkPipelineLayout clipmap_gfx_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &clip_gfx_pl_ci, nullptr, &clipmap_gfx_pipeline_layout));

    VkGraphicsPipelineCreateInfo clip_gfx_pipe_ci = gfx_pipe_ci;
    clip_gfx_pipe_ci.layout = clipmap_gfx_pipeline_layout;

    VkPipeline clipmap_terrain_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &clip_gfx_pipe_ci, nullptr, &clipmap_terrain_pipeline));

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
    water_depth_stencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

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

    // ---- Cloud raymarch pipeline (fullscreen, alpha blend, no depth) ---------
    auto rm_vs_spirv = load_spirv("shaders/cloud_raymarch_vs.spv");
    auto rm_fs_spirv = load_spirv("shaders/cloud_raymarch_fs.spv");

    VkShaderModuleCreateInfo rm_vs_sm_ci{};
    rm_vs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    rm_vs_sm_ci.codeSize = rm_vs_spirv.size() * sizeof(uint32_t);
    rm_vs_sm_ci.pCode = rm_vs_spirv.data();
    VkShaderModule raymarch_vs = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &rm_vs_sm_ci, nullptr, &raymarch_vs));

    VkShaderModuleCreateInfo rm_fs_sm_ci{};
    rm_fs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    rm_fs_sm_ci.codeSize = rm_fs_spirv.size() * sizeof(uint32_t);
    rm_fs_sm_ci.pCode = rm_fs_spirv.data();
    VkShaderModule raymarch_fs = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &rm_fs_sm_ci, nullptr, &raymarch_fs));

    VkPipelineShaderStageCreateInfo rm_stages[2]{};
    rm_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rm_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    rm_stages[0].module = raymarch_vs;
    rm_stages[0].pName = "main";
    rm_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rm_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    rm_stages[1].module = raymarch_fs;
    rm_stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo rm_vi{};
    rm_vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineDepthStencilStateCreateInfo rm_ds{};
    rm_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    rm_ds.depthTestEnable = VK_FALSE;
    rm_ds.depthWriteEnable = VK_FALSE;

    VkGraphicsPipelineCreateInfo rm_pipe_ci = gfx_pipe_ci;
    rm_pipe_ci.pStages = rm_stages;
    rm_pipe_ci.pVertexInputState = &rm_vi;
    rm_pipe_ci.pDepthStencilState = &rm_ds;
    rm_pipe_ci.pColorBlendState = &water_color_blend;
    rm_pipe_ci.layout = raymarch_pipeline_layout;

    VkPipeline raymarch_pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &rm_pipe_ci, nullptr, &raymarch_pipeline));

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

    // ---- Atmosphere descriptor set layout --------------------------------------
    VkDescriptorSetLayoutBinding atmo_bindings[6]{};
    atmo_bindings[0].binding = 0;
    atmo_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    atmo_bindings[0].descriptorCount = 1;
    atmo_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[1].binding = 1;
    atmo_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    atmo_bindings[1].descriptorCount = 1;
    atmo_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[2].binding = 2;
    atmo_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    atmo_bindings[2].descriptorCount = 1;
    atmo_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[3].binding = 3;
    atmo_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    atmo_bindings[3].descriptorCount = 1;
    atmo_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[4].binding = 4;
    atmo_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    atmo_bindings[4].descriptorCount = 1;
    atmo_bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[5].binding = 5;
    atmo_bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    atmo_bindings[5].descriptorCount = 1;
    atmo_bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo atmo_dsl_ci{};
    atmo_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    atmo_dsl_ci.bindingCount = 6;
    atmo_dsl_ci.pBindings = atmo_bindings;

    VkDescriptorSetLayout atmo_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &atmo_dsl_ci, nullptr, &atmo_desc_layout));

    // ---- Terrain gen compute descriptor set layout + pipeline layout -----------
    VkDescriptorSetLayoutBinding tgen_bindings[2]{};
    tgen_bindings[0].binding = 0;
    tgen_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    tgen_bindings[0].descriptorCount = 1;
    tgen_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tgen_bindings[1].binding = 1;
    tgen_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    tgen_bindings[1].descriptorCount = 1;
    tgen_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo tgen_dsl_ci{};
    tgen_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    tgen_dsl_ci.bindingCount = 2;
    tgen_dsl_ci.pBindings = tgen_bindings;

    VkDescriptorSetLayout terrain_gen_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &tgen_dsl_ci, nullptr, &terrain_gen_desc_layout));

    VkPushConstantRange tgen_push{};
    tgen_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tgen_push.offset = 0;
    tgen_push.size = sizeof(PlanetGenPC);

    VkPipelineLayoutCreateInfo tgen_pl_ci{};
    tgen_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tgen_pl_ci.setLayoutCount = 1;
    tgen_pl_ci.pSetLayouts = &terrain_gen_desc_layout;
    tgen_pl_ci.pushConstantRangeCount = 1;
    tgen_pl_ci.pPushConstantRanges = &tgen_push;

    VkPipelineLayout terrain_gen_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &tgen_pl_ci, nullptr, &terrain_gen_pipeline_layout));

    VkShaderModule terrain_gen_shader = VK_NULL_HANDLE;
    {
        auto spv = load_spirv("shaders/planet_gen_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &terrain_gen_shader));
    }

    VkPipeline terrain_gen_pipeline = VK_NULL_HANDLE;
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = terrain_gen_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = terrain_gen_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &terrain_gen_pipeline));
    }

    // ---- Atmosphere pipeline layout + pipeline ---------------------------------
    VkPushConstantRange atmo_push{};
    atmo_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    atmo_push.offset = 0;
    atmo_push.size = sizeof(Atmo3DPC);

    VkPipelineLayoutCreateInfo atmo_pl_ci{};
    atmo_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    atmo_pl_ci.setLayoutCount = 1;
    atmo_pl_ci.pSetLayouts = &atmo_desc_layout;
    atmo_pl_ci.pushConstantRangeCount = 1;
    atmo_pl_ci.pPushConstantRanges = &atmo_push;

    VkPipelineLayout atmo_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &atmo_pl_ci, nullptr, &atmo_pipeline_layout));

    VkShaderModule atmo_shader = VK_NULL_HANDLE;
    {
        auto spv = load_spirv("shaders/atmosphere3d_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &atmo_shader));
    }

    VkPipeline atmo_pipeline = VK_NULL_HANDLE;
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = atmo_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = atmo_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &atmo_pipeline));
    }

    // ---- Sand compute descriptor set layout ------------------------------------
    VkDescriptorSetLayoutBinding sand_sim_bindings[4]{};
    sand_sim_bindings[0].binding = 0;
    sand_sim_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sand_sim_bindings[0].descriptorCount = 1;
    sand_sim_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    sand_sim_bindings[1].binding = 1;
    sand_sim_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sand_sim_bindings[1].descriptorCount = 1;
    sand_sim_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    sand_sim_bindings[2].binding = 2;
    sand_sim_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sand_sim_bindings[2].descriptorCount = 1;
    sand_sim_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    sand_sim_bindings[3].binding = 3;
    sand_sim_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sand_sim_bindings[3].descriptorCount = 1;
    sand_sim_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo sand_sim_dsl_ci{};
    sand_sim_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sand_sim_dsl_ci.bindingCount = 4;
    sand_sim_dsl_ci.pBindings = sand_sim_bindings;

    VkDescriptorSetLayout sand_sim_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &sand_sim_dsl_ci, nullptr, &sand_sim_desc_layout));

    // ---- Sand compute pipeline layout + pipeline --------------------------------
    VkPushConstantRange sand_sim_push{};
    sand_sim_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    sand_sim_push.offset = 0;
    sand_sim_push.size = sizeof(SandSimPC);

    VkPipelineLayoutCreateInfo sand_sim_pl_ci{};
    sand_sim_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    sand_sim_pl_ci.setLayoutCount = 1;
    sand_sim_pl_ci.pSetLayouts = &sand_sim_desc_layout;
    sand_sim_pl_ci.pushConstantRangeCount = 1;
    sand_sim_pl_ci.pPushConstantRanges = &sand_sim_push;

    VkPipelineLayout sand_sim_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &sand_sim_pl_ci, nullptr, &sand_sim_pipeline_layout));

    VkShaderModule sand_sim_shader = VK_NULL_HANDLE;
    {
        auto spv = load_spirv("shaders/sand_sim_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &sand_sim_shader));
    }

    VkPipeline sand_sim_pipeline = VK_NULL_HANDLE;
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = sand_sim_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = sand_sim_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &sand_sim_pipeline));
    }

    // ---- Sand render descriptor set layout + pipeline layout --------------------
    VkDescriptorSetLayoutBinding sand_render_bindings[2]{};
    sand_render_bindings[0].binding = 0;
    sand_render_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sand_render_bindings[0].descriptorCount = 1;
    sand_render_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    sand_render_bindings[1].binding = 1;
    sand_render_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sand_render_bindings[1].descriptorCount = 1;
    sand_render_bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo sand_render_dsl_ci{};
    sand_render_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sand_render_dsl_ci.bindingCount = 2;
    sand_render_dsl_ci.pBindings = sand_render_bindings;

    VkDescriptorSetLayout sand_render_desc_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &sand_render_dsl_ci, nullptr, &sand_render_desc_layout));

    VkPushConstantRange sand_render_push{};
    sand_render_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    sand_render_push.offset = 0;
    sand_render_push.size = sizeof(SandRenderPC);

    VkPipelineLayoutCreateInfo sand_render_pl_ci{};
    sand_render_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    sand_render_pl_ci.setLayoutCount = 1;
    sand_render_pl_ci.pSetLayouts = &sand_render_desc_layout;
    sand_render_pl_ci.pushConstantRangeCount = 1;
    sand_render_pl_ci.pPushConstantRanges = &sand_render_push;

    VkPipelineLayout sand_render_pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &sand_render_pl_ci, nullptr, &sand_render_pipeline_layout));

    // ---- Sand render shaders + pipeline -----------------------------------------
    VkShaderModule sand_render_vs = VK_NULL_HANDLE;
    {
        auto spv = load_spirv("shaders/sand_render_vs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &sand_render_vs));
    }
    VkShaderModule sand_render_fs = VK_NULL_HANDLE;
    {
        auto spv = load_spirv("shaders/sand_render_fs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &sand_render_fs));
    }

    VkPipeline sand_render_pipeline = VK_NULL_HANDLE;
    {
        VkPipelineShaderStageCreateInfo sand_stages[2]{};
        sand_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sand_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        sand_stages[0].module = sand_render_vs;
        sand_stages[0].pName = "main";
        sand_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sand_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        sand_stages[1].module = sand_render_fs;
        sand_stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo sand_vi{};
        sand_vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo sand_ia{};
        sand_ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        sand_ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

        VkPipelineDepthStencilStateCreateInfo sand_ds{};
        sand_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        sand_ds.depthTestEnable = VK_TRUE;
        sand_ds.depthWriteEnable = VK_FALSE;
        sand_ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

        VkPipelineColorBlendAttachmentState sand_ba{};
        sand_ba.blendEnable = VK_TRUE;
        sand_ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        sand_ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        sand_ba.colorBlendOp = VK_BLEND_OP_ADD;
        sand_ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        sand_ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        sand_ba.alphaBlendOp = VK_BLEND_OP_ADD;
        sand_ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                               | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo sand_cb{};
        sand_cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        sand_cb.attachmentCount = 1;
        sand_cb.pAttachments = &sand_ba;

        VkGraphicsPipelineCreateInfo sand_pipe_ci = gfx_pipe_ci;
        sand_pipe_ci.pStages = sand_stages;
        sand_pipe_ci.pVertexInputState = &sand_vi;
        sand_pipe_ci.pInputAssemblyState = &sand_ia;
        sand_pipe_ci.pDepthStencilState = &sand_ds;
        sand_pipe_ci.pColorBlendState = &sand_cb;
        sand_pipe_ci.layout = sand_render_pipeline_layout;

        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &sand_pipe_ci, nullptr, &sand_render_pipeline));
    }

    // ---- Descriptor pool + sets ---------------------------------------------
    // Counts: SWE init(2) + SWE step×2(8) + terrain brush(1) + gfx(4 incl sediment) + erosion×2(8) = needs ~23 descriptors
    VkDescriptorPoolSize pool_sizes[5]{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 25;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[1].descriptorCount = 16;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[2].descriptorCount = 4;
    pool_sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[3].descriptorCount = 42;
    pool_sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[4].descriptorCount = 5;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 20;
    dp_ci.poolSizeCount = 5;
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
    VkDescriptorSetLayout gfx_layouts_2[2] = {gfx_desc_layout, gfx_desc_layout};
    VkDescriptorSetAllocateInfo gfx_ds_ai{};
    gfx_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    gfx_ds_ai.descriptorPool = desc_pool;
    gfx_ds_ai.descriptorSetCount = 2;
    gfx_ds_ai.pSetLayouts = gfx_layouts_2;

    VkDescriptorSet gfx_desc_sets[2] = {};
    VK_CHECK(vkAllocateDescriptorSets(device, &gfx_ds_ai, gfx_desc_sets));

    // Terrain brush descriptor set
    VkDescriptorSetAllocateInfo tb_ds_ai{};
    tb_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    tb_ds_ai.descriptorPool = desc_pool;
    tb_ds_ai.descriptorSetCount = 1;
    tb_ds_ai.pSetLayouts = &tb_desc_layout;

    VkDescriptorSet tb_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &tb_ds_ai, &tb_desc_set));

    // Sand brush descriptor set (reuses tb_desc_layout — just a storage image at binding 0)
    VkDescriptorSet sand_brush_desc_set = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo sb_ds_ai{};
        sb_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        sb_ds_ai.descriptorPool = desc_pool;
        sb_ds_ai.descriptorSetCount = 1;
        sb_ds_ai.pSetLayouts = &tb_desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &sb_ds_ai, &sand_brush_desc_set));
    }

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

    // Graphics descriptors (2 sets — differ only in binding 5: cloud volume a vs b)
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

        VkDescriptorImageInfo shadow_info{};
        shadow_info.imageView = atmo_shadow.view;
        shadow_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        shadow_info.sampler = sampler;

        VkDescriptorImageInfo vol_a_info{};
        vol_a_info.imageView = atmo_state_a.view;
        vol_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_a_info.sampler = sampler;

        VkDescriptorImageInfo vol_b_info{};
        vol_b_info.imageView = atmo_state_b.view;
        vol_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_b_info.sampler = sampler;

        VkDescriptorImageInfo vol_infos[2] = {vol_a_info, vol_b_info};

        VkDescriptorImageInfo wind_a_gfx{};
        wind_a_gfx.imageView = wind_field_a.view;
        wind_a_gfx.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wind_a_gfx.sampler = sampler;

        VkDescriptorImageInfo wind_b_gfx{};
        wind_b_gfx.imageView = wind_field_b.view;
        wind_b_gfx.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wind_b_gfx.sampler = sampler;

        VkDescriptorImageInfo wind_infos[2] = {wind_a_gfx, wind_b_gfx};

        VkDescriptorImageInfo sand_dep_info{};
        sand_dep_info.imageView = sand_deposit.view;
        sand_dep_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sand_dep_info.sampler = sampler;

        for (int s = 0; s < 2; s++) {
            VkWriteDescriptorSet writes[8]{};

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = gfx_desc_sets[s];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &ubo_buf_info;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = gfx_desc_sets[s];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &heightmap_info;

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = gfx_desc_sets[s];
            writes[2].dstBinding = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[2].pImageInfo = &swe_out_info;

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = gfx_desc_sets[s];
            writes[3].dstBinding = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[3].pImageInfo = &sediment_info;

            writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = gfx_desc_sets[s];
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[4].pImageInfo = &shadow_info;

            writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[5].dstSet = gfx_desc_sets[s];
            writes[5].dstBinding = 5;
            writes[5].descriptorCount = 1;
            writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[5].pImageInfo = &vol_infos[s];

            writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[6].dstSet = gfx_desc_sets[s];
            writes[6].dstBinding = 6;
            writes[6].descriptorCount = 1;
            writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[6].pImageInfo = &wind_infos[s];

            writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[7].dstSet = gfx_desc_sets[s];
            writes[7].dstBinding = 7;
            writes[7].descriptorCount = 1;
            writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[7].pImageInfo = &sand_dep_info;

            vkUpdateDescriptorSets(device, 8, writes, 0, nullptr);
        }
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

    // Sand brush descriptor (same layout as terrain brush, points to sand_deposit)
    {
        VkDescriptorImageInfo sb_img_info{};
        sb_img_info.imageView = sand_deposit.view;
        sb_img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet sb_write{};
        sb_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sb_write.dstSet = sand_brush_desc_set;
        sb_write.dstBinding = 0;
        sb_write.descriptorCount = 1;
        sb_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sb_write.pImageInfo = &sb_img_info;

        vkUpdateDescriptorSets(device, 1, &sb_write, 0, nullptr);
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

    // Atmosphere descriptor sets (2 for ping-pong)
    VkDescriptorSetLayout atmo_layouts[2] = {atmo_desc_layout, atmo_desc_layout};
    VkDescriptorSetAllocateInfo atmo_ds_ai{};
    atmo_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    atmo_ds_ai.descriptorPool = desc_pool;
    atmo_ds_ai.descriptorSetCount = 2;
    atmo_ds_ai.pSetLayouts = atmo_layouts;

    VkDescriptorSet atmo_desc_sets[2] = {};
    VK_CHECK(vkAllocateDescriptorSets(device, &atmo_ds_ai, atmo_desc_sets));

    // Write atmosphere descriptors (3D volumes + 2D shadow)
    {
        VkDescriptorImageInfo terrain_sampler_info{};
        terrain_sampler_info.sampler = terrain_linear_sampler;
        terrain_sampler_info.imageView = heightmap_gpu.view;
        terrain_sampler_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo atmo_a_info{};
        atmo_a_info.imageView = atmo_state_a.view;
        atmo_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo atmo_b_info{};
        atmo_b_info.imageView = atmo_state_b.view;
        atmo_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo wind_a_info{};
        wind_a_info.imageView = wind_field_a.view;
        wind_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo wind_b_info{};
        wind_b_info.imageView = wind_field_b.view;
        wind_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo shadow_info{};
        shadow_info.imageView = atmo_shadow.view;
        shadow_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Set 0: state_read=a, state_write=b, wind_read=a, wind_write=b
        VkWriteDescriptorSet aw0[6]{};
        aw0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[0].dstSet = atmo_desc_sets[0];
        aw0[0].dstBinding = 0;
        aw0[0].descriptorCount = 1;
        aw0[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        aw0[0].pImageInfo = &terrain_sampler_info;

        aw0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[1].dstSet = atmo_desc_sets[0];
        aw0[1].dstBinding = 1;
        aw0[1].descriptorCount = 1;
        aw0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw0[1].pImageInfo = &atmo_a_info;

        aw0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[2].dstSet = atmo_desc_sets[0];
        aw0[2].dstBinding = 2;
        aw0[2].descriptorCount = 1;
        aw0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw0[2].pImageInfo = &atmo_b_info;

        aw0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[3].dstSet = atmo_desc_sets[0];
        aw0[3].dstBinding = 3;
        aw0[3].descriptorCount = 1;
        aw0[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw0[3].pImageInfo = &wind_a_info;

        aw0[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[4].dstSet = atmo_desc_sets[0];
        aw0[4].dstBinding = 4;
        aw0[4].descriptorCount = 1;
        aw0[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw0[4].pImageInfo = &wind_b_info;

        aw0[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[5].dstSet = atmo_desc_sets[0];
        aw0[5].dstBinding = 5;
        aw0[5].descriptorCount = 1;
        aw0[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw0[5].pImageInfo = &shadow_info;

        vkUpdateDescriptorSets(device, 6, aw0, 0, nullptr);

        // Set 1: state_read=b, state_write=a, wind_read=b, wind_write=a
        VkWriteDescriptorSet aw1[6]{};
        aw1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[0].dstSet = atmo_desc_sets[1];
        aw1[0].dstBinding = 0;
        aw1[0].descriptorCount = 1;
        aw1[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        aw1[0].pImageInfo = &terrain_sampler_info;

        aw1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[1].dstSet = atmo_desc_sets[1];
        aw1[1].dstBinding = 1;
        aw1[1].descriptorCount = 1;
        aw1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw1[1].pImageInfo = &atmo_b_info;

        aw1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[2].dstSet = atmo_desc_sets[1];
        aw1[2].dstBinding = 2;
        aw1[2].descriptorCount = 1;
        aw1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw1[2].pImageInfo = &atmo_a_info;

        aw1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[3].dstSet = atmo_desc_sets[1];
        aw1[3].dstBinding = 3;
        aw1[3].descriptorCount = 1;
        aw1[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw1[3].pImageInfo = &wind_b_info;

        aw1[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[4].dstSet = atmo_desc_sets[1];
        aw1[4].dstBinding = 4;
        aw1[4].descriptorCount = 1;
        aw1[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw1[4].pImageInfo = &wind_a_info;

        aw1[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[5].dstSet = atmo_desc_sets[1];
        aw1[5].dstBinding = 5;
        aw1[5].descriptorCount = 1;
        aw1[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw1[5].pImageInfo = &shadow_info;

        vkUpdateDescriptorSets(device, 6, aw1, 0, nullptr);
    }

    // Sand compute descriptor sets (2 for wind ping-pong)
    VkDescriptorSetLayout sand_sim_layouts[2] = {sand_sim_desc_layout, sand_sim_desc_layout};
    VkDescriptorSetAllocateInfo sand_sim_ds_ai{};
    sand_sim_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sand_sim_ds_ai.descriptorPool = desc_pool;
    sand_sim_ds_ai.descriptorSetCount = 2;
    sand_sim_ds_ai.pSetLayouts = sand_sim_layouts;

    VkDescriptorSet sand_sim_desc_sets[2] = {};
    VK_CHECK(vkAllocateDescriptorSets(device, &sand_sim_ds_ai, sand_sim_desc_sets));

    // Sand render descriptor set
    VkDescriptorSetAllocateInfo sand_render_ds_ai{};
    sand_render_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sand_render_ds_ai.descriptorPool = desc_pool;
    sand_render_ds_ai.descriptorSetCount = 1;
    sand_render_ds_ai.pSetLayouts = &sand_render_desc_layout;

    VkDescriptorSet sand_render_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &sand_render_ds_ai, &sand_render_desc_set));

    // Write sand compute descriptors
    {
        VkDescriptorImageInfo terrain_sampler_info{};
        terrain_sampler_info.sampler = terrain_linear_sampler;
        terrain_sampler_info.imageView = heightmap_gpu.view;
        terrain_sampler_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo wind_a_info{};
        wind_a_info.sampler = sampler;
        wind_a_info.imageView = wind_field_a.view;
        wind_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo wind_b_info{};
        wind_b_info.sampler = sampler;
        wind_b_info.imageView = wind_field_b.view;
        wind_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo sand_buf_info{};
        sand_buf_info.buffer = sand_particle_buf;
        sand_buf_info.offset = 0;
        sand_buf_info.range = SAND_MAX_PARTICLES * 32;

        VkDescriptorImageInfo sand_dep_sim_info{};
        sand_dep_sim_info.sampler = sampler;
        sand_dep_sim_info.imageView = sand_deposit.view;
        sand_dep_sim_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Set 0: terrain + wind_a + particles + sand_deposit
        VkWriteDescriptorSet sw0[4]{};
        sw0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[0].dstSet = sand_sim_desc_sets[0];
        sw0[0].dstBinding = 0;
        sw0[0].descriptorCount = 1;
        sw0[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw0[0].pImageInfo = &terrain_sampler_info;

        sw0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[1].dstSet = sand_sim_desc_sets[0];
        sw0[1].dstBinding = 1;
        sw0[1].descriptorCount = 1;
        sw0[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw0[1].pImageInfo = &wind_a_info;

        sw0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[2].dstSet = sand_sim_desc_sets[0];
        sw0[2].dstBinding = 2;
        sw0[2].descriptorCount = 1;
        sw0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sw0[2].pBufferInfo = &sand_buf_info;

        sw0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[3].dstSet = sand_sim_desc_sets[0];
        sw0[3].dstBinding = 3;
        sw0[3].descriptorCount = 1;
        sw0[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw0[3].pImageInfo = &sand_dep_sim_info;

        vkUpdateDescriptorSets(device, 4, sw0, 0, nullptr);

        // Set 1: terrain + wind_b + particles + sand_deposit
        VkWriteDescriptorSet sw1[4]{};
        sw1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[0].dstSet = sand_sim_desc_sets[1];
        sw1[0].dstBinding = 0;
        sw1[0].descriptorCount = 1;
        sw1[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw1[0].pImageInfo = &terrain_sampler_info;

        sw1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[1].dstSet = sand_sim_desc_sets[1];
        sw1[1].dstBinding = 1;
        sw1[1].descriptorCount = 1;
        sw1[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw1[1].pImageInfo = &wind_b_info;

        sw1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[2].dstSet = sand_sim_desc_sets[1];
        sw1[2].dstBinding = 2;
        sw1[2].descriptorCount = 1;
        sw1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sw1[2].pBufferInfo = &sand_buf_info;

        sw1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[3].dstSet = sand_sim_desc_sets[1];
        sw1[3].dstBinding = 3;
        sw1[3].descriptorCount = 1;
        sw1[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw1[3].pImageInfo = &sand_dep_sim_info;

        vkUpdateDescriptorSets(device, 4, sw1, 0, nullptr);
    }

    // Write sand render descriptors
    {
        VkDescriptorBufferInfo ubo_buf_info{};
        ubo_buf_info.buffer = camera_ubo;
        ubo_buf_info.offset = 0;
        ubo_buf_info.range = sizeof(CameraData);

        VkDescriptorBufferInfo sand_buf_info{};
        sand_buf_info.buffer = sand_particle_buf;
        sand_buf_info.offset = 0;
        sand_buf_info.range = SAND_MAX_PARTICLES * 32;

        VkWriteDescriptorSet srw[2]{};
        srw[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        srw[0].dstSet = sand_render_desc_set;
        srw[0].dstBinding = 0;
        srw[0].descriptorCount = 1;
        srw[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        srw[0].pBufferInfo = &ubo_buf_info;

        srw[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        srw[1].dstSet = sand_render_desc_set;
        srw[1].dstBinding = 1;
        srw[1].descriptorCount = 1;
        srw[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        srw[1].pBufferInfo = &sand_buf_info;

        vkUpdateDescriptorSets(device, 2, srw, 0, nullptr);
    }

    // Terrain gen descriptor set
    VkDescriptorSetAllocateInfo tgen_ds_ai{};
    tgen_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    tgen_ds_ai.descriptorPool = desc_pool;
    tgen_ds_ai.descriptorSetCount = 1;
    tgen_ds_ai.pSetLayouts = &terrain_gen_desc_layout;

    VkDescriptorSet terrain_gen_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &tgen_ds_ai, &terrain_gen_desc_set));

    // Write terrain gen descriptor (binding 0 = clipmap heightmap array as storage image)
    {
        VkDescriptorImageInfo tgen_img_info{};
        tgen_img_info.imageView = clipmap_hm_view;
        tgen_img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo stamp_desc_info{};
        stamp_desc_info.buffer = stamp_buf;
        stamp_desc_info.offset = 0;
        stamp_desc_info.range = MAX_STAMPS * sizeof(TerrainStamp);

        VkWriteDescriptorSet tgen_writes[2]{};
        tgen_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        tgen_writes[0].dstSet = terrain_gen_desc_set;
        tgen_writes[0].dstBinding = 0;
        tgen_writes[0].descriptorCount = 1;
        tgen_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        tgen_writes[0].pImageInfo = &tgen_img_info;

        tgen_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        tgen_writes[1].dstSet = terrain_gen_desc_set;
        tgen_writes[1].dstBinding = 1;
        tgen_writes[1].descriptorCount = 1;
        tgen_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        tgen_writes[1].pBufferInfo = &stamp_desc_info;

        vkUpdateDescriptorSets(device, 2, tgen_writes, 0, nullptr);
    }

    // Clipmap graphics descriptor sets (2 for atmosphere ping-pong)
    VkDescriptorSetLayout clip_gfx_layouts[2] = {gfx_desc_layout, gfx_desc_layout};
    VkDescriptorSetAllocateInfo clip_gfx_ds_ai{};
    clip_gfx_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    clip_gfx_ds_ai.descriptorPool = desc_pool;
    clip_gfx_ds_ai.descriptorSetCount = 2;
    clip_gfx_ds_ai.pSetLayouts = clip_gfx_layouts;

    VkDescriptorSet clipmap_gfx_desc_sets[2] = {};
    VK_CHECK(vkAllocateDescriptorSets(device, &clip_gfx_ds_ai, clipmap_gfx_desc_sets));

    // Write clipmap gfx descriptors (same as gfx_desc_sets but binding 1 = clipmap heightmap array view)
    {
        VkDescriptorBufferInfo ubo_buf_info{};
        ubo_buf_info.buffer = camera_ubo;
        ubo_buf_info.offset = 0;
        ubo_buf_info.range = sizeof(CameraData);

        VkDescriptorImageInfo clip_heightmap_info{};
        clip_heightmap_info.imageView = clipmap_hm_view;
        clip_heightmap_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        clip_heightmap_info.sampler = sampler;

        VkDescriptorImageInfo swe_out_info{};
        swe_out_info.imageView = swe_output.view;
        swe_out_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        swe_out_info.sampler = sampler;

        VkDescriptorImageInfo sediment_info{};
        sediment_info.imageView = sediment_a.view;
        sediment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sediment_info.sampler = sampler;

        VkDescriptorImageInfo shadow_info{};
        shadow_info.imageView = atmo_shadow.view;
        shadow_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        shadow_info.sampler = sampler;

        VkDescriptorImageInfo vol_a_info{};
        vol_a_info.imageView = atmo_state_a.view;
        vol_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_a_info.sampler = sampler;

        VkDescriptorImageInfo vol_b_info{};
        vol_b_info.imageView = atmo_state_b.view;
        vol_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_b_info.sampler = sampler;

        VkDescriptorImageInfo vol_infos[2] = {vol_a_info, vol_b_info};

        VkDescriptorImageInfo wind_a_gfx{};
        wind_a_gfx.imageView = wind_field_a.view;
        wind_a_gfx.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wind_a_gfx.sampler = sampler;

        VkDescriptorImageInfo wind_b_gfx{};
        wind_b_gfx.imageView = wind_field_b.view;
        wind_b_gfx.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wind_b_gfx.sampler = sampler;

        VkDescriptorImageInfo wind_infos[2] = {wind_a_gfx, wind_b_gfx};

        VkDescriptorImageInfo sand_dep_info{};
        sand_dep_info.imageView = sand_deposit.view;
        sand_dep_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sand_dep_info.sampler = sampler;

        for (int s = 0; s < 2; s++) {
            VkWriteDescriptorSet writes[8]{};

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = clipmap_gfx_desc_sets[s];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &ubo_buf_info;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = clipmap_gfx_desc_sets[s];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &clip_heightmap_info;

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = clipmap_gfx_desc_sets[s];
            writes[2].dstBinding = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[2].pImageInfo = &swe_out_info;

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = clipmap_gfx_desc_sets[s];
            writes[3].dstBinding = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[3].pImageInfo = &sediment_info;

            writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = clipmap_gfx_desc_sets[s];
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[4].pImageInfo = &shadow_info;

            writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[5].dstSet = clipmap_gfx_desc_sets[s];
            writes[5].dstBinding = 5;
            writes[5].descriptorCount = 1;
            writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[5].pImageInfo = &vol_infos[s];

            writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[6].dstSet = clipmap_gfx_desc_sets[s];
            writes[6].dstBinding = 6;
            writes[6].descriptorCount = 1;
            writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[6].pImageInfo = &wind_infos[s];

            writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[7].dstSet = clipmap_gfx_desc_sets[s];
            writes[7].dstBinding = 7;
            writes[7].descriptorCount = 1;
            writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[7].pImageInfo = &sand_dep_info;

            vkUpdateDescriptorSets(device, 8, writes, 0, nullptr);
        }
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

        VkImageMemoryBarrier2 init_barriers[12]{};
        for (int i = 0; i < 12; ++i) {
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

        // Atmosphere images (UNDEFINED -> GENERAL) — 4 3D volumes + 1 2D shadow
        for (int i = 5; i < 10; ++i) {
            init_barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            init_barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        }
        init_barriers[5].image = atmo_state_a.image;
        init_barriers[6].image = atmo_state_b.image;
        init_barriers[7].image = wind_field_a.image;
        init_barriers[8].image = wind_field_b.image;
        init_barriers[9].image = atmo_shadow.image;

        init_barriers[10] = init_barriers[3]; // copy template from sediment (TRANSFER_DST)
        init_barriers[10].image = sand_deposit.image;

        // Tile pool (UNDEFINED -> GENERAL)
        init_barriers[11].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        init_barriers[11].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        init_barriers[11].image = clipmap_hm_image;
        init_barriers[11].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};

        VkDependencyInfo dep_init{};
        dep_init.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_init.imageMemoryBarrierCount = 12;
        dep_init.pImageMemoryBarriers = init_barriers;
        vkCmdPipelineBarrier2(cmd, &dep_init);

        VkClearColorValue clear_zero{};
        clear_zero.float32[0] = 0.0f;
        VkImageSubresourceRange clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, sediment_a.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
        vkCmdClearColorImage(cmd, sediment_b.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
        vkCmdClearColorImage(cmd, sand_deposit.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
        vkCmdFillBuffer(cmd, sand_particle_buf, 0, SAND_MAX_PARTICLES * 32, 0);

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

    // ---- Shader hot-reload helper ------------------------------------------
    auto reload_all_shaders = [&]() {
        std::fprintf(stderr, "[shader reload] Compiling shaders...\n");

        // Shader entries: {hlsl_relative, spv_relative, stage}
        struct ShaderEntry {
            const char* hlsl;
            const char* spv;
            const char* stage;
        };

        ShaderEntry entries[] = {
            {"../shaders/swe_init.hlsl",       "shaders/swe_init.spv",       "compute"},
            {"../shaders/swe_step.hlsl",       "shaders/swe_step.spv",       "compute"},
            {"../shaders/terrain.vs.hlsl",     "shaders/terrain_vs.spv",     "vertex"},
            {"../shaders/terrain.fs.hlsl",     "shaders/terrain_fs.spv",     "fragment"},
            {"../shaders/water.vs.hlsl",       "shaders/water_vs.spv",       "vertex"},
            {"../shaders/water.fs.hlsl",       "shaders/water_fs.spv",       "fragment"},
            {"../shaders/terrain_brush.hlsl",  "shaders/terrain_brush.spv",  "compute"},
            {"../shaders/erosion.hlsl",        "shaders/erosion.spv",        "compute"},
            {"../shaders/atmosphere3d.cs.hlsl", "shaders/atmosphere3d_cs.spv", "compute"},
            {"../shaders/cloud_raymarch.vs.hlsl", "shaders/cloud_raymarch_vs.spv", "vertex"},
            {"../shaders/cloud_raymarch.fs.hlsl", "shaders/cloud_raymarch_fs.spv", "fragment"},
            {"../shaders/sand_sim.cs.hlsl",      "shaders/sand_sim_cs.spv",      "compute"},
            {"../shaders/sand_render.vs.hlsl",   "shaders/sand_render_vs.spv",   "vertex"},
            {"../shaders/sand_render.fs.hlsl",   "shaders/sand_render_fs.spv",   "fragment"},
            {"../shaders/planet_gen.cs.hlsl",    "shaders/planet_gen_cs.spv",    "compute"},
        };
        constexpr int NUM_SHADERS = 15;

        // Compile all shaders first; abort on any failure
        for (int i = 0; i < NUM_SHADERS; ++i) {
            char cmd[512];
            std::snprintf(cmd, sizeof(cmd),
                "glslc -x hlsl -fshader-stage=%s -fentry-point=main "
                "--target-env=vulkan1.3 %s -o %s 2>&1",
                entries[i].stage, entries[i].hlsl, entries[i].spv);

            FILE* proc = popen(cmd, "r");
            if (!proc) {
                std::fprintf(stderr, "[shader reload] Failed to run glslc for %s\n", entries[i].hlsl);
                return;
            }

            char line[1024];
            while (std::fgets(line, sizeof(line), proc)) {
                std::fprintf(stderr, "  %s", line);
            }

            int status = pclose(proc);
            if (status != 0) {
                std::fprintf(stderr, "[shader reload] FAILED to compile %s (exit %d). Pipelines unchanged.\n",
                             entries[i].hlsl, status);
                return;
            }
        }

        std::fprintf(stderr, "[shader reload] All %d shaders compiled. Recreating pipelines...\n", NUM_SHADERS);

        // Wait for GPU idle before touching anything
        vkDeviceWaitIdle(device);

        // Destroy old shader modules
        vkDestroyShaderModule(device, swe_init_shader, nullptr);
        vkDestroyShaderModule(device, swe_step_shader, nullptr);
        vkDestroyShaderModule(device, terrain_vs, nullptr);
        vkDestroyShaderModule(device, terrain_fs, nullptr);
        vkDestroyShaderModule(device, water_vs, nullptr);
        vkDestroyShaderModule(device, water_fs, nullptr);
        vkDestroyShaderModule(device, terrain_brush_shader, nullptr);
        vkDestroyShaderModule(device, erosion_shader, nullptr);
        vkDestroyShaderModule(device, atmo_shader, nullptr);
        vkDestroyShaderModule(device, raymarch_vs, nullptr);
        vkDestroyShaderModule(device, raymarch_fs, nullptr);
        vkDestroyShaderModule(device, sand_sim_shader, nullptr);
        vkDestroyShaderModule(device, sand_render_vs, nullptr);
        vkDestroyShaderModule(device, sand_render_fs, nullptr);
        vkDestroyShaderModule(device, terrain_gen_shader, nullptr);

        // Destroy old pipelines
        vkDestroyPipeline(device, swe_init_pipeline, nullptr);
        vkDestroyPipeline(device, swe_step_pipeline, nullptr);
        vkDestroyPipeline(device, gfx_pipeline, nullptr);
        vkDestroyPipeline(device, water_pipeline, nullptr);
        vkDestroyPipeline(device, raymarch_pipeline, nullptr);
        vkDestroyPipeline(device, terrain_brush_pipeline, nullptr);
        vkDestroyPipeline(device, erosion_pipeline, nullptr);
        vkDestroyPipeline(device, atmo_pipeline, nullptr);
        vkDestroyPipeline(device, sand_sim_pipeline, nullptr);
        vkDestroyPipeline(device, sand_render_pipeline, nullptr);
        vkDestroyPipeline(device, terrain_gen_pipeline, nullptr);
        vkDestroyPipeline(device, clipmap_terrain_pipeline, nullptr);

        // Destroy old pipeline layouts
        vkDestroyPipelineLayout(device, swe_init_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, swe_step_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, gfx_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, raymarch_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, tb_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, ero_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, atmo_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, sand_sim_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, sand_render_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, terrain_gen_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(device, clipmap_gfx_pipeline_layout, nullptr);

        // Reload SPIR-V and recreate shader modules
        {
            auto spv = load_spirv("shaders/swe_init.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &swe_init_shader));
        }
        {
            auto spv = load_spirv("shaders/swe_step.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &swe_step_shader));
        }
        {
            auto spv = load_spirv("shaders/terrain_vs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &terrain_vs));
        }
        {
            auto spv = load_spirv("shaders/terrain_fs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &terrain_fs));
        }
        {
            auto spv = load_spirv("shaders/water_vs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &water_vs));
        }
        {
            auto spv = load_spirv("shaders/water_fs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &water_fs));
        }
        {
            auto spv = load_spirv("shaders/terrain_brush.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &terrain_brush_shader));
        }
        {
            auto spv = load_spirv("shaders/erosion.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &erosion_shader));
        }
        {
            auto spv = load_spirv("shaders/atmosphere3d_cs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &atmo_shader));
        }
        {
            auto spv = load_spirv("shaders/cloud_raymarch_vs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &raymarch_vs));
        }
        {
            auto spv = load_spirv("shaders/cloud_raymarch_fs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &raymarch_fs));
        }
        {
            auto spv = load_spirv("shaders/sand_sim_cs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &sand_sim_shader));
        }
        {
            auto spv = load_spirv("shaders/sand_render_vs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &sand_render_vs));
        }
        {
            auto spv = load_spirv("shaders/sand_render_fs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &sand_render_fs));
        }
        {
            auto spv = load_spirv("shaders/planet_gen_cs.spv");
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(uint32_t);
            ci.pCode = spv.data();
            VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &terrain_gen_shader));
        }

        // Recreate pipeline layouts
        // (descriptor set layouts are unchanged, only pipeline layouts are rebuilt)
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.offset = 0;
            push.size = sizeof(SweInitPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &swe_init_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &swe_init_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.offset = 0;
            push.size = sizeof(SweStepPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &swe_step_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &swe_step_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            push.offset = 0;
            push.size = sizeof(GfxPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &gfx_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &gfx_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            push.offset = 0;
            push.size = sizeof(RaymarchPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &gfx_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &raymarch_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.offset = 0;
            push.size = sizeof(TerrainBrushPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &tb_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &tb_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.offset = 0;
            push.size = sizeof(ErosionPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &ero_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &ero_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.offset = 0;
            push.size = sizeof(Atmo3DPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &atmo_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &atmo_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.offset = 0;
            push.size = sizeof(SandSimPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &sand_sim_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &sand_sim_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            push.offset = 0;
            push.size = sizeof(SandRenderPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &sand_render_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &sand_render_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.offset = 0;
            push.size = sizeof(PlanetGenPC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &terrain_gen_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &terrain_gen_pipeline_layout));
        }
        {
            VkPushConstantRange push{};
            push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            push.offset = 0;
            push.size = sizeof(PlanetTilePC);

            VkPipelineLayoutCreateInfo pl_ci{};
            pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_ci.setLayoutCount = 1;
            pl_ci.pSetLayouts = &gfx_desc_layout;
            pl_ci.pushConstantRangeCount = 1;
            pl_ci.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &clipmap_gfx_pipeline_layout));
        }

        // Recreate compute pipelines
        {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = swe_init_shader;
            stage.pName = "main";

            VkComputePipelineCreateInfo cp_ci{};
            cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp_ci.stage = stage;
            cp_ci.layout = swe_init_pipeline_layout;
            VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &swe_init_pipeline));
        }
        {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = swe_step_shader;
            stage.pName = "main";

            VkComputePipelineCreateInfo cp_ci{};
            cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp_ci.stage = stage;
            cp_ci.layout = swe_step_pipeline_layout;
            VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &swe_step_pipeline));
        }
        {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = terrain_brush_shader;
            stage.pName = "main";

            VkComputePipelineCreateInfo cp_ci{};
            cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp_ci.stage = stage;
            cp_ci.layout = tb_pipeline_layout;
            VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &terrain_brush_pipeline));
        }
        {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = erosion_shader;
            stage.pName = "main";

            VkComputePipelineCreateInfo cp_ci{};
            cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp_ci.stage = stage;
            cp_ci.layout = ero_pipeline_layout;
            VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &erosion_pipeline));
        }
        {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = atmo_shader;
            stage.pName = "main";

            VkComputePipelineCreateInfo cp_ci{};
            cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp_ci.stage = stage;
            cp_ci.layout = atmo_pipeline_layout;
            VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &atmo_pipeline));
        }
        {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = sand_sim_shader;
            stage.pName = "main";

            VkComputePipelineCreateInfo cp_ci{};
            cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp_ci.stage = stage;
            cp_ci.layout = sand_sim_pipeline_layout;
            VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &sand_sim_pipeline));
        }
        {
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = terrain_gen_shader;
            stage.pName = "main";

            VkComputePipelineCreateInfo cp_ci{};
            cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp_ci.stage = stage;
            cp_ci.layout = terrain_gen_pipeline_layout;
            VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &terrain_gen_pipeline));
        }

        // Recreate graphics pipelines (terrain + water)
        {
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = terrain_vs;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = terrain_fs;
            stages[1].pName = "main";

            VkVertexInputBindingDescription vb{};
            vb.binding = 0;
            vb.stride = sizeof(float) * 2;
            vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription va{};
            va.binding = 0;
            va.location = 0;
            va.format = VK_FORMAT_R32G32_SFLOAT;
            va.offset = 0;

            VkPipelineVertexInputStateCreateInfo vi{};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vi.vertexBindingDescriptionCount = 1;
            vi.pVertexBindingDescriptions = &vb;
            vi.vertexAttributeDescriptionCount = 1;
            vi.pVertexAttributeDescriptions = &va;

            VkPipelineInputAssemblyStateCreateInfo ia{};
            ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo vp{};
            vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            vp.viewportCount = 1;
            vp.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rs{};
            rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rs.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo ms{};
            ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            ds.depthTestEnable = VK_TRUE;
            ds.depthWriteEnable = VK_TRUE;
            ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

            VkPipelineColorBlendAttachmentState ba{};
            ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                              | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo cb{};
            cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments = &ba;

            VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dy{};
            dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dy.dynamicStateCount = 2;
            dy.pDynamicStates = dyn;

            VkFormat cf = VK_FORMAT_B8G8R8A8_UNORM;
            VkFormat df = VK_FORMAT_D32_SFLOAT;

            VkPipelineRenderingCreateInfo ri{};
            ri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachmentFormats = &cf;
            ri.depthAttachmentFormat = df;

            VkGraphicsPipelineCreateInfo gp_ci{};
            gp_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gp_ci.pNext = &ri;
            gp_ci.stageCount = 2;
            gp_ci.pStages = stages;
            gp_ci.pVertexInputState = &vi;
            gp_ci.pInputAssemblyState = &ia;
            gp_ci.pViewportState = &vp;
            gp_ci.pRasterizationState = &rs;
            gp_ci.pMultisampleState = &ms;
            gp_ci.pDepthStencilState = &ds;
            gp_ci.pColorBlendState = &cb;
            gp_ci.pDynamicState = &dy;
            gp_ci.layout = gfx_pipeline_layout;

            VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp_ci, nullptr, &gfx_pipeline));

            // Clipmap terrain pipeline (same state, different layout)
            VkGraphicsPipelineCreateInfo clip_ci = gp_ci;
            clip_ci.layout = clipmap_gfx_pipeline_layout;
            VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &clip_ci, nullptr, &clipmap_terrain_pipeline));

            // Water pipeline (alpha blend, no depth write)
            VkPipelineShaderStageCreateInfo w_stages[2]{};
            w_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            w_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            w_stages[0].module = water_vs;
            w_stages[0].pName = "main";
            w_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            w_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            w_stages[1].module = water_fs;
            w_stages[1].pName = "main";

            VkPipelineDepthStencilStateCreateInfo w_ds{};
            w_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            w_ds.depthTestEnable = VK_TRUE;
            w_ds.depthWriteEnable = VK_FALSE;
            w_ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

            VkPipelineColorBlendAttachmentState w_ba{};
            w_ba.blendEnable = VK_TRUE;
            w_ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            w_ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            w_ba.colorBlendOp = VK_BLEND_OP_ADD;
            w_ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            w_ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            w_ba.alphaBlendOp = VK_BLEND_OP_ADD;
            w_ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo w_cb{};
            w_cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            w_cb.attachmentCount = 1;
            w_cb.pAttachments = &w_ba;

            VkGraphicsPipelineCreateInfo w_ci = gp_ci;
            w_ci.pStages = w_stages;
            w_ci.pDepthStencilState = &w_ds;
            w_ci.pColorBlendState = &w_cb;

            VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &w_ci, nullptr, &water_pipeline));

            // Raymarch pipeline (fullscreen, no depth, alpha blend)
            VkPipelineShaderStageCreateInfo rm_stg[2]{};
            rm_stg[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            rm_stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            rm_stg[0].module = raymarch_vs;
            rm_stg[0].pName = "main";
            rm_stg[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            rm_stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            rm_stg[1].module = raymarch_fs;
            rm_stg[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo rm_vi_s{};
            rm_vi_s.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

            VkPipelineDepthStencilStateCreateInfo rm_ds_s{};
            rm_ds_s.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            rm_ds_s.depthTestEnable = VK_FALSE;
            rm_ds_s.depthWriteEnable = VK_FALSE;

            VkGraphicsPipelineCreateInfo rm_ci = gp_ci;
            rm_ci.pStages = rm_stg;
            rm_ci.pVertexInputState = &rm_vi_s;
            rm_ci.pDepthStencilState = &rm_ds_s;
            rm_ci.pColorBlendState = &w_cb;
            rm_ci.layout = raymarch_pipeline_layout;

            VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &rm_ci, nullptr, &raymarch_pipeline));

            // Sand render pipeline (line list, no vertex input, depth test no write, alpha blend)
            VkPipelineShaderStageCreateInfo sand_stg[2]{};
            sand_stg[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            sand_stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            sand_stg[0].module = sand_render_vs;
            sand_stg[0].pName = "main";
            sand_stg[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            sand_stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            sand_stg[1].module = sand_render_fs;
            sand_stg[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo sand_vi_s{};
            sand_vi_s.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

            VkPipelineInputAssemblyStateCreateInfo sand_ia_s{};
            sand_ia_s.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            sand_ia_s.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

            VkPipelineDepthStencilStateCreateInfo sand_ds_s{};
            sand_ds_s.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            sand_ds_s.depthTestEnable = VK_TRUE;
            sand_ds_s.depthWriteEnable = VK_FALSE;
            sand_ds_s.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

            VkPipelineColorBlendAttachmentState sand_ba_s{};
            sand_ba_s.blendEnable = VK_TRUE;
            sand_ba_s.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            sand_ba_s.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            sand_ba_s.colorBlendOp = VK_BLEND_OP_ADD;
            sand_ba_s.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            sand_ba_s.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            sand_ba_s.alphaBlendOp = VK_BLEND_OP_ADD;
            sand_ba_s.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                     | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo sand_cb_s{};
            sand_cb_s.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            sand_cb_s.attachmentCount = 1;
            sand_cb_s.pAttachments = &sand_ba_s;

            VkGraphicsPipelineCreateInfo sand_ci = gp_ci;
            sand_ci.pStages = sand_stg;
            sand_ci.pVertexInputState = &sand_vi_s;
            sand_ci.pInputAssemblyState = &sand_ia_s;
            sand_ci.pDepthStencilState = &sand_ds_s;
            sand_ci.pColorBlendState = &sand_cb_s;
            sand_ci.layout = sand_render_pipeline_layout;

            VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &sand_ci, nullptr, &sand_render_pipeline));
        }

        std::fprintf(stderr, "[shader reload] Done — %d shaders reloaded, all pipelines recreated.\n", NUM_SHADERS);
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
    std::printf("Shaders: press F5 to hot-reload all shaders\n");
    std::fflush(stdout);

    // ---- Planet quadtree helpers ---------------------------------------------
    auto face_uv_to_cube = [](double u, double v, uint32_t face) -> glm::dvec3 {
        switch (face) {
            case 0: return {1.0, v, -u};
            case 1: return {-1.0, v, u};
            case 2: return {u, 1.0, -v};
            case 3: return {u, -1.0, v};
            case 4: return {u, v, 1.0};
            case 5: return {-u, v, -1.0};
            default: return {0, 0, 0};
        }
    };

    auto cube_to_sphere_d = [](glm::dvec3 p) -> glm::dvec3 {
        double x2 = p.x*p.x, y2 = p.y*p.y, z2 = p.z*p.z;
        return {
            p.x * std::sqrt(std::max(0.0, 1.0 - y2*0.5 - z2*0.5 + y2*z2/3.0)),
            p.y * std::sqrt(std::max(0.0, 1.0 - x2*0.5 - z2*0.5 + x2*z2/3.0)),
            p.z * std::sqrt(std::max(0.0, 1.0 - x2*0.5 - y2*0.5 + x2*y2/3.0))
        };
    };

    auto tile_center_dir = [&](const QuadNode& node) -> glm::dvec3 {
        double ts = 2.0 / (1 << node.level);
        double u = -1.0 + (node.x + 0.5) * ts;
        double v = -1.0 + (node.y + 0.5) * ts;
        glm::dvec3 cube_pt = face_uv_to_cube(u, v, node.face);
        return glm::normalize(cube_to_sphere_d(cube_pt));
    };

    auto tile_center_on_sphere = [&](const QuadNode& node) -> glm::dvec3 {
        return tile_center_dir(node) * static_cast<double>(PLANET_RADIUS);
    };

    struct TileCandidate {
        QuadNode node;
        double screen_error;
        bool operator<(const TileCandidate& o) const { return screen_error < o.screen_error; }
    };

    auto compute_tile_error = [&](const QuadNode& node, const glm::dvec3& cam_pos_d,
                                   const glm::dvec3& cam_fwd_d, float screen_height,
                                   float fov_y) -> double {
        glm::dvec3 center = tile_center_on_sphere(node);
        glm::dvec3 rel = center - cam_pos_d;
        double dist = glm::length(rel);

        // Horizon cull
        double cam_height = glm::length(cam_pos_d);
        if (cam_height > PLANET_RADIUS * 1.01) {
            glm::dvec3 cam_dir = glm::normalize(cam_pos_d);
            glm::dvec3 tile_dir = glm::normalize(center);
            double angle = std::acos(std::clamp(glm::dot(cam_dir, tile_dir), -1.0, 1.0));
            double horizon = std::acos(static_cast<double>(PLANET_RADIUS) / cam_height);
            double tile_angular_size = 2.0 / (1 << node.level) * 1.5;
            if (angle - tile_angular_size > horizon)
                return -1.0;
        }

        double tile_world_size = (2.0 / (1 << node.level)) * PLANET_RADIUS * 1.5708;
        return (tile_world_size / std::max(dist, 1.0)) * screen_height / fov_y;
    };

    // ---- Frame loop ---------------------------------------------------------
    uint32_t current_frame = 0;
    uint32_t swe_ping_pong = 0;
    uint32_t sediment_ping_pong = 0;
    uint32_t atmo_ping_pong = 0;
    bool atmo_needs_init = true;
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

        if (g_reload_shaders) {
            reload_all_shaders();
            g_reload_shaders = false;
        }

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
                ImGui::Text("Altitude AGL: %.1f m", g_altitude_above_terrain);
                ImGui::Text("Terrain height: %.1f m", g_terrain_height_at_cam);
                ImGui::Text("Tiles: %u", g_visible_tile_count);
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
                ImGui::SliderFloat("Stamp angular scale", &g_stamp_angular_scale, 0.00001f, 0.01f, "%.5f");
                ImGui::SliderFloat("Pulse amount (m)", &g_pulse_amount, 1.0f, 200.0f);
                ImGui::SliderFloat("Pulse radius (cells)", &g_pulse_radius_cells, 5.0f, 200.0f);
                ImGui::Text("Stamps: %u / %u", static_cast<uint32_t>(g_stamps.size()), MAX_STAMPS);
                if (ImGui::Button("Undo stamp") && !g_stamps.empty()) {
                    g_stamps.pop_back();
                    g_stamps_dirty = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Clear stamps") && !g_stamps.empty()) {
                    g_stamps.clear();
                    g_stamps_dirty = true;
                }
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

            if (ImGui::CollapsingHeader("Atmosphere")) {
                ImGui::Checkbox("Atmosphere enabled", &g_atmosphere_enabled);
                if (g_atmosphere_enabled) {
                    ImGui::SliderFloat("Cloud opacity", &g_cloud_opacity, 0.0f, 3.0f);
                    ImGui::SliderFloat("Cloud altitude", &g_cloud_altitude, 0.0f, 2000.0f, "%.0f m");
                    ImGui::SliderFloat("Orographic lift", &g_orographic_lift, 0.0f, 2.0f);
                    ImGui::SliderFloat("Adiabatic cooling", &g_adiabatic_cooling, 0.0f, 0.02f, "%.5f");
                    ImGui::SliderFloat("Rain shadow", &g_rain_shadow, 0.0f, 1.0f);
                    ImGui::Separator();
                    ImGui::Checkbox("Sand enabled", &g_sand_enabled);
                    if (g_sand_enabled) {
                        ImGui::SliderFloat("Loft threshold", &g_sand_loft_threshold, 0.5f, 4.0f);
                        ImGui::SliderFloat("Loft rate", &g_sand_loft_rate, 0.1f, 3.0f);
                        ImGui::SliderFloat("Settling speed", &g_sand_settling, 0.5f, 10.0f);
                        ImGui::SliderFloat("Streak length", &g_sand_streak, 0.01f, 0.3f, "%.3f");
                        ImGui::SliderFloat("Particle alpha", &g_sand_alpha, 0.1f, 1.0f);
                        ImGui::SliderFloat("Bounce energy", &g_sand_bounce, 0.0f, 0.8f);
                        ImGui::SliderFloat("Gravity", &g_sand_gravity, 1.0f, 20.0f);
                    }
                    if (ImGui::Button("Reset atmosphere")) g_request_atmo_reset = true;
                }
            }

            if (ImGui::CollapsingHeader("Scene")) {
                if (ImGui::Button("Regenerate basin")) g_request_basin_reset = true;
                ImGui::TextDisabled("Pulse: SPACE | Modes: 1=raise 2=lower 3=water 4=sand");
                ImGui::TextDisabled("Look: RMB+drag | Move: WASD+QE | Menu: `");
                ImGui::TextDisabled("F5: hot-reload shaders");
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

        // ---- Per-frame camera movement (double precision) ----
        {
            glm::dvec3 cam_pos_d(g_camera.pos_x, g_camera.pos_y, g_camera.pos_z);
            double cam_dist = glm::length(cam_pos_d);
            glm::vec3 cam_dir_f = glm::normalize(glm::vec3(cam_pos_d));
            g_terrain_height_at_cam = static_cast<double>(cpu_terrain_height_with_stamps(cam_dir_f));
            double terrain_radius = static_cast<double>(PLANET_RADIUS) + g_terrain_height_at_cam;
            g_altitude_above_terrain = cam_dist - terrain_radius;

            float base_speed = static_cast<float>(std::max(g_altitude_above_terrain, 10.0)) * 1.5f;
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

            if (glm::dot(move, move) > 0.0f) {
                glm::vec3 dir = glm::normalize(move);
                g_camera.pos_x += static_cast<double>(dir.x) * speed * dt;
                g_camera.pos_y += static_cast<double>(dir.y) * speed * dt;
                g_camera.pos_z += static_cast<double>(dir.z) * speed * dt;
            }

            // Clamp camera above terrain surface
            cam_pos_d = glm::dvec3(g_camera.pos_x, g_camera.pos_y, g_camera.pos_z);
            cam_dist = glm::length(cam_pos_d);
            cam_dir_f = glm::normalize(glm::vec3(cam_pos_d));
            double h_at_new_pos = static_cast<double>(cpu_terrain_height_with_stamps(cam_dir_f));
            double min_radius = static_cast<double>(PLANET_RADIUS) + h_at_new_pos + 2.0;
            if (cam_dist < min_radius) {
                glm::dvec3 cam_dir_d = cam_pos_d / cam_dist;
                g_camera.pos_x = cam_dir_d.x * min_radius;
                g_camera.pos_y = cam_dir_d.y * min_radius;
                g_camera.pos_z = cam_dir_d.z * min_radius;
                g_altitude_above_terrain = 2.0;
            }
        }

        // ---- Camera UBO update (camera-relative, reversed-Z infinite far) ----
        glm::mat4 cam_view, cam_proj;
        {
            float aspect = static_cast<float>(extent.width) / extent.height;

            // Reversed-Z infinite far plane
            float f = 1.0f / std::tan(g_camera.fov_y * 0.5f);
            cam_proj = glm::mat4(0.0f);
            cam_proj[0][0] = f / aspect;
            cam_proj[1][1] = -f;  // Vulkan Y-flip
            cam_proj[2][2] = 0.0f;
            cam_proj[2][3] = -1.0f;
            cam_proj[3][2] = g_camera.near_plane;

            float cy = std::cos(g_camera.yaw);
            float sy = std::sin(g_camera.yaw);
            float cp = std::cos(g_camera.pitch);
            float sp = std::sin(g_camera.pitch);
            glm::vec3 forward(cy * cp, sp, sy * cp);

            // Camera-relative: camera is always at origin
            cam_view = glm::lookAtRH(glm::vec3(0.0f), forward, glm::vec3(0.0f, 1.0f, 0.0f));
        }

        // ---- Ray-pick cursor on sphere surface ----
        float grid_x = 0.0f, grid_y = 0.0f;
        glm::vec3 stamp_sphere_dir(0.0f);
        g_cursor_on_world = false;
        {
            int win_w, win_h;
            glfwGetWindowSize(window, &win_w, &win_h);
            float ndc_x = static_cast<float>(g_cursor_x) / win_w * 2.0f - 1.0f;
            float ndc_y = static_cast<float>(g_cursor_y) / win_h * 2.0f - 1.0f;

            glm::mat4 inv_vp = glm::inverse(cam_proj * cam_view);
            glm::vec4 near_clip = inv_vp * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
            glm::vec4 far_clip  = inv_vp * glm::vec4(ndc_x, ndc_y, 0.0f, 1.0f);
            near_clip /= near_clip.w;
            far_clip /= far_clip.w;

            glm::dvec3 ray_origin(g_camera.pos_x, g_camera.pos_y, g_camera.pos_z);
            glm::dvec3 ray_dir = glm::normalize(glm::dvec3(far_clip) - glm::dvec3(near_clip));

            double sphere_r = static_cast<double>(PLANET_RADIUS) + g_terrain_height_at_cam;
            double a = glm::dot(ray_dir, ray_dir);
            double b = 2.0 * glm::dot(ray_origin, ray_dir);
            double c = glm::dot(ray_origin, ray_origin) - sphere_r * sphere_r;
            double disc = b * b - 4.0 * a * c;

            if (disc >= 0.0) {
                double t = (-b - std::sqrt(disc)) / (2.0 * a);
                if (t > 0.0) {
                    glm::dvec3 hit = ray_origin + t * ray_dir;
                    stamp_sphere_dir = glm::normalize(glm::vec3(hit));
                    g_cursor_on_world = true;
                }
            }
        }
        bool brush_active = g_lmb_held && !g_rmb_held;
        bool brush_hit = brush_active && g_cursor_on_world;

        // ---- Place terrain stamp on LMB ----
        if (brush_hit &&
            (g_brush_mode == BrushMode::Raise || g_brush_mode == BrushMode::Lower) &&
            g_stamps.size() < MAX_STAMPS)
        {
            static double last_stamp_time = 0.0;
            double now_s = glfwGetTime();
            if (now_s - last_stamp_time > 0.1) {
                float sign = (g_brush_mode == BrushMode::Raise) ? 1.0f : -1.0f;
                float angular_radius = g_brush_radius_grid * g_stamp_angular_scale;
                angular_radius = std::clamp(angular_radius, 0.0001f, 0.2f);

                TerrainStamp stamp{};
                stamp.pos_x = stamp_sphere_dir.x;
                stamp.pos_y = stamp_sphere_dir.y;
                stamp.pos_z = stamp_sphere_dir.z;
                stamp.radius = angular_radius;
                stamp.delta_h = sign * g_terrain_strength;
                stamp.cos_radius = std::cos(angular_radius);
                g_stamps.push_back(stamp);
                g_stamps_dirty = true;
                last_stamp_time = now_s;
            }
        }

        // ---- Camera UBO update (perspective + brush ring) ----
        {
            float brush_radius_world = g_brush_radius_grid * bp.cell_spacing;
            glm::vec4 brush_color;
            if (g_brush_mode == BrushMode::Raise)
                brush_color = glm::vec4(0.30f, 0.95f, 0.40f, 1.0f);
            else if (g_brush_mode == BrushMode::Lower)
                brush_color = glm::vec4(0.95f, 0.45f, 0.20f, 1.0f);
            else if (g_brush_mode == BrushMode::Sand)
                brush_color = glm::vec4(0.85f, 0.70f, 0.45f, 1.0f);
            else
                brush_color = glm::vec4(0.30f, 0.85f, 0.95f, 1.0f);

            CameraData cam{};
            cam.view = cam_view;
            cam.proj = cam_proj;
            cam.sun_dir = glm::normalize(glm::vec3(0.4f, 0.7f, -0.3f));
            cam._pad0 = 0.0f;
            cam.sun_color = glm::vec3(1.0f, 0.95f, 0.85f);
            cam._pad1 = 0.0f;
            cam.cam_pos = glm::vec3(0.0f);  // camera-relative rendering
            cam._pad2 = g_mud_visibility;
            float angular_r = g_brush_radius_grid * g_stamp_angular_scale;
            cam.brush_world = glm::vec4(
                stamp_sphere_dir.x, stamp_sphere_dir.y, stamp_sphere_dir.z,
                g_cursor_on_world ? angular_r : 0.0f);
            cam.brush_color = brush_color;
            cam.inv_view_proj = glm::inverse(cam_proj * cam_view);
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

        // ---- Sand brush dispatch ----
        if (brush_hit && g_brush_mode == BrushMode::Sand) {
            TerrainBrushPC sb_pc{};
            sb_pc.brush_x = grid_x;
            sb_pc.brush_y = grid_y;
            sb_pc.brush_radius = g_brush_radius_grid;
            sb_pc.brush_amount = g_brush_strength * std::min(dt, 0.033f);
            sb_pc.grid_w = bp.grid_w;
            sb_pc.grid_h = bp.grid_h;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, terrain_brush_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    tb_pipeline_layout, 0, 1, &sand_brush_desc_set, 0, nullptr);
            vkCmdPushConstants(frame.cmd, tb_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(sb_pc), &sb_pc);
            vkCmdDispatch(frame.cmd, (bp.grid_w + 15) / 16, (bp.grid_h + 15) / 16, 1);

            VkMemoryBarrier2 sb_bar{};
            sb_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            sb_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            sb_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            sb_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            sb_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

            VkDependencyInfo sb_dep{};
            sb_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            sb_dep.memoryBarrierCount = 1;
            sb_dep.pMemoryBarriers = &sb_bar;
            vkCmdPipelineBarrier2(frame.cmd, &sb_dep);
        }

        // ---- Planet tile selection and generation ----
        std::vector<QuadNode> visible_tiles;
        visible_tiles.reserve(PLANET_TILE_POOL);
        {
            glm::dvec3 cam_pos_d(g_camera.pos_x, g_camera.pos_y, g_camera.pos_z);
            float screen_h = static_cast<float>(extent.height);

            glm::dvec3 cam_fwd_d(
                std::cos(g_camera.yaw) * std::cos(g_camera.pitch),
                std::sin(g_camera.pitch),
                std::sin(g_camera.yaw) * std::cos(g_camera.pitch)
            );

            // Dynamic max LOD: cap based on altitude to limit LOD range
            uint32_t effective_max_level = PLANET_MAX_LEVEL;
            if (g_altitude_above_terrain > 100.0) {
                int cap = static_cast<int>(14.0 - std::log2(g_altitude_above_terrain / 100.0));
                effective_max_level = static_cast<uint32_t>(std::clamp(cap, 5, static_cast<int>(PLANET_MAX_LEVEL)));
            }

            // Priority queue: always process the tile with largest screen error first
            std::priority_queue<TileCandidate> pq;

            for (uint32_t f = 0; f < 6; f++) {
                QuadNode root{f, 0, 0, 0};
                double err = compute_tile_error(root, cam_pos_d, cam_fwd_d, screen_h, g_camera.fov_y);
                if (err > 0.0) pq.push({root, err});
            }

            while (!pq.empty() && visible_tiles.size() < PLANET_TILE_POOL) {
                auto top = pq.top();
                pq.pop();

                if (top.screen_error > TILE_SUBDIVIDE_PX && top.node.level < effective_max_level) {
                    for (uint32_t cy = 0; cy < 2; cy++)
                        for (uint32_t cx = 0; cx < 2; cx++) {
                            QuadNode child{top.node.face, top.node.level + 1,
                                           top.node.x * 2 + cx, top.node.y * 2 + cy};
                            double err = compute_tile_error(child, cam_pos_d, cam_fwd_d, screen_h, g_camera.fov_y);
                            if (err > 0.0) pq.push({child, err});
                        }
                } else {
                    visible_tiles.push_back(top.node);
                }
            }

            // Drain remaining tiles that don't need subdivision
            while (!pq.empty() && visible_tiles.size() < PLANET_TILE_POOL) {
                visible_tiles.push_back(pq.top().node);
                pq.pop();
            }

            g_visible_tile_count = static_cast<uint32_t>(visible_tiles.size());

            // Upload stamps to GPU if changed
            if (g_stamps_dirty && !g_stamps.empty()) {
                size_t copy_size = g_stamps.size() * sizeof(TerrainStamp);
                std::memcpy(stamp_buf_info.pMappedData, g_stamps.data(), copy_size);
                vmaFlushAllocation(allocator, stamp_buf_alloc, 0, copy_size);
                g_stamps_dirty = false;
            }

            // Dispatch planet_gen compute for each visible tile
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, terrain_gen_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                terrain_gen_pipeline_layout, 0, 1, &terrain_gen_desc_set, 0, nullptr);

            for (uint32_t i = 0; i < visible_tiles.size(); i++) {
                const auto& tile = visible_tiles[i];
                float ts = 2.0f / static_cast<float>(1u << tile.level);

                PlanetGenPC gen_pc{};
                gen_pc.u_min = -1.0f + tile.x * ts;
                gen_pc.v_min = -1.0f + tile.y * ts;
                gen_pc.tile_size = ts;
                gen_pc.face = tile.face;
                gen_pc.pool_index = i;
                gen_pc.tex_res = PLANET_TILE_RES;
                gen_pc.seed = 42;
                gen_pc.stamp_count = static_cast<uint32_t>(g_stamps.size());

                vkCmdPushConstants(frame.cmd, terrain_gen_pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gen_pc), &gen_pc);
                vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
            }

            // Barrier: compute -> vertex shader
            VkMemoryBarrier2 gen_bar{};
            gen_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            gen_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            gen_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            gen_bar.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            gen_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            VkDependencyInfo gen_dep{};
            gen_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            gen_dep.memoryBarrierCount = 1;
            gen_dep.pMemoryBarriers = &gen_bar;
            vkCmdPipelineBarrier2(frame.cmd, &gen_dep);
        }

        // ---- Atmosphere 3D dispatch ----
        if (g_atmosphere_enabled) {
            g_accumulated_atmo_time += dt * g_time_scale;

            Atmo3DPC apc{};
            apc.dt = std::min(dt, 0.033f) * g_time_scale;
            apc.accumulated_time = g_accumulated_atmo_time;
            apc.grid_w = ATMO_W;
            apc.grid_h = ATMO_H;
            apc.grid_d = ATMO_D;
            apc.terrain_scale = bp.cell_spacing;
            apc.layer_height = ATMO_LAYER_HEIGHT;
            apc.max_elevation = 2000.0f;
            apc.orographic_lift_coeff = g_orographic_lift;
            apc.adiabatic_cooling_rate = g_adiabatic_cooling;
            apc.rain_shadow_intensity = g_rain_shadow;
            apc.force_init = (atmo_needs_init || g_request_atmo_reset) ? 1u : 0u;
            apc.sand_enabled = g_sand_enabled ? 1u : 0u;
            apc.sand_loft_threshold = g_sand_loft_threshold;
            apc.sand_loft_rate = g_sand_loft_rate;
            apc.sand_settling = g_sand_settling;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atmo_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                atmo_pipeline_layout, 0, 1, &atmo_desc_sets[atmo_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, atmo_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(Atmo3DPC), &apc);
            vkCmdDispatch(frame.cmd, (ATMO_W + 3) / 4, (ATMO_H + 3) / 4, (ATMO_D + 3) / 4);

            // Barrier: atmosphere writes -> SWE reads + fragment reads
            VkMemoryBarrier2 atmo_bar{};
            atmo_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            atmo_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            atmo_bar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            atmo_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            atmo_bar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

            VkDependencyInfo atmo_dep{};
            atmo_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            atmo_dep.memoryBarrierCount = 1;
            atmo_dep.pMemoryBarriers = &atmo_bar;
            vkCmdPipelineBarrier2(frame.cmd, &atmo_dep);

            atmo_ping_pong ^= 1;

            if (atmo_needs_init || g_request_atmo_reset) {
                atmo_needs_init = false;
                g_request_atmo_reset = false;
            }
        }

        // ---- Sand particle dispatch ----
        if (g_atmosphere_enabled && g_sand_enabled) {
            static uint32_t sand_emit_offset = 0;

            SandSimPC spc{};
            spc.dt = std::min(dt, 0.033f) * g_time_scale;
            spc.terrain_size = terrain_size;
            spc.loft_threshold = g_sand_loft_threshold;
            spc.loft_rate = g_sand_loft_rate;
            spc.gravity = g_sand_gravity;
            spc.accumulated_time = g_accumulated_atmo_time;
            spc.max_particles = SAND_MAX_PARTICLES;
            spc.emit_offset = sand_emit_offset;
            spc.emit_count = SAND_EMIT_PER_FRAME;
            spc.grid_d = ATMO_D;
            spc.layer_height = ATMO_LAYER_HEIGHT;
            spc.bounce_energy = g_sand_bounce;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sand_sim_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                sand_sim_pipeline_layout, 0, 1, &sand_sim_desc_sets[atmo_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, sand_sim_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(SandSimPC), &spc);
            vkCmdDispatch(frame.cmd, (SAND_MAX_PARTICLES + 63) / 64, 1, 1);

            // Barrier: sand compute -> vertex read
            VkMemoryBarrier2 sand_bar{};
            sand_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            sand_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            sand_bar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            sand_bar.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            sand_bar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

            VkDependencyInfo sand_dep{};
            sand_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            sand_dep.memoryBarrierCount = 1;
            sand_dep.pMemoryBarriers = &sand_bar;
            vkCmdPipelineBarrier2(frame.cmd, &sand_dep);

            sand_emit_offset = (sand_emit_offset + SAND_EMIT_PER_FRAME) % SAND_MAX_PARTICLES;
        }

        // ---- SWE step dispatch (CFL sub-stepping) ----
        {
            vkCmdResetQueryPool(frame.cmd, query_pool, current_frame * 2, 2);

            float swe_total_dt = std::min(dt, 0.033f) * g_time_scale;
            float c_max = std::sqrt(g_gravity * 200.0f);
            float cfl_dt = bp.cell_spacing / (c_max * 6.0f);
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
            depth_attachment.clearValue.depthStencil = {0.0f, 0};

            VkRenderingInfo rendering_info{};
            rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering_info.renderArea = {{0, 0}, extent};
            rendering_info.layerCount = 1;
            rendering_info.colorAttachmentCount = 1;
            rendering_info.pColorAttachments = &color_attachment;
            rendering_info.pDepthAttachment = &depth_attachment;

            vkCmdBeginRendering(frame.cmd, &rendering_info);

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

            // Planet terrain draw
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, clipmap_terrain_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                clipmap_gfx_pipeline_layout, 0, 1, &clipmap_gfx_desc_sets[atmo_ping_pong], 0, nullptr);

            VkDeviceSize clip_offset = 0;
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &clipmap_vbo, &clip_offset);
            vkCmdBindIndexBuffer(frame.cmd, clipmap_ibo, 0, VK_INDEX_TYPE_UINT32);

            glm::dvec3 cam_pos_d(g_camera.pos_x, g_camera.pos_y, g_camera.pos_z);

            for (uint32_t i = 0; i < visible_tiles.size(); i++) {
                const auto& tile = visible_tiles[i];
                float ts = 2.0f / static_cast<float>(1u << tile.level);

                // rel_xyz = center_dir * R - cam (VS adds height displacement)
                glm::dvec3 dir = tile_center_dir(tile);
                glm::dvec3 center_at_R = dir * static_cast<double>(PLANET_RADIUS);
                glm::dvec3 rel_d = center_at_R - cam_pos_d;

                PlanetTilePC tpc{};
                tpc.rel_x = static_cast<float>(rel_d.x);
                tpc.rel_y = static_cast<float>(rel_d.y);
                tpc.rel_z = static_cast<float>(rel_d.z);
                tpc.u_min = -1.0f + tile.x * ts;
                tpc.v_min = -1.0f + tile.y * ts;
                tpc.tile_size = ts;
                tpc.face = tile.face;
                tpc.pool_index = i;
                tpc.planet_radius = PLANET_RADIUS;
                tpc.max_elevation = PLANET_MAX_ELEVATION;
                tpc.heightmap_texel = 1.0f / static_cast<float>(PLANET_TILE_RES);
                tpc.cloud_opacity = 0.0f;

                vkCmdPushConstants(frame.cmd, clipmap_gfx_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(tpc), &tpc);
                vkCmdDrawIndexed(frame.cmd, clipmap_index_count, 1, 0, 0, 0);
            }

            // Water pass (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, water_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    gfx_pipeline_layout, 0, 1, &gfx_desc_sets[atmo_ping_pong], 0, nullptr);
                GfxPC wpc{};
                wpc.terrain_size = terrain_size;
                wpc.heightmap_texel = 1.0f / static_cast<float>(bp.grid_w);
                wpc.max_elevation = 2000.0f;
                wpc.cloud_opacity = g_atmosphere_enabled ? g_cloud_opacity : 0.0f;
                vkCmdPushConstants(frame.cmd, gfx_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(wpc), &wpc);
                VkDeviceSize water_offset = 0;
                vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer, &water_offset);
                vkCmdBindIndexBuffer(frame.cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(frame.cmd, index_count, 1, 0, 0, 0);
            }

            // Cloud raymarch pass (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, raymarch_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    raymarch_pipeline_layout, 0, 1, &gfx_desc_sets[atmo_ping_pong], 0, nullptr);

                RaymarchPC rpc{};
                rpc.terrain_size = terrain_size;
                rpc.max_elevation = 2000.0f;
                rpc.cloud_opacity = g_cloud_opacity;
                rpc.cloud_base = g_cloud_altitude;
                rpc.vol_w = ATMO_W;
                rpc.vol_h = ATMO_H;
                rpc.vol_d = ATMO_D;
                rpc.layer_height = ATMO_LAYER_HEIGHT;
                vkCmdPushConstants(frame.cmd, raymarch_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(rpc), &rpc);

                vkCmdDraw(frame.cmd, 3, 1, 0, 0);
            }

            // Sand particle draw (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sand_render_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    sand_render_pipeline_layout, 0, 1, &sand_render_desc_set, 0, nullptr);

                SandRenderPC srpc{};
                srpc.streak_length = g_sand_streak;
                srpc.particle_alpha = g_sand_alpha;
                srpc.max_particles = SAND_MAX_PARTICLES;
                vkCmdPushConstants(frame.cmd, sand_render_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(srpc), &srpc);

                vkCmdDraw(frame.cmd, SAND_MAX_PARTICLES * 2, 1, 0, 0);
            }

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

    vkDestroyPipeline(device, clipmap_terrain_pipeline, nullptr);
    vkDestroyPipelineLayout(device, clipmap_gfx_pipeline_layout, nullptr);

    vkDestroyPipeline(device, terrain_gen_pipeline, nullptr);
    vkDestroyPipelineLayout(device, terrain_gen_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, terrain_gen_desc_layout, nullptr);
    vkDestroyShaderModule(device, terrain_gen_shader, nullptr);

    vkDestroyImageView(device, clipmap_hm_view, nullptr);
    vmaDestroyImage(allocator, clipmap_hm_image, clipmap_hm_alloc);

    vmaDestroyBuffer(allocator, clipmap_ibo, clipmap_ibo_alloc);
    vmaDestroyBuffer(allocator, clipmap_vbo, clipmap_vbo_alloc);

    vkDestroyPipeline(device, sand_render_pipeline, nullptr);
    vkDestroyPipelineLayout(device, sand_render_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, sand_render_desc_layout, nullptr);
    vkDestroyShaderModule(device, sand_render_fs, nullptr);
    vkDestroyShaderModule(device, sand_render_vs, nullptr);

    vkDestroyPipeline(device, sand_sim_pipeline, nullptr);
    vkDestroyPipelineLayout(device, sand_sim_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, sand_sim_desc_layout, nullptr);
    vkDestroyShaderModule(device, sand_sim_shader, nullptr);

    vkDestroyPipeline(device, water_pipeline, nullptr);
    vkDestroyShaderModule(device, water_fs, nullptr);
    vkDestroyShaderModule(device, water_vs, nullptr);

    vkDestroyPipeline(device, raymarch_pipeline, nullptr);
    vkDestroyPipelineLayout(device, raymarch_pipeline_layout, nullptr);
    vkDestroyShaderModule(device, raymarch_fs, nullptr);
    vkDestroyShaderModule(device, raymarch_vs, nullptr);

    vkDestroyPipeline(device, atmo_pipeline, nullptr);
    vkDestroyPipelineLayout(device, atmo_pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, atmo_desc_layout, nullptr);
    vkDestroyShaderModule(device, atmo_shader, nullptr);

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

    destroy_swe_image(device, allocator, sand_deposit);
    destroy_swe_image(device, allocator, atmo_shadow);
    destroy_swe_image(device, allocator, wind_field_b);
    destroy_swe_image(device, allocator, wind_field_a);
    destroy_swe_image(device, allocator, atmo_state_b);
    destroy_swe_image(device, allocator, atmo_state_a);

    vmaDestroyBuffer(allocator, sand_particle_buf, sand_particle_alloc);
    vmaDestroyBuffer(allocator, index_buffer, index_alloc);
    vmaDestroyBuffer(allocator, vertex_buffer, vertex_alloc);
    vmaDestroyBuffer(allocator, camera_ubo, camera_ubo_alloc);
    vmaDestroyBuffer(allocator, stamp_buf, stamp_buf_alloc);

    vkDestroySampler(device, terrain_linear_sampler, nullptr);
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

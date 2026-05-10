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

#include "camera.h"
#include "input.h"
#include "pipeline.h"
#include "planet.h"
#include "renderer.h"
#include "resources.h"
#include "terrain.h"
#include "ui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

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

bool cursor_on_world = false;
Camera g_camera;
double g_terrain_height_at_cam = 0.0;
double g_altitude_above_terrain = 100000.0;
std::vector<TerrainStamp> g_stamps;
bool g_stamps_dirty = false;

std::vector<WaterStamp> g_water_stamps;
bool g_water_stamps_dirty = false;
UIState g_ui;
float g_accumulated_atmo_time = 0.0f;

} // namespace

int main()
{
#ifdef __APPLE__
    setenv("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "0", 0);
#endif

    Renderer renderer{};
    renderer_init(renderer, 1280, 720, "drift_engine");
    VkDevice device = renderer.device;
    VmaAllocator allocator = renderer.allocator;
    VkQueue graphics_queue = renderer.graphics_queue;
    uint32_t gfx_family = renderer.gfx_family;
    GLFWwindow* window = renderer.window;

    camera_initialize_orientation(g_camera);

    InputState input{};
    CallbackContext cb_ctx{&input, &g_camera, &g_ui};
    input_install_callbacks(window, &cb_ctx);
    ImGui_ImplGlfw_InstallCallbacks(window);

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

    // ---- Water stamp buffer (parallel to terrain stamps) --------------------
    VkBuffer water_stamp_buf = VK_NULL_HANDLE;
    VmaAllocation water_stamp_buf_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo water_stamp_buf_info{};
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = MAX_WATER_STAMPS * sizeof(WaterStamp);
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VK_CHECK(vmaCreateBuffer(allocator, &bci, &ai,
            &water_stamp_buf, &water_stamp_buf_alloc, &water_stamp_buf_info));
    }

    // ---- Planet SWE edge-flag buffer (per-pool-slot, host-readable) ----------
    // The step shader InterlockedOrs bits into edge_flags[pool_index] when water
    // crosses an edge above static depth. CPU reads this each frame to anchor
    // missing neighbors and grow the simulation domain.
    VkBuffer edge_flags_buf = VK_NULL_HANDLE;
    VmaAllocation edge_flags_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo edge_flags_info{};
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = PLANET_TILE_POOL * sizeof(uint32_t);
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VK_CHECK(vmaCreateBuffer(allocator, &bci, &ai,
            &edge_flags_buf, &edge_flags_alloc, &edge_flags_info));
    }

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

    // ---- Water state tile pools (per-tile SWE) ---------------------------------
    auto create_water_array = [&](VkImage& img, VmaAllocation& alloc, VkImageView& view) {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        ci.extent = {PLANET_TILE_RES, PLANET_TILE_RES, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = PLANET_TILE_POOL;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                 | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        VK_CHECK(vmaCreateImage(allocator, &ci, &ai, &img, &alloc, nullptr));

        VkImageViewCreateInfo v_ci{};
        v_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v_ci.image = img;
        v_ci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        v_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        v_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};
        VK_CHECK(vkCreateImageView(device, &v_ci, nullptr, &view));
    };

    VkImage water_state_a_img = VK_NULL_HANDLE, water_state_b_img = VK_NULL_HANDLE, water_output_img = VK_NULL_HANDLE;
    VmaAllocation water_state_a_alloc = VK_NULL_HANDLE, water_state_b_alloc = VK_NULL_HANDLE, water_output_alloc = VK_NULL_HANDLE;
    VkImageView water_state_a_view = VK_NULL_HANDLE, water_state_b_view = VK_NULL_HANDLE, water_output_view = VK_NULL_HANDLE;

    create_water_array(water_state_a_img, water_state_a_alloc, water_state_a_view);
    create_water_array(water_state_b_img, water_state_b_alloc, water_state_b_view);
    create_water_array(water_output_img, water_output_alloc, water_output_view);

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

    Pipelines pipelines{};
    pipelines_create(pipelines, device);

    // ---- Descriptor pool + sets ---------------------------------------------
    // Counts: SWE init(2) + SWE step×2(8) + terrain brush(1) + gfx(4 incl sediment) + erosion×2(8) = needs ~23 descriptors
    VkDescriptorPoolSize pool_sizes[5]{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 35;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[1].descriptorCount = 22;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[2].descriptorCount = 4;
    pool_sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[3].descriptorCount = 48;
    pool_sizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[4].descriptorCount = 12;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 25;
    dp_ci.poolSizeCount = 5;
    dp_ci.pPoolSizes = pool_sizes;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(device, &dp_ci, nullptr, &desc_pool));

    // SWE init descriptor set
    VkDescriptorSetAllocateInfo swe_init_ds_ai{};
    swe_init_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    swe_init_ds_ai.descriptorPool = desc_pool;
    swe_init_ds_ai.descriptorSetCount = 1;
    swe_init_ds_ai.pSetLayouts = &pipelines.swe_init_desc_layout;

    VkDescriptorSet swe_init_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &swe_init_ds_ai, &swe_init_desc_set));

    // SWE step descriptor sets (2 for ping-pong)
    VkDescriptorSetLayout swe_step_layouts[2] = {pipelines.swe_step_desc_layout, pipelines.swe_step_desc_layout};
    VkDescriptorSetAllocateInfo swe_step_ds_ai{};
    swe_step_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    swe_step_ds_ai.descriptorPool = desc_pool;
    swe_step_ds_ai.descriptorSetCount = 2;
    swe_step_ds_ai.pSetLayouts = swe_step_layouts;

    VkDescriptorSet swe_step_desc_sets[2] = {};
    VK_CHECK(vkAllocateDescriptorSets(device, &swe_step_ds_ai, swe_step_desc_sets));

    // Graphics descriptor set
    VkDescriptorSetLayout gfx_layouts_2[2] = {pipelines.gfx_desc_layout, pipelines.gfx_desc_layout};
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
    tb_ds_ai.pSetLayouts = &pipelines.tb_desc_layout;

    VkDescriptorSet tb_desc_set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &tb_ds_ai, &tb_desc_set));

    // Sand brush descriptor set (reuses pipelines.tb_desc_layout — just a storage image at binding 0)
    VkDescriptorSet sand_brush_desc_set = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo sb_ds_ai{};
        sb_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        sb_ds_ai.descriptorPool = desc_pool;
        sb_ds_ai.descriptorSetCount = 1;
        sb_ds_ai.pSetLayouts = &pipelines.tb_desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &sb_ds_ai, &sand_brush_desc_set));
    }

    // Erosion descriptor sets (2 for ping-pong)
    VkDescriptorSetLayout ero_layouts[2] = {pipelines.ero_desc_layout, pipelines.ero_desc_layout};
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
    VkDescriptorSetLayout atmo_layouts[2] = {pipelines.atmo_desc_layout, pipelines.atmo_desc_layout};
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
    VkDescriptorSetLayout sand_sim_layouts[2] = {pipelines.sand_sim_desc_layout, pipelines.sand_sim_desc_layout};
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
    sand_render_ds_ai.pSetLayouts = &pipelines.sand_render_desc_layout;

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
    tgen_ds_ai.pSetLayouts = &pipelines.terrain_gen_desc_layout;

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
    VkDescriptorSetLayout clip_gfx_layouts[2] = {pipelines.gfx_desc_layout, pipelines.gfx_desc_layout};
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
        swe_out_info.imageView = water_output_view;
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

    // ---- Planet SWE descriptor sets -------------------------------------------
    VkDescriptorSet planet_swe_init_desc_set = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &pipelines.planet_swe_init_desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, &planet_swe_init_desc_set));

        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = clipmap_hm_view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_a_info{};
        state_a_info.imageView = water_state_a_view;
        state_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_b_info{};
        state_b_info.imageView = water_state_b_view;
        state_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo output_info{};
        output_info.imageView = water_output_view;
        output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo water_stamp_bi{};
        water_stamp_bi.buffer = water_stamp_buf;
        water_stamp_bi.offset = 0;
        water_stamp_bi.range = MAX_WATER_STAMPS * sizeof(WaterStamp);

        VkWriteDescriptorSet writes[5]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = planet_swe_init_desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].pImageInfo = &terrain_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = planet_swe_init_desc_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &state_a_info;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = planet_swe_init_desc_set;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &state_b_info;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = planet_swe_init_desc_set;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[3].pImageInfo = &output_info;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = planet_swe_init_desc_set;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &water_stamp_bi;

        vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    }

    // Planet SWE h-adjust descriptor: state_a + state_b (storage images, RW).
    VkDescriptorSet planet_swe_h_adjust_desc_set = VK_NULL_HANDLE;
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &pipelines.planet_swe_h_adjust_desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, &planet_swe_h_adjust_desc_set));

        VkDescriptorImageInfo state_a_info{};
        state_a_info.imageView = water_state_a_view;
        state_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo state_b_info{};
        state_b_info.imageView = water_state_b_view;
        state_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = planet_swe_h_adjust_desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &state_a_info;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = planet_swe_h_adjust_desc_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &state_b_info;
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }

    VkDescriptorSet planet_swe_step_desc_sets[2] = {};
    {
        VkDescriptorSetLayout layouts[2] = {pipelines.planet_swe_step_desc_layout, pipelines.planet_swe_step_desc_layout};
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = desc_pool;
        ai.descriptorSetCount = 2;
        ai.pSetLayouts = layouts;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, planet_swe_step_desc_sets));

        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = clipmap_hm_view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_a_info{};
        state_a_info.imageView = water_state_a_view;
        state_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_b_info{};
        state_b_info.imageView = water_state_b_view;
        state_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo output_info{};
        output_info.imageView = water_output_view;
        output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo edge_flags_bi{};
        edge_flags_bi.buffer = edge_flags_buf;
        edge_flags_bi.offset = 0;
        edge_flags_bi.range = VK_WHOLE_SIZE;

        // Set 0: read A -> write B
        VkWriteDescriptorSet w0[5]{};
        w0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[0].dstSet = planet_swe_step_desc_sets[0];
        w0[0].dstBinding = 0;
        w0[0].descriptorCount = 1;
        w0[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[0].pImageInfo = &terrain_info;
        w0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[1].dstSet = planet_swe_step_desc_sets[0];
        w0[1].dstBinding = 1;
        w0[1].descriptorCount = 1;
        w0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[1].pImageInfo = &state_a_info;
        w0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[2].dstSet = planet_swe_step_desc_sets[0];
        w0[2].dstBinding = 2;
        w0[2].descriptorCount = 1;
        w0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0[2].pImageInfo = &state_b_info;
        w0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[3].dstSet = planet_swe_step_desc_sets[0];
        w0[3].dstBinding = 3;
        w0[3].descriptorCount = 1;
        w0[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0[3].pImageInfo = &output_info;
        w0[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[4].dstSet = planet_swe_step_desc_sets[0];
        w0[4].dstBinding = 4;
        w0[4].descriptorCount = 1;
        w0[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w0[4].pBufferInfo = &edge_flags_bi;
        vkUpdateDescriptorSets(device, 5, w0, 0, nullptr);

        // Set 1: read B -> write A
        VkWriteDescriptorSet w1[5]{};
        w1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[0].dstSet = planet_swe_step_desc_sets[1];
        w1[0].dstBinding = 0;
        w1[0].descriptorCount = 1;
        w1[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[0].pImageInfo = &terrain_info;
        w1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[1].dstSet = planet_swe_step_desc_sets[1];
        w1[1].dstBinding = 1;
        w1[1].descriptorCount = 1;
        w1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[1].pImageInfo = &state_b_info;
        w1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[2].dstSet = planet_swe_step_desc_sets[1];
        w1[2].dstBinding = 2;
        w1[2].descriptorCount = 1;
        w1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1[2].pImageInfo = &state_a_info;
        w1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[3].dstSet = planet_swe_step_desc_sets[1];
        w1[3].dstBinding = 3;
        w1[3].descriptorCount = 1;
        w1[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1[3].pImageInfo = &output_info;
        w1[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[4].dstSet = planet_swe_step_desc_sets[1];
        w1[4].dstBinding = 4;
        w1[4].descriptorCount = 1;
        w1[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w1[4].pBufferInfo = &edge_flags_bi;
        vkUpdateDescriptorSets(device, 5, w1, 0, nullptr);
    }

    uint32_t planet_swe_ping_pong = 0;

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

        VkImageMemoryBarrier2 init_barriers[15]{};
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

        // Water tile pools (UNDEFINED -> GENERAL)
        for (int i = 12; i < 15; ++i) {
            init_barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            init_barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            init_barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};
        }
        init_barriers[12].image = water_state_a_img;
        init_barriers[13].image = water_state_b_img;
        init_barriers[14].image = water_output_img;

        VkDependencyInfo dep_init{};
        dep_init.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_init.imageMemoryBarrierCount = 15;
        dep_init.pImageMemoryBarriers = init_barriers;
        vkCmdPipelineBarrier2(cmd, &dep_init);

        VkClearColorValue clear_zero{};
        clear_zero.float32[0] = 0.0f;
        VkImageSubresourceRange clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, sediment_a.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
        vkCmdClearColorImage(cmd, sediment_b.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
        vkCmdClearColorImage(cmd, sand_deposit.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);

        VkImageSubresourceRange water_clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};
        vkCmdClearColorImage(cmd, water_state_a_img, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &water_clear_range);
        vkCmdClearColorImage(cmd, water_state_b_img, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &water_clear_range);
        vkCmdClearColorImage(cmd, water_output_img, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &water_clear_range);
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

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.swe_init_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelines.swe_init_pipeline_layout, 0, 1, &swe_init_desc_set, 0, nullptr);

        SweInitPC init_pc{};
        init_pc.grid_w = SWE_GRID_W;
        init_pc.grid_h = SWE_GRID_H;
        init_pc.initial_water_level = bp.floor_height + bp.initial_water;
        init_pc._pad = 0.0f;
        vkCmdPushConstants(cmd, pipelines.swe_init_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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

    // ---- Startup printout ---------------------------------------------------
    std::printf("drift_engine v0.4.1 — Vulkan up.\n");
    std::printf("Device:   %s\n", renderer.vkb_phys.name.c_str());
    std::printf("Queues:   graphics=%u  present=%u\n",
                gfx_family,
                renderer.vkb_device.get_queue_index(vkb::QueueType::present).value());
    std::printf("Swapchain: %ux%u, %u images\n",
                renderer.vkb_swapchain.extent.width, renderer.vkb_swapchain.extent.height,
                static_cast<uint32_t>(renderer.swapchain_images.size()));
    std::printf("Terrain: %ux%u R32_SFLOAT, min=%.1f max=%.1f\n",
                hm.width, hm.height, *hm_min, *hm_max);
    std::printf("Mesh: %ux%u vertices, %u triangles\n",
                MESH_RES, MESH_RES, index_count / 3);
    std::printf("SWE: %ux%u RGBA16F, dx=%.1f m, init water=%.0f m\n",
                SWE_GRID_W, SWE_GRID_H, bp.cell_spacing, bp.initial_water);
    std::printf("Graphics: perspective free-cam (RMB look, WASD move, Shift fast, Alt slow)\n");
    std::printf("Shaders: press F5 to hot-reload all shaders\n");
    std::fflush(stdout);

    // Planet quadtree helpers are now in planet.h/.cpp

    // ---- Frame loop ---------------------------------------------------------
    uint32_t current_frame = 0;
    uint32_t swe_ping_pong = 0;
    uint32_t sediment_ping_pong = 0;
    uint32_t atmo_ping_pong = 0;
    bool atmo_needs_init = true;
    // Stable slot allocator: a QuadNode keeps the same pool_index for as long as it
    // remains visible OR is disturbed. Slots are recycled (LIFO) when a tile leaves
    // the visible set AND is not disturbed. Disturbed tiles stay anchored so the
    // brushed water keeps simulating across camera moves and LOD transitions.
    std::unordered_map<QuadNode, uint32_t, QuadNodeHash> tile_slot_map;
    auto INVALID_TILE = QuadNode{0xFFFFFFFFu, 0u, 0u, 0u};
    std::vector<QuadNode> slot_to_tile(PLANET_TILE_POOL, INVALID_TILE);
    std::vector<uint32_t> free_slots;
    free_slots.reserve(PLANET_TILE_POOL);
    for (uint32_t s = PLANET_TILE_POOL; s-- > 0; ) free_slots.push_back(s);
    bool planet_swe_needs_full_init = true;
    std::unordered_set<QuadNode, QuadNodeHash> disturbed_tiles;
    // Tiles that need their SWE state initialized on the next init pass — a single
    // queue fed by the visible-tile allocator and by the cross-tile auto-anchor path.
    std::unordered_set<QuadNode, QuadNodeHash> pending_init;
    // Number of terrain stamps already applied to disturbed tiles' h column.
    // When new stamps are added, the delta is dispatched as h-adjust per disturbed tile.
    uint32_t last_processed_stamp_count = 0;
    double last_time = glfwGetTime();

    constexpr int AVG_FRAMES = 30;
    double cpu_times[AVG_FRAMES] = {};
    double gpu_times[AVG_FRAMES] = {};
    int timing_index = 0;
    int timing_count = 0;
    double last_title_update = 0.0;
    double cpu_avg_ms = 0.0, gpu_avg_ms = 0.0;
    double ns_per_tick = static_cast<double>(renderer.vkb_phys.properties.limits.timestampPeriod);
    bool queries_valid = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (input.reload_shaders) {
            pipelines_reload(pipelines, device);
            input.reload_shaders = false;
        }

        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        if (fb_w == 0 || fb_h == 0)
            continue;

        FrameData& frame = renderer.frames[renderer.current_frame];

        VK_CHECK(vkWaitForFences(device, 1, &frame.in_flight, VK_TRUE, UINT64_MAX));

        if (queries_valid) {
            uint64_t timestamps[2] = {};
            VkResult qr = vkGetQueryPoolResults(device, renderer.query_pool,
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
            device, renderer.vkb_swapchain.swapchain, UINT64_MAX,
            frame.image_available, VK_NULL_HANDLE, &image_index);

        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
            renderer_rebuild_swapchain(renderer);
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

        VkExtent2D extent = renderer.vkb_swapchain.extent;

        // ---- ImGui frame ----
        g_ui.cpu_avg_ms = cpu_avg_ms;
        g_ui.gpu_avg_ms = gpu_avg_ms;
        g_ui.altitude_above_terrain = g_altitude_above_terrain;
        g_ui.terrain_height_at_cam = g_terrain_height_at_cam;
        g_ui.stamp_count = static_cast<uint32_t>(g_stamps.size());
        g_ui.max_stamps = MAX_STAMPS;
        ui_draw(g_ui);

        if (g_ui.undo_stamp && !g_stamps.empty()) {
            g_stamps.pop_back();
            g_stamps_dirty = true;
            g_ui.undo_stamp = false;
        }
        if (g_ui.clear_stamps && !g_stamps.empty()) {
            g_stamps.clear();
            g_stamps_dirty = true;
            g_ui.clear_stamps = false;
        }

        // ---- Reset handlers (heavy, stall GPU) ----
        if (g_ui.request_basin_reset || g_ui.request_water_reset) {
            vkDeviceWaitIdle(device);

            if (g_ui.request_basin_reset) {
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
                g_ui.request_basin_reset = false;
                g_ui.request_water_reset = true;
            }

            if (g_ui.request_water_reset) {
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

                vkCmdBindPipeline(tmp_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.swe_init_pipeline);
                vkCmdBindDescriptorSets(tmp_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipelines.swe_init_pipeline_layout, 0, 1, &swe_init_desc_set, 0, nullptr);
                SweInitPC init_pc{};
                init_pc.grid_w = SWE_GRID_W;
                init_pc.grid_h = SWE_GRID_H;
                init_pc.initial_water_level = bp.floor_height + bp.initial_water;
                vkCmdPushConstants(tmp_cmd, pipelines.swe_init_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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
                g_ui.request_water_reset = false;
                g_ui.request_sediment_reset = true;
            }
        }

        if (g_ui.request_sediment_reset) {
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
            g_ui.request_sediment_reset = false;
        }

        // ---- Per-frame camera movement (double precision) ----
        {
            auto height_fn = [](glm::vec3 dir) {
                float h = cpu_terrain_height_with_stamps(dir, g_stamps.data(),
                    static_cast<uint32_t>(g_stamps.size()));
                if (g_ui.ocean_enabled) h = std::max(h, g_ui.sea_level);
                return h;
            };

            if (input.toggle_camera_mode) {
                if (g_camera.mode == CameraMode::Orbital) {
                    camera_switch_to_first_person(g_camera, PLANET_RADIUS, height_fn);
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    input.first_mouse = true;
                    input.space_held = false;
                } else {
                    camera_switch_to_orbital(g_camera);
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                    input.rmb_held = false;
                    input.space_held = false;
                }
                input.toggle_camera_mode = false;
            }
            g_ui.first_person_mode = (g_camera.mode == CameraMode::FirstPerson);

            auto cam_result = camera_update(g_camera, window, dt, PLANET_RADIUS, height_fn);
            g_terrain_height_at_cam = cam_result.terrain_height_at_cam;
            g_altitude_above_terrain = cam_result.altitude_above_terrain;
        }

        // ---- Camera UBO update (camera-relative, reversed-Z infinite far) ----
        float aspect = static_cast<float>(extent.width) / extent.height;
        glm::mat4 cam_view = camera_build_view(g_camera);
        glm::mat4 cam_proj = camera_build_proj(g_camera, aspect);

        // ---- Ray-pick cursor against heightfield ----
        float grid_x = 0.0f, grid_y = 0.0f;
        glm::vec3 stamp_sphere_dir(0.0f);
        cursor_on_world = false;
        {
            int win_w, win_h;
            glfwGetWindowSize(window, &win_w, &win_h);
            bool fp = (g_camera.mode == CameraMode::FirstPerson);
            float ndc_x = fp ? 0.0f : static_cast<float>(input.cursor_x) / win_w * 2.0f - 1.0f;
            float ndc_y = fp ? 0.0f : static_cast<float>(input.cursor_y) / win_h * 2.0f - 1.0f;

            glm::mat4 inv_proj = glm::inverse(cam_proj);
            glm::vec4 clip_pt = inv_proj * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
            clip_pt /= clip_pt.w;
            glm::vec3 ray_local = glm::normalize(glm::vec3(clip_pt));
            glm::dvec3 ray_origin = camera_eye_position(g_camera);
            glm::dvec3 ray_dir = glm::normalize(glm::dvec3(camera_orientation(g_camera) * ray_local));

            // Adaptive forward march against the heightfield, then bisect.
            // Sphere-intersect is too coarse near the surface (FP) and ignores
            // stamps; ray-march catches both.
            auto sample = [&](const glm::dvec3& p) -> double {
                double r = glm::length(p);
                glm::vec3 dir = glm::vec3(p / r);
                float h = cpu_terrain_height_with_stamps(dir, g_stamps.data(),
                    static_cast<uint32_t>(g_stamps.size()));
                if (g_ui.ocean_enabled) h = std::max(h, g_ui.sea_level);
                return r - (static_cast<double>(PLANET_RADIUS) + static_cast<double>(h));
            };

            const double max_dist = 200000.0;  // 200 km
            double t = 0.0;
            double prev_t = 0.0;
            double prev_d = sample(ray_origin);
            // If we're somehow already below surface, nothing to pick.
            if (prev_d > 0.0) {
                double step = std::max(prev_d * 0.5, 1.0);
                bool found = false;
                for (int i = 0; i < 64; ++i) {
                    t += step;
                    if (t > max_dist) break;
                    double d = sample(ray_origin + t * ray_dir);
                    if (d <= 0.0) { found = true; break; }
                    prev_t = t;
                    prev_d = d;
                    // Geometric step growth, bounded.
                    step = std::clamp(d * 0.8, 1.0, 5000.0);
                }
                if (found) {
                    double lo = prev_t, hi = t;
                    for (int i = 0; i < 20; ++i) {
                        double mid = 0.5 * (lo + hi);
                        if (sample(ray_origin + mid * ray_dir) > 0.0) lo = mid;
                        else hi = mid;
                    }
                    glm::dvec3 hit = ray_origin + hi * ray_dir;
                    stamp_sphere_dir = glm::normalize(glm::vec3(hit));
                    cursor_on_world = true;
                }
            }
        }
        bool brush_active = input.lmb_held && !input.rmb_held;
        bool brush_hit = brush_active && cursor_on_world;

        // ---- Place terrain stamp on LMB ----
        if (brush_hit &&
            (input.brush_mode == BrushMode::Raise || input.brush_mode == BrushMode::Lower) &&
            g_stamps.size() < MAX_STAMPS)
        {
            static double last_stamp_time = 0.0;
            double now_s = glfwGetTime();
            if (now_s - last_stamp_time > 0.1) {
                float sign = (input.brush_mode == BrushMode::Raise) ? 1.0f : -1.0f;
                float angular_radius = g_ui.brush_radius_grid * g_ui.stamp_angular_scale;
                angular_radius = std::clamp(angular_radius, 0.0001f, 0.2f);

                TerrainStamp stamp{};
                stamp.pos_x = stamp_sphere_dir.x;
                stamp.pos_y = stamp_sphere_dir.y;
                stamp.pos_z = stamp_sphere_dir.z;
                stamp.radius = angular_radius;
                stamp.delta_h = sign * g_ui.terrain_strength;
                stamp.cos_radius = std::cos(angular_radius);
                g_stamps.push_back(stamp);
                g_stamps_dirty = true;
                last_stamp_time = now_s;
            }
        }

        // ---- Place water stamp on LMB (parallel to terrain stamps) ----
        // Persistent record of brushed water; applied to every LOD's SWE init so
        // a brushed lake stays visible from any zoom. The dynamic SWE step still
        // adds waves on top at the brushed level — this is the static skeleton.
        if (brush_hit && input.brush_mode == BrushMode::Water &&
            g_water_stamps.size() < MAX_WATER_STAMPS)
        {
            static double last_water_stamp_time = 0.0;
            double now_s = glfwGetTime();
            if (now_s - last_water_stamp_time > 0.1) {
                float angular_radius = g_ui.brush_radius_grid * g_ui.stamp_angular_scale;
                angular_radius = std::clamp(angular_radius, 0.0001f, 0.2f);

                WaterStamp s{};
                s.pos_x = stamp_sphere_dir.x;
                s.pos_y = stamp_sphere_dir.y;
                s.pos_z = stamp_sphere_dir.z;
                s.radius = angular_radius;
                // The brush deposits a *column*: per stamp adds brush_strength
                // metres at the center, falling off via the gaussian. This makes
                // brushing build up a lake over a couple of seconds.
                s.water_amount = g_ui.brush_strength;
                s.cos_radius = std::cos(angular_radius);
                g_water_stamps.push_back(s);
                g_water_stamps_dirty = true;
                last_water_stamp_time = now_s;
            }
        }

        // ---- Camera UBO update (perspective + brush ring) ----
        {
            float brush_radius_world = g_ui.brush_radius_grid * bp.cell_spacing;
            glm::vec4 brush_color;
            if (input.brush_mode == BrushMode::Raise)
                brush_color = glm::vec4(0.30f, 0.95f, 0.40f, 1.0f);
            else if (input.brush_mode == BrushMode::Lower)
                brush_color = glm::vec4(0.95f, 0.45f, 0.20f, 1.0f);
            else if (input.brush_mode == BrushMode::Sand)
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
            cam._pad2 = g_ui.mud_visibility;
            float angular_r = g_ui.brush_radius_grid * g_ui.stamp_angular_scale;
            cam.brush_world = glm::vec4(
                stamp_sphere_dir.x, stamp_sphere_dir.y, stamp_sphere_dir.z,
                cursor_on_world ? angular_r : 0.0f);
            cam.brush_color = brush_color;
            cam.inv_view_proj = glm::inverse(cam_proj * cam_view);
            std::memcpy(camera_ubo_info.pMappedData, &cam, sizeof(cam));
            vmaFlushAllocation(allocator, camera_ubo_alloc, 0, VK_WHOLE_SIZE);
        }

        // ---- Terrain brush dispatch (before SWE) ----
        if (brush_hit && (input.brush_mode == BrushMode::Raise || input.brush_mode == BrushMode::Lower)) {
            float sign = (input.brush_mode == BrushMode::Raise) ? 1.0f : -1.0f;

            TerrainBrushPC tb_pc{};
            tb_pc.brush_x = grid_x;
            tb_pc.brush_y = grid_y;
            tb_pc.brush_radius = g_ui.brush_radius_grid;
            tb_pc.brush_amount = sign * g_ui.terrain_strength * std::min(dt, 0.033f);
            tb_pc.grid_w = bp.grid_w;
            tb_pc.grid_h = bp.grid_h;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.terrain_brush_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipelines.tb_pipeline_layout, 0, 1, &tb_desc_set, 0, nullptr);
            vkCmdPushConstants(frame.cmd, pipelines.tb_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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
        if (brush_hit && input.brush_mode == BrushMode::Sand) {
            TerrainBrushPC sb_pc{};
            sb_pc.brush_x = grid_x;
            sb_pc.brush_y = grid_y;
            sb_pc.brush_radius = g_ui.brush_radius_grid;
            sb_pc.brush_amount = g_ui.brush_strength * std::min(dt, 0.033f);
            sb_pc.grid_w = bp.grid_w;
            sb_pc.grid_h = bp.grid_h;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.terrain_brush_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipelines.tb_pipeline_layout, 0, 1, &sand_brush_desc_set, 0, nullptr);
            vkCmdPushConstants(frame.cmd, pipelines.tb_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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
        PlanetTraversalParams tp{};
        tp.cam_pos = camera_eye_position(g_camera);
        tp.cam_forward = glm::dvec3(camera_forward(g_camera));
        tp.screen_height = static_cast<float>(extent.height);
        tp.fov_y = g_camera.fov_y;
        tp.planet_radius = PLANET_RADIUS;
        tp.subdivide_threshold = TILE_SUBDIVIDE_PX;
        tp.max_level = PLANET_MAX_LEVEL;
        tp.max_tiles = PLANET_TILE_POOL;
        tp.altitude_above_terrain = g_altitude_above_terrain;
        auto visible_tiles = planet_select_visible_tiles(tp);
        std::sort(visible_tiles.begin(), visible_tiles.end(), [](const QuadNode& a, const QuadNode& b) {
            if (a.face != b.face) return a.face < b.face;
            if (a.level != b.level) return a.level < b.level;
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        });
        g_ui.visible_tile_count = static_cast<uint32_t>(visible_tiles.size());

        // ---- Auto-anchor neighbors from previous frame's edge flags -----------
        // The step shader InterlockedOrs which edges of each disturbed tile have
        // water above static. We resolve those flagged tiles' missing same-face
        // same-level neighbors and anchor them so water can flow into them next
        // frame. Face-seam and LOD-mismatch crossings are deferred to Phase 2/3
        // (planet_neighbor_same_face returns invalid; that edge stays reflective).
        if (g_ui.ocean_enabled) {
            const uint32_t* flags = static_cast<const uint32_t*>(edge_flags_info.pMappedData);
            if (flags) {
                for (uint32_t s = 0; s < PLANET_TILE_POOL; ++s) {
                    if (flags[s] == 0) continue;
                    QuadNode tile = slot_to_tile[s];
                    if (tile.face == INVALID_TILE.face) continue;  // slot is unassigned
                    for (int dir = 0; dir < 4; ++dir) {
                        uint32_t bit = 1u << dir;
                        if (!(flags[s] & bit)) continue;
                        QuadNeighbor nb = planet_neighbor_same_face(tile, dir);
                        if (!nb.valid) continue;
                        if (tile_slot_map.count(nb.tile)) {
                            // Already anchored. Just join the simulation.
                            disturbed_tiles.insert(nb.tile);
                            continue;
                        }
                        if (free_slots.empty()) continue;
                        uint32_t ns = free_slots.back();
                        free_slots.pop_back();
                        tile_slot_map.emplace(nb.tile, ns);
                        slot_to_tile[ns] = nb.tile;
                        disturbed_tiles.insert(nb.tile);
                        pending_init.insert(nb.tile);
                    }
                }
            }
        }

        // ---- Stable slot allocation -------------------------------------------
        // tile_slots[i] is the pool_index for visible_tiles[i]. Tiles new to the
        // visible set are pushed into pending_init for the SWE init pass.
        std::vector<uint32_t> tile_slots(visible_tiles.size());
        {
            std::unordered_set<QuadNode, QuadNodeHash> visible_set;
            visible_set.reserve(visible_tiles.size());
            for (const auto& t : visible_tiles) visible_set.insert(t);

            // Free slots whose tile has left the visible set — but never free a
            // disturbed (brushed) tile. Those stay anchored to their slot so the
            // SWE simulation keeps running and the water survives camera moves
            // / LOD transitions, like terrain stamps survive.
            for (auto it = tile_slot_map.begin(); it != tile_slot_map.end(); ) {
                if (!visible_set.count(it->first) && !disturbed_tiles.count(it->first)) {
                    slot_to_tile[it->second] = INVALID_TILE;
                    free_slots.push_back(it->second);
                    it = tile_slot_map.erase(it);
                } else {
                    ++it;
                }
            }

            // Resolve / allocate slots for visible tiles.
            for (uint32_t i = 0; i < visible_tiles.size(); i++) {
                auto it = tile_slot_map.find(visible_tiles[i]);
                if (it != tile_slot_map.end()) {
                    tile_slots[i] = it->second;
                } else {
                    // PLANET_TILE_POOL >> typical visible count; underrun should not happen.
                    uint32_t slot = free_slots.empty() ? 0u : free_slots.back();
                    if (!free_slots.empty()) free_slots.pop_back();
                    tile_slot_map.emplace(visible_tiles[i], slot);
                    slot_to_tile[slot] = visible_tiles[i];
                    tile_slots[i] = slot;
                    pending_init.insert(visible_tiles[i]);
                }
            }
        }
        if (planet_swe_needs_full_init) {
            for (const auto& t : visible_tiles) pending_init.insert(t);
            planet_swe_needs_full_init = false;
        }
        {

            // Upload stamps to GPU if changed
            if (g_stamps_dirty && !g_stamps.empty()) {
                size_t copy_size = g_stamps.size() * sizeof(TerrainStamp);
                std::memcpy(stamp_buf_info.pMappedData, g_stamps.data(), copy_size);
                vmaFlushAllocation(allocator, stamp_buf_alloc, 0, copy_size);
                g_stamps_dirty = false;
            }
            if (g_water_stamps_dirty && !g_water_stamps.empty()) {
                size_t copy_size = g_water_stamps.size() * sizeof(WaterStamp);
                std::memcpy(water_stamp_buf_info.pMappedData, g_water_stamps.data(), copy_size);
                vmaFlushAllocation(allocator, water_stamp_buf_alloc, 0, copy_size);
                g_water_stamps_dirty = false;
            }

            // Dispatch planet_gen compute for each visible tile
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.terrain_gen_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelines.terrain_gen_pipeline_layout, 0, 1, &terrain_gen_desc_set, 0, nullptr);

            for (uint32_t i = 0; i < visible_tiles.size(); i++) {
                const auto& tile = visible_tiles[i];
                float ts = 2.0f / static_cast<float>(1u << tile.level);

                PlanetGenPC gen_pc{};
                gen_pc.u_min = -1.0f + tile.x * ts;
                gen_pc.v_min = -1.0f + tile.y * ts;
                gen_pc.tile_size = ts;
                gen_pc.face = tile.face;
                gen_pc.pool_index = tile_slots[i];
                gen_pc.tex_res = PLANET_TILE_RES;
                gen_pc.seed = 42;
                gen_pc.stamp_count = static_cast<uint32_t>(g_stamps.size());

                vkCmdPushConstants(frame.cmd, pipelines.terrain_gen_pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gen_pc), &gen_pc);
                vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
            }

            // Barrier: compute -> compute+vertex (SWE reads heightmap, then VS reads both)
            VkMemoryBarrier2 gen_bar{};
            gen_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            gen_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            gen_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            gen_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            gen_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            VkDependencyInfo gen_dep{};
            gen_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            gen_dep.memoryBarrierCount = 1;
            gen_dep.pMemoryBarriers = &gen_bar;
            vkCmdPipelineBarrier2(frame.cmd, &gen_dep);
        }

        // ---- Planet SWE init + step (per-tile, only on disturbed tiles) ----
        if (g_ui.ocean_enabled) {
            // Clear edge_flags before this frame's step; barrier so InterlockedOrs
            // see a zeroed buffer.
            vkCmdFillBuffer(frame.cmd, edge_flags_buf, 0, VK_WHOLE_SIZE, 0u);
            {
                VkMemoryBarrier2 ef_bar{};
                ef_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                ef_bar.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                ef_bar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                ef_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                ef_bar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                     | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers = &ef_bar;
                vkCmdPipelineBarrier2(frame.cmd, &dep);
            }

            // Init dispatch — consume pending_init.
            bool any_init = !pending_init.empty();
            if (any_init) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelines.planet_swe_init_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelines.planet_swe_init_pipeline_layout, 0, 1,
                    &planet_swe_init_desc_set, 0, nullptr);
                for (const auto& t : pending_init) {
                    auto it = tile_slot_map.find(t);
                    if (it == tile_slot_map.end()) continue;
                    float ts = 2.0f / static_cast<float>(1u << t.level);
                    PlanetSweInitPC ipc{};
                    ipc.grid_w = PLANET_TILE_RES;
                    ipc.grid_h = PLANET_TILE_RES;
                    ipc.sea_level = g_ui.sea_level;
                    ipc.pool_index = it->second;
                    ipc.u_min = -1.0f + t.x * ts;
                    ipc.v_min = -1.0f + t.y * ts;
                    ipc.tile_size = ts;
                    ipc.face = t.face;
                    ipc.water_stamp_count = static_cast<uint32_t>(g_water_stamps.size());
                    vkCmdPushConstants(frame.cmd, pipelines.planet_swe_init_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ipc), &ipc);
                    vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
                }
                pending_init.clear();
            }

            // Pick target tile under the cursor for the water brush.
            PlanetTilePick pick{};
            bool water_brush_active = brush_hit && input.brush_mode == BrushMode::Water;
            if (water_brush_active) {
                pick = planet_pick_tile(stamp_sphere_dir, visible_tiles, PLANET_TILE_RES);
                if (pick.hit) disturbed_tiles.insert(pick.node);
            }

            if (any_init) {
                VkMemoryBarrier2 init_bar{};
                init_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                init_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                init_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                init_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                init_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                       | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers = &init_bar;
                vkCmdPipelineBarrier2(frame.cmd, &dep);
            }

            // ---- h-adjust: subtract any newly-applied terrain stamp's local Δz
            // from disturbed tiles' water column h. Preserves water surface across
            // bed changes (gated on previously-wet cells, so dry land doesn't
            // spontaneously fill when terrain is lowered).
            uint32_t cur_stamp_count = static_cast<uint32_t>(g_stamps.size());
            // Discard the unprocessed range if the user hit "clear" or "undo" so
            // we don't re-apply stamps that no longer exist.
            if (cur_stamp_count < last_processed_stamp_count)
                last_processed_stamp_count = cur_stamp_count;

            bool any_h_adjust = (cur_stamp_count > last_processed_stamp_count) && !disturbed_tiles.empty();
            if (any_h_adjust) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelines.planet_swe_h_adjust_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelines.planet_swe_h_adjust_pipeline_layout, 0, 1,
                    &planet_swe_h_adjust_desc_set, 0, nullptr);
                for (uint32_t si = last_processed_stamp_count; si < cur_stamp_count; ++si) {
                    const TerrainStamp& s = g_stamps[si];
                    for (const auto& tile : disturbed_tiles) {
                        auto it = tile_slot_map.find(tile);
                        if (it == tile_slot_map.end()) continue;
                        float ts = 2.0f / static_cast<float>(1u << tile.level);
                        PlanetSweHAdjustPC hpc{};
                        hpc.u_min = -1.0f + tile.x * ts;
                        hpc.v_min = -1.0f + tile.y * ts;
                        hpc.tile_size = ts;
                        hpc.face = tile.face;
                        hpc.pool_index = it->second;
                        hpc.grid_w = PLANET_TILE_RES;
                        hpc.grid_h = PLANET_TILE_RES;
                        hpc.stamp_pos_x = s.pos_x;
                        hpc.stamp_pos_y = s.pos_y;
                        hpc.stamp_pos_z = s.pos_z;
                        hpc.stamp_cos_radius = s.cos_radius;
                        hpc.stamp_delta_h = s.delta_h;
                        vkCmdPushConstants(frame.cmd, pipelines.planet_swe_h_adjust_pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hpc), &hpc);
                        vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
                    }
                }
                last_processed_stamp_count = cur_stamp_count;

                VkMemoryBarrier2 ha_bar{};
                ha_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                ha_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                ha_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                ha_bar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                ha_bar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                     | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo ha_dep{};
                ha_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                ha_dep.memoryBarrierCount = 1;
                ha_dep.pMemoryBarriers = &ha_bar;
                vkCmdPipelineBarrier2(frame.cmd, &ha_dep);
            } else if (cur_stamp_count > last_processed_stamp_count) {
                // No disturbed tiles to adjust, but still mark stamps as processed
                // so they aren't re-applied if water is brushed later.
                last_processed_stamp_count = cur_stamp_count;
            }

            // Step pass: simulate every disturbed tile, even ones that are not
            // currently visible. The slot allocator keeps disturbed tiles anchored,
            // so their heightmap and water state survive in the pool.
            if (!disturbed_tiles.empty()) {
                // Resolve disturbed tiles → pool slots + neighbor slots once.
                // Tiles with no slot (shouldn't happen with anchored allocation,
                // but guard) are skipped. Neighbor slots are 0xFFFFFFFFu when the
                // neighbor is across a face seam / different LOD / unallocated;
                // the shader treats those edges as reflective.
                constexpr uint32_t NO_NEIGHBOR = 0xFFFFFFFFu;
                struct DisturbedDispatch {
                    QuadNode tile;
                    uint32_t slot;
                    float    tile_dx;
                    uint32_t nb[4];  // L, R, D, U
                };
                std::vector<DisturbedDispatch> dd;
                dd.reserve(disturbed_tiles.size());
                for (const auto& tile : disturbed_tiles) {
                    auto it = tile_slot_map.find(tile);
                    if (it == tile_slot_map.end()) continue;
                    float ts = 2.0f / static_cast<float>(1u << tile.level);
                    float tile_dx = ts * PLANET_RADIUS / static_cast<float>(PLANET_TILE_RES - 1);
                    DisturbedDispatch entry{tile, it->second, tile_dx,
                                            {NO_NEIGHBOR, NO_NEIGHBOR, NO_NEIGHBOR, NO_NEIGHBOR}};
                    for (int dir = 0; dir < 4; ++dir) {
                        QuadNeighbor nb = planet_neighbor_same_face(tile, dir);
                        if (!nb.valid) continue;
                        auto nit = tile_slot_map.find(nb.tile);
                        if (nit != tile_slot_map.end()) entry.nb[dir] = nit->second;
                    }
                    dd.push_back(entry);
                }

                // CFL from the finest disturbed tile.
                float swe_total_dt = std::min(dt, 0.033f) * g_ui.time_scale;
                float min_dx = 1e30f;
                for (const auto& d : dd) min_dx = std::min(min_dx, d.tile_dx);
                if (min_dx < 1.0f) min_dx = 1.0f;

                float c_max = std::sqrt(g_ui.gravity * 200.0f);
                float cfl_dt = min_dx / (c_max * 6.0f);
                int substeps = std::max(1, static_cast<int>(std::ceil(swe_total_dt / cfl_dt)));
                substeps = std::min(substeps, 16);
                float sub_dt = swe_total_dt / substeps;

                // World brush radius derived the same way the cursor ring is sized:
                // brush_world.w (in main.cpp ~2018) is an *angular* radius (radians);
                // terrain.fs.hlsl draws the ring against acos(dot(frag_dir, brush_dir)).
                // The arc length on the planet surface is angular * R.
                float angular_r = g_ui.brush_radius_grid * g_ui.stamp_angular_scale;
                float brush_world_r = angular_r * PLANET_RADIUS;

                // Identify which entry in dd is the brush target (only meaningful while
                // the brush is active and pointing at a visible disturbed tile).
                int pick_dd_index = -1;
                if (water_brush_active && pick.hit) {
                    for (size_t k = 0; k < dd.size(); ++k) {
                        if (dd[k].tile == pick.node) { pick_dd_index = (int)k; break; }
                    }
                }

                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelines.planet_swe_step_pipeline);

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
                        pipelines.planet_swe_step_pipeline_layout, 0, 1,
                        &planet_swe_step_desc_sets[planet_swe_ping_pong], 0, nullptr);

                    for (size_t k = 0; k < dd.size(); ++k) {
                        const auto& d = dd[k];
                        PlanetSweStepPC spc{};
                        spc.time = static_cast<float>(now);
                        spc.dt = sub_dt;
                        spc.gravity = g_ui.gravity;
                        spc.friction = g_ui.friction;
                        spc.dx = d.tile_dx;
                        spc.sea_level = g_ui.sea_level;
                        spc.damping = g_ui.damping;
                        spc.pool_index = d.slot;
                        spc.grid_w = PLANET_TILE_RES;
                        spc.grid_h = PLANET_TILE_RES;

                        // Only the picked tile, and only on the first substep,
                        // injects the brush pulse — keeps total injection rate-like
                        // regardless of substep count.
                        bool inject_here = (step == 0) && ((int)k == pick_dd_index);
                        if (inject_here) {
                            float radius_cells = std::max(1.0f, brush_world_r / d.tile_dx);
                            spc.pulse_x = pick.grid_x;
                            spc.pulse_y = pick.grid_y;
                            spc.pulse_radius = radius_cells;
                            spc.pulse_amount = g_ui.brush_strength * sub_dt;
                        } else {
                            spc.pulse_amount = 0.0f;
                        }

                        spc.neighbor_left  = d.nb[0];
                        spc.neighbor_right = d.nb[1];
                        spc.neighbor_down  = d.nb[2];
                        spc.neighbor_up    = d.nb[3];

                        vkCmdPushConstants(frame.cmd, pipelines.planet_swe_step_pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(spc), &spc);
                        vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
                    }

                    planet_swe_ping_pong ^= 1;
                }

                // Make this frame's edge_flags writes visible to the host so the
                // next frame's auto-anchor pass can read them.
                VkMemoryBarrier2 ef_host_bar{};
                ef_host_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                ef_host_bar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                ef_host_bar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                ef_host_bar.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
                ef_host_bar.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
                VkDependencyInfo ef_host_dep{};
                ef_host_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                ef_host_dep.memoryBarrierCount = 1;
                ef_host_dep.pMemoryBarriers = &ef_host_bar;
                vkCmdPipelineBarrier2(frame.cmd, &ef_host_dep);
            }
        }

        // ---- Atmosphere 3D dispatch ----
        if (g_ui.atmosphere_enabled) {
            g_accumulated_atmo_time += dt * g_ui.time_scale;

            Atmo3DPC apc{};
            apc.dt = std::min(dt, 0.033f) * g_ui.time_scale;
            apc.accumulated_time = g_accumulated_atmo_time;
            apc.grid_w = ATMO_W;
            apc.grid_h = ATMO_H;
            apc.grid_d = ATMO_D;
            apc.terrain_scale = bp.cell_spacing;
            apc.layer_height = ATMO_LAYER_HEIGHT;
            apc.max_elevation = 2000.0f;
            apc.orographic_lift_coeff = g_ui.orographic_lift;
            apc.adiabatic_cooling_rate = g_ui.adiabatic_cooling;
            apc.rain_shadow_intensity = g_ui.rain_shadow;
            apc.force_init = (atmo_needs_init || g_ui.request_atmo_reset) ? 1u : 0u;
            apc.sand_enabled = g_ui.sand_enabled ? 1u : 0u;
            apc.sand_loft_threshold = g_ui.sand_loft_threshold;
            apc.sand_loft_rate = g_ui.sand_loft_rate;
            apc.sand_settling = g_ui.sand_settling;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.atmo_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelines.atmo_pipeline_layout, 0, 1, &atmo_desc_sets[atmo_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, pipelines.atmo_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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

            if (atmo_needs_init || g_ui.request_atmo_reset) {
                atmo_needs_init = false;
                g_ui.request_atmo_reset = false;
            }
        }

        // ---- Sand particle dispatch ----
        if (g_ui.atmosphere_enabled && g_ui.sand_enabled) {
            static uint32_t sand_emit_offset = 0;

            SandSimPC spc{};
            spc.dt = std::min(dt, 0.033f) * g_ui.time_scale;
            spc.terrain_size = terrain_size;
            spc.loft_threshold = g_ui.sand_loft_threshold;
            spc.loft_rate = g_ui.sand_loft_rate;
            spc.gravity = g_ui.sand_gravity;
            spc.accumulated_time = g_accumulated_atmo_time;
            spc.max_particles = SAND_MAX_PARTICLES;
            spc.emit_offset = sand_emit_offset;
            spc.emit_count = SAND_EMIT_PER_FRAME;
            spc.grid_d = ATMO_D;
            spc.layer_height = ATMO_LAYER_HEIGHT;
            spc.bounce_energy = g_ui.sand_bounce;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.sand_sim_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelines.sand_sim_pipeline_layout, 0, 1, &sand_sim_desc_sets[atmo_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, pipelines.sand_sim_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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

        // ---- SWE step dispatch (CFL sub-stepping) — flat grid, disabled when ocean is on ----
        if (!g_ui.ocean_enabled) {
            vkCmdResetQueryPool(frame.cmd, renderer.query_pool, current_frame * 2, 2);

            float swe_total_dt = std::min(dt, 0.033f) * g_ui.time_scale;
            float c_max = std::sqrt(g_ui.gravity * 200.0f);
            float cfl_dt = bp.cell_spacing / (c_max * 6.0f);
            int substeps = std::max(1, static_cast<int>(std::ceil(swe_total_dt / cfl_dt)));
            substeps = std::min(substeps, 16);
            float sub_dt = swe_total_dt / substeps;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.swe_step_pipeline);

            vkCmdWriteTimestamp2(frame.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                renderer.query_pool, current_frame * 2 + 0);

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
                                        pipelines.swe_step_pipeline_layout, 0, 1,
                                        &swe_step_desc_sets[swe_ping_pong], 0, nullptr);

                SweStepPC swe_pc{};
                swe_pc.time = static_cast<float>(now);
                swe_pc.dt = sub_dt;
                swe_pc.gravity = g_ui.gravity;
                swe_pc.friction = g_ui.friction;
                swe_pc.dx = bp.cell_spacing;
                swe_pc.sea_level = bp.floor_height + bp.initial_water;
                swe_pc.damping = g_ui.damping;
                swe_pc._pad0 = 0.0f;
                swe_pc.grid_w = SWE_GRID_W;
                swe_pc.grid_h = SWE_GRID_H;

                if (step == 0) {
                    bool water_brush_active = brush_hit && input.brush_mode == BrushMode::Water;
                    if (input.pulse_pending) {
                        swe_pc.pulse_x = SWE_GRID_W * 0.5f;
                        swe_pc.pulse_y = SWE_GRID_H * 0.5f;
                        swe_pc.pulse_radius = g_ui.pulse_radius_cells;
                        swe_pc.pulse_amount = g_ui.pulse_amount;
                        input.pulse_pending = false;
                    } else if (water_brush_active) {
                        swe_pc.pulse_x = grid_x;
                        swe_pc.pulse_y = grid_y;
                        swe_pc.pulse_radius = g_ui.brush_radius_grid;
                        swe_pc.pulse_amount = g_ui.brush_strength;
                    } else {
                        swe_pc.pulse_amount = 0.0f;
                    }
                } else {
                    swe_pc.pulse_amount = 0.0f;
                }

                vkCmdPushConstants(frame.cmd, pipelines.swe_step_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(swe_pc), &swe_pc);
                vkCmdDispatch(frame.cmd, (SWE_GRID_W + 7) / 8, (SWE_GRID_H + 7) / 8, 1);

                swe_ping_pong ^= 1;
            }

            vkCmdWriteTimestamp2(frame.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                renderer.query_pool, current_frame * 2 + 1);
        }

        // ---- Erosion dispatch (after SWE, before graphics) ----
        if (g_ui.erosion_enabled) {
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

            float ero_dt = std::min(dt, 0.033f) * g_ui.time_scale;

            ErosionPC ero_pc{};
            ero_pc.dt = ero_dt;
            ero_pc.dx = bp.cell_spacing;
            ero_pc.grid_w = SWE_GRID_W;
            ero_pc.grid_h = SWE_GRID_H;
            ero_pc.k_erosion = g_ui.k_erosion;
            ero_pc.k_deposit = g_ui.k_deposit;
            ero_pc.k_capacity = g_ui.k_capacity;
            ero_pc.min_slope = g_ui.min_slope;
            ero_pc.min_depth = g_ui.min_erosion_depth;
            ero_pc.max_change = g_ui.max_change_m;
            ero_pc.max_sediment = g_ui.max_sediment;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.erosion_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipelines.ero_pipeline_layout, 0, 1,
                                    &ero_desc_sets[sediment_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, pipelines.ero_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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
            sc_barrier.image = renderer.swapchain_images[image_index];
            sc_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkImageMemoryBarrier2 depth_barrier{};
            depth_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            depth_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            depth_barrier.srcAccessMask = VK_ACCESS_2_NONE;
            depth_barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            depth_barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depth_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depth_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_barrier.image = renderer.depth_buffer.image;
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
            color_attachment.imageView = renderer.swapchain_views[image_index];
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attachment.clearValue.color = {{0.02f, 0.02f, 0.05f, 1.0f}};

            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = renderer.depth_buffer.view;
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
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.clipmap_terrain_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelines.clipmap_gfx_pipeline_layout, 0, 1, &clipmap_gfx_desc_sets[atmo_ping_pong], 0, nullptr);

            VkDeviceSize clip_offset = 0;
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &clipmap_vbo, &clip_offset);
            vkCmdBindIndexBuffer(frame.cmd, clipmap_ibo, 0, VK_INDEX_TYPE_UINT32);

            glm::dvec3 cam_pos_d = camera_eye_position(g_camera);

            for (uint32_t i = 0; i < visible_tiles.size(); i++) {
                const auto& tile = visible_tiles[i];
                float ts = 2.0f / static_cast<float>(1u << tile.level);

                // rel_xyz = center_dir * R - cam (VS adds height displacement)
                glm::dvec3 dir = planet_tile_center_dir(tile);
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
                tpc.pool_index = tile_slots[i];
                tpc.planet_radius = PLANET_RADIUS;
                tpc.max_elevation = PLANET_MAX_ELEVATION;
                tpc.heightmap_texel = 1.0f / static_cast<float>(PLANET_TILE_RES);
                tpc.cloud_opacity = 0.0f;
                tpc.sea_level = g_ui.ocean_enabled ? g_ui.sea_level : -1.0f;

                vkCmdPushConstants(frame.cmd, pipelines.clipmap_gfx_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(tpc), &tpc);
                vkCmdDrawIndexed(frame.cmd, clipmap_index_count, 1, 0, 0, 0);
            }

            // Water pass (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.water_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipelines.gfx_pipeline_layout, 0, 1, &gfx_desc_sets[atmo_ping_pong], 0, nullptr);
                GfxPC wpc{};
                wpc.terrain_size = terrain_size;
                wpc.heightmap_texel = 1.0f / static_cast<float>(bp.grid_w);
                wpc.max_elevation = 2000.0f;
                wpc.cloud_opacity = g_ui.atmosphere_enabled ? g_ui.cloud_opacity : 0.0f;
                vkCmdPushConstants(frame.cmd, pipelines.gfx_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(wpc), &wpc);
                VkDeviceSize water_offset = 0;
                vkCmdBindVertexBuffers(frame.cmd, 0, 1, &vertex_buffer, &water_offset);
                vkCmdBindIndexBuffer(frame.cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(frame.cmd, index_count, 1, 0, 0, 0);
            }

            // Cloud raymarch pass (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.raymarch_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipelines.raymarch_pipeline_layout, 0, 1, &gfx_desc_sets[atmo_ping_pong], 0, nullptr);

                RaymarchPC rpc{};
                rpc.terrain_size = terrain_size;
                rpc.max_elevation = 2000.0f;
                rpc.cloud_opacity = g_ui.cloud_opacity;
                rpc.cloud_base = g_ui.cloud_altitude;
                rpc.vol_w = ATMO_W;
                rpc.vol_h = ATMO_H;
                rpc.vol_d = ATMO_D;
                rpc.layer_height = ATMO_LAYER_HEIGHT;
                vkCmdPushConstants(frame.cmd, pipelines.raymarch_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(rpc), &rpc);

                vkCmdDraw(frame.cmd, 3, 1, 0, 0);
            }

            // Sand particle draw (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.sand_render_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipelines.sand_render_pipeline_layout, 0, 1, &sand_render_desc_set, 0, nullptr);

                SandRenderPC srpc{};
                srpc.streak_length = g_ui.sand_streak;
                srpc.particle_alpha = g_ui.sand_alpha;
                srpc.max_particles = SAND_MAX_PARTICLES;
                vkCmdPushConstants(frame.cmd, pipelines.sand_render_pipeline_layout,
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
            imgui_color.imageView = renderer.swapchain_views[image_index];
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
            barrier.image = renderer.swapchain_images[image_index];
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
        present_info.pSwapchains = &renderer.vkb_swapchain.swapchain;
        present_info.pImageIndices = &image_index;

        VkResult present_result = vkQueuePresentKHR(renderer.present_queue, &present_info);

        if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
            present_result == VK_SUBOPTIMAL_KHR || input.framebuffer_resized) {
            renderer_rebuild_swapchain(renderer);
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
            if (input.brush_mode == BrushMode::Raise) mode_str = "raise";
            else if (input.brush_mode == BrushMode::Lower) mode_str = "lower";
            float strength_display = (input.brush_mode == BrushMode::Water) ? g_ui.brush_strength : g_ui.terrain_strength;
            const char* unit = (input.brush_mode == BrushMode::Water) ? "m/frame" : "m/s";
            char title[256];
            std::snprintf(title, sizeof(title),
                "drift_engine — CPU %.1f ms | GPU %.1f ms | %.0f fps | %s | size %.0f | strength %.1f %s",
                cpu_avg_ms, gpu_avg_ms, 1000.0 / cpu_avg_ms, mode_str, g_ui.brush_radius_grid, strength_display, unit);
            glfwSetWindowTitle(window, title);
            last_title_update = frame_end;
        }

        current_frame = (current_frame + 1) % FRAMES_IN_FLIGHT;
    }

    // ---- Cleanup (LIFO) -----------------------------------------------------
    vkDeviceWaitIdle(device);

    pipelines_destroy(pipelines, device);

    vkDestroyImageView(device, clipmap_hm_view, nullptr);
    vmaDestroyImage(allocator, clipmap_hm_image, clipmap_hm_alloc);

    vmaDestroyBuffer(allocator, clipmap_ibo, clipmap_ibo_alloc);
    vmaDestroyBuffer(allocator, clipmap_vbo, clipmap_vbo_alloc);

    vkDestroyDescriptorPool(device, desc_pool, nullptr);

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

    renderer_shutdown(renderer);
    return 0;
}

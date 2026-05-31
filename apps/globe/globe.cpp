// globe.cpp — Globe (planet simulation) module implementation.

#include "globe.h"

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
#include <vector>

// Alias constants from globe.h for code compatibility.
static constexpr uint32_t SWE_GRID_W = GLOBE_SWE_GRID_W;
static constexpr uint32_t SWE_GRID_H = GLOBE_SWE_GRID_H;
static constexpr uint32_t ATMO_W = GLOBE_ATMO_W;
static constexpr uint32_t ATMO_H = GLOBE_ATMO_H;
static constexpr uint32_t ATMO_D = GLOBE_ATMO_D;
static constexpr float ATMO_LAYER_HEIGHT = GLOBE_ATMO_LAYER_HEIGHT;
static constexpr uint32_t SAND_MAX_PARTICLES = GLOBE_SAND_MAX_PARTICLES;
static constexpr uint32_t SAND_EMIT_PER_FRAME = GLOBE_SAND_EMIT_PER_FRAME;
static constexpr uint32_t MESH_RES = GLOBE_MESH_RES;
static constexpr uint32_t PLANET_TILE_RES = GLOBE_TILE_RES;
static constexpr uint32_t PLANET_TILE_POOL = GLOBE_TILE_POOL;
static constexpr uint32_t PLANET_MAX_LEVEL = GLOBE_MAX_LEVEL;
static constexpr float    PLANET_RADIUS = GLOBE_PLANET_RADIUS;
static constexpr float    PLANET_MAX_ELEVATION = GLOBE_MAX_ELEVATION;
static constexpr float    TILE_SUBDIVIDE_PX = GLOBE_TILE_SUBDIVIDE_PX;

void globe_init(GlobeState& s, Renderer& r)
{
#ifdef __APPLE__
    setenv("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "0", 0);
#endif

    VkDevice device = r.device;
    VmaAllocator allocator = r.allocator;
    VkQueue graphics_queue = r.graphics_queue;
    uint32_t gfx_family = r.gfx_family;
    GLFWwindow* window = r.window;

    camera_initialize_orientation(s.camera);

    s.cb_ctx = {&s.input, &s.camera, &s.ui};
    if (s.embedded) {
        ImGui_ImplGlfw_RestoreCallbacks(window);
        input_install_callbacks(window, &s.cb_ctx);
        ImGui_ImplGlfw_InstallCallbacks(window);
    } else {
        input_install_callbacks(window, &s.cb_ctx);
        ImGui_ImplGlfw_InstallCallbacks(window);
    }

    // ---- Procedural basin ---------------------------------------------------
    s.basin_params = BasinParams{};
    auto basin = generate_crater_basin(s.basin_params);
    HeightmapData hm{std::move(basin), s.basin_params.grid_w, s.basin_params.grid_h};

    auto [hm_min, hm_max] = std::minmax_element(hm.values.begin(), hm.values.end());
    (void)hm_min; (void)hm_max;
    s.terrain_size = static_cast<float>(s.basin_params.grid_w) * s.basin_params.cell_spacing;
    float terrain_size = s.terrain_size;

    s.heightmap_gpu = upload_heightmap(device, allocator, graphics_queue, gfx_family, hm);

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
    s.index_count = static_cast<uint32_t>(mesh_indices.size());

    VkBufferCreateInfo vbo_ci{};
    vbo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vbo_ci.size = mesh_vertices.size() * sizeof(float);
    vbo_ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VmaAllocationCreateInfo vbo_ai{};
    vbo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    vbo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo vbo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &vbo_ci, &vbo_ai, &s.vertex_buffer, &s.vertex_alloc, &vbo_info));
    std::memcpy(vbo_info.pMappedData, mesh_vertices.data(), vbo_ci.size);
    vmaFlushAllocation(allocator, s.vertex_alloc, 0, VK_WHOLE_SIZE);

    VkBufferCreateInfo ibo_ci{};
    ibo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ibo_ci.size = mesh_indices.size() * sizeof(uint32_t);
    ibo_ci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    VmaAllocationCreateInfo ibo_ai{};
    ibo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    ibo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo ibo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &ibo_ci, &ibo_ai, &s.index_buffer, &s.index_alloc, &ibo_info));
    std::memcpy(ibo_info.pMappedData, mesh_indices.data(), ibo_ci.size);
    vmaFlushAllocation(allocator, s.index_alloc, 0, VK_WHOLE_SIZE);

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

    s.clipmap_index_count = static_cast<uint32_t>(clip_indices.size());

    VkBufferCreateInfo clip_vbo_ci{};
    clip_vbo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    clip_vbo_ci.size = clip_verts.size() * sizeof(float);
    clip_vbo_ci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VmaAllocationCreateInfo clip_vbo_ai{};
    clip_vbo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    clip_vbo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo clip_vbo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &clip_vbo_ci, &clip_vbo_ai, &s.clipmap_vbo, &s.clipmap_vbo_alloc, &clip_vbo_info));
    std::memcpy(clip_vbo_info.pMappedData, clip_verts.data(), clip_vbo_ci.size);
    vmaFlushAllocation(allocator, s.clipmap_vbo_alloc, 0, VK_WHOLE_SIZE);

    VkBufferCreateInfo clip_ibo_ci{};
    clip_ibo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    clip_ibo_ci.size = clip_indices.size() * sizeof(uint32_t);
    clip_ibo_ci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    VmaAllocationCreateInfo clip_ibo_ai{};
    clip_ibo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    clip_ibo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo clip_ibo_info{};
    VK_CHECK(vmaCreateBuffer(allocator, &clip_ibo_ci, &clip_ibo_ai, &s.clipmap_ibo, &s.clipmap_ibo_alloc, &clip_ibo_info));
    std::memcpy(clip_ibo_info.pMappedData, clip_indices.data(), clip_ibo_ci.size);
    vmaFlushAllocation(allocator, s.clipmap_ibo_alloc, 0, VK_WHOLE_SIZE);

    // ---- Camera UBO ---------------------------------------------------------
    VkBufferCreateInfo ubo_ci{};
    ubo_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ubo_ci.size = sizeof(CameraData);
    ubo_ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo ubo_ai{};
    ubo_ai.usage = VMA_MEMORY_USAGE_AUTO;
    ubo_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VK_CHECK(vmaCreateBuffer(allocator, &ubo_ci, &ubo_ai, &s.camera_ubo, &s.camera_ubo_alloc, &s.camera_ubo_info));

    // ---- Stamp buffer (SSBO for terrain edit stamps) ------------------------
    VkBufferCreateInfo stamp_buf_ci{};
    stamp_buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stamp_buf_ci.size = MAX_STAMPS * sizeof(TerrainStamp);
    stamp_buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo stamp_buf_ai{};
    stamp_buf_ai.usage = VMA_MEMORY_USAGE_AUTO;
    stamp_buf_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                       | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VK_CHECK(vmaCreateBuffer(allocator, &stamp_buf_ci, &stamp_buf_ai,
        &s.stamp_buf, &s.stamp_buf_alloc, &s.stamp_buf_info));

    // ---- Water stamp buffer (parallel to terrain stamps) --------------------
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
            &s.water_stamp_buf, &s.water_stamp_buf_alloc, &s.water_stamp_buf_info));
    }

    // ---- Planet SWE edge-flag buffer (per-pool-slot, host-readable) ----------
    // The step shader InterlockedOrs bits into edge_flags[pool_index] when water
    // crosses an edge above static depth. CPU reads this each frame to anchor
    // missing neighbors and grow the simulation domain.
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
            &s.edge_flags_buf, &s.edge_flags_alloc, &s.edge_flags_info));
    }

    // ---- Atmosphere debug readback (bottom layer of wind field for pressure stats) ----
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = ATMO_W * ATMO_H * sizeof(uint16_t) * 4;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VK_CHECK(vmaCreateBuffer(allocator, &bci, &ai,
            &s.atmo_debug_buf, &s.atmo_debug_alloc, &s.atmo_debug_info));
    }

    // ---- SWE images (ping-pong state + output) ------------------------------
    s.swe_state_a = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    s.swe_state_b = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    s.swe_output  = create_swe_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

    // ---- Sediment images (ping-pong) ----------------------------------------
    s.sediment_a = create_sediment_image(device, allocator, SWE_GRID_W, SWE_GRID_H);
    s.sediment_b = create_sediment_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

    // ---- Atmosphere images -----------------------------------------------------
    s.atmo_state_a = create_volume_image(device, allocator, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    s.atmo_state_b = create_volume_image(device, allocator, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    s.wind_field_a = create_volume_image(device, allocator, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    s.wind_field_b = create_volume_image(device, allocator, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    s.atmo_shadow  = create_sediment_image(device, allocator, ATMO_W, ATMO_H);
    s.ground_cond  = create_swe_image(device, allocator, ATMO_W, ATMO_H);
    s.ground_wind  = create_wind_image(device, allocator, ATMO_W, ATMO_H);
    s.sand_deposit = create_sediment_image(device, allocator, SWE_GRID_W, SWE_GRID_H);

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

    VK_CHECK(vmaCreateImage(allocator, &clip_hm_ci, &clip_hm_ai, &s.clipmap_hm_image, &s.clipmap_hm_alloc, nullptr));

    VkImageViewCreateInfo clip_hm_view_ci{};
    clip_hm_view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    clip_hm_view_ci.image = s.clipmap_hm_image;
    clip_hm_view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    clip_hm_view_ci.format = VK_FORMAT_R32_SFLOAT;
    clip_hm_view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};

    VK_CHECK(vkCreateImageView(device, &clip_hm_view_ci, nullptr, &s.clipmap_hm_view));

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


    create_water_array(s.water_state_a_img, s.water_state_a_alloc, s.water_state_a_view);
    create_water_array(s.water_state_b_img, s.water_state_b_alloc, s.water_state_b_view);
    create_water_array(s.water_output_img, s.water_output_alloc, s.water_output_view);

    // ---- Sand particle buffer --------------------------------------------------
    VkBufferCreateInfo sand_buf_ci{};
    sand_buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    sand_buf_ci.size = SAND_MAX_PARTICLES * 32; // 8 floats per particle
    sand_buf_ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo sand_buf_ai{};
    sand_buf_ai.usage = VMA_MEMORY_USAGE_AUTO;
    sand_buf_ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VK_CHECK(vmaCreateBuffer(allocator, &sand_buf_ci, &sand_buf_ai, &s.sand_particle_buf, &s.sand_particle_alloc, nullptr));

    // ---- Sampler ------------------------------------------------------------
    VkSamplerCreateInfo sampler_ci{};
    sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_ci.magFilter = VK_FILTER_LINEAR;
    sampler_ci.minFilter = VK_FILTER_LINEAR;
    sampler_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    VK_CHECK(vkCreateSampler(device, &sampler_ci, nullptr, &s.sampler));

    // ---- Terrain linear s.sampler (for atmosphere terrain reads) ----
    VkSamplerCreateInfo terrain_samp_ci{};
    terrain_samp_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    terrain_samp_ci.magFilter = VK_FILTER_LINEAR;
    terrain_samp_ci.minFilter = VK_FILTER_LINEAR;
    terrain_samp_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    terrain_samp_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    terrain_samp_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device, &terrain_samp_ci, nullptr, &s.terrain_linear_sampler));

    pipelines_create(s.pipelines, device);

    // ---- Descriptor pool + sets ---------------------------------------------
    // Counts: SWE init(2) + SWE step×2(8) + terrain brush(1) + gfx(4 incl sediment) + erosion×2(8) = needs ~23 descriptors
    VkDescriptorPoolSize pool_sizes[5]{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 45;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_sizes[1].descriptorCount = 48;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[2].descriptorCount = 4;
    pool_sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[3].descriptorCount = 12;
    pool_sizes[4].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    pool_sizes[4].descriptorCount = 32;

    VkDescriptorPoolCreateInfo dp_ci{};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 25;
    dp_ci.poolSizeCount = 5;
    dp_ci.pPoolSizes = pool_sizes;

    VK_CHECK(vkCreateDescriptorPool(device, &dp_ci, nullptr, &s.desc_pool));

    // SWE init descriptor set
    VkDescriptorSetAllocateInfo swe_init_ds_ai{};
    swe_init_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    swe_init_ds_ai.descriptorPool = s.desc_pool;
    swe_init_ds_ai.descriptorSetCount = 1;
    swe_init_ds_ai.pSetLayouts = &s.pipelines.swe_init_desc_layout;

    VK_CHECK(vkAllocateDescriptorSets(device, &swe_init_ds_ai, &s.swe_init_desc_set));

    // SWE step descriptor sets (2 for ping-pong)
    VkDescriptorSetLayout swe_step_layouts[2] = {s.pipelines.swe_step_desc_layout, s.pipelines.swe_step_desc_layout};
    VkDescriptorSetAllocateInfo swe_step_ds_ai{};
    swe_step_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    swe_step_ds_ai.descriptorPool = s.desc_pool;
    swe_step_ds_ai.descriptorSetCount = 2;
    swe_step_ds_ai.pSetLayouts = swe_step_layouts;

    VK_CHECK(vkAllocateDescriptorSets(device, &swe_step_ds_ai, s.swe_step_desc_sets));

    // Graphics descriptor set
    VkDescriptorSetLayout gfx_layouts_2[2] = {s.pipelines.gfx_desc_layout, s.pipelines.gfx_desc_layout};
    VkDescriptorSetAllocateInfo gfx_ds_ai{};
    gfx_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    gfx_ds_ai.descriptorPool = s.desc_pool;
    gfx_ds_ai.descriptorSetCount = 2;
    gfx_ds_ai.pSetLayouts = gfx_layouts_2;

    VK_CHECK(vkAllocateDescriptorSets(device, &gfx_ds_ai, s.gfx_desc_sets));

    // Terrain brush descriptor set
    VkDescriptorSetAllocateInfo tb_ds_ai{};
    tb_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    tb_ds_ai.descriptorPool = s.desc_pool;
    tb_ds_ai.descriptorSetCount = 1;
    tb_ds_ai.pSetLayouts = &s.pipelines.tb_desc_layout;

    VK_CHECK(vkAllocateDescriptorSets(device, &tb_ds_ai, &s.tb_desc_set));

    // Sand brush descriptor set (reuses s.pipelines.tb_desc_layout — just a storage image at binding 0)
    {
        VkDescriptorSetAllocateInfo sb_ds_ai{};
        sb_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        sb_ds_ai.descriptorPool = s.desc_pool;
        sb_ds_ai.descriptorSetCount = 1;
        sb_ds_ai.pSetLayouts = &s.pipelines.tb_desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &sb_ds_ai, &s.sand_brush_desc_set));
    }

    // Erosion descriptor sets (2 for ping-pong)
    VkDescriptorSetLayout ero_layouts[2] = {s.pipelines.ero_desc_layout, s.pipelines.ero_desc_layout};
    VkDescriptorSetAllocateInfo ero_ds_ai{};
    ero_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ero_ds_ai.descriptorPool = s.desc_pool;
    ero_ds_ai.descriptorSetCount = 2;
    ero_ds_ai.pSetLayouts = ero_layouts;

    VK_CHECK(vkAllocateDescriptorSets(device, &ero_ds_ai, s.ero_desc_sets));

    // ---- Write descriptor sets ----------------------------------------------
    // SWE init descriptors
    {
        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = s.heightmap_gpu.view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_info{};
        state_info.imageView = s.swe_state_a.view;
        state_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = s.swe_init_desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].pImageInfo = &terrain_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = s.swe_init_desc_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &state_info;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }

    // SWE step descriptors
    {
        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = s.heightmap_gpu.view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_a_info{};
        state_a_info.imageView = s.swe_state_a.view;
        state_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_b_info{};
        state_b_info.imageView = s.swe_state_b.view;
        state_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo output_info{};
        output_info.imageView = s.swe_output.view;
        output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo ground_cond_info{};
        ground_cond_info.imageView = s.ground_cond.view;
        ground_cond_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo ground_cond_sampler_info{};
        ground_cond_sampler_info.sampler = s.sampler;

        // Set 0: read A -> write B
        VkWriteDescriptorSet writes_0[6]{};
        writes_0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[0].dstSet = s.swe_step_desc_sets[0];
        writes_0[0].dstBinding = 0;
        writes_0[0].descriptorCount = 1;
        writes_0[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_0[0].pImageInfo = &terrain_info;

        writes_0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[1].dstSet = s.swe_step_desc_sets[0];
        writes_0[1].dstBinding = 1;
        writes_0[1].descriptorCount = 1;
        writes_0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_0[1].pImageInfo = &state_a_info;

        writes_0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[2].dstSet = s.swe_step_desc_sets[0];
        writes_0[2].dstBinding = 2;
        writes_0[2].descriptorCount = 1;
        writes_0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_0[2].pImageInfo = &state_b_info;

        writes_0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[3].dstSet = s.swe_step_desc_sets[0];
        writes_0[3].dstBinding = 3;
        writes_0[3].descriptorCount = 1;
        writes_0[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_0[3].pImageInfo = &output_info;

        writes_0[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[4].dstSet = s.swe_step_desc_sets[0];
        writes_0[4].dstBinding = 4;
        writes_0[4].descriptorCount = 1;
        writes_0[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_0[4].pImageInfo = &ground_cond_info;

        writes_0[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_0[5].dstSet = s.swe_step_desc_sets[0];
        writes_0[5].dstBinding = 5;
        writes_0[5].descriptorCount = 1;
        writes_0[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes_0[5].pImageInfo = &ground_cond_sampler_info;

        vkUpdateDescriptorSets(device, 6, writes_0, 0, nullptr);

        // Set 1: read B -> write A
        VkWriteDescriptorSet writes_1[6]{};
        writes_1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[0].dstSet = s.swe_step_desc_sets[1];
        writes_1[0].dstBinding = 0;
        writes_1[0].descriptorCount = 1;
        writes_1[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_1[0].pImageInfo = &terrain_info;

        writes_1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[1].dstSet = s.swe_step_desc_sets[1];
        writes_1[1].dstBinding = 1;
        writes_1[1].descriptorCount = 1;
        writes_1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_1[1].pImageInfo = &state_b_info;

        writes_1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[2].dstSet = s.swe_step_desc_sets[1];
        writes_1[2].dstBinding = 2;
        writes_1[2].descriptorCount = 1;
        writes_1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_1[2].pImageInfo = &state_a_info;

        writes_1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[3].dstSet = s.swe_step_desc_sets[1];
        writes_1[3].dstBinding = 3;
        writes_1[3].descriptorCount = 1;
        writes_1[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes_1[3].pImageInfo = &output_info;

        writes_1[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[4].dstSet = s.swe_step_desc_sets[1];
        writes_1[4].dstBinding = 4;
        writes_1[4].descriptorCount = 1;
        writes_1[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes_1[4].pImageInfo = &ground_cond_info;

        writes_1[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes_1[5].dstSet = s.swe_step_desc_sets[1];
        writes_1[5].dstBinding = 5;
        writes_1[5].descriptorCount = 1;
        writes_1[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes_1[5].pImageInfo = &ground_cond_sampler_info;

        vkUpdateDescriptorSets(device, 6, writes_1, 0, nullptr);
    }

    // Graphics descriptors (2 sets — differ only in binding 5: cloud volume a vs b)
    {
        VkDescriptorBufferInfo ubo_buf_info{};
        ubo_buf_info.buffer = s.camera_ubo;
        ubo_buf_info.offset = 0;
        ubo_buf_info.range = sizeof(CameraData);

        VkDescriptorImageInfo heightmap_info{};
        heightmap_info.imageView = s.heightmap_gpu.view;
        heightmap_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        heightmap_info.sampler = s.sampler;

        VkDescriptorImageInfo swe_out_info{};
        swe_out_info.imageView = s.swe_output.view;
        swe_out_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        swe_out_info.sampler = s.sampler;

        VkDescriptorImageInfo sediment_info{};
        sediment_info.imageView = s.sediment_a.view;
        sediment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sediment_info.sampler = s.sampler;

        VkDescriptorImageInfo shadow_info{};
        shadow_info.imageView = s.atmo_shadow.view;
        shadow_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        shadow_info.sampler = s.sampler;

        VkDescriptorImageInfo vol_a_info{};
        vol_a_info.imageView = s.atmo_state_a.view;
        vol_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_a_info.sampler = s.sampler;

        VkDescriptorImageInfo vol_b_info{};
        vol_b_info.imageView = s.atmo_state_b.view;
        vol_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_b_info.sampler = s.sampler;

        VkDescriptorImageInfo vol_infos[2] = {vol_a_info, vol_b_info};

        VkDescriptorImageInfo wind_a_gfx{};
        wind_a_gfx.imageView = s.wind_field_a.view;
        wind_a_gfx.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wind_a_gfx.sampler = s.sampler;

        VkDescriptorImageInfo wind_b_gfx{};
        wind_b_gfx.imageView = s.wind_field_b.view;
        wind_b_gfx.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wind_b_gfx.sampler = s.sampler;

        VkDescriptorImageInfo wind_infos[2] = {wind_a_gfx, wind_b_gfx};

        VkDescriptorImageInfo sand_dep_info{};
        sand_dep_info.imageView = s.sand_deposit.view;
        sand_dep_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sand_dep_info.sampler = s.sampler;

        VkDescriptorImageInfo sampler_only_info{};
        sampler_only_info.sampler = s.sampler;

        for (int si = 0; si < 2; si++) {
            VkWriteDescriptorSet writes[9]{};

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = s.gfx_desc_sets[si];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &ubo_buf_info;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = s.gfx_desc_sets[si];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[1].pImageInfo = &heightmap_info;

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = s.gfx_desc_sets[si];
            writes[2].dstBinding = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[2].pImageInfo = &swe_out_info;

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = s.gfx_desc_sets[si];
            writes[3].dstBinding = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[3].pImageInfo = &sediment_info;

            writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = s.gfx_desc_sets[si];
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[4].pImageInfo = &shadow_info;

            writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[5].dstSet = s.gfx_desc_sets[si];
            writes[5].dstBinding = 5;
            writes[5].descriptorCount = 1;
            writes[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[5].pImageInfo = &vol_infos[si];

            writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[6].dstSet = s.gfx_desc_sets[si];
            writes[6].dstBinding = 6;
            writes[6].descriptorCount = 1;
            writes[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[6].pImageInfo = &wind_infos[si];

            writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[7].dstSet = s.gfx_desc_sets[si];
            writes[7].dstBinding = 7;
            writes[7].descriptorCount = 1;
            writes[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[7].pImageInfo = &sand_dep_info;

            writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[8].dstSet = s.gfx_desc_sets[si];
            writes[8].dstBinding = 8;
            writes[8].descriptorCount = 1;
            writes[8].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            writes[8].pImageInfo = &sampler_only_info;

            vkUpdateDescriptorSets(device, 9, writes, 0, nullptr);
        }
    }

    // Terrain brush descriptor
    {
        VkDescriptorImageInfo tb_img_info{};
        tb_img_info.imageView = s.heightmap_gpu.view;
        tb_img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet tb_write{};
        tb_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        tb_write.dstSet = s.tb_desc_set;
        tb_write.dstBinding = 0;
        tb_write.descriptorCount = 1;
        tb_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        tb_write.pImageInfo = &tb_img_info;

        vkUpdateDescriptorSets(device, 1, &tb_write, 0, nullptr);
    }

    // Sand brush descriptor (same layout as terrain brush, points to s.sand_deposit)
    {
        VkDescriptorImageInfo sb_img_info{};
        sb_img_info.imageView = s.sand_deposit.view;
        sb_img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet sb_write{};
        sb_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sb_write.dstSet = s.sand_brush_desc_set;
        sb_write.dstBinding = 0;
        sb_write.descriptorCount = 1;
        sb_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sb_write.pImageInfo = &sb_img_info;

        vkUpdateDescriptorSets(device, 1, &sb_write, 0, nullptr);
    }

    // Erosion descriptors (2 sets for sediment ping-pong, always read s.swe_state_a)
    {
        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = s.heightmap_gpu.view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo swe_info{};
        swe_info.imageView = s.swe_state_a.view;
        swe_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo sed_a_info{};
        sed_a_info.imageView = s.sediment_a.view;
        sed_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo sed_b_info{};
        sed_b_info.imageView = s.sediment_b.view;
        sed_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo gc_info{};
        gc_info.imageView = s.ground_cond.view;
        gc_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo gc_sampler_info{};
        gc_sampler_info.sampler = s.sampler;

        VkDescriptorImageInfo gw_info{};
        gw_info.imageView = s.ground_wind.view;
        gw_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo gw_sampler_info{};
        gw_sampler_info.sampler = s.sampler;

        // Set 0: read s.sediment_a -> write s.sediment_b
        VkWriteDescriptorSet w0[8]{};
        w0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[0].dstSet = s.ero_desc_sets[0];
        w0[0].dstBinding = 0;
        w0[0].descriptorCount = 1;
        w0[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0[0].pImageInfo = &terrain_info;

        w0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[1].dstSet = s.ero_desc_sets[0];
        w0[1].dstBinding = 1;
        w0[1].descriptorCount = 1;
        w0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[1].pImageInfo = &swe_info;

        w0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[2].dstSet = s.ero_desc_sets[0];
        w0[2].dstBinding = 2;
        w0[2].descriptorCount = 1;
        w0[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[2].pImageInfo = &sed_a_info;

        w0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[3].dstSet = s.ero_desc_sets[0];
        w0[3].dstBinding = 3;
        w0[3].descriptorCount = 1;
        w0[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0[3].pImageInfo = &sed_b_info;

        w0[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[4].dstSet = s.ero_desc_sets[0];
        w0[4].dstBinding = 4;
        w0[4].descriptorCount = 1;
        w0[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[4].pImageInfo = &gc_info;

        w0[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[5].dstSet = s.ero_desc_sets[0];
        w0[5].dstBinding = 5;
        w0[5].descriptorCount = 1;
        w0[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w0[5].pImageInfo = &gc_sampler_info;

        w0[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[6].dstSet = s.ero_desc_sets[0];
        w0[6].dstBinding = 6;
        w0[6].descriptorCount = 1;
        w0[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[6].pImageInfo = &gw_info;

        w0[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[7].dstSet = s.ero_desc_sets[0];
        w0[7].dstBinding = 7;
        w0[7].descriptorCount = 1;
        w0[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w0[7].pImageInfo = &gw_sampler_info;

        vkUpdateDescriptorSets(device, 8, w0, 0, nullptr);

        // Set 1: read s.sediment_b -> write s.sediment_a
        VkWriteDescriptorSet w1[8]{};
        w1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[0].dstSet = s.ero_desc_sets[1];
        w1[0].dstBinding = 0;
        w1[0].descriptorCount = 1;
        w1[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1[0].pImageInfo = &terrain_info;

        w1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[1].dstSet = s.ero_desc_sets[1];
        w1[1].dstBinding = 1;
        w1[1].descriptorCount = 1;
        w1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[1].pImageInfo = &swe_info;

        w1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[2].dstSet = s.ero_desc_sets[1];
        w1[2].dstBinding = 2;
        w1[2].descriptorCount = 1;
        w1[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[2].pImageInfo = &sed_b_info;

        w1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[3].dstSet = s.ero_desc_sets[1];
        w1[3].dstBinding = 3;
        w1[3].descriptorCount = 1;
        w1[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1[3].pImageInfo = &sed_a_info;

        w1[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[4].dstSet = s.ero_desc_sets[1];
        w1[4].dstBinding = 4;
        w1[4].descriptorCount = 1;
        w1[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[4].pImageInfo = &gc_info;

        w1[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[5].dstSet = s.ero_desc_sets[1];
        w1[5].dstBinding = 5;
        w1[5].descriptorCount = 1;
        w1[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w1[5].pImageInfo = &gc_sampler_info;

        w1[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[6].dstSet = s.ero_desc_sets[1];
        w1[6].dstBinding = 6;
        w1[6].descriptorCount = 1;
        w1[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[6].pImageInfo = &gw_info;

        w1[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[7].dstSet = s.ero_desc_sets[1];
        w1[7].dstBinding = 7;
        w1[7].descriptorCount = 1;
        w1[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w1[7].pImageInfo = &gw_sampler_info;

        vkUpdateDescriptorSets(device, 8, w1, 0, nullptr);
    }

    // Atmosphere descriptor sets (2 for ping-pong)
    VkDescriptorSetLayout atmo_layouts[2] = {s.pipelines.atmo_desc_layout, s.pipelines.atmo_desc_layout};
    VkDescriptorSetAllocateInfo atmo_ds_ai{};
    atmo_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    atmo_ds_ai.descriptorPool = s.desc_pool;
    atmo_ds_ai.descriptorSetCount = 2;
    atmo_ds_ai.pSetLayouts = atmo_layouts;

    VK_CHECK(vkAllocateDescriptorSets(device, &atmo_ds_ai, s.atmo_desc_sets));

    // Write atmosphere descriptors (3D volumes + 2D shadow + ground outputs + samplers)
    {
        VkDescriptorImageInfo terrain_tex_info{};
        terrain_tex_info.imageView = s.heightmap_gpu.view;
        terrain_tex_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo atmo_a_info{};
        atmo_a_info.imageView = s.atmo_state_a.view;
        atmo_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo atmo_b_info{};
        atmo_b_info.imageView = s.atmo_state_b.view;
        atmo_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo wind_a_info{};
        wind_a_info.imageView = s.wind_field_a.view;
        wind_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo wind_b_info{};
        wind_b_info.imageView = s.wind_field_b.view;
        wind_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo shadow_info{};
        shadow_info.imageView = s.atmo_shadow.view;
        shadow_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo gc_out_info{};
        gc_out_info.imageView = s.ground_cond.view;
        gc_out_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo gw_out_info{};
        gw_out_info.imageView = s.ground_wind.view;
        gw_out_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo swe_out_atmo_info{};
        swe_out_atmo_info.imageView = s.swe_output.view;
        swe_out_atmo_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo terrain_samp_only{};
        terrain_samp_only.sampler = s.terrain_linear_sampler;

        VkDescriptorImageInfo swe_samp_only{};
        swe_samp_only.sampler = s.sampler;

        // Set 0: state_read=a, state_write=b, wind_read=a, wind_write=b
        VkWriteDescriptorSet aw0[11]{};
        aw0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[0].dstSet = s.atmo_desc_sets[0];
        aw0[0].dstBinding = 0;
        aw0[0].descriptorCount = 1;
        aw0[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw0[0].pImageInfo = &terrain_tex_info;

        aw0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[1].dstSet = s.atmo_desc_sets[0];
        aw0[1].dstBinding = 1;
        aw0[1].descriptorCount = 1;
        aw0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw0[1].pImageInfo = &atmo_a_info;

        aw0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[2].dstSet = s.atmo_desc_sets[0];
        aw0[2].dstBinding = 2;
        aw0[2].descriptorCount = 1;
        aw0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw0[2].pImageInfo = &atmo_b_info;

        aw0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[3].dstSet = s.atmo_desc_sets[0];
        aw0[3].dstBinding = 3;
        aw0[3].descriptorCount = 1;
        aw0[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw0[3].pImageInfo = &wind_a_info;

        aw0[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[4].dstSet = s.atmo_desc_sets[0];
        aw0[4].dstBinding = 4;
        aw0[4].descriptorCount = 1;
        aw0[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw0[4].pImageInfo = &wind_b_info;

        aw0[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[5].dstSet = s.atmo_desc_sets[0];
        aw0[5].dstBinding = 5;
        aw0[5].descriptorCount = 1;
        aw0[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw0[5].pImageInfo = &shadow_info;

        aw0[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[6].dstSet = s.atmo_desc_sets[0];
        aw0[6].dstBinding = 6;
        aw0[6].descriptorCount = 1;
        aw0[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw0[6].pImageInfo = &gc_out_info;

        aw0[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[7].dstSet = s.atmo_desc_sets[0];
        aw0[7].dstBinding = 7;
        aw0[7].descriptorCount = 1;
        aw0[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw0[7].pImageInfo = &gw_out_info;

        aw0[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[8].dstSet = s.atmo_desc_sets[0];
        aw0[8].dstBinding = 8;
        aw0[8].descriptorCount = 1;
        aw0[8].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw0[8].pImageInfo = &swe_out_atmo_info;

        aw0[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[9].dstSet = s.atmo_desc_sets[0];
        aw0[9].dstBinding = 9;
        aw0[9].descriptorCount = 1;
        aw0[9].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        aw0[9].pImageInfo = &terrain_samp_only;

        aw0[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw0[10].dstSet = s.atmo_desc_sets[0];
        aw0[10].dstBinding = 10;
        aw0[10].descriptorCount = 1;
        aw0[10].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        aw0[10].pImageInfo = &swe_samp_only;

        vkUpdateDescriptorSets(device, 11, aw0, 0, nullptr);

        // Set 1: state_read=b, state_write=a, wind_read=b, wind_write=a
        VkWriteDescriptorSet aw1[11]{};
        aw1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[0].dstSet = s.atmo_desc_sets[1];
        aw1[0].dstBinding = 0;
        aw1[0].descriptorCount = 1;
        aw1[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw1[0].pImageInfo = &terrain_tex_info;

        aw1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[1].dstSet = s.atmo_desc_sets[1];
        aw1[1].dstBinding = 1;
        aw1[1].descriptorCount = 1;
        aw1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw1[1].pImageInfo = &atmo_b_info;

        aw1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[2].dstSet = s.atmo_desc_sets[1];
        aw1[2].dstBinding = 2;
        aw1[2].descriptorCount = 1;
        aw1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw1[2].pImageInfo = &atmo_a_info;

        aw1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[3].dstSet = s.atmo_desc_sets[1];
        aw1[3].dstBinding = 3;
        aw1[3].descriptorCount = 1;
        aw1[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw1[3].pImageInfo = &wind_b_info;

        aw1[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[4].dstSet = s.atmo_desc_sets[1];
        aw1[4].dstBinding = 4;
        aw1[4].descriptorCount = 1;
        aw1[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw1[4].pImageInfo = &wind_a_info;

        aw1[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[5].dstSet = s.atmo_desc_sets[1];
        aw1[5].dstBinding = 5;
        aw1[5].descriptorCount = 1;
        aw1[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw1[5].pImageInfo = &shadow_info;

        aw1[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[6].dstSet = s.atmo_desc_sets[1];
        aw1[6].dstBinding = 6;
        aw1[6].descriptorCount = 1;
        aw1[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw1[6].pImageInfo = &gc_out_info;

        aw1[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[7].dstSet = s.atmo_desc_sets[1];
        aw1[7].dstBinding = 7;
        aw1[7].descriptorCount = 1;
        aw1[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        aw1[7].pImageInfo = &gw_out_info;

        aw1[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[8].dstSet = s.atmo_desc_sets[1];
        aw1[8].dstBinding = 8;
        aw1[8].descriptorCount = 1;
        aw1[8].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        aw1[8].pImageInfo = &swe_out_atmo_info;

        aw1[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[9].dstSet = s.atmo_desc_sets[1];
        aw1[9].dstBinding = 9;
        aw1[9].descriptorCount = 1;
        aw1[9].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        aw1[9].pImageInfo = &terrain_samp_only;

        aw1[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        aw1[10].dstSet = s.atmo_desc_sets[1];
        aw1[10].dstBinding = 10;
        aw1[10].descriptorCount = 1;
        aw1[10].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        aw1[10].pImageInfo = &swe_samp_only;

        vkUpdateDescriptorSets(device, 11, aw1, 0, nullptr);
    }

    // Sand compute descriptor sets (2 for wind ping-pong)
    VkDescriptorSetLayout sand_sim_layouts[2] = {s.pipelines.sand_sim_desc_layout, s.pipelines.sand_sim_desc_layout};
    VkDescriptorSetAllocateInfo sand_sim_ds_ai{};
    sand_sim_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sand_sim_ds_ai.descriptorPool = s.desc_pool;
    sand_sim_ds_ai.descriptorSetCount = 2;
    sand_sim_ds_ai.pSetLayouts = sand_sim_layouts;

    VK_CHECK(vkAllocateDescriptorSets(device, &sand_sim_ds_ai, s.sand_sim_desc_sets));

    // Sand render descriptor set
    VkDescriptorSetAllocateInfo sand_render_ds_ai{};
    sand_render_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sand_render_ds_ai.descriptorPool = s.desc_pool;
    sand_render_ds_ai.descriptorSetCount = 1;
    sand_render_ds_ai.pSetLayouts = &s.pipelines.sand_render_desc_layout;

    VK_CHECK(vkAllocateDescriptorSets(device, &sand_render_ds_ai, &s.sand_render_desc_set));

    // Write sand compute descriptors
    {
        VkDescriptorImageInfo terrain_tex_info{};
        terrain_tex_info.imageView = s.heightmap_gpu.view;
        terrain_tex_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo wind_a_info{};
        wind_a_info.imageView = s.wind_field_a.view;
        wind_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo wind_b_info{};
        wind_b_info.imageView = s.wind_field_b.view;
        wind_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo sand_buf_info{};
        sand_buf_info.buffer = s.sand_particle_buf;
        sand_buf_info.offset = 0;
        sand_buf_info.range = SAND_MAX_PARTICLES * 32;

        VkDescriptorImageInfo sand_dep_tex_info{};
        sand_dep_tex_info.imageView = s.sand_deposit.view;
        sand_dep_tex_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo terrain_samp_info{};
        terrain_samp_info.sampler = s.terrain_linear_sampler;

        VkDescriptorImageInfo wind_samp_info{};
        wind_samp_info.sampler = s.sampler;

        VkDescriptorImageInfo sand_dep_samp_info{};
        sand_dep_samp_info.sampler = s.sampler;

        // Set 0: terrain + wind_a + particles + s.sand_deposit + 3 samplers
        VkWriteDescriptorSet sw0[7]{};
        sw0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[0].dstSet = s.sand_sim_desc_sets[0];
        sw0[0].dstBinding = 0;
        sw0[0].descriptorCount = 1;
        sw0[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        sw0[0].pImageInfo = &terrain_tex_info;

        sw0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[1].dstSet = s.sand_sim_desc_sets[0];
        sw0[1].dstBinding = 1;
        sw0[1].descriptorCount = 1;
        sw0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        sw0[1].pImageInfo = &wind_a_info;

        sw0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[2].dstSet = s.sand_sim_desc_sets[0];
        sw0[2].dstBinding = 2;
        sw0[2].descriptorCount = 1;
        sw0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sw0[2].pBufferInfo = &sand_buf_info;

        sw0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[3].dstSet = s.sand_sim_desc_sets[0];
        sw0[3].dstBinding = 3;
        sw0[3].descriptorCount = 1;
        sw0[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        sw0[3].pImageInfo = &sand_dep_tex_info;

        sw0[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[4].dstSet = s.sand_sim_desc_sets[0];
        sw0[4].dstBinding = 4;
        sw0[4].descriptorCount = 1;
        sw0[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        sw0[4].pImageInfo = &terrain_samp_info;

        sw0[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[5].dstSet = s.sand_sim_desc_sets[0];
        sw0[5].dstBinding = 5;
        sw0[5].descriptorCount = 1;
        sw0[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        sw0[5].pImageInfo = &wind_samp_info;

        sw0[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw0[6].dstSet = s.sand_sim_desc_sets[0];
        sw0[6].dstBinding = 6;
        sw0[6].descriptorCount = 1;
        sw0[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        sw0[6].pImageInfo = &sand_dep_samp_info;

        vkUpdateDescriptorSets(device, 7, sw0, 0, nullptr);

        // Set 1: terrain + wind_b + particles + s.sand_deposit + 3 samplers
        VkWriteDescriptorSet sw1[7]{};
        sw1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[0].dstSet = s.sand_sim_desc_sets[1];
        sw1[0].dstBinding = 0;
        sw1[0].descriptorCount = 1;
        sw1[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        sw1[0].pImageInfo = &terrain_tex_info;

        sw1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[1].dstSet = s.sand_sim_desc_sets[1];
        sw1[1].dstBinding = 1;
        sw1[1].descriptorCount = 1;
        sw1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        sw1[1].pImageInfo = &wind_b_info;

        sw1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[2].dstSet = s.sand_sim_desc_sets[1];
        sw1[2].dstBinding = 2;
        sw1[2].descriptorCount = 1;
        sw1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sw1[2].pBufferInfo = &sand_buf_info;

        sw1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[3].dstSet = s.sand_sim_desc_sets[1];
        sw1[3].dstBinding = 3;
        sw1[3].descriptorCount = 1;
        sw1[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        sw1[3].pImageInfo = &sand_dep_tex_info;

        sw1[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[4].dstSet = s.sand_sim_desc_sets[1];
        sw1[4].dstBinding = 4;
        sw1[4].descriptorCount = 1;
        sw1[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        sw1[4].pImageInfo = &terrain_samp_info;

        sw1[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[5].dstSet = s.sand_sim_desc_sets[1];
        sw1[5].dstBinding = 5;
        sw1[5].descriptorCount = 1;
        sw1[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        sw1[5].pImageInfo = &wind_samp_info;

        sw1[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw1[6].dstSet = s.sand_sim_desc_sets[1];
        sw1[6].dstBinding = 6;
        sw1[6].descriptorCount = 1;
        sw1[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        sw1[6].pImageInfo = &sand_dep_samp_info;

        vkUpdateDescriptorSets(device, 7, sw1, 0, nullptr);
    }

    // Write sand render descriptors
    {
        VkDescriptorBufferInfo ubo_buf_info{};
        ubo_buf_info.buffer = s.camera_ubo;
        ubo_buf_info.offset = 0;
        ubo_buf_info.range = sizeof(CameraData);

        VkDescriptorBufferInfo sand_buf_info{};
        sand_buf_info.buffer = s.sand_particle_buf;
        sand_buf_info.offset = 0;
        sand_buf_info.range = SAND_MAX_PARTICLES * 32;

        VkWriteDescriptorSet srw[2]{};
        srw[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        srw[0].dstSet = s.sand_render_desc_set;
        srw[0].dstBinding = 0;
        srw[0].descriptorCount = 1;
        srw[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        srw[0].pBufferInfo = &ubo_buf_info;

        srw[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        srw[1].dstSet = s.sand_render_desc_set;
        srw[1].dstBinding = 1;
        srw[1].descriptorCount = 1;
        srw[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        srw[1].pBufferInfo = &sand_buf_info;

        vkUpdateDescriptorSets(device, 2, srw, 0, nullptr);
    }

    // Terrain gen descriptor set
    VkDescriptorSetAllocateInfo tgen_ds_ai{};
    tgen_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    tgen_ds_ai.descriptorPool = s.desc_pool;
    tgen_ds_ai.descriptorSetCount = 1;
    tgen_ds_ai.pSetLayouts = &s.pipelines.terrain_gen_desc_layout;

    VK_CHECK(vkAllocateDescriptorSets(device, &tgen_ds_ai, &s.terrain_gen_desc_set));

    // Write terrain gen descriptor (binding 0 = clipmap heightmap array as storage image)
    {
        VkDescriptorImageInfo tgen_img_info{};
        tgen_img_info.imageView = s.clipmap_hm_view;
        tgen_img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo stamp_desc_info{};
        stamp_desc_info.buffer = s.stamp_buf;
        stamp_desc_info.offset = 0;
        stamp_desc_info.range = MAX_STAMPS * sizeof(TerrainStamp);

        VkWriteDescriptorSet tgen_writes[2]{};
        tgen_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        tgen_writes[0].dstSet = s.terrain_gen_desc_set;
        tgen_writes[0].dstBinding = 0;
        tgen_writes[0].descriptorCount = 1;
        tgen_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        tgen_writes[0].pImageInfo = &tgen_img_info;

        tgen_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        tgen_writes[1].dstSet = s.terrain_gen_desc_set;
        tgen_writes[1].dstBinding = 1;
        tgen_writes[1].descriptorCount = 1;
        tgen_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        tgen_writes[1].pBufferInfo = &stamp_desc_info;

        vkUpdateDescriptorSets(device, 2, tgen_writes, 0, nullptr);
    }

    // Clipmap graphics descriptor sets (2 for atmosphere ping-pong)
    VkDescriptorSetLayout clip_gfx_layouts[2] = {s.pipelines.gfx_desc_layout, s.pipelines.gfx_desc_layout};
    VkDescriptorSetAllocateInfo clip_gfx_ds_ai{};
    clip_gfx_ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    clip_gfx_ds_ai.descriptorPool = s.desc_pool;
    clip_gfx_ds_ai.descriptorSetCount = 2;
    clip_gfx_ds_ai.pSetLayouts = clip_gfx_layouts;

    VK_CHECK(vkAllocateDescriptorSets(device, &clip_gfx_ds_ai, s.clipmap_gfx_desc_sets));

    // Write clipmap gfx descriptors (same as s.gfx_desc_sets but binding 1 = clipmap heightmap array view)
    {
        VkDescriptorBufferInfo ubo_buf_info{};
        ubo_buf_info.buffer = s.camera_ubo;
        ubo_buf_info.offset = 0;
        ubo_buf_info.range = sizeof(CameraData);

        VkDescriptorImageInfo clip_heightmap_info{};
        clip_heightmap_info.imageView = s.clipmap_hm_view;
        clip_heightmap_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        clip_heightmap_info.sampler = s.sampler;

        VkDescriptorImageInfo swe_out_info{};
        swe_out_info.imageView = s.water_output_view;
        swe_out_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        swe_out_info.sampler = s.sampler;

        VkDescriptorImageInfo sediment_info{};
        sediment_info.imageView = s.sediment_a.view;
        sediment_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sediment_info.sampler = s.sampler;

        VkDescriptorImageInfo shadow_info{};
        shadow_info.imageView = s.atmo_shadow.view;
        shadow_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        shadow_info.sampler = s.sampler;

        VkDescriptorImageInfo vol_a_info{};
        vol_a_info.imageView = s.atmo_state_a.view;
        vol_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_a_info.sampler = s.sampler;

        VkDescriptorImageInfo vol_b_info{};
        vol_b_info.imageView = s.atmo_state_b.view;
        vol_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vol_b_info.sampler = s.sampler;

        VkDescriptorImageInfo vol_infos[2] = {vol_a_info, vol_b_info};

        VkDescriptorImageInfo wind_a_gfx{};
        wind_a_gfx.imageView = s.wind_field_a.view;
        wind_a_gfx.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wind_a_gfx.sampler = s.sampler;

        VkDescriptorImageInfo wind_b_gfx{};
        wind_b_gfx.imageView = s.wind_field_b.view;
        wind_b_gfx.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        wind_b_gfx.sampler = s.sampler;

        VkDescriptorImageInfo wind_infos[2] = {wind_a_gfx, wind_b_gfx};

        VkDescriptorImageInfo sand_dep_info{};
        sand_dep_info.imageView = s.sand_deposit.view;
        sand_dep_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sand_dep_info.sampler = s.sampler;

        VkDescriptorImageInfo sampler_only_info{};
        sampler_only_info.sampler = s.sampler;

        for (int si = 0; si < 2; si++) {
            VkWriteDescriptorSet writes[9]{};

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].pBufferInfo = &ubo_buf_info;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[1].pImageInfo = &clip_heightmap_info;

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[2].dstBinding = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[2].pImageInfo = &swe_out_info;

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[3].dstBinding = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[3].pImageInfo = &sediment_info;

            writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[4].pImageInfo = &shadow_info;

            writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[5].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[5].dstBinding = 5;
            writes[5].descriptorCount = 1;
            writes[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[5].pImageInfo = &vol_infos[si];

            writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[6].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[6].dstBinding = 6;
            writes[6].descriptorCount = 1;
            writes[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[6].pImageInfo = &wind_infos[si];

            writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[7].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[7].dstBinding = 7;
            writes[7].descriptorCount = 1;
            writes[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writes[7].pImageInfo = &sand_dep_info;

            writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[8].dstSet = s.clipmap_gfx_desc_sets[si];
            writes[8].dstBinding = 8;
            writes[8].descriptorCount = 1;
            writes[8].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            writes[8].pImageInfo = &sampler_only_info;

            vkUpdateDescriptorSets(device, 9, writes, 0, nullptr);
        }
    }

    // ---- Planet SWE descriptor sets -------------------------------------------
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = s.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &s.pipelines.planet_swe_init_desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, &s.planet_swe_init_desc_set));

        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = s.clipmap_hm_view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_a_info{};
        state_a_info.imageView = s.water_state_a_view;
        state_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_b_info{};
        state_b_info.imageView = s.water_state_b_view;
        state_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo output_info{};
        output_info.imageView = s.water_output_view;
        output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo water_stamp_bi{};
        water_stamp_bi.buffer = s.water_stamp_buf;
        water_stamp_bi.offset = 0;
        water_stamp_bi.range = MAX_WATER_STAMPS * sizeof(WaterStamp);

        VkWriteDescriptorSet writes[5]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = s.planet_swe_init_desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].pImageInfo = &terrain_info;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = s.planet_swe_init_desc_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &state_a_info;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = s.planet_swe_init_desc_set;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &state_b_info;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = s.planet_swe_init_desc_set;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[3].pImageInfo = &output_info;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = s.planet_swe_init_desc_set;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &water_stamp_bi;

        vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
    }

    // Planet SWE h-adjust descriptor: state_a + state_b (storage images, RW).
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = s.desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &s.pipelines.planet_swe_h_adjust_desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, &s.planet_swe_h_adjust_desc_set));

        VkDescriptorImageInfo state_a_info{};
        state_a_info.imageView = s.water_state_a_view;
        state_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo state_b_info{};
        state_b_info.imageView = s.water_state_b_view;
        state_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = s.planet_swe_h_adjust_desc_set;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &state_a_info;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = s.planet_swe_h_adjust_desc_set;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &state_b_info;
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }

    {
        VkDescriptorSetLayout layouts[2] = {s.pipelines.planet_swe_step_desc_layout, s.pipelines.planet_swe_step_desc_layout};
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = s.desc_pool;
        ai.descriptorSetCount = 2;
        ai.pSetLayouts = layouts;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, s.planet_swe_step_desc_sets));

        VkDescriptorImageInfo terrain_info{};
        terrain_info.imageView = s.clipmap_hm_view;
        terrain_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_a_info{};
        state_a_info.imageView = s.water_state_a_view;
        state_a_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo state_b_info{};
        state_b_info.imageView = s.water_state_b_view;
        state_b_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo output_info{};
        output_info.imageView = s.water_output_view;
        output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo edge_flags_bi{};
        edge_flags_bi.buffer = s.edge_flags_buf;
        edge_flags_bi.offset = 0;
        edge_flags_bi.range = VK_WHOLE_SIZE;

        // Set 0: read A -> write B
        VkWriteDescriptorSet w0[5]{};
        w0[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[0].dstSet = s.planet_swe_step_desc_sets[0];
        w0[0].dstBinding = 0;
        w0[0].descriptorCount = 1;
        w0[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[0].pImageInfo = &terrain_info;
        w0[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[1].dstSet = s.planet_swe_step_desc_sets[0];
        w0[1].dstBinding = 1;
        w0[1].descriptorCount = 1;
        w0[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w0[1].pImageInfo = &state_a_info;
        w0[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[2].dstSet = s.planet_swe_step_desc_sets[0];
        w0[2].dstBinding = 2;
        w0[2].descriptorCount = 1;
        w0[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0[2].pImageInfo = &state_b_info;
        w0[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[3].dstSet = s.planet_swe_step_desc_sets[0];
        w0[3].dstBinding = 3;
        w0[3].descriptorCount = 1;
        w0[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0[3].pImageInfo = &output_info;
        w0[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w0[4].dstSet = s.planet_swe_step_desc_sets[0];
        w0[4].dstBinding = 4;
        w0[4].descriptorCount = 1;
        w0[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w0[4].pBufferInfo = &edge_flags_bi;
        vkUpdateDescriptorSets(device, 5, w0, 0, nullptr);

        // Set 1: read B -> write A
        VkWriteDescriptorSet w1[5]{};
        w1[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[0].dstSet = s.planet_swe_step_desc_sets[1];
        w1[0].dstBinding = 0;
        w1[0].descriptorCount = 1;
        w1[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[0].pImageInfo = &terrain_info;
        w1[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[1].dstSet = s.planet_swe_step_desc_sets[1];
        w1[1].dstBinding = 1;
        w1[1].descriptorCount = 1;
        w1[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w1[1].pImageInfo = &state_b_info;
        w1[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[2].dstSet = s.planet_swe_step_desc_sets[1];
        w1[2].dstBinding = 2;
        w1[2].descriptorCount = 1;
        w1[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1[2].pImageInfo = &state_a_info;
        w1[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[3].dstSet = s.planet_swe_step_desc_sets[1];
        w1[3].dstBinding = 3;
        w1[3].descriptorCount = 1;
        w1[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1[3].pImageInfo = &output_info;
        w1[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w1[4].dstSet = s.planet_swe_step_desc_sets[1];
        w1[4].dstBinding = 4;
        w1[4].descriptorCount = 1;
        w1[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w1[4].pBufferInfo = &edge_flags_bi;
        vkUpdateDescriptorSets(device, 5, w1, 0, nullptr);
    }

    s.swe_ping_pong = 0;

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

        VkImageMemoryBarrier2 init_barriers[17]{};
        for (int i = 0; i < 14; ++i) {
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
        init_barriers[0].image = s.swe_state_a.image;
        init_barriers[1].image = s.swe_state_b.image;
        init_barriers[2].image = s.swe_output.image;

        for (int i = 3; i < 5; ++i) {
            init_barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            init_barriers[i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        }
        init_barriers[3].image = s.sediment_a.image;
        init_barriers[4].image = s.sediment_b.image;

        // Atmosphere images (UNDEFINED -> GENERAL) — 4 3D volumes + 1 2D shadow + 2 ground output
        for (int i = 5; i < 12; ++i) {
            init_barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            init_barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        }
        init_barriers[5].image = s.atmo_state_a.image;
        init_barriers[6].image = s.atmo_state_b.image;
        init_barriers[7].image = s.wind_field_a.image;
        init_barriers[8].image = s.wind_field_b.image;
        init_barriers[9].image = s.atmo_shadow.image;
        init_barriers[10].image = s.ground_cond.image;
        init_barriers[11].image = s.ground_wind.image;

        init_barriers[12] = init_barriers[3]; // copy template from sediment (TRANSFER_DST)
        init_barriers[12].image = s.sand_deposit.image;

        // Tile pool (UNDEFINED -> GENERAL)
        init_barriers[13].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        init_barriers[13].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        init_barriers[13].image = s.clipmap_hm_image;
        init_barriers[13].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};

        // Water tile pools (UNDEFINED -> GENERAL)
        for (int i = 14; i < 17; ++i) {
            init_barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            init_barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            init_barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};
        }
        init_barriers[14].image = s.water_state_a_img;
        init_barriers[15].image = s.water_state_b_img;
        init_barriers[16].image = s.water_output_img;

        VkDependencyInfo dep_init{};
        dep_init.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_init.imageMemoryBarrierCount = 17;
        dep_init.pImageMemoryBarriers = init_barriers;
        vkCmdPipelineBarrier2(cmd, &dep_init);

        VkClearColorValue clear_zero{};
        clear_zero.float32[0] = 0.0f;
        VkImageSubresourceRange clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, s.sediment_a.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
        vkCmdClearColorImage(cmd, s.sediment_b.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
        vkCmdClearColorImage(cmd, s.sand_deposit.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);

        VkImageSubresourceRange water_clear_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PLANET_TILE_POOL};
        vkCmdClearColorImage(cmd, s.water_state_a_img, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &water_clear_range);
        vkCmdClearColorImage(cmd, s.water_state_b_img, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &water_clear_range);
        vkCmdClearColorImage(cmd, s.water_output_img, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &water_clear_range);
        vkCmdFillBuffer(cmd, s.sand_particle_buf, 0, SAND_MAX_PARTICLES * 32, 0);

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
        sed_after_clear[0].image = s.sediment_a.image;
        sed_after_clear[1].image = s.sediment_b.image;

        VkDependencyInfo dep_sed_clear{};
        dep_sed_clear.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_sed_clear.imageMemoryBarrierCount = 2;
        dep_sed_clear.pImageMemoryBarriers = sed_after_clear;
        vkCmdPipelineBarrier2(cmd, &dep_sed_clear);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.swe_init_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s.pipelines.swe_init_pipeline_layout, 0, 1, &s.swe_init_desc_set, 0, nullptr);

        SweInitPC init_pc{};
        init_pc.grid_w = SWE_GRID_W;
        init_pc.grid_h = SWE_GRID_H;
        init_pc.initial_water_level = s.basin_params.floor_height + s.basin_params.initial_water;
        init_pc._pad = 0.0f;
        vkCmdPushConstants(cmd, s.pipelines.swe_init_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(init_pc), &init_pc);

        vkCmdDispatch(cmd, (SWE_GRID_W + 7) / 8, (SWE_GRID_H + 7) / 8, 1);

        VkMemoryBarrier2 mem_barrier{};
        mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mem_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mem_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mem_barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mem_barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo dep_afterenderer_init{};
        dep_afterenderer_init.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_afterenderer_init.memoryBarrierCount = 1;
        dep_afterenderer_init.pMemoryBarriers = &mem_barrier;
        vkCmdPipelineBarrier2(cmd, &dep_afterenderer_init);

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

    // ---- Frame loop state init ------------------------------------------------
    // remains visible OR is disturbed. Slots are recycled (LIFO) when a tile leaves
    // the visible set AND is not disturbed. Disturbed tiles stay anchored so the
    // brushed water keeps simulating across camera moves and LOD transitions.
    s.free_slots.reserve(PLANET_TILE_POOL);
    // Tiles that need their SWE state initialized on the next init pass — a single
    // queue fed by the visible-tile allocator and by the cross-tile auto-anchor path.
    // Number of terrain stamps already applied to disturbed tiles' h column.
    // When new stamps are added, the delta is dispatched as h-adjust per disturbed tile.



    // Initialize frame state
    s.slot_to_tile.assign(PLANET_TILE_POOL, QuadNode{0xFFFFFFFFu, 0u, 0u, 0u});
    s.free_slots.reserve(PLANET_TILE_POOL);
    for (uint32_t sl = PLANET_TILE_POOL; sl-- > 0; ) s.free_slots.push_back(sl);
    s.last_time = glfwGetTime();
    s.ns_per_tick = static_cast<double>(r.vkb_phys.properties.limits.timestampPeriod);
    s.terrain_size = static_cast<float>(s.basin_params.grid_w) * s.basin_params.cell_spacing;
    s.initialized = true;
}

// ---------------------------------------------------------------------------
bool globe_tick(GlobeState& s, Renderer& r, float dt)
{
    VkDevice device = r.device;
    VmaAllocator allocator = r.allocator;
    VkQueue graphics_queue = r.graphics_queue;
    uint32_t gfx_family = r.gfx_family;
    GLFWwindow* window = r.window;
    float terrain_size = s.terrain_size;

    s.last_dt = dt;
    s.ui.embedded = s.embedded;
    s.ui.first_person_mode = (s.camera.mode == CameraMode::FirstPerson);

        if (s.input.reload_shaders) {
            pipelines_reload(s.pipelines, device);
            s.input.reload_shaders = false;
        }

        int fb_w, fb_h;



        if (s.queries_valid) {
            uint64_t timestamps[2] = {};
            VkResult qr = vkGetQueryPoolResults(device, r.query_pool,
                r.current_frame * 2, 2,
                sizeof(timestamps), timestamps, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT);
            if (qr == VK_SUCCESS) {
                double swe_ms = static_cast<double>(timestamps[1] - timestamps[0]) * s.ns_per_tick / 1e6;
                s.gpu_times[s.timing_index] = swe_ms;
            }

            // Atmosphere pressure debug readback
            if (s.atmo_debug_info.pMappedData && s.ui.atmosphere_enabled) {
                const uint16_t* px = static_cast<const uint16_t*>(s.atmo_debug_info.pMappedData);
                float p_min = 1e30f, p_max = -1e30f, p_sum = 0.0f;
                float ws_max = 0.0f;
                uint32_t count = 0;
                for (uint32_t i = 0; i < ATMO_W * ATMO_H; ++i) {
                    uint16_t hx = px[i * 4 + 0];
                    uint16_t hy = px[i * 4 + 1];
                    uint16_t hp = px[i * 4 + 3];
                    // IEEE 754 half-to-float conversion
                    auto h2f = [](uint16_t h) -> float {
                        uint32_t sign = (h >> 15) & 1;
                        uint32_t exp  = (h >> 10) & 0x1F;
                        uint32_t mant = h & 0x3FF;
                        if (exp == 0) return sign ? -0.0f : 0.0f;
                        if (exp == 31) return sign ? -1e30f : 1e30f;
                        float f = std::ldexp(static_cast<float>(1024 + mant), static_cast<int>(exp) - 25);
                        return sign ? -f : f;
                    };
                    float pressure = h2f(hp);
                    float wx = h2f(hx);
                    float wy = h2f(hy);
                    if (pressure > 10.0f) {
                        p_min = std::min(p_min, pressure);
                        p_max = std::max(p_max, pressure);
                        p_sum += pressure;
                        ++count;
                    }
                    float ws = std::sqrt(wx * wx + wy * wy);
                    ws_max = std::max(ws_max, ws);
                }
                if (count > 0) {
                    s.ui.pressure_min = p_min;
                    s.ui.pressure_max = p_max;
                    s.ui.pressure_mean = p_sum / count;
                    s.ui.wind_speed_max = ws_max;
                }
            }
        }

        // ---- ImGui frame ----
        s.ui.cpu_avg_ms = s.cpu_avg_ms;
        s.ui.gpu_avg_ms = s.gpu_avg_ms;
        s.ui.altitude_above_terrain = s.altitude_above_terrain;
        s.ui.terrain_height_at_cam = s.terrain_height_at_cam;
        s.ui.stamp_count = static_cast<uint32_t>(s.stamps.size());
        s.ui.max_stamps = MAX_STAMPS;
        ui_draw(s.ui);

        if (s.ui.undo_stamp && !s.stamps.empty()) {
            s.stamps.pop_back();
            s.stamps_dirty = true;
            s.ui.undo_stamp = false;
        }
        if (s.ui.clear_stamps && !s.stamps.empty()) {
            s.stamps.clear();
            s.stamps_dirty = true;
            s.ui.clear_stamps = false;
        }

        // ---- Reset handlers (heavy, stall GPU) ----
        if (s.ui.request_basin_reset || s.ui.request_water_reset) {
            vkDeviceWaitIdle(device);

            if (s.ui.request_basin_reset) {
                auto new_basin = generate_crater_basin(s.basin_params);
                VkDeviceSize buf_size = static_cast<VkDeviceSize>(s.basin_params.grid_w) * s.basin_params.grid_h * sizeof(float);

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
                to_dst.image = s.heightmap_gpu.image;
                to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo dep_dst{};
                dep_dst.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep_dst.imageMemoryBarrierCount = 1;
                dep_dst.pImageMemoryBarriers = &to_dst;
                vkCmdPipelineBarrier2(tmp_cmd, &dep_dst);

                VkBufferImageCopy2 copy_region{};
                copy_region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
                copy_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy_region.imageExtent = {s.basin_params.grid_w, s.basin_params.grid_h, 1};
                VkCopyBufferToImageInfo2 copy_info{};
                copy_info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
                copy_info.srcBuffer = stg_buf;
                copy_info.dstImage = s.heightmap_gpu.image;
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
                to_gen.image = s.heightmap_gpu.image;
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
                s.ui.request_basin_reset = false;
                s.ui.request_water_reset = true;
            }

            if (s.ui.request_water_reset) {
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

                vkCmdBindPipeline(tmp_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.swe_init_pipeline);
                vkCmdBindDescriptorSets(tmp_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        s.pipelines.swe_init_pipeline_layout, 0, 1, &s.swe_init_desc_set, 0, nullptr);
                SweInitPC init_pc{};
                init_pc.grid_w = SWE_GRID_W;
                init_pc.grid_h = SWE_GRID_H;
                init_pc.initial_water_level = s.basin_params.floor_height + s.basin_params.initial_water;
                vkCmdPushConstants(tmp_cmd, s.pipelines.swe_init_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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
                s.swe_ping_pong = 0;
                s.ui.request_water_reset = false;
                s.ui.request_sediment_reset = true;
            }
        }

        if (s.ui.request_sediment_reset) {
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
            vkCmdClearColorImage(tmp_cmd, s.sediment_a.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);
            vkCmdClearColorImage(tmp_cmd, s.sediment_b.image, VK_IMAGE_LAYOUT_GENERAL, &clear_zero, 1, &clear_range);

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
            s.sediment_ping_pong = 0;
            s.ui.request_sediment_reset = false;
        }

        // ---- Per-frame camera movement (double precision) ----
        {
            auto height_fn = [&s](glm::vec3 dir) {
                float h = cpu_terrain_height_with_stamps(dir, s.stamps.data(),
                    static_cast<uint32_t>(s.stamps.size()));
                if (s.ui.ocean_enabled) h = std::max(h, s.ui.sea_level);
                return h;
            };

            if (s.input.toggle_camera_mode) {
                if (s.camera.mode == CameraMode::Orbital)
                    camera_switch_to_first_person(s.camera, PLANET_RADIUS, height_fn);
                else
                    camera_switch_to_orbital(s.camera);
                s.input.rmb_held = false;
                s.input.toggle_camera_mode = false;
            }
            s.ui.first_person_mode = (s.camera.mode == CameraMode::FirstPerson);

            glm::dvec3 prev_eye = camera_eye_position(s.camera);
            auto cam_result = camera_update(s.camera, window, s.last_dt, PLANET_RADIUS, height_fn);
            s.terrain_height_at_cam = cam_result.terrain_height_at_cam;
            s.altitude_above_terrain = cam_result.altitude_above_terrain;

            // Per-frame movement speed for the HUD (smoothed).
            glm::dvec3 cur_eye = camera_eye_position(s.camera);
            double frame_speed = (s.last_dt > 1e-6) ? glm::length(cur_eye - prev_eye) / s.last_dt : 0.0;
            s.ui.move_speed_mps = static_cast<float>(0.85 * s.ui.move_speed_mps + 0.15 * frame_speed);
            s.ui.walk_speed_setting = s.camera.fp.walk_speed;
        }

        // ---- Cursor visibility (UE pattern) ----
        // OS cursor hidden over the world (3D brush ring/dot is the cursor),
        // captured during RMB-look, normal over ImGui. Switch only on change
        // so we're not pinging GLFW every frame.
        {
            bool over_ui = ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse;
            int desired = over_ui     ? GLFW_CURSOR_NORMAL
                        : s.input.rmb_held ? GLFW_CURSOR_DISABLED
                                         : GLFW_CURSOR_HIDDEN;
            if (desired != s.current_cursor_mode) {
                glfwSetInputMode(window, GLFW_CURSOR, desired);
                if (desired == GLFW_CURSOR_DISABLED) s.input.first_mouse = true;
                s.current_cursor_mode = desired;
            }
        }

        // ---- Camera UBO update (camera-relative, reversed-Z infinite far) ----
        int _fb_w, _fb_h;
        glfwGetFramebufferSize(window, &_fb_w, &_fb_h);
        float aspect = (_fb_h > 0) ? static_cast<float>(_fb_w) / _fb_h : 1.0f;
        glm::mat4 cam_view = camera_build_view(s.camera);
        glm::mat4 cam_proj = camera_build_proj(s.camera, aspect);

        // ---- Ray-pick cursor against heightfield ----
        s.grid_x = 0.0f; s.grid_y = 0.0f;
        s.stamp_sphere_dir = glm::vec3(0.0f);
        glm::dvec3 stamp_world_pos(0.0);
        s.cursor_on_world = false;
        {
            int win_w, win_h;
            glfwGetWindowSize(window, &win_w, &win_h);
            float ndc_x = static_cast<float>(s.input.cursor_x) / win_w * 2.0f - 1.0f;
            float ndc_y = static_cast<float>(s.input.cursor_y) / win_h * 2.0f - 1.0f;

            glm::mat4 inv_proj = glm::inverse(cam_proj);
            glm::vec4 clip_pt = inv_proj * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
            clip_pt /= clip_pt.w;
            glm::vec3 ray_local = glm::normalize(glm::vec3(clip_pt));
            glm::dvec3 ray_origin = camera_eye_position(s.camera);
            glm::dvec3 ray_dir = glm::normalize(glm::dvec3(camera_orientation(s.camera) * ray_local));

            auto sample = [&](const glm::dvec3& p) -> double {
                double r = glm::length(p);
                glm::vec3 dir = glm::vec3(p / r);
                float h = cpu_terrain_height_with_stamps(dir, s.stamps.data(),
                    static_cast<uint32_t>(s.stamps.size()));
                if (s.ui.ocean_enabled) h = std::max(h, s.ui.sea_level);
                return r - (static_cast<double>(PLANET_RADIUS) + static_cast<double>(h));
            };

            // Coarse sphere intersect at the planet base radius gives us the
            // smooth-planet hit. We start the heightfield march ~10 km before
            // that to capture mountains, and use the sphere hit as fallback
            // when the heightfield doesn't converge (avoids the cursor
            // jumping to the horizon-lock point on near-tangent rays).
            const double R_base = static_cast<double>(PLANET_RADIUS);
            double sphere_t = -1.0;
            {
                double b = 2.0 * glm::dot(ray_origin, ray_dir);
                double c = glm::dot(ray_origin, ray_origin) - R_base * R_base;
                double disc = b * b - 4.0 * c;
                if (disc >= 0.0) {
                    double t_hit = 0.5 * (-b - std::sqrt(disc));
                    if (t_hit > 0.0) sphere_t = t_hit;
                }
            }

            const double max_dist = 5.0e7;  // 50 000 km
            double t_seed = (sphere_t > 0.0) ? std::max(0.0, sphere_t - 10000.0) : 0.0;
            double t = t_seed;
            double prev_t = t_seed;
            double prev_d = sample(ray_origin + t * ray_dir);

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
                    s.stamp_sphere_dir = glm::normalize(glm::vec3(hit));
                    stamp_world_pos = hit;
                    s.cursor_on_world = true;
                }
            }

            // Sphere-hit fallback: heightfield march didn't find a crossing
            // but the smooth planet would have been hit. Land on the smooth
            // sphere so the cursor stays put on the planet rather than
            // jumping out to the horizon-lock point.
            if (!s.cursor_on_world && sphere_t > 0.0) {
                glm::dvec3 hit = ray_origin + sphere_t * ray_dir;
                glm::vec3 dir = glm::normalize(glm::vec3(hit));
                float h = cpu_terrain_height_with_stamps(dir, s.stamps.data(),
                    static_cast<uint32_t>(s.stamps.size()));
                if (s.ui.ocean_enabled) h = std::max(h, s.ui.sea_level);
                double surf_r = R_base + static_cast<double>(h);
                s.stamp_sphere_dir = dir;
                stamp_world_pos = glm::dvec3(dir) * surf_r;
                s.cursor_on_world = true;
            }

            // Horizon-lock (UE port): ray genuinely missed the planet (sky).
            // Take the ray's closest approach to the planet center, but cap
            // at ~200 km along the ray so we don't land on the far side of
            // the planet when looking near the horizon.
            if (!s.cursor_on_world) {
                double t_close = -glm::dot(ray_origin, ray_dir);
                if (t_close > 0.0) {
                    double t_use = std::min(t_close, 200000.0);
                    glm::dvec3 close_pt = ray_origin + t_use * ray_dir;
                    glm::vec3 dir = glm::normalize(glm::vec3(close_pt));
                    float h = cpu_terrain_height_with_stamps(dir, s.stamps.data(),
                        static_cast<uint32_t>(s.stamps.size()));
                    if (s.ui.ocean_enabled) h = std::max(h, s.ui.sea_level);
                    double surf_r = R_base + static_cast<double>(h);
                    s.stamp_sphere_dir = dir;
                    stamp_world_pos = glm::dvec3(dir) * surf_r;
                    s.cursor_on_world = true;
                }
            }
        }
        // ---- Fly-to-cursor (C key, FP only) ----
        if (s.input.warp_to_cursor) {
            if (s.camera.mode == CameraMode::FirstPerson && s.cursor_on_world) {
                float h = cpu_terrain_height_with_stamps(s.stamp_sphere_dir, s.stamps.data(),
                    static_cast<uint32_t>(s.stamps.size()));
                if (s.ui.ocean_enabled) h = std::max(h, s.ui.sea_level);
                double target_radius = static_cast<double>(PLANET_RADIUS) + static_cast<double>(h)
                                     + static_cast<double>(s.camera.fp.eye_height_offset);
                glm::dvec3 target_eye = glm::dvec3(s.stamp_sphere_dir) * target_radius;
                camera_begin_warp_to(s.camera, target_eye);
            }
            s.input.warp_to_cursor = false;
        }

        // FP is move + warp only. Terrain/water painting happens in orbital
        // (god's-eye) mode where the brush ring + LMB are the editing tools.
        bool can_paint = (s.camera.mode == CameraMode::Orbital);
        s.brush_active = can_paint && s.input.lmb_held && !s.input.rmb_held;
        s.brush_hit = s.brush_active && s.cursor_on_world;

        // Brush angular radius scales with view distance so the ring stays a
        // sensible on-screen size at any altitude (UE pattern). Drives both the
        // visualization and the stamp angular radius — WYSIWYG.
        s.effective_angular = 0.0f;
        if (s.cursor_on_world) {
            glm::dvec3 eye_w = camera_eye_position(s.camera);
            double pick_dist = glm::length(stamp_world_pos - eye_w);
            double scale = std::clamp(pick_dist / 5000.0, 0.0005, 4.0);
            float base = s.ui.brush_radius_grid * s.ui.stamp_angular_scale;
            s.effective_angular = std::clamp(static_cast<float>(base * scale), 1e-6f, 0.2f);
        }

        // ---- Place terrain stamp on LMB ----
        if (s.brush_hit &&
            (s.input.brush_mode == BrushMode::Raise || s.input.brush_mode == BrushMode::Lower) &&
            s.stamps.size() < MAX_STAMPS)
        {
            double now_s = glfwGetTime();
            if (now_s - s.last_stamp_time > 0.1) {
                float sign = (s.input.brush_mode == BrushMode::Raise) ? 1.0f : -1.0f;
                float angular_radius = s.effective_angular;

                TerrainStamp stamp{};
                stamp.pos_x = s.stamp_sphere_dir.x;
                stamp.pos_y = s.stamp_sphere_dir.y;
                stamp.pos_z = s.stamp_sphere_dir.z;
                stamp.radius = angular_radius;
                stamp.delta_h = sign * s.ui.terrain_strength;
                stamp.cos_radius = std::cos(angular_radius);
                s.stamps.push_back(stamp);
                s.stamps_dirty = true;
                s.last_stamp_time = now_s;
            }
        }

        // ---- Place water stamp on LMB (parallel to terrain stamps) ----
        // Persistent record of brushed water; applied to every LOD's SWE init so
        // a brushed lake stays visible from any zoom. The dynamic SWE step still
        // adds waves on top at the brushed level — this is the static skeleton.
        if (s.brush_hit && s.input.brush_mode == BrushMode::Water &&
            s.water_stamps.size() < MAX_WATER_STAMPS)
        {
            double now_s = glfwGetTime();
            if (now_s - s.last_water_stamp_time > 0.1) {
                float angular_radius = s.effective_angular;

                WaterStamp ws{};
                ws.pos_x = s.stamp_sphere_dir.x;
                ws.pos_y = s.stamp_sphere_dir.y;
                ws.pos_z = s.stamp_sphere_dir.z;
                ws.radius = angular_radius;
                // The brush deposits a *column*: per stamp adds brush_strength
                // metres at the center, falling off via the gaussian. This makes
                // brushing build up a lake over a couple of seconds.
                ws.water_amount = s.ui.brush_strength;
                ws.cos_radius = std::cos(angular_radius);
                s.water_stamps.push_back(ws);
                s.water_stamps_dirty = true;
                s.last_water_stamp_time = now_s;
            }
        }

        // ---- Camera UBO update (perspective + brush cursor) ----
        {
            // brush_color.a = 0 → ring (orbital), 1 → filled dot (FP).
            float cursor_alpha = (s.camera.mode == CameraMode::FirstPerson) ? 1.0f : 0.0f;
            glm::vec4 brush_color;
            if (s.input.brush_mode == BrushMode::Raise)
                brush_color = glm::vec4(0.30f, 0.95f, 0.40f, cursor_alpha);
            else if (s.input.brush_mode == BrushMode::Lower)
                brush_color = glm::vec4(0.95f, 0.45f, 0.20f, cursor_alpha);
            else if (s.input.brush_mode == BrushMode::Sand)
                brush_color = glm::vec4(0.85f, 0.70f, 0.45f, cursor_alpha);
            else
                brush_color = glm::vec4(0.30f, 0.85f, 0.95f, cursor_alpha);

            CameraData cam{};
            cam.view = cam_view;
            cam.proj = cam_proj;
            cam.sun_dir = glm::normalize(glm::vec3(0.4f, 0.7f, -0.3f));
            cam._pad0 = 0.0f;
            cam.sun_color = glm::vec3(1.0f, 0.95f, 0.85f);
            cam._pad1 = 0.0f;
            cam.cam_pos = glm::vec3(0.0f);  // camera-relative rendering
            cam._pad2 = s.ui.mud_visibility;
            // Two protocols, picked by brush_color.a:
            //   FP (alpha=1): camera-relative world pos + meter radius.
            //   Orbital (alpha=0): sphere direction + angular radius.
            if (s.camera.mode == CameraMode::FirstPerson) {
                glm::dvec3 eye_w = camera_eye_position(s.camera);
                glm::vec3 cam_rel = glm::vec3(stamp_world_pos - eye_w);
                double pick_dist = glm::length(stamp_world_pos - eye_w);
                // Dot world-radius scales with view distance so it stays
                // ~0.3° on screen — visible at any reach. No upper cap; at
                // 100 km distance the dot is 500 m, at 1000 km it's 5 km,
                // which still reads as a small pip on screen.
                float dot_r = static_cast<float>(std::clamp(pick_dist * 0.005, 0.25, 1.0e6));
                cam.brush_world = glm::vec4(cam_rel.x, cam_rel.y, cam_rel.z,
                                            s.cursor_on_world ? dot_r : 0.0f);
            } else {
                cam.brush_world = glm::vec4(
                    s.stamp_sphere_dir.x, s.stamp_sphere_dir.y, s.stamp_sphere_dir.z,
                    s.cursor_on_world ? s.effective_angular : 0.0f);
            }
            cam.brush_color = brush_color;
            cam.inv_view_proj = glm::inverse(cam_proj * cam_view);
            std::memcpy(s.camera_ubo_info.pMappedData, &cam, sizeof(cam));
            vmaFlushAllocation(allocator, s.camera_ubo_alloc, 0, VK_WHOLE_SIZE);
        }

        double frame_end = glfwGetTime();
        s.cpu_times[s.timing_index] = (frame_end - (s.last_time - static_cast<double>(dt))) * 1000.0;
        s.timing_index = (s.timing_index + 1) % GlobeState::AVG_FRAMES;
        if (s.timing_count < GlobeState::AVG_FRAMES) s.timing_count++;
        s.queries_valid = true;

        if (frame_end - s.last_title_update >= 1.0) {
            double cpu_sum = 0.0, gpu_sum = 0.0;
            for (int i = 0; i < s.timing_count; ++i) {
                cpu_sum += s.cpu_times[i];
                gpu_sum += s.gpu_times[i];
            }
            s.cpu_avg_ms = cpu_sum / s.timing_count;
            s.gpu_avg_ms = gpu_sum / s.timing_count;
            s.last_title_update = frame_end;
        }

    // Return to launcher menu if the user requested it (ESC or "< Back").
    if (s.embedded && s.ui.wants_back) {
        s.ui.wants_back = false;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
void globe_render(GlobeState& s, Renderer& r,
                  FrameData& frame, uint32_t image_index, VkExtent2D extent)
{
    auto INVALID_TILE = QuadNode{0xFFFFFFFFu, 0u, 0u, 0u};
    VkDevice device = r.device;
    VmaAllocator allocator = r.allocator;
    VkQueue graphics_queue = r.graphics_queue;
    uint32_t gfx_family = r.gfx_family;
    GLFWwindow* window = r.window;
    float terrain_size = s.terrain_size;
    (void)allocator; (void)graphics_queue; (void)gfx_family; (void)window;

        // ---- Terrain brush dispatch (flat-grid only, before SWE) ----
        if (!s.ui.ocean_enabled && s.brush_hit && (s.input.brush_mode == BrushMode::Raise || s.input.brush_mode == BrushMode::Lower)) {
            float sign = (s.input.brush_mode == BrushMode::Raise) ? 1.0f : -1.0f;

            TerrainBrushPC tb_pc{};
            tb_pc.brush_x = s.grid_x;
            tb_pc.brush_y = s.grid_y;
            tb_pc.brush_radius = s.ui.brush_radius_grid;
            tb_pc.brush_amount = sign * s.ui.terrain_strength * std::min(s.last_dt, 0.033f);
            tb_pc.grid_w = s.basin_params.grid_w;
            tb_pc.grid_h = s.basin_params.grid_h;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.terrain_brush_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    s.pipelines.tb_pipeline_layout, 0, 1, &s.tb_desc_set, 0, nullptr);
            vkCmdPushConstants(frame.cmd, s.pipelines.tb_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(tb_pc), &tb_pc);
            vkCmdDispatch(frame.cmd, (s.basin_params.grid_w + 15) / 16, (s.basin_params.grid_h + 15) / 16, 1);

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

        // ---- Sand brush dispatch (flat-grid only) ----
        if (!s.ui.ocean_enabled && s.brush_hit && s.input.brush_mode == BrushMode::Sand) {
            TerrainBrushPC sb_pc{};
            sb_pc.brush_x = s.grid_x;
            sb_pc.brush_y = s.grid_y;
            sb_pc.brush_radius = s.ui.brush_radius_grid;
            sb_pc.brush_amount = s.ui.brush_strength * std::min(s.last_dt, 0.033f);
            sb_pc.grid_w = s.basin_params.grid_w;
            sb_pc.grid_h = s.basin_params.grid_h;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.terrain_brush_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    s.pipelines.tb_pipeline_layout, 0, 1, &s.sand_brush_desc_set, 0, nullptr);
            vkCmdPushConstants(frame.cmd, s.pipelines.tb_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(sb_pc), &sb_pc);
            vkCmdDispatch(frame.cmd, (s.basin_params.grid_w + 15) / 16, (s.basin_params.grid_h + 15) / 16, 1);

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
        tp.cam_pos = camera_eye_position(s.camera);
        tp.cam_forward = glm::dvec3(camera_forward(s.camera));
        tp.screen_height = static_cast<float>(extent.height);
        tp.fov_y = s.camera.fov_y;
        tp.planet_radius = PLANET_RADIUS;
        tp.max_elevation = PLANET_MAX_ELEVATION;
        tp.subdivide_threshold = TILE_SUBDIVIDE_PX;
        tp.max_level = PLANET_MAX_LEVEL;
        tp.max_tiles = PLANET_TILE_POOL;
        tp.altitude_above_terrain = s.altitude_above_terrain;
        auto visible_tiles = planet_select_visible_tiles(tp);
        std::sort(visible_tiles.begin(), visible_tiles.end(), [](const QuadNode& a, const QuadNode& b) {
            if (a.face != b.face) return a.face < b.face;
            if (a.level != b.level) return a.level < b.level;
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        });
        s.ui.visible_tile_count = static_cast<uint32_t>(visible_tiles.size());

        // ---- Auto-anchor neighbors from previous frame's edge flags -----------
        // The step shader InterlockedOrs which edges of each disturbed tile have
        // water above static. We resolve those flagged tiles' missing same-face
        // same-level neighbors and anchor them so water can flow into them next
        // frame. Face-seam and LOD-mismatch crossings are deferred to Phase 2/3
        // (planet_neighbor_same_face returns invalid; that edge stays reflective).
        if (s.ui.ocean_enabled) {
            const uint32_t* flags = static_cast<const uint32_t*>(s.edge_flags_info.pMappedData);
            if (flags) {
                for (uint32_t sl = 0; sl < PLANET_TILE_POOL; ++sl) {
                    if (flags[sl] == 0) continue;
                    QuadNode tile = s.slot_to_tile[sl];
                    if (tile.face == INVALID_TILE.face) continue;  // slot is unassigned
                    for (int dir = 0; dir < 4; ++dir) {
                        uint32_t bit = 1u << dir;
                        if (!(flags[sl] & bit)) continue;
                        QuadNeighbor nb = planet_neighbor_same_face(tile, dir);
                        if (!nb.valid) continue;
                        if (s.tile_slot_map.count(nb.tile)) {
                            // Already anchored. Just join the simulation.
                            s.disturbed_tiles.insert(nb.tile);
                            continue;
                        }
                        if (s.free_slots.empty()) continue;
                        uint32_t ns = s.free_slots.back();
                        s.free_slots.pop_back();
                        s.tile_slot_map.emplace(nb.tile, ns);
                        s.slot_to_tile[ns] = nb.tile;
                        s.disturbed_tiles.insert(nb.tile);
                        s.pending_init.insert(nb.tile);
                    }
                }
            }
        }

        // ---- Stable slot allocation -------------------------------------------
        // tile_slots[i] is the pool_index for visible_tiles[i]. Tiles new to the
        // visible set are pushed into s.pending_init for the SWE init pass.
        std::vector<uint32_t> tile_slots(visible_tiles.size());
        {
            std::unordered_set<QuadNode, QuadNodeHash> visible_set;
            visible_set.reserve(visible_tiles.size());
            for (const auto& t : visible_tiles) visible_set.insert(t);

            // Free slots whose tile has left the visible set — but never free a
            // disturbed (brushed) tile. Those stay anchored to their slot so the
            // SWE simulation keeps running and the water survives camera moves
            // / LOD transitions, like terrain stamps survive.
            for (auto it = s.tile_slot_map.begin(); it != s.tile_slot_map.end(); ) {
                if (!visible_set.count(it->first) && !s.disturbed_tiles.count(it->first)) {
                    s.slot_to_tile[it->second] = INVALID_TILE;
                    s.free_slots.push_back(it->second);
                    it = s.tile_slot_map.erase(it);
                } else {
                    ++it;
                }
            }

            // Resolve / allocate slots for visible tiles.
            for (uint32_t i = 0; i < visible_tiles.size(); i++) {
                auto it = s.tile_slot_map.find(visible_tiles[i]);
                if (it != s.tile_slot_map.end()) {
                    tile_slots[i] = it->second;
                } else {
                    // PLANET_TILE_POOL >> typical visible count; underrun should not happen.
                    uint32_t slot = s.free_slots.empty() ? 0u : s.free_slots.back();
                    if (!s.free_slots.empty()) s.free_slots.pop_back();
                    s.tile_slot_map.emplace(visible_tiles[i], slot);
                    s.slot_to_tile[slot] = visible_tiles[i];
                    tile_slots[i] = slot;
                    s.pending_init.insert(visible_tiles[i]);
                }
            }
        }
        if (s.planet_swe_needs_full_init) {
            for (const auto& t : visible_tiles) s.pending_init.insert(t);
            s.planet_swe_needs_full_init = false;
        }
        {

            // Upload stamps to GPU if changed
            if (s.stamps_dirty && !s.stamps.empty()) {
                size_t copy_size = s.stamps.size() * sizeof(TerrainStamp);
                std::memcpy(s.stamp_buf_info.pMappedData, s.stamps.data(), copy_size);
                vmaFlushAllocation(allocator, s.stamp_buf_alloc, 0, copy_size);
                s.stamps_dirty = false;
            }
            if (s.water_stamps_dirty && !s.water_stamps.empty()) {
                size_t copy_size = s.water_stamps.size() * sizeof(WaterStamp);
                std::memcpy(s.water_stamp_buf_info.pMappedData, s.water_stamps.data(), copy_size);
                vmaFlushAllocation(allocator, s.water_stamp_buf_alloc, 0, copy_size);
                s.water_stamps_dirty = false;
            }

            // Build the set of tiles that need terrain (re)generation:
            // - newly visible tiles (s.pending_init)
            // - all allocated tiles when stamps change (stamps are global)
            bool stamps_changed = (static_cast<uint32_t>(s.stamps.size()) != s.terrain_gen_stamp_count);
            std::vector<std::pair<QuadNode, uint32_t>> tiles_to_gen;
            if (stamps_changed) {
                for (const auto& [tile, slot] : s.tile_slot_map)
                    tiles_to_gen.push_back({tile, slot});
                s.terrain_gen_stamp_count = static_cast<uint32_t>(s.stamps.size());
            } else {
                for (const auto& tile : s.pending_init) {
                    auto it = s.tile_slot_map.find(tile);
                    if (it != s.tile_slot_map.end())
                        tiles_to_gen.push_back({tile, it->second});
                }
            }

            if (!tiles_to_gen.empty()) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.terrain_gen_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    s.pipelines.terrain_gen_pipeline_layout, 0, 1, &s.terrain_gen_desc_set, 0, nullptr);

                for (const auto& [tile, slot] : tiles_to_gen) {
                    float ts = 2.0f / static_cast<float>(1u << tile.level);

                    PlanetGenPC gen_pc{};
                    gen_pc.u_min = -1.0f + tile.x * ts;
                    gen_pc.v_min = -1.0f + tile.y * ts;
                    gen_pc.tile_size = ts;
                    gen_pc.face = tile.face;
                    gen_pc.pool_index = slot;
                    gen_pc.tex_res = PLANET_TILE_RES;
                    gen_pc.seed = 42;
                    gen_pc.stamp_count = static_cast<uint32_t>(s.stamps.size());

                    vkCmdPushConstants(frame.cmd, s.pipelines.terrain_gen_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gen_pc), &gen_pc);
                    vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
                }

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
        }

        // ---- Planet SWE init + step (per-tile, only on disturbed tiles) ----
        if (s.ui.ocean_enabled) {
            // Clear edge_flags before this frame's step; barrier so InterlockedOrs
            // see a zeroed buffer.
            vkCmdFillBuffer(frame.cmd, s.edge_flags_buf, 0, VK_WHOLE_SIZE, 0u);
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

            // Init dispatch — consume s.pending_init.
            bool any_init = !s.pending_init.empty();
            if (any_init) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    s.pipelines.planet_swe_init_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    s.pipelines.planet_swe_init_pipeline_layout, 0, 1,
                    &s.planet_swe_init_desc_set, 0, nullptr);
                for (const auto& t : s.pending_init) {
                    auto it = s.tile_slot_map.find(t);
                    if (it == s.tile_slot_map.end()) continue;
                    float ts = 2.0f / static_cast<float>(1u << t.level);
                    PlanetSweInitPC ipc{};
                    ipc.grid_w = PLANET_TILE_RES;
                    ipc.grid_h = PLANET_TILE_RES;
                    ipc.sea_level = s.ui.sea_level;
                    ipc.pool_index = it->second;
                    ipc.u_min = -1.0f + t.x * ts;
                    ipc.v_min = -1.0f + t.y * ts;
                    ipc.tile_size = ts;
                    ipc.face = t.face;
                    ipc.water_stamp_count = static_cast<uint32_t>(s.water_stamps.size());
                    vkCmdPushConstants(frame.cmd, s.pipelines.planet_swe_init_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ipc), &ipc);
                    vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
                }
                s.pending_init.clear();
            }

            // Pick target tile under the cursor for the water brush.
            PlanetTilePick pick{};
            bool water_brush_active = s.brush_hit && s.input.brush_mode == BrushMode::Water;
            if (water_brush_active) {
                pick = planet_pick_tile(s.stamp_sphere_dir, visible_tiles, PLANET_TILE_RES);
                if (pick.hit) s.disturbed_tiles.insert(pick.node);
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
            uint32_t cur_stamp_count = static_cast<uint32_t>(s.stamps.size());
            // Discard the unprocessed range if the user hit "clear" or "undo" so
            // we don't re-apply stamps that no longer exist.
            if (cur_stamp_count < s.last_processed_stamp_count)
                s.last_processed_stamp_count = cur_stamp_count;

            bool any_h_adjust = (cur_stamp_count > s.last_processed_stamp_count) && !s.disturbed_tiles.empty();
            if (any_h_adjust) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    s.pipelines.planet_swe_h_adjust_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    s.pipelines.planet_swe_h_adjust_pipeline_layout, 0, 1,
                    &s.planet_swe_h_adjust_desc_set, 0, nullptr);
                for (uint32_t si = s.last_processed_stamp_count; si < cur_stamp_count; ++si) {
                    const TerrainStamp& stmp = s.stamps[si];
                    for (const auto& tile : s.disturbed_tiles) {
                        auto it = s.tile_slot_map.find(tile);
                        if (it == s.tile_slot_map.end()) continue;
                        float ts = 2.0f / static_cast<float>(1u << tile.level);
                        PlanetSweHAdjustPC hpc{};
                        hpc.u_min = -1.0f + tile.x * ts;
                        hpc.v_min = -1.0f + tile.y * ts;
                        hpc.tile_size = ts;
                        hpc.face = tile.face;
                        hpc.pool_index = it->second;
                        hpc.grid_w = PLANET_TILE_RES;
                        hpc.grid_h = PLANET_TILE_RES;
                        hpc.stamp_pos_x = stmp.pos_x;
                        hpc.stamp_pos_y = stmp.pos_y;
                        hpc.stamp_pos_z = stmp.pos_z;
                        hpc.stamp_cos_radius = stmp.cos_radius;
                        hpc.stamp_delta_h = stmp.delta_h;
                        vkCmdPushConstants(frame.cmd, s.pipelines.planet_swe_h_adjust_pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hpc), &hpc);
                        vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
                    }
                }
                s.last_processed_stamp_count = cur_stamp_count;

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
            } else if (cur_stamp_count > s.last_processed_stamp_count) {
                // No disturbed tiles to adjust, but still mark stamps as processed
                // so they aren't re-applied if water is brushed later.
                s.last_processed_stamp_count = cur_stamp_count;
            }

            // Step pass: simulate every disturbed tile, even ones that are not
            // currently visible. The slot allocator keeps disturbed tiles anchored,
            // so their heightmap and water state survive in the pool.
            if (!s.disturbed_tiles.empty()) {
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
                dd.reserve(s.disturbed_tiles.size());
                for (const auto& tile : s.disturbed_tiles) {
                    auto it = s.tile_slot_map.find(tile);
                    if (it == s.tile_slot_map.end()) continue;
                    float ts = 2.0f / static_cast<float>(1u << tile.level);
                    float tile_dx = ts * PLANET_RADIUS / static_cast<float>(PLANET_TILE_RES - 1);
                    DisturbedDispatch entry{tile, it->second, tile_dx,
                                            {NO_NEIGHBOR, NO_NEIGHBOR, NO_NEIGHBOR, NO_NEIGHBOR}};
                    for (int dir = 0; dir < 4; ++dir) {
                        QuadNeighbor nb = planet_neighbor_same_face(tile, dir);
                        if (!nb.valid) continue;
                        auto nit = s.tile_slot_map.find(nb.tile);
                        if (nit != s.tile_slot_map.end()) entry.nb[dir] = nit->second;
                    }
                    dd.push_back(entry);
                }

                // CFL from the finest disturbed tile.
                float swe_total_dt = std::min(s.last_dt, 0.033f) * s.ui.time_scale;
                float min_dx = 1e30f;
                for (const auto& d : dd) min_dx = std::min(min_dx, d.tile_dx);
                if (min_dx < 1.0f) min_dx = 1.0f;

                float c_max = std::sqrt(s.ui.gravity * 200.0f);
                float cfl_dt = min_dx / (c_max * 6.0f);
                int substeps = std::max(1, static_cast<int>(std::ceil(swe_total_dt / cfl_dt)));
                substeps = std::min(substeps, 16);
                float sub_dt = swe_total_dt / substeps;

                // World brush radius derived the same way the cursor ring is sized:
                // brush_world.w (in main.cpp ~2018) is an *angular* radius (radians);
                // terrain.fs.hlsl draws the ring against acos(dot(frag_dir, brush_dir)).
                // The arc length on the planet surface is angular * R.
                float angular_r = s.ui.brush_radius_grid * s.ui.stamp_angular_scale;
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
                    s.pipelines.planet_swe_step_pipeline);

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
                        s.pipelines.planet_swe_step_pipeline_layout, 0, 1,
                        &s.planet_swe_step_desc_sets[s.swe_ping_pong], 0, nullptr);

                    for (size_t k = 0; k < dd.size(); ++k) {
                        const auto& d = dd[k];
                        PlanetSweStepPC spc{};
                        spc.time = static_cast<float>(glfwGetTime());
                        spc.dt = sub_dt;
                        spc.gravity = s.ui.gravity;
                        spc.friction = s.ui.friction;
                        spc.dx = d.tile_dx;
                        spc.sea_level = s.ui.sea_level;
                        spc.damping = s.ui.damping;
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
                            spc.pulse_amount = s.ui.brush_strength * sub_dt;
                        } else {
                            spc.pulse_amount = 0.0f;
                        }

                        spc.neighbor_left  = d.nb[0];
                        spc.neighbor_right = d.nb[1];
                        spc.neighbor_down  = d.nb[2];
                        spc.neighbor_up    = d.nb[3];

                        vkCmdPushConstants(frame.cmd, s.pipelines.planet_swe_step_pipeline_layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(spc), &spc);
                        vkCmdDispatch(frame.cmd, (PLANET_TILE_RES + 7) / 8, (PLANET_TILE_RES + 7) / 8, 1);
                    }

                    s.swe_ping_pong ^= 1;
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

        // ---- SWE step dispatch (CFL sub-stepping) — flat grid, disabled when ocean is on ----
        // Authority: fastest phenomenon — water responds to brush edits immediately
        if (!s.ui.ocean_enabled) {
            vkCmdResetQueryPool(frame.cmd, r.query_pool, r.current_frame * 2, 2);

            float swe_total_dt = std::min(s.last_dt, 0.033f) * s.ui.time_scale;
            float c_max = std::sqrt(s.ui.gravity * 200.0f);
            float cfl_dt = s.basin_params.cell_spacing / (c_max * 6.0f);
            int substeps = std::max(1, static_cast<int>(std::ceil(swe_total_dt / cfl_dt)));
            substeps = std::min(substeps, 16);
            float sub_dt = swe_total_dt / substeps;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.swe_step_pipeline);

            vkCmdWriteTimestamp2(frame.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                r.query_pool, r.current_frame * 2 + 0);

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
                                        s.pipelines.swe_step_pipeline_layout, 0, 1,
                                        &s.swe_step_desc_sets[s.swe_ping_pong], 0, nullptr);

                SweStepPC swe_pc{};
                swe_pc.time = static_cast<float>(glfwGetTime());
                swe_pc.dt = sub_dt;
                swe_pc.gravity = s.ui.gravity;
                swe_pc.friction = s.ui.friction;
                swe_pc.dx = s.basin_params.cell_spacing;
                swe_pc.sea_level = s.basin_params.floor_height + s.basin_params.initial_water;
                swe_pc.damping = s.ui.damping;
                swe_pc.k_rain = 0.0f;
                swe_pc.grid_w = SWE_GRID_W;
                swe_pc.grid_h = SWE_GRID_H;

                if (step == 0) {
                    bool water_brush_active = s.brush_hit && s.input.brush_mode == BrushMode::Water;
                    if (s.input.pulse_pending) {
                        swe_pc.pulse_x = SWE_GRID_W * 0.5f;
                        swe_pc.pulse_y = SWE_GRID_H * 0.5f;
                        swe_pc.pulse_radius = s.ui.pulse_radius_cells;
                        swe_pc.pulse_amount = s.ui.pulse_amount;
                        s.input.pulse_pending = false;
                    } else if (water_brush_active) {
                        swe_pc.pulse_x = s.grid_x;
                        swe_pc.pulse_y = s.grid_y;
                        swe_pc.pulse_radius = s.ui.brush_radius_grid;
                        swe_pc.pulse_amount = s.ui.brush_strength;
                    } else {
                        swe_pc.pulse_amount = 0.0f;
                    }
                } else {
                    swe_pc.pulse_amount = 0.0f;
                }

                vkCmdPushConstants(frame.cmd, s.pipelines.swe_step_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(swe_pc), &swe_pc);
                vkCmdDispatch(frame.cmd, (SWE_GRID_W + 7) / 8, (SWE_GRID_H + 7) / 8, 1);

                s.swe_ping_pong ^= 1;
            }

            vkCmdWriteTimestamp2(frame.cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                r.query_pool, r.current_frame * 2 + 1);
        }

        // ---- Atmosphere 3D dispatch ----
        // Authority: slower phenomenon — reads terrain + water state, computes pressure/wind/rain
        if (s.ui.atmosphere_enabled) {
            s.accumulated_atmo_time += s.last_dt * s.ui.time_scale;

            Atmo3DPC apc{};
            apc.dt = std::min(s.last_dt, 0.033f) * s.ui.time_scale;
            apc.accumulated_time = s.accumulated_atmo_time;
            apc.grid_w = ATMO_W;
            apc.grid_h = ATMO_H;
            apc.grid_d = ATMO_D;
            apc.terrain_scale = s.basin_params.cell_spacing;
            apc.layer_height = ATMO_LAYER_HEIGHT;
            apc.max_elevation = 2000.0f;
            apc.orographic_lift_coeff = s.ui.orographic_lift;
            apc.adiabatic_cooling_rate = s.ui.adiabatic_cooling;
            apc.rain_shadow_intensity = s.ui.rain_shadow;
            apc.force_init = (s.atmo_needs_init || s.ui.request_atmo_reset) ? 1u : 0u;
            apc.k_pressure = s.ui.k_pressure;
            apc.wind_strength = 1.0f;
            apc.k_evaporation = 0.0f;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.atmo_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                s.pipelines.atmo_pipeline_layout, 0, 1, &s.atmo_desc_sets[s.atmo_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, s.pipelines.atmo_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, sizeof(Atmo3DPC), &apc);
            vkCmdDispatch(frame.cmd, (ATMO_W + 3) / 4, (ATMO_H + 3) / 4, (ATMO_D + 3) / 4);

            // Barrier: atmosphere writes -> erosion reads + fragment reads
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

            // Copy bottom layer of wind field for CPU pressure debug readback
            {
                SweImage& wind_src = (s.atmo_ping_pong == 0) ? s.wind_field_b : s.wind_field_a;
                VkBufferImageCopy2 region{};
                region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {ATMO_W, ATMO_H, 1};
                VkCopyImageToBufferInfo2 copy{};
                copy.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2;
                copy.srcImage = wind_src.image;
                copy.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
                copy.dstBuffer = s.atmo_debug_buf;
                copy.regionCount = 1;
                copy.pRegions = &region;
                vkCmdCopyImageToBuffer2(frame.cmd, &copy);
            }

            s.atmo_ping_pong ^= 1;

            if (s.atmo_needs_init || s.ui.request_atmo_reset) {
                s.atmo_needs_init = false;
                s.ui.request_atmo_reset = false;
            }
        }

        // ---- Erosion dispatch (after SWE + atmosphere) ----
        // Authority: geological timescale — reads water flow + ground conditions
        if (s.ui.erosion_enabled) {
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

            float ero_dt = std::min(s.last_dt, 0.033f) * s.ui.time_scale;

            ErosionPC ero_pc{};
            ero_pc.dt = ero_dt;
            ero_pc.dx = s.basin_params.cell_spacing;
            ero_pc.grid_w = SWE_GRID_W;
            ero_pc.grid_h = SWE_GRID_H;
            ero_pc.k_erosion = s.ui.k_erosion;
            ero_pc.k_deposit = s.ui.k_deposit;
            ero_pc.k_capacity = s.ui.k_capacity;
            ero_pc.min_slope = s.ui.min_slope;
            ero_pc.min_depth = s.ui.min_erosion_depth;
            ero_pc.max_change = s.ui.max_change_m;
            ero_pc.max_sediment = s.ui.max_sediment;
            ero_pc.k_wind = 0.0f;
            ero_pc.k_thermal = 0.0f;
            ero_pc.wind_threshold = 1.0f;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.erosion_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    s.pipelines.ero_pipeline_layout, 0, 1,
                                    &s.ero_desc_sets[s.sediment_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, s.pipelines.ero_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(ero_pc), &ero_pc);
            vkCmdDispatch(frame.cmd, (SWE_GRID_W + 15) / 16, (SWE_GRID_H + 15) / 16, 1);

            s.sediment_ping_pong ^= 1;
        }

        // ---- Sand particle dispatch (consumer — after all authority systems) ----
        if (s.ui.atmosphere_enabled && s.ui.sand_enabled) {
            static uint32_t sand_emit_offset = 0;

            SandSimPC spc{};
            spc.dt = std::min(s.last_dt, 0.033f) * s.ui.time_scale;
            spc.terrain_size = terrain_size;
            spc.loft_threshold = s.ui.sand_loft_threshold;
            spc.loft_rate = s.ui.sand_loft_rate;
            spc.gravity = s.ui.sand_gravity;
            spc.accumulated_time = s.accumulated_atmo_time;
            spc.max_particles = SAND_MAX_PARTICLES;
            spc.emit_offset = sand_emit_offset;
            spc.emit_count = SAND_EMIT_PER_FRAME;
            spc.grid_d = ATMO_D;
            spc.layer_height = ATMO_LAYER_HEIGHT;
            spc.bounce_energy = s.ui.sand_bounce;

            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelines.sand_sim_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                s.pipelines.sand_sim_pipeline_layout, 0, 1, &s.sand_sim_desc_sets[s.atmo_ping_pong], 0, nullptr);
            vkCmdPushConstants(frame.cmd, s.pipelines.sand_sim_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
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
            sc_barrier.image = r.swapchain_images[image_index];
            sc_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkImageMemoryBarrier2 depth_barrier{};
            depth_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            depth_barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            depth_barrier.srcAccessMask = VK_ACCESS_2_NONE;
            depth_barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            depth_barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depth_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depth_barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_barrier.image = r.depth_buffer.image;
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
            color_attachment.imageView = r.swapchain_views[image_index];
            color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color_attachment.clearValue.color = {{0.02f, 0.02f, 0.05f, 1.0f}};

            VkRenderingAttachmentInfo depth_attachment{};
            depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView = r.depth_buffer.view;
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
            vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelines.clipmap_terrain_pipeline);
            vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                s.pipelines.clipmap_gfx_pipeline_layout, 0, 1, &s.clipmap_gfx_desc_sets[s.atmo_ping_pong], 0, nullptr);

            VkDeviceSize clip_offset = 0;
            vkCmdBindVertexBuffers(frame.cmd, 0, 1, &s.clipmap_vbo, &clip_offset);
            vkCmdBindIndexBuffer(frame.cmd, s.clipmap_ibo, 0, VK_INDEX_TYPE_UINT32);

            glm::dvec3 cam_pos_d = camera_eye_position(s.camera);

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
                tpc.sea_level = s.ui.ocean_enabled ? s.ui.sea_level : -1.0f;

                vkCmdPushConstants(frame.cmd, s.pipelines.clipmap_gfx_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(tpc), &tpc);
                vkCmdDrawIndexed(frame.cmd, s.clipmap_index_count, 1, 0, 0, 0);
            }

            // Water pass (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelines.water_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    s.pipelines.gfx_pipeline_layout, 0, 1, &s.gfx_desc_sets[s.atmo_ping_pong], 0, nullptr);
                GfxPC wpc{};
                wpc.terrain_size = terrain_size;
                wpc.heightmap_texel = 1.0f / static_cast<float>(s.basin_params.grid_w);
                wpc.max_elevation = 2000.0f;
                wpc.cloud_opacity = s.ui.atmosphere_enabled ? s.ui.cloud_opacity : 0.0f;
                vkCmdPushConstants(frame.cmd, s.pipelines.gfx_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(wpc), &wpc);
                VkDeviceSize water_offset = 0;
                vkCmdBindVertexBuffers(frame.cmd, 0, 1, &s.vertex_buffer, &water_offset);
                vkCmdBindIndexBuffer(frame.cmd, s.index_buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(frame.cmd, s.index_count, 1, 0, 0, 0);
            }

            // Cloud raymarch pass (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelines.raymarch_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    s.pipelines.raymarch_pipeline_layout, 0, 1, &s.gfx_desc_sets[s.atmo_ping_pong], 0, nullptr);

                RaymarchPC rpc{};
                rpc.terrain_size = terrain_size;
                rpc.max_elevation = 2000.0f;
                rpc.cloud_opacity = s.ui.cloud_opacity;
                rpc.cloud_base = s.ui.cloud_altitude;
                rpc.vol_w = ATMO_W;
                rpc.vol_h = ATMO_H;
                rpc.vol_d = ATMO_D;
                rpc.layer_height = ATMO_LAYER_HEIGHT;
                vkCmdPushConstants(frame.cmd, s.pipelines.raymarch_pipeline_layout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(rpc), &rpc);

                vkCmdDraw(frame.cmd, 3, 1, 0, 0);
            }

            // Sand particle draw (disabled for planet mode — Phase 2)
            if (false) {
                vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelines.sand_render_pipeline);
                vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    s.pipelines.sand_render_pipeline_layout, 0, 1, &s.sand_render_desc_set, 0, nullptr);

                SandRenderPC srpc{};
                srpc.streak_length = s.ui.sand_streak;
                srpc.particle_alpha = s.ui.sand_alpha;
                srpc.max_particles = SAND_MAX_PARTICLES;
                vkCmdPushConstants(frame.cmd, s.pipelines.sand_render_pipeline_layout,
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
            imgui_color.imageView = r.swapchain_views[image_index];
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
            barrier.image = r.swapchain_images[image_index];
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(frame.cmd, &dep);
        }

}

// ---------------------------------------------------------------------------
void globe_shutdown(GlobeState& s, Renderer& r)
{
    if (s.embedded) {
        ImGui_ImplGlfw_RestoreCallbacks(r.window);
        glfwSetKeyCallback(r.window, nullptr);
        glfwSetMouseButtonCallback(r.window, nullptr);
        glfwSetCursorPosCallback(r.window, nullptr);
        glfwSetScrollCallback(r.window, nullptr);
        glfwSetFramebufferSizeCallback(r.window, nullptr);
        glfwSetWindowUserPointer(r.window, nullptr);
        glfwSetInputMode(r.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        ImGui_ImplGlfw_InstallCallbacks(r.window);
    }

    VkDevice device = r.device;
    VmaAllocator allocator = r.allocator;


    pipelines_destroy(s.pipelines, device);

    vkDestroyImageView(device, s.clipmap_hm_view, nullptr);
    vmaDestroyImage(allocator, s.clipmap_hm_image, s.clipmap_hm_alloc);

    vmaDestroyBuffer(allocator, s.clipmap_ibo, s.clipmap_ibo_alloc);
    vmaDestroyBuffer(allocator, s.clipmap_vbo, s.clipmap_vbo_alloc);

    vkDestroyDescriptorPool(device, s.desc_pool, nullptr);

    destroy_swe_image(device, allocator, s.swe_output);
    destroy_swe_image(device, allocator, s.swe_state_b);
    destroy_swe_image(device, allocator, s.swe_state_a);

    destroy_swe_image(device, allocator, s.sediment_b);
    destroy_swe_image(device, allocator, s.sediment_a);

    destroy_swe_image(device, allocator, s.sand_deposit);
    destroy_swe_image(device, allocator, s.ground_wind);
    destroy_swe_image(device, allocator, s.ground_cond);
    destroy_swe_image(device, allocator, s.atmo_shadow);
    destroy_swe_image(device, allocator, s.wind_field_b);
    destroy_swe_image(device, allocator, s.wind_field_a);
    destroy_swe_image(device, allocator, s.atmo_state_b);
    destroy_swe_image(device, allocator, s.atmo_state_a);

    vmaDestroyBuffer(allocator, s.sand_particle_buf, s.sand_particle_alloc);
    vmaDestroyBuffer(allocator, s.index_buffer, s.index_alloc);
    vmaDestroyBuffer(allocator, s.vertex_buffer, s.vertex_alloc);
    vmaDestroyBuffer(allocator, s.camera_ubo, s.camera_ubo_alloc);
    vmaDestroyBuffer(allocator, s.stamp_buf, s.stamp_buf_alloc);

    vkDestroySampler(device, s.terrain_linear_sampler, nullptr);
    vkDestroySampler(device, s.sampler, nullptr);
    vkDestroyImageView(device, s.heightmap_gpu.view, nullptr);
    vmaDestroyImage(allocator, s.heightmap_gpu.image, s.heightmap_gpu.allocation);


    vkDestroyImageView(device, s.water_state_a_view, nullptr);
    vmaDestroyImage(allocator, s.water_state_a_img, s.water_state_a_alloc);
    vkDestroyImageView(device, s.water_state_b_view, nullptr);
    vmaDestroyImage(allocator, s.water_state_b_img, s.water_state_b_alloc);
    vkDestroyImageView(device, s.water_output_view, nullptr);
    vmaDestroyImage(allocator, s.water_output_img, s.water_output_alloc);
    vmaDestroyBuffer(allocator, s.water_stamp_buf, s.water_stamp_buf_alloc);
    vmaDestroyBuffer(allocator, s.edge_flags_buf, s.edge_flags_alloc);
    vmaDestroyBuffer(allocator, s.atmo_debug_buf, s.atmo_debug_alloc);
}

// globe.h — embeddable Globe (planet simulation) module.
//
// Provides init/tick/render/shutdown so the globe can be driven by either
// a standalone executable or the unified launcher.
#pragma once

#include "renderer.h"
#include "resources.h"
#include "camera.h"
#include "input.h"
#include "pipeline.h"
#include "planet.h"
#include "terrain.h"
#include "ui.h"
#include "vk_util.h"
#include "input_frame.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr uint32_t GLOBE_SWE_GRID_W        = 1024;
constexpr uint32_t GLOBE_SWE_GRID_H        = 1024;
constexpr uint32_t GLOBE_ATMO_W            = 128;
constexpr uint32_t GLOBE_ATMO_H            = 128;
constexpr uint32_t GLOBE_ATMO_D            = 32;
constexpr float    GLOBE_ATMO_LAYER_HEIGHT  = 100.0f;
constexpr uint32_t GLOBE_SAND_MAX_PARTICLES = 131072;
constexpr uint32_t GLOBE_SAND_EMIT_PER_FRAME = 2048;
constexpr uint32_t GLOBE_MESH_RES          = 512;
constexpr uint32_t GLOBE_TILE_RES          = 64;
constexpr uint32_t GLOBE_TILE_POOL         = 2048;
constexpr uint32_t GLOBE_MAX_LEVEL         = 14;
constexpr float    GLOBE_PLANET_RADIUS     = 6371000.0f;
constexpr float    GLOBE_MAX_ELEVATION     = 8000.0f;
constexpr float    GLOBE_TILE_SUBDIVIDE_PX = 512.0f;

// ---------------------------------------------------------------------------
// All globe state, bundled into a single struct.
// ---------------------------------------------------------------------------
struct GlobeState {
    bool embedded    = false;
    bool initialized = false;

    // --- Basin / heightmap ---
    BasinParams basin_params{};
    float       terrain_size = 0.0f;
    HeightmapGPU heightmap_gpu{};

    // --- Flat grid mesh ---
    VkBuffer      vertex_buffer = VK_NULL_HANDLE;
    VmaAllocation vertex_alloc  = VK_NULL_HANDLE;
    VkBuffer      index_buffer  = VK_NULL_HANDLE;
    VmaAllocation index_alloc   = VK_NULL_HANDLE;
    uint32_t      index_count   = 0;

    // --- Planet clipmap mesh ---
    VkBuffer      clipmap_vbo       = VK_NULL_HANDLE;
    VmaAllocation clipmap_vbo_alloc = VK_NULL_HANDLE;
    VkBuffer      clipmap_ibo       = VK_NULL_HANDLE;
    VmaAllocation clipmap_ibo_alloc = VK_NULL_HANDLE;
    uint32_t      clipmap_index_count = 0;

    // --- Camera UBO ---
    VkBuffer          camera_ubo       = VK_NULL_HANDLE;
    VmaAllocation     camera_ubo_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo camera_ubo_info{};

    // --- Stamp buffers ---
    VkBuffer          stamp_buf       = VK_NULL_HANDLE;
    VmaAllocation     stamp_buf_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo stamp_buf_info{};

    VkBuffer          water_stamp_buf       = VK_NULL_HANDLE;
    VmaAllocation     water_stamp_buf_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo water_stamp_buf_info{};

    // --- Edge flags (planet SWE cross-tile anchoring) ---
    VkBuffer          edge_flags_buf       = VK_NULL_HANDLE;
    VmaAllocation     edge_flags_alloc     = VK_NULL_HANDLE;
    VmaAllocationInfo edge_flags_info{};

    // --- Atmosphere debug readback ---
    VkBuffer          atmo_debug_buf       = VK_NULL_HANDLE;
    VmaAllocation     atmo_debug_alloc     = VK_NULL_HANDLE;
    VmaAllocationInfo atmo_debug_info{};

    // --- SWE images (ping-pong + output) ---
    SweImage swe_state_a{}, swe_state_b{}, swe_output{};

    // --- Sediment images (ping-pong) ---
    SweImage sediment_a{}, sediment_b{};

    // --- Atmosphere images ---
    SweImage atmo_state_a{}, atmo_state_b{};
    SweImage wind_field_a{}, wind_field_b{};
    SweImage atmo_shadow{}, ground_cond{}, ground_wind{};
    SweImage sand_deposit{};

    // --- Planet tile pools (2D array textures) ---
    VkImage       clipmap_hm_image = VK_NULL_HANDLE;
    VmaAllocation clipmap_hm_alloc = VK_NULL_HANDLE;
    VkImageView   clipmap_hm_view  = VK_NULL_HANDLE;

    VkImage       water_state_a_img = VK_NULL_HANDLE;
    VmaAllocation water_state_a_alloc = VK_NULL_HANDLE;
    VkImageView   water_state_a_view = VK_NULL_HANDLE;

    VkImage       water_state_b_img = VK_NULL_HANDLE;
    VmaAllocation water_state_b_alloc = VK_NULL_HANDLE;
    VkImageView   water_state_b_view = VK_NULL_HANDLE;

    VkImage       water_output_img = VK_NULL_HANDLE;
    VmaAllocation water_output_alloc = VK_NULL_HANDLE;
    VkImageView   water_output_view = VK_NULL_HANDLE;

    // --- Sand particle buffer ---
    VkBuffer      sand_particle_buf   = VK_NULL_HANDLE;
    VmaAllocation sand_particle_alloc = VK_NULL_HANDLE;

    // --- Samplers ---
    VkSampler sampler                = VK_NULL_HANDLE;
    VkSampler terrain_linear_sampler = VK_NULL_HANDLE;

    // --- Pipelines ---
    Pipelines pipelines{};

    // --- Descriptor pool + sets ---
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    VkDescriptorSet swe_init_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet swe_step_desc_sets[2]{};
    VkDescriptorSet gfx_desc_sets[2]{};
    VkDescriptorSet tb_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet sand_brush_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet ero_desc_sets[2]{};
    VkDescriptorSet atmo_desc_sets[2]{};
    VkDescriptorSet sand_sim_desc_sets[2]{};
    VkDescriptorSet sand_render_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet terrain_gen_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet clipmap_gfx_desc_sets[2]{};
    VkDescriptorSet planet_swe_init_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet planet_swe_h_adjust_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet planet_swe_step_desc_sets[2]{};

    // --- Frame state ---
    uint32_t swe_ping_pong      = 0;
    uint32_t sediment_ping_pong = 0;
    uint32_t atmo_ping_pong     = 0;
    bool     atmo_needs_init    = true;
    bool     queries_valid      = false;

    // --- Planet tile management ---
    std::unordered_map<QuadNode, uint32_t, QuadNodeHash> tile_slot_map;
    std::vector<QuadNode>   slot_to_tile;
    std::vector<uint32_t>   free_slots;
    bool                    planet_swe_needs_full_init = true;
    std::unordered_set<QuadNode, QuadNodeHash> disturbed_tiles;
    std::unordered_set<QuadNode, QuadNodeHash> pending_init;
    uint32_t last_processed_stamp_count = 0;
    uint32_t terrain_gen_stamp_count    = 0;

    // --- Terrain stamps (persistent brush edits) ---
    std::vector<TerrainStamp> stamps;
    bool stamps_dirty = false;
    std::vector<WaterStamp> water_stamps;
    bool water_stamps_dirty = false;

    // --- Camera + input + UI ---
    Camera     camera{};
    InputState input{};
    UIState    ui{};
    CallbackContext cb_ctx{};

    bool   cursor_on_world        = false;
    double terrain_height_at_cam  = 0.0;
    double altitude_above_terrain = 100000.0;
    float  accumulated_atmo_time  = 0.0f;

    // --- Timing ---
    double last_time         = 0.0;
    double ns_per_tick       = 0.0;
    static constexpr int AVG_FRAMES = 30;
    double cpu_times[AVG_FRAMES]{};
    double gpu_times[AVG_FRAMES]{};
    int    timing_index = 0;
    int    timing_count = 0;
    double last_title_update = 0.0;
    double cpu_avg_ms = 0.0;
    double gpu_avg_ms = 0.0;

    // --- Per-frame tick→render transfer ---
    float  last_dt           = 0.0f;
    bool   brush_hit         = false;
    bool   brush_active      = false;
    float  grid_x            = 0.0f;
    float  grid_y            = 0.0f;
    glm::vec3 stamp_sphere_dir{0.0f};
    float  effective_angular = 0.0f;
    glm::mat4 cam_view{1.0f};
    glm::mat4 cam_proj{1.0f};

    // --- Static statics that need to be per-instance ---
    double last_stamp_time       = 0.0;
    double last_water_stamp_time = 0.0;
    int    current_cursor_mode   = -1;
};

// Module API ---------------------------------------------------------------

void globe_init(GlobeState& s, Renderer& r);

/// Returns false if the user pressed "Back" (only when s.embedded).
bool globe_tick(GlobeState& s, Renderer& r, const InputFrame& in, float dt);

void globe_render(GlobeState& s, Renderer& r,
                  FrameData& frame, uint32_t image_index, VkExtent2D extent);

void globe_shutdown(GlobeState& s, Renderer& r);

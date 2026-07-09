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
#include "hydrology.h"
#include "ui.h"
#include "vk_util.h"
#include "input_frame.h"
#include "environment.h"   // bestiary — planet-scale EnvironmentField bridge

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

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
constexpr uint32_t GLOBE_HYDRO_RES         = 128;  // coarse global drainage grid, per cube face
                                                   // (128 keeps the synchronous build fast; raised
                                                   //  once the solve moves to a worker thread)
// Bytes of one published field (hydro or climate): 6 cube faces of RGBA32F.
constexpr VkDeviceSize HYDRO_FIELD_BYTES =
    VkDeviceSize{GLOBE_HYDRO_RES} * GLOBE_HYDRO_RES * 6 * 4 * sizeof(float);
constexpr float    WATER_BRUSH_DEPOSIT     = 75.0f; // brush_strength → field water units per deposit
                                                    // (feel knob for the field-only flood pulse)

// Persistent background hydrology worker. Owns a live water sim (LiveHydrology)
// that it steps continuously and publishes; the main thread uploads published
// snapshots and signals structure rebuilds on terrain edits. The field is the
// live ground truth, so edits make the water re-route organically (not snap).
// Held behind a unique_ptr so GlobeState stays move-assignable (the launcher
// does `globe_state = {}` on re-entry); the destructor stops + joins the worker.
// A one-shot water-brush deposit, handed main-thread → hydrology worker. The
// worker drains these into LiveHydrology::water before each step (field-only
// water: the brush feeds the field, the field drives everything downstream).
struct WaterDeposit { glm::vec3 dir; float cos_radius; float amount; };

// A baked-back observation from the fine SWE sim, handed main-thread →
// hydrology worker when a disturbed tile quiesces: one field cell's observed
// water-surface elevation (m) + mean water depth (m) over its wet texels.
// The worker applies these via LiveHydrology::apply_surface_set.
struct SurfaceSet { int cell; float surf; float mean_depth; };

// Seconds a disturbed tile must go untouched (no brush, no cross-edge flow)
// before its live SWE water is baked back into the field and the tile returns
// to the re-seed pool. Long enough for a splash to settle under damping.
constexpr double GLOBE_SWE_QUIESCE_S = 6.0;

// Hard ceiling on how long cross-edge flow (edge flags) can keep extending a
// tile's disturbance after its last brush contact. Prevents any feedback loop
// (however exotic) from pinning tiles in the live sim forever; a tile baked
// back mid-flow just freezes its observed surface into the field.
constexpr double GLOBE_SWE_DISTURB_MAX_S = 30.0;

// Slots the edge-flag auto-anchor must leave free for camera-driven tile
// streaming. The anchor cascade grows flood-fill-style (each anchored tile's
// waves can anchor its neighbors), so without a reserve one big splash — or
// the animated ocean in first-person — consumes the whole pool and visible
// tiles start aliasing slots.
constexpr size_t GLOBE_TILE_ANCHOR_RESERVE = 256;

struct HydroAsync {
    std::thread       worker;
    std::atomic<bool> stop{false};
    std::atomic<bool> structure_dirty{true};   // recompute drainage from pending_stamps
    std::atomic<bool> publish_ready{false};     // a fresh baked field is available

    std::mutex                stamps_mtx;
    std::vector<TerrainStamp> pending_stamps;   // guarded by stamps_mtx

    std::mutex                deposits_mtx;
    std::vector<WaterDeposit> pending_deposits; // brush deposits, guarded by deposits_mtx

    std::mutex               sets_mtx;
    std::vector<SurfaceSet>  pending_surface_sets; // SWE bake-backs, guarded by sets_mtx

    std::mutex             pub_mtx;
    std::vector<glm::vec4> published;           // latest baked hydrology field, guarded by pub_mtx
    std::vector<glm::vec4> climate_published;   // latest baked climate field (sst/current), guarded by pub_mtx

    uint32_t res = 0;
    // Live-synced from the UI slider (main thread stores + sets structure_dirty;
    // worker re-reads it at every structure rebuild).
    std::atomic<float> sea_level{800.0f};

    ~HydroAsync() {
        stop.store(true, std::memory_order_release);
        if (worker.joinable()) worker.join();
    }
};

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
    // Built by tick, written to the UBO by globe_render via vkCmdUpdateBuffer
    // — a host memcpy raced the previous in-flight frame's shader reads.
    CameraData        camera_frame_data{};

    // --- Stamp buffers ---
    VkBuffer          stamp_buf       = VK_NULL_HANDLE;
    VmaAllocation     stamp_buf_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo stamp_buf_info{};


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

    // --- Hydrology: coarse global drainage field (6-layer RGBA32F, GLOBE_HYDRO_RES²) ---
    VkImage       hydrology_img   = VK_NULL_HANDLE;
    VmaAllocation hydrology_alloc = VK_NULL_HANDLE;
    VkImageView   hydrology_view  = VK_NULL_HANDLE;
    // Persistent per-in-flight-frame staging for hydro+climate field uploads
    // (slot layout: [hydro][climate], HYDRO_FIELD_BYTES each). tick snapshots
    // the published fields and sets the pending flags; globe_render memcpys
    // into this frame's slot and records the copies into the frame cmd buffer.
    VkBuffer          hydro_staging_buf[FRAMES_IN_FLIGHT]   = {};
    VmaAllocation     hydro_staging_alloc[FRAMES_IN_FLIGHT] = {};
    VmaAllocationInfo hydro_staging_info[FRAMES_IN_FLIGHT]  = {};
    bool              hydro_upload_pending   = false;
    bool              climate_upload_pending = false;
    std::unique_ptr<HydroAsync> hydro;          // persistent live-water worker (see HydroAsync)
    uint32_t      hydro_solved_stamp_count = 0; // stamp count the structure is based on
    double        hydro_last_upload = 0.0;      // throttle for uploading published fields
    // Latest published hydrology field retained on the CPU so the ecosystem can
    // sample it (moisture/river/lake) by sphere direction — see sample_planet_field.
    std::vector<glm::vec4> hydro_field_cpu;
    uint32_t      hydro_field_res = 0;
    // Per-face river presence blocks (6 * RIVER_MASK_BLOCKS^2, rebuilt each
    // field upload) — lets the river overlay skip tiles with no channels.
    std::vector<uint8_t>   river_block_mask;
    bool          env_bridge_logged = false;  // one-shot Phase-1 acceptance log fired

    // --- Climate: coarse ocean circulation field (6-layer RGBA32F, GLOBE_HYDRO_RES²) ---
    // r=sst, g=sst_base(reserved for humidity), b=current angle, a=current speed.
    VkImage       climate_img   = VK_NULL_HANDLE;
    VmaAllocation climate_alloc = VK_NULL_HANDLE;
    VkImageView   climate_view  = VK_NULL_HANDLE;
    std::vector<glm::vec4> climate_field_cpu;

    // --- Sand particle buffer ---
    VkBuffer      sand_particle_buf   = VK_NULL_HANDLE;
    VmaAllocation sand_particle_alloc = VK_NULL_HANDLE;

    // --- HDR scene target (RGBA16F; scene renders here, tonemap writes the
    //     swapchain). Recreated when the swapchain extent changes. ---
    VkImage       hdr_img    = VK_NULL_HANDLE;
    VmaAllocation hdr_alloc  = VK_NULL_HANDLE;
    VkImageView   hdr_view   = VK_NULL_HANDLE;
    VkExtent2D    hdr_extent{0, 0};

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
    VkDescriptorSet river_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet sky_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet tonemap_desc_set = VK_NULL_HANDLE;

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
    // Last time each disturbed tile was actively driven (brush pulse or
    // cross-edge flow). Once a tile goes GLOBE_SWE_QUIESCE_S untouched, its
    // water is baked back into the hydrology field and it is un-disturbed.
    std::unordered_map<QuadNode, double, QuadNodeHash> disturbed_touch;
    // When each disturbed tile was last brushed (not merely flow-touched).
    // Edge flags may extend disturbance only GLOBE_SWE_DISTURB_MAX_S past this.
    std::unordered_map<QuadNode, double, QuadNodeHash> disturbed_since;
    std::unordered_set<QuadNode, QuadNodeHash> pending_init;
    // Resident tiles whose SWE water should be re-seeded from the hydrology
    // field (queued on every field publish; skips disturbed tiles so live SWE
    // dynamics are never overwritten). Init-only — no terrain regeneration.
    std::unordered_set<QuadNode, QuadNodeHash> pending_water_reseed;
    uint32_t last_processed_stamp_count = 0;
    uint32_t terrain_gen_stamp_count    = 0;

    // --- Terrain stamps (persistent brush edits) ---
    std::vector<TerrainStamp> stamps;
    bool stamps_dirty = false;

    // --- Camera + input + UI ---
    Camera     camera{};
    UIState    ui{};
    // Persistent brush mode (set by the 1/2/3/4 keys via InputFrame). All other
    // input is read transiently from the per-frame InputFrame; see globe_tick.
    BrushMode  brush_mode = BrushMode::Water;
    // Space requests a water pulse; set in tick (edge), consumed in render.
    bool       pulse_pending = false;

    bool   cursor_on_world        = false;
    double terrain_height_at_cam  = 0.0;
    double altitude_above_terrain = 100000.0;
    float  accumulated_atmo_time  = 0.0f;

    // --- Perf diagnostics (stderr spike attribution + 5 s summaries) ---
    double perf_report_t      = 0.0;
    int    perf_frames        = 0;
    double perf_dt_sum        = 0.0;
    float  perf_dt_max        = 0.0f;
    int    perf_uploads       = 0;   // hydro/climate field uploads this window
    double perf_upload_ms_sum = 0.0;
    double perf_upload_ms_max = 0.0;
    int    perf_bakebacks     = 0;
    double perf_bakeback_ms_max = 0.0;
    // What the PREVIOUS frame did — read when this frame's dt spikes.
    float  perf_last_upload_ms   = 0.0f;
    float  perf_last_bakeback_ms = 0.0f;
    int    perf_last_tilegen     = 0;
    int    perf_last_swe_init    = 0;

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
    uint32_t sand_emit_offset    = 0;
    double last_stamp_time       = 0.0;
    double last_water_stamp_time = 0.0;
    int    current_cursor_mode   = -1;
};

// Module API ---------------------------------------------------------------

// Planet-scale ecosystem environment (docs/PLANETARY_ECOSYSTEM.md Phase 1).
// Builds a bestiary::EnvironmentField over a local tangent frame at
// `center_dir` (unit sphere direction): sample(x, z) maps x metres east /
// z metres north of the center to watershed moisture from the live hydrology
// field (standing water forces moisture to 1) and latitude/altitude
// temperature. Snapshots the field + stamps, so the returned closure stays
// valid and consistent while the worker publishes fresher fields.
bestiary::EnvironmentField globe_environment_field(const GlobeState& s, glm::vec3 center_dir);

void globe_init(GlobeState& s, Renderer& r);

/// Returns false if the user pressed "Back" (only when s.embedded).
bool globe_tick(GlobeState& s, Renderer& r, const InputFrame& in, float dt);

void globe_render(GlobeState& s, Renderer& r,
                  FrameData& frame, uint32_t image_index, VkExtent2D extent);

void globe_shutdown(GlobeState& s, Renderer& r);

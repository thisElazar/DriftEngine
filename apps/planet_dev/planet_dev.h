// planet_dev.h — PLANET (DEV): multi-tile world architecture sandbox.
//
// An N x N world of World-Lab-scale tiles (256 m each) STREAMED through a
// fixed pool of GPU slots: tiles activate by camera distance and by water
// reaching a resident tile's edge (edge-flag auto-anchor), with water flowing
// across tile seams and plants continuous across the world. This is the flat
// proving ground for the planet's tile machinery: water uses the SAME compute
// shader as the planet (shaders/planet_swe_step.cs.hlsl) — tile state in
// texture array layers, neighbor-slot push constants, edge-flags readback —
// so what works here is what runs on the sphere.
//
// Provides init/tick/render/shutdown so it can be driven by a standalone
// executable or the unified launcher.
#pragma once

#include "renderer.h"
#include "shared/lab_common.h"

#include "resources.h"
#include "grid_util.h"
#include "pipeline.h"          // PlanetSweStepPC, ClumpPC

#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"
#include "morphology/wildflower.h"
#include "morphology/lplant.h"
#include "environment.h"
#include "distribution.h"
#include "creature/agent.h"
#include "creature/creature_profile.h"
#include "creature/creature_mesh.h"

#include <future>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// World layout: N x N tiles of 256 m, world spans +-(N/2 * 256) m. Tile index
// t = ty*N + tx. Only a POOL of tiles is GPU-resident at a time: tile_slot[]
// maps world tile -> pool slot (array layer), slot_tile[] maps back, and a
// free list + cooldown recycle slots — the Globe's tile_slot_map pattern,
// flat. Tiles stream in by camera distance and by edge-flag auto-anchoring
// (water reaching the edge of a resident tile pulls its neighbor in).
// ---------------------------------------------------------------------------
constexpr uint32_t PD_GRID       = 256;     // cells per tile side
constexpr float    PD_DX         = 1.0f;    // meters per cell
constexpr uint32_t PD_TILES_X    = 8;
constexpr uint32_t PD_TILES_Y    = 8;
constexpr uint32_t PD_TILE_TOTAL = PD_TILES_X * PD_TILES_Y;       // world tiles
constexpr uint32_t PD_POOL       = 16;                            // resident GPU slots
constexpr float    PD_TILE_SIZE  = PD_GRID * PD_DX;               // 256 m
constexpr float    PD_WORLD_HALF = 0.5f * PD_TILES_X * PD_TILE_SIZE;  // +-1024 m

// The far field: one extra array layer holding a low-res heightmap of the
// WHOLE world, drawn for every non-resident tile so terrain never visually
// disappears — beyond the simulation frontier the world just loses water /
// plants / detail (the open-world model; on the sphere this is the LOD
// stream). 256 texels over 2048 m = 8 m cells; a far tile draw is a 32-cell
// mesh sampling its 1/8 x 1/8 sub-rect of the far layer.
constexpr uint32_t PD_LAYERS        = PD_POOL + 1;   // pool slots + far layer
constexpr uint32_t PD_FAR_LAYER     = PD_POOL;
constexpr float    PD_FAR_CELL      = PD_TILE_SIZE * PD_TILES_X / PD_GRID;  // 8 m
constexpr uint32_t PD_FAR_TILE_GRID = PD_GRID / 8;   // far mesh cells per tile

constexpr uint32_t PD_NO_NEIGHBOR = 0xFFFFFFFFu;   // SWE shader: reflective wall
constexpr uint32_t PD_NO_SLOT     = 0xFFFFFFFFu;   // tile not GPU-resident
constexpr uint32_t PD_NO_TILE     = 0xFFFFFFFFu;   // slot unassigned

// World-grid neighbor tile in the SWE shader's L/R/D/U order (D/U follow
// grid +y == world +z); PD_NO_TILE at the world boundary.
inline uint32_t pd_neighbor_tile(uint32_t t, int dir)
{
    uint32_t tx = t % PD_TILES_X, ty = t / PD_TILES_X;
    switch (dir) {
    case 0: return (tx > 0)               ? t - 1          : PD_NO_TILE;
    case 1: return (tx + 1 < PD_TILES_X)  ? t + 1          : PD_NO_TILE;
    case 2: return (ty > 0)               ? t - PD_TILES_X : PD_NO_TILE;
    default:return (ty + 1 < PD_TILES_Y)  ? t + PD_TILES_X : PD_NO_TILE;
    }
}

inline float pd_tile_origin_x(uint32_t t) { return -PD_WORLD_HALF + static_cast<float>(t % PD_TILES_X) * PD_TILE_SIZE; }
inline float pd_tile_origin_z(uint32_t t) { return -PD_WORLD_HALF + static_cast<float>(t / PD_TILES_X) * PD_TILE_SIZE; }

// Tile containing a world position (clamped to the world).
inline uint32_t pd_tile_of(float wx, float wz)
{
    int tx = static_cast<int>((wx + PD_WORLD_HALF) / PD_TILE_SIZE);
    int ty = static_cast<int>((wz + PD_WORLD_HALF) / PD_TILE_SIZE);
    tx = std::clamp(tx, 0, static_cast<int>(PD_TILES_X) - 1);
    ty = std::clamp(ty, 0, static_cast<int>(PD_TILES_Y) - 1);
    return static_cast<uint32_t>(ty) * PD_TILES_X + static_cast<uint32_t>(tx);
}

// ---------------------------------------------------------------------------
// Push constant for tiled terrain rendering (128 bytes, the guaranteed max).
// ---------------------------------------------------------------------------
struct PlanetDevTerrainPC {
    glm::mat4 mvp;
    float     grid_w_f;
    float     grid_h_f;
    float     cell_size;
    float     sea_level;
    float     brush_x;          // cursor-tile grid coords
    float     brush_y;
    float     brush_radius;     // grid cells
    float     brush_active;     // only 1 on the cursor tile's draw
    float     moisture_overlay;
    float     tile_origin_x;    // world-space tile corner
    float     tile_origin_z;
    float     layer;            // texture array layer (pool slot or far layer)
    float     seam_highlight;
    float     uv_off_x;         // sub-rect of the layer this draw samples:
    float     uv_off_y;         //   uv' = mesh_uv * uv_scale + uv_off
    float     uv_scale;         // 1/offset 0 for pool slots; 1/8 for far tiles
};
static_assert(sizeof(PlanetDevTerrainPC) == 128, "terrain PC must fit min push range");

// Push constant for the array-layer terrain brush (planet_dev_brush.cs.hlsl).
struct PD_BrushPC {
    float    brush_x;       // tile-local grid coords (may be off-tile)
    float    brush_y;
    float    brush_radius;
    float    brush_amount;
    uint32_t grid_w;
    uint32_t grid_h;
    uint32_t layer;
    uint32_t _pad0;
};

enum class PD_BrushMode { Raise, Lower, Water };

// ---------------------------------------------------------------------------
// Module-local GPU helper types
// ---------------------------------------------------------------------------
struct PD_ArrayImage {              // 2D array image, PD_POOL layers
    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkImageView   view  = VK_NULL_HANDLE;   // array view over all layers
};

struct PD_TerrainPipeline {
    VkShaderModule        vs       = VK_NULL_HANDLE;
    VkShaderModule        fs       = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl      = VK_NULL_HANDLE;
    VkPipelineLayout      layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
};

struct PD_ClumpPipeline {
    VkShaderModule   vs       = VK_NULL_HANDLE;
    VkShaderModule   fs       = VK_NULL_HANDLE;
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
};

struct PD_PlantMesh {
    GpuBuffer vbo{};
    GpuBuffer ibo{};
    uint32_t  index_count = 0;
};

// Streaming GPU work staged by tick, recorded into the FRAME command buffer
// by render (no oneshot fence waits — staging buffers ride the retire queue).
// `layer` is the array layer (a pool slot, or PD_FAR_LAYER for terrain-only
// far-field rebuilds). Any null buffer is skipped.
struct PD_PendingUpload {
    uint32_t  layer = 0;
    bool      clear_state = false;   // clear SWE A/B/output layers (activation)
    GpuBuffer terrain{};             // R32F heightmap
    GpuBuffer water{};               // RGBA16F depth restore (h in r, rest 0)
    GpuBuffer moisture{};            // R32F moisture slice
};

// ---------------------------------------------------------------------------
// All module state
// ---------------------------------------------------------------------------
struct PlanetDevState {
    bool embedded    = false;
    bool initialized = false;

    // --- Terrain + water pools (one array layer per tile) ---
    PD_ArrayImage terrain_pool{};    // R32_SFLOAT
    PD_ArrayImage swe_state_a{};     // R16G16B16A16_SFLOAT (h, hu, hv, foam)
    PD_ArrayImage swe_state_b{};
    PD_ArrayImage swe_output{};      // render-facing (r = depth, a = foam)
    PD_ArrayImage moisture_pool{};   // R32_SFLOAT

    // --- Edge flags (one uint per pool slot, host-visible; drives streaming) ---
    VkBuffer          edge_flags_buf   = VK_NULL_HANDLE;
    VmaAllocation     edge_flags_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo edge_flags_info{};

    // --- CPU mirrors (ALL world tiles, persistent across streaming) ---
    std::vector<std::vector<float>> hm_cpu;   // heightmaps; brush edits survive stream-out
    std::vector<float> moisture_slice;        // scratch: per-slot GPU upload slice
    // Seam-free WORLD grids (PD_TILES_X*PD_GRID squared), persistent: resident
    // tiles refresh their region from GPU readbacks; deactivating a tile saves
    // its water depth here, activation restores it. The capillary blur runs
    // over the resident tiles' bounding box so moisture crosses seams exactly
    // like the water does.
    std::vector<float> water_world;
    std::vector<float> moisture_world;

    // --- Tile streaming (Globe tile_slot_map pattern, flat) ---
    uint32_t tile_slot[PD_TILE_TOTAL];   // world tile -> pool slot or PD_NO_SLOT
    uint32_t slot_tile[PD_POOL];         // pool slot -> world tile or PD_NO_TILE
    std::vector<uint32_t> free_slots;
    // Freed slots wait FRAMES_IN_FLIGHT frames before reuse so no in-flight
    // frame can sample a layer while its new tenant uploads.
    std::vector<std::pair<uint32_t, uint64_t>> slot_cooldown;
    float ui_stream_radius = 400.0f;     // activation distance from camera target
    std::vector<PD_PendingUpload> pending_uploads;
    bool  far_dirty = false;             // far layer stale (brush/deactivation)
    float far_rebuild_timer = 0.0f;

    // Plant placement runs on worker threads against a SNAPSHOT of the tile's
    // environment (placement is ~350 ms of poisson fill — way too slow for the
    // frame loop). Results land in the population when the future completes;
    // a job whose tile streamed back out is dropped.
    struct PlacementJob {
        uint32_t tile;
        double   queued_at;
        std::future<std::vector<bestiary::PlantInstance>> fut;
    };
    std::vector<PlacementJob> placement_jobs;

    // --- Sampler ---
    VkSampler sampler = VK_NULL_HANDLE;

    // --- Pipelines ---
    ComputePipeline    pipe_swe_step{};   // shaders/planet_swe_step_cs.spv (shared with the planet)
    ComputePipeline    pipe_brush{};      // array-layer terrain brush
    PD_TerrainPipeline pipe_terrain{};
    PD_ClumpPipeline   pipe_clump{};

    // --- Descriptors ---
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet  ds_step[2]{};       // ping/pong; tile + neighbors ride push constants
    VkDescriptorSet  ds_brush = VK_NULL_HANDLE;
    VkDescriptorSet  ds_terrain = VK_NULL_HANDLE;

    // --- Terrain meshes (full-res shared by resident draws; low-res far) ---
    StaticGridMesh terrain_mesh{};
    StaticGridMesh far_mesh{};

    // --- Camera ---
    OrbitCamera camera{};

    // --- Plants (ONE world-spanning population; world coords) ---
    std::vector<bestiary::PlantSpecies>  plant_roster;
    std::vector<PD_PlantMesh>            plant_canonical;
    std::vector<GpuBuffer>               plant_inst;
    std::vector<uint32_t>                plant_inst_count;
    std::vector<bestiary::PlantInstance> plant_population;
    bestiary::EnvironmentField           persistent_env;
    float    plant_growth_rate = 0.15f;
    float    plant_decay_rate  = 0.03f;
    uint32_t sprout_seed       = 100;

    // House species params (lab-saved world_*.toml profiles)
    bestiary::ClumpParams          clump_params{};
    bestiary::ClumpExpression      clump_expr{};
    bestiary::BushParams           bush_params{};
    bestiary::BushExpression       bush_expr{};
    bestiary::TreeParams           tree_params{};
    bestiary::TreeExpression       tree_expr{};
    bestiary::ClumpParams          reed_params{};
    bestiary::ClumpExpression      reed_expr{};
    bestiary::WildflowerParams     wildflower_params{};
    bestiary::WildflowerExpression wildflower_expr{};
    bestiary::LPlantParams         lplant_params{};
    bestiary::LPlantExpression     lplant_expr{};
    bestiary::EcosystemParams      eco_params{};

    // --- Creatures (world-spanning agents; seams don't exist for them) ---
    std::vector<bestiary::Agent>           agents;
    std::vector<bestiary::CreatureProfile> creature_profiles;
    std::vector<std::string>               creature_names;
    PD_PlantMesh                           creature_mesh_gpu{};
    PD_ClumpPipeline                       pipe_creature{};
    bool     ui_creatures_enabled = true;
    int      ui_creature_count    = 24;
    float    ui_creature_speed    = 1.0f;
    int      respawn_pending      = 0;
    uint32_t creature_tick        = 0;

    // --- Cursor pick (world plane) ---
    bool     cursor_valid = false;
    float    cursor_wx = 0.0f, cursor_wz = 0.0f;
    uint32_t cursor_tile = 0;
    float    cursor_gx = 0.0f, cursor_gy = 0.0f;   // cursor-tile grid coords

    // --- UI / sim controls ---
    float ui_gravity        = 9.81f;
    float ui_friction       = 0.05f;
    float ui_damping        = 0.02f;
    int   ui_swe_substeps   = 4;
    float ui_swe_dt         = 0.05f;
    bool  ui_paused         = false;
    int   ui_brush_radius   = 12;       // grid cells
    float ui_brush_strength = 1.5f;     // water (m/s of column)
    float ui_terrain_strength = 4.0f;   // terrain (m/s of height)
    PD_BrushMode brush_mode = PD_BrushMode::Water;
    float ui_capillary_depth = 0.05f;
    int   ui_capillary_blur  = 4;
    bool  ui_show_plants    = true;
    bool  ui_show_moisture  = false;
    bool  ui_seam_highlight = true;
    int   replant_pending   = 0;
    uint32_t edge_flags_ui[PD_POOL]{};   // per-SLOT, latched before the frame clears them

    // --- Deferred buffer destruction (Globe veg retire-queue pattern) ---
    // Buffers replaced mid-flight (creature mesh, plant instances) are parked
    // here with the frame they were retired on and destroyed once no in-flight
    // frame can still reference them — no vkDeviceWaitIdle needed.
    std::vector<std::pair<GpuBuffer, uint64_t>> retire;
    uint64_t frame_counter = 0;

    // --- Frame state ---
    int   swe_ping_pong       = 0;
    bool  brushing            = false;
    float veg_rebuild_timer   = 0.0f;
    float auto_replant_timer  = 0.0f;
    float accumulated_time    = 0.0f;
    float last_dt             = 0.0f;
};

// Module API ---------------------------------------------------------------

void planet_dev_init(PlanetDevState& s, Renderer& r);

/// Returns false if the user pressed "Back" (only when s.embedded).
bool planet_dev_tick(PlanetDevState& s, Renderer& r, const InputFrame& in, float dt);

void planet_dev_render(PlanetDevState& s, Renderer& r,
                       FrameData& frame, uint32_t image_index, VkExtent2D extent);

void planet_dev_shutdown(PlanetDevState& s, Renderer& r);

// planet_dev.h — PLANET (DEV): multi-tile world architecture sandbox.
//
// A 2x2 grid of World-Lab-scale tiles (256 m each) with water flowing across
// tile seams and plants continuous across the world. This is the flat proving
// ground for the planet's tile machinery: water uses the SAME compute shader
// as the planet (shaders/planet_swe_step.cs.hlsl) — tile state in texture
// array layers, neighbor-slot push constants, edge-flags readback — so what
// works here is what runs on the sphere. Creatures land in a later increment.
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

// ---------------------------------------------------------------------------
// World layout: 2x2 tiles, 256 m each, world spans [-256, +256]^2.
// Tile index t = ty*2 + tx (t0 = -x/-z, t1 = +x/-z, t2 = -x/+z, t3 = +x/+z).
// ---------------------------------------------------------------------------
constexpr uint32_t PD_GRID       = 256;     // cells per tile side
constexpr float    PD_DX         = 1.0f;    // meters per cell
constexpr uint32_t PD_TILES_X    = 2;
constexpr uint32_t PD_TILES_Y    = 2;
constexpr uint32_t PD_TILE_COUNT = PD_TILES_X * PD_TILES_Y;
constexpr float    PD_TILE_SIZE  = PD_GRID * PD_DX;          // 256 m
constexpr float    PD_WORLD_HALF = PD_TILE_SIZE;             // 2x2 -> +-256 m

constexpr uint32_t PD_NO_NEIGHBOR = 0xFFFFFFFFu;

// Neighbor slot table per tile, in the SWE shader's L/R/D/U order (D/U follow
// grid +y == world +z). Edges of the world stay PD_NO_NEIGHBOR = reflective.
constexpr uint32_t PD_NEIGHBORS[PD_TILE_COUNT][4] = {
    {PD_NO_NEIGHBOR, 1u,             PD_NO_NEIGHBOR, 2u            },  // t0
    {0u,             PD_NO_NEIGHBOR, PD_NO_NEIGHBOR, 3u            },  // t1
    {PD_NO_NEIGHBOR, 3u,             0u,             PD_NO_NEIGHBOR},  // t2
    {2u,             PD_NO_NEIGHBOR, 1u,             PD_NO_NEIGHBOR},  // t3
};

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
    float     layer;            // texture array layer (= tile index)
    float     seam_highlight;
    float     _pad0, _pad1, _pad2;
};
static_assert(sizeof(PlanetDevTerrainPC) == 128, "terrain PC must fit min push range");

// ---------------------------------------------------------------------------
// Module-local GPU helper types
// ---------------------------------------------------------------------------
struct PD_ArrayImage {              // 2D array image, PD_TILE_COUNT layers
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

    // --- Edge flags (one uint per tile, host-visible; debug readout only) ---
    VkBuffer          edge_flags_buf   = VK_NULL_HANDLE;
    VmaAllocation     edge_flags_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo edge_flags_info{};

    // --- CPU mirrors ---
    std::vector<float> hm_cpu[PD_TILE_COUNT];
    std::vector<float> moisture_cpu[PD_TILE_COUNT];   // GPU-upload slices of the world grid
    // Seam-free WORLD grids (PD_GRID*2 squared): the per-tile water readbacks
    // are stitched into one grid, the capillary blur runs globally (so moisture
    // crosses seams exactly like the water does), then sliced back per tile.
    std::vector<float> water_world;
    std::vector<float> moisture_world;

    // --- Sampler ---
    VkSampler sampler = VK_NULL_HANDLE;

    // --- Pipelines ---
    ComputePipeline    pipe_swe_step{};   // shaders/planet_swe_step_cs.spv (shared with the planet)
    PD_TerrainPipeline pipe_terrain{};
    PD_ClumpPipeline   pipe_clump{};

    // --- Descriptors ---
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet  ds_step[2]{};       // ping/pong; tile + neighbors ride push constants
    VkDescriptorSet  ds_terrain = VK_NULL_HANDLE;

    // --- Terrain mesh (shared by all 4 tile draws) ---
    StaticGridMesh terrain_mesh{};

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
    float ui_brush_strength = 1.5f;
    float ui_capillary_depth = 0.05f;
    int   ui_capillary_blur  = 4;
    bool  ui_show_plants    = true;
    bool  ui_show_moisture  = false;
    bool  ui_seam_highlight = true;
    int   replant_pending   = 0;
    uint32_t edge_flags_ui[PD_TILE_COUNT]{};   // latched before the frame clears them

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

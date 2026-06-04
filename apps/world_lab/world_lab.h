// world_lab.h — embeddable World Lab module.
//
// Provides init/tick/render/shutdown so the lab can be driven by either
// a standalone executable or a unified launcher.
#pragma once

#include "renderer.h"
#include "shared/lab_common.h"

#include "resources.h"
#include "grid_util.h"
#include "pipeline.h"

#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"
#include "morphology/wildflower.h"
#include "environment.h"
#include "distribution.h"
#include "creature/agent.h"
#include "creature/creature_profile.h"
#include "creature/creature_mesh.h"

// ---------------------------------------------------------------------------
// Tile constants
// ---------------------------------------------------------------------------
constexpr uint32_t WL_GRID_W   = 256;
constexpr uint32_t WL_GRID_H   = 256;
constexpr float    WL_DX       = 1.0f;        // 1 m per cell -> tile is 256m x 256m
constexpr float    WL_TILE_HALF_X = WL_GRID_W * WL_DX * 0.5f;
constexpr float    WL_TILE_HALF_Z = WL_GRID_H * WL_DX * 0.5f;

constexpr uint32_t WL_ATMO_W = WL_GRID_W;
constexpr uint32_t WL_ATMO_H = WL_GRID_H;
constexpr uint32_t WL_ATMO_D = 32;

constexpr float WL_SPRING_GX   = 70.0f;
constexpr float WL_SPRING_GY   = 80.0f;
constexpr float WL_SPRING_RATE = 0.15f;
constexpr float WL_POOL_GX     = 140.0f;
constexpr float WL_POOL_GY     = 150.0f;

// ---------------------------------------------------------------------------
// Push constant for world terrain rendering
// ---------------------------------------------------------------------------
struct WorldTerrainPC {
    glm::mat4 mvp;
    float     grid_w_f;
    float     grid_h_f;
    float     cell_size;
    float     sea_level;
    float     brush_x;
    float     brush_y;
    float     brush_radius;
    float     brush_active;
    float     moisture_overlay; // 0 = normal terrain, 1 = moisture heatmap
    float     _pad0;
    float     _pad1;
    float     _pad2;
};

// ---------------------------------------------------------------------------
// Camera UBO for cloud raymarch
// ---------------------------------------------------------------------------
struct WL_CameraUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 sun_dir;   float _p0;
    glm::vec3 sun_color; float _p1;
    glm::vec3 cam_pos;   float _p2;
    glm::vec4 brush_world;
    glm::vec4 brush_color;
    glm::mat4 inv_view_proj;
};

// ---------------------------------------------------------------------------
// Local pipeline types (world_lab uses different shaders from plant_lab)
// ---------------------------------------------------------------------------
struct WL_TerrainPipeline {
    VkShaderModule        vs       = VK_NULL_HANDLE;
    VkShaderModule        fs       = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl      = VK_NULL_HANDLE;
    VkPipelineLayout      layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
};

struct WL_ClumpPipeline {
    VkShaderModule   vs       = VK_NULL_HANDLE;
    VkShaderModule   fs       = VK_NULL_HANDLE;
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
};

struct WL_CloudPipeline {
    VkShaderModule        vs       = VK_NULL_HANDLE;
    VkShaderModule        fs       = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl      = VK_NULL_HANDLE;
    VkPipelineLayout      layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
};

// ---------------------------------------------------------------------------
// Plant mesh (rebuilt on replant/growth)
// ---------------------------------------------------------------------------
struct WL_PlantMesh {
    GpuBuffer vbo{};
    GpuBuffer ibo{};
    uint32_t  index_count  = 0;
    uint32_t  vertex_count = 0;
};

// ---------------------------------------------------------------------------
// Population graph ring buffers
// ---------------------------------------------------------------------------
static constexpr int WL_GRAPH_HISTORY       = 300;
static constexpr float WL_GRAPH_SAMPLE_INTERVAL = 0.5f;

struct WL_GraphState {
    float sample_timer = 0.0f;
    int   write_idx    = 0;
    int   count        = 0;
    float pop_sprinter[WL_GRAPH_HISTORY] = {};
    float pop_grazer[WL_GRAPH_HISTORY]   = {};
    float pop_browser[WL_GRAPH_HISTORY]  = {};
    float pop_wolf[WL_GRAPH_HISTORY]     = {};
    float pop_rabbit[WL_GRAPH_HISTORY]   = {};
    float pop_bird[WL_GRAPH_HISTORY]     = {};
    float pop_raptor[WL_GRAPH_HISTORY]   = {};
    float pop_snake[WL_GRAPH_HISTORY]    = {};
    float pop_total[WL_GRAPH_HISTORY]    = {};
    float energy_avg[WL_GRAPH_HISTORY]   = {};
    float energy_min[WL_GRAPH_HISTORY]   = {};
    float energy_max[WL_GRAPH_HISTORY]   = {};
};

// ---------------------------------------------------------------------------
// All world-lab state, bundled into a single struct.
// ---------------------------------------------------------------------------
struct WorldLabState {
    bool embedded     = false;  // true when driven by launcher (shows Back button)
    bool initialized  = false;
    bool preview_mode = false;  // auto-orbit, no ImGui, simulation runs autonomously

    // --- GPU terrain / heightmap ---
    HeightmapData hm_data{};
    HeightmapGPU  hm_gpu{};
    std::vector<float> hm_cpu;

    HeightmapData moist_data{};
    HeightmapGPU  moist_gpu{};

    // --- SWE images ---
    SweImage state_a{};
    SweImage state_b{};
    SweImage water_out{};

    // --- Atmosphere 3D images ---
    SweImage atmo_state[2]{};
    SweImage wind_field[2]{};
    SweImage atmo_shadow{};
    SweImage ground_cond{};
    SweImage ground_wind{};

    // --- Sediment images ---
    SweImage sediment[2]{};

    // --- Readback buffers ---
    GpuBuffer wind_readback[FRAMES_IN_FLIGHT]{};
    GpuBuffer gcond_readback[FRAMES_IN_FLIGHT]{};

    // --- Sampler ---
    VkSampler sampler = VK_NULL_HANDLE;

    // --- Compute pipelines ---
    ComputePipeline pipe_swe_init{};
    ComputePipeline pipe_swe_step{};
    ComputePipeline pipe_brush{};
    ComputePipeline pipe_atmo{};
    ComputePipeline pipe_erosion{};

    // --- Descriptor pool + sets ---
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet  ds_swe_init = VK_NULL_HANDLE;
    VkDescriptorSet  ds_swe_step[2]{};
    VkDescriptorSet  ds_brush = VK_NULL_HANDLE;
    VkDescriptorSet  ds_atmo[2]{};
    VkDescriptorSet  ds_erosion[2]{};
    VkDescriptorSet  ds_terrain = VK_NULL_HANDLE;
    VkDescriptorSet  ds_cloud[2]{};

    // --- Graphics pipelines ---
    WL_TerrainPipeline pipe_terrain{};
    WL_ClumpPipeline   pipe_clump{};
    WL_ClumpPipeline   pipe_creature{};
    WL_CloudPipeline   pipe_cloud{};

    // --- Camera UBO ---
    VkBuffer        camera_ubo       = VK_NULL_HANDLE;
    VmaAllocation   camera_ubo_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo camera_ubo_info{};

    // --- Terrain mesh ---
    StaticGridMesh terrain_mesh{};

    // --- Plant rendering (instanced) — one entry per roster species ---
    std::vector<bestiary::PlantSpecies> plant_roster;     // built from the species library
    std::vector<WL_PlantMesh>           plant_canonical;  // canonical mesh per species
    std::vector<GpuBuffer>              plant_inst;        // per-species instance buffer
    std::vector<uint32_t>               plant_inst_count;

    // --- Camera ---
    OrbitCamera camera{};

    // --- Plant + ecosystem params ---
    bestiary::ClumpParams      clump_params{};
    bestiary::ClumpExpression  clump_expr{};
    bestiary::BushParams       bush_params{};
    bestiary::BushExpression   bush_expr{};
    bestiary::TreeParams       tree_params{};
    bestiary::TreeExpression   tree_expr{};
    bestiary::EcosystemParams  eco_params{};
    bestiary::ClumpParams           reed_params{};
    bestiary::ClumpExpression       reed_expr{};
    bestiary::WildflowerParams      wildflower_params{};
    bestiary::WildflowerExpression  wildflower_expr{};
    bestiary::LPlantParams          lplant_params{};
    bestiary::LPlantExpression      lplant_expr{};

    // --- Brush + sim controls ---
    int   ui_brush_radius_cells = 12;
    float ui_brush_amount       = 1.5f;
    bool  ui_swe_enabled        = true;
    int   ui_swe_substeps       = 4;
    float ui_swe_dt             = 0.05f;
    float ui_sim_speed          = 1.0f;
    bool  ui_paused             = false;
    float ui_gravity            = 9.81f;
    float ui_friction           = 0.05f;
    float ui_damping            = 0.02f;
    float ui_capillary_depth    = 0.05f;
    int   ui_capillary_blur     = 4;
    bool  ui_show_plants        = true;
    bool  ui_show_moisture      = false;
    int   replant_pending       = 0;

    // --- Atmosphere params ---
    bool  ui_atmo_enabled        = false;
    float ui_cloud_opacity       = 1.0f;
    float ui_orographic_lift     = 0.15f;
    float ui_adiabatic_cooling   = 0.0065f;
    float ui_rain_shadow         = 0.3f;
    float ui_k_pressure          = 1.0f;
    float ui_wind_strength       = 1.0f;
    float ui_k_evaporation       = 0.3f;
    float ui_cloud_base          = -10.0f;
    float ui_layer_height        = 10.0f;
    bool  ui_atmo_reset          = false;

    // --- Erosion params ---
    bool  ui_erosion_enabled     = false;
    float ui_k_erosion           = 0.01f;
    float ui_k_deposit           = 0.02f;
    float ui_k_capacity          = 0.5f;
    float ui_min_slope           = 0.001f;
    float ui_k_wind              = 0.05f;
    float ui_k_thermal           = 0.02f;
    float ui_wind_threshold      = 1.0f;

    // --- Water physics ---
    float ui_k_rain              = 0.0f;

    // --- Ping-pong state ---
    int   atmo_ping_pong  = 0;
    int   ero_ping_pong   = 0;
    bool  atmo_force_init = true;
    int   swe_ping_pong   = 0;
    int   wt_water_pulse_active = 0;

    // --- Debug readback ---
    float debug_pressure_min = 0, debug_pressure_mean = 0, debug_pressure_max = 0;
    float debug_wind_speed_max = 0;
    float debug_precip_max = 0, debug_precip_mean = 0;
    float debug_temp_mean = 0, debug_humidity_mean = 0;
    float debug_gc_wind_mean = 0, debug_gc_wind_max = 0;

    // --- Input state ---
    bool brushing            = false;
    bool brushed_this_stroke = false;
    bool prev_lmb            = false;
    bool prev_rain_key       = false;

    // --- Creature state ---
    std::vector<bestiary::Agent> agents;
    std::vector<bestiary::CreatureProfile> creature_profiles;
    WL_PlantMesh creature_mesh_gpu{};
    std::vector<float> persistent_water_depth;
    bestiary::EnvironmentField persistent_env;
    bool  ui_creatures_enabled = true;
    int   ui_creature_count    = 20;
    float ui_creature_speed    = 1.0f;
    int   ui_follow_agent      = -1;   // -1 = free camera; >= 0 = following agent
    float veg_rebuild_timer    = 0.0f;
    uint32_t creature_tick     = 0;
    int   ui_species_sel       = 0;

    // --- Plant population (persistent individuals) ---
    std::vector<bestiary::PlantInstance> plant_population;
    float plant_growth_rate = 0.15f;
    float plant_decay_rate  = 0.03f;
    uint32_t sprout_seed    = 100;

    // --- Autorun state ---
    bool  ui_autorun          = true;
    bool  ui_spring_enabled   = true;
    float ui_spring_rate      = WL_SPRING_RATE;
    float auto_replant_timer  = 0.0f;
    bool  initial_spawn_done  = false;

    // --- Population graph ---
    WL_GraphState graph{};

    // --- Timing ---
    float accumulated_time = 0.0f;
    float last_dt          = 0.0f;  // cached from tick for render
};

// Module API ---------------------------------------------------------------

/// Create pipelines, allocate images, build terrain mesh, init creature profiles.
void world_lab_init(WorldLabState& s, Renderer& r);

/// Poll camera, run simulation, handle input, draw ImGui panel.
/// Returns false if the user pressed "Back" (only possible when s.embedded).
bool world_lab_tick(WorldLabState& s, Renderer& r, const InputFrame& in, float dt);

/// Record the command buffer: compute dispatches, terrain draw, plant draw,
/// creature draw, cloud draw, ImGui draw, present barrier.
void world_lab_render(WorldLabState& s, Renderer& r,
                      FrameData& frame, uint32_t image_index, VkExtent2D extent);

/// Destroy all pipelines, images, buffers.  Must be called before renderer_shutdown.
void world_lab_shutdown(WorldLabState& s, Renderer& r);

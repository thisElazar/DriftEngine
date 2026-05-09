#pragma once

#include "vk_common.h"
#include <cstdint>
#include <vector>

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

struct Pipelines {
    VkShaderModule swe_init_shader = VK_NULL_HANDLE;
    VkShaderModule swe_step_shader = VK_NULL_HANDLE;
    VkShaderModule terrain_vs = VK_NULL_HANDLE;
    VkShaderModule terrain_fs = VK_NULL_HANDLE;
    VkShaderModule water_vs = VK_NULL_HANDLE;
    VkShaderModule water_fs = VK_NULL_HANDLE;
    VkShaderModule terrain_brush_shader = VK_NULL_HANDLE;
    VkShaderModule erosion_shader = VK_NULL_HANDLE;
    VkShaderModule atmo_shader = VK_NULL_HANDLE;
    VkShaderModule raymarch_vs = VK_NULL_HANDLE;
    VkShaderModule raymarch_fs = VK_NULL_HANDLE;
    VkShaderModule sand_sim_shader = VK_NULL_HANDLE;
    VkShaderModule sand_render_vs = VK_NULL_HANDLE;
    VkShaderModule sand_render_fs = VK_NULL_HANDLE;
    VkShaderModule terrain_gen_shader = VK_NULL_HANDLE;

    VkDescriptorSetLayout swe_init_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout swe_step_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout gfx_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout tb_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout ero_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout atmo_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout terrain_gen_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout sand_sim_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout sand_render_desc_layout = VK_NULL_HANDLE;

    VkPipelineLayout swe_init_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout swe_step_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout gfx_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout raymarch_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout clipmap_gfx_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout tb_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout ero_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout terrain_gen_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout atmo_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout sand_sim_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout sand_render_pipeline_layout = VK_NULL_HANDLE;

    VkPipeline swe_init_pipeline = VK_NULL_HANDLE;
    VkPipeline swe_step_pipeline = VK_NULL_HANDLE;
    VkPipeline gfx_pipeline = VK_NULL_HANDLE;
    VkPipeline clipmap_terrain_pipeline = VK_NULL_HANDLE;
    VkPipeline water_pipeline = VK_NULL_HANDLE;
    VkPipeline raymarch_pipeline = VK_NULL_HANDLE;
    VkPipeline terrain_brush_pipeline = VK_NULL_HANDLE;
    VkPipeline erosion_pipeline = VK_NULL_HANDLE;
    VkPipeline terrain_gen_pipeline = VK_NULL_HANDLE;
    VkPipeline atmo_pipeline = VK_NULL_HANDLE;
    VkPipeline sand_sim_pipeline = VK_NULL_HANDLE;
    VkPipeline sand_render_pipeline = VK_NULL_HANDLE;
};

std::vector<uint32_t> load_spirv(const char* path);
void pipelines_create(Pipelines& p, VkDevice device);
void pipelines_reload(Pipelines& p, VkDevice device);
void pipelines_destroy(Pipelines& p, VkDevice device);

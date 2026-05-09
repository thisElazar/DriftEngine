#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct TerrainStamp;

float cpu_hash31(glm::vec3 p);
float cpu_gradient_noise_3d(glm::vec3 p);
float cpu_fbm3d(glm::vec3 p, int octaves, float lacunarity, float gain);
float cpu_ridged3d(glm::vec3 p, int octaves);
float cpu_terrain_height(glm::vec3 sphere_dir);
float cpu_terrain_height_with_stamps(glm::vec3 sphere_dir,
                                     const TerrainStamp* stamps, uint32_t count);

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

std::vector<float> generate_crater_basin(const BasinParams& p);

#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct TerrainStamp;

// THE planet seed — the single value behind both terrain paths. The GPU
// (planet_gen.cs.hlsl) receives it as a push constant; the CPU mirror
// (cpu_terrain_height) folds it into its noise offsets. Define it once here
// so changing the seed can never silently break CPU/GPU lockstep (camera
// collision, cursor picking, hydrology, and water heights all sample the CPU
// side and must see the same planet the GPU renders).
constexpr uint32_t PLANET_SEED = 42;   // one seed -> one planet (M3: many)

float cpu_hash31(glm::vec3 p);
float cpu_gradient_noise_3d(glm::vec3 p);
float cpu_fbm3d(glm::vec3 p, int octaves, float lacunarity, float gain);
float cpu_ridged3d(glm::vec3 p, int octaves);
float cpu_terrain_height(glm::vec3 sphere_dir);
float cpu_terrain_height_with_stamps(glm::vec3 sphere_dir,
                                     const TerrainStamp* stamps, uint32_t count);

// CPU mirror of shaders/planet_climate.hlsli — must stay in lockstep with it,
// same rule as cpu_terrain_height / planet_gen.cs.hlsl. temperature: 0 polar ..
// 1 equatorial at sea level; moisture: 0 arid .. 1 wet (climate provinces, NOT
// the watershed field). Biome weights: x=desert, y=grass, z=forest, w=tundra.
struct CpuPlanetClimate {
    float temperature = 0.5f;
    float moisture    = 0.5f;
};
CpuPlanetClimate cpu_planet_climate(glm::vec3 sphere_dir, float seed_f);
glm::vec4 cpu_planet_biome_weights(CpuPlanetClimate c);

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

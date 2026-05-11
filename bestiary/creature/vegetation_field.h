#pragma once

#include <cstdint>
#include <vector>

namespace bestiary {

struct EnvironmentField;

struct VegetationDensityField {
    std::vector<float> density;
    std::vector<float> capacity;
    uint32_t grid_w      = 0;
    uint32_t grid_h      = 0;
    float    cell_size    = 1.0f;
    float    tile_half_x  = 0.0f;
    float    tile_half_z  = 0.0f;
    float    regrowth_rate = 0.02f;
    float    decay_rate    = 0.01f;
};

void init_vegetation_field(VegetationDensityField& field,
                           uint32_t grid_w, uint32_t grid_h,
                           float cell_size,
                           float tile_half_x, float tile_half_z);

void accumulate_vegetation_capacity(VegetationDensityField& field,
                                    float x, float z,
                                    float amount);

void fill_vegetation_density(VegetationDensityField& field);

float sample_vegetation_density(const VegetationDensityField& field,
                                float wx, float wz);

float consume_vegetation(VegetationDensityField& field,
                         float wx, float wz, float amount);

void tick_vegetation(VegetationDensityField& field, float dt);

void update_capacity_from_environment(VegetationDensityField& field,
                                      const EnvironmentField& env);

float sample_vegetation_fullness(const VegetationDensityField& field,
                                 float wx, float wz);

} // namespace bestiary

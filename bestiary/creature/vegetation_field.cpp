#include "vegetation_field.h"
#include "../environment.h"

#include <algorithm>
#include <cmath>

namespace bestiary {

void init_vegetation_field(VegetationDensityField& field,
                           uint32_t grid_w, uint32_t grid_h,
                           float cell_size,
                           float tile_half_x, float tile_half_z)
{
    field.grid_w     = grid_w;
    field.grid_h     = grid_h;
    field.cell_size  = cell_size;
    field.tile_half_x = tile_half_x;
    field.tile_half_z = tile_half_z;

    size_t n = static_cast<size_t>(grid_w) * grid_h;
    field.density.assign(n, 0.0f);
    field.capacity.assign(n, 0.0f);
}

static void world_to_grid(const VegetationDensityField& field,
                           float wx, float wz,
                           int& gx, int& gz)
{
    gx = static_cast<int>(std::floor((wx + field.tile_half_x) / field.cell_size));
    gz = static_cast<int>(std::floor((wz + field.tile_half_z) / field.cell_size));
    gx = std::clamp(gx, 0, static_cast<int>(field.grid_w) - 1);
    gz = std::clamp(gz, 0, static_cast<int>(field.grid_h) - 1);
}

void accumulate_vegetation_capacity(VegetationDensityField& field,
                                    float x, float z,
                                    float amount)
{
    int gx, gz;
    world_to_grid(field, x, z, gx, gz);
    size_t idx = static_cast<size_t>(gz) * field.grid_w + static_cast<size_t>(gx);
    field.capacity[idx] = std::min(field.capacity[idx] + amount, 1.0f);
}

void fill_vegetation_density(VegetationDensityField& field)
{
    field.density = field.capacity;
}

float sample_vegetation_density(const VegetationDensityField& field,
                                float wx, float wz)
{
    float fx = (wx + field.tile_half_x) / field.cell_size;
    float fz = (wz + field.tile_half_z) / field.cell_size;

    int x0 = static_cast<int>(std::floor(fx));
    int z0 = static_cast<int>(std::floor(fz));
    int x1 = x0 + 1;
    int z1 = z0 + 1;

    float tx = fx - static_cast<float>(x0);
    float tz = fz - static_cast<float>(z0);

    int w = static_cast<int>(field.grid_w);
    int h = static_cast<int>(field.grid_h);
    x0 = std::clamp(x0, 0, w - 1);
    x1 = std::clamp(x1, 0, w - 1);
    z0 = std::clamp(z0, 0, h - 1);
    z1 = std::clamp(z1, 0, h - 1);

    float d00 = field.density[static_cast<size_t>(z0) * field.grid_w + static_cast<size_t>(x0)];
    float d10 = field.density[static_cast<size_t>(z0) * field.grid_w + static_cast<size_t>(x1)];
    float d01 = field.density[static_cast<size_t>(z1) * field.grid_w + static_cast<size_t>(x0)];
    float d11 = field.density[static_cast<size_t>(z1) * field.grid_w + static_cast<size_t>(x1)];

    float top    = d00 + (d10 - d00) * tx;
    float bottom = d01 + (d11 - d01) * tx;
    return top + (bottom - top) * tz;
}

float consume_vegetation(VegetationDensityField& field,
                         float wx, float wz, float amount)
{
    int gx, gz;
    world_to_grid(field, wx, wz, gx, gz);
    size_t idx = static_cast<size_t>(gz) * field.grid_w + static_cast<size_t>(gx);

    float available = field.density[idx];
    float consumed = std::min(available, amount);
    field.density[idx] -= consumed;
    return consumed;
}

void tick_vegetation(VegetationDensityField& field, float dt)
{
    float growth = field.regrowth_rate * dt;
    float decay  = field.decay_rate * dt;

    size_t n = field.density.size();
    for (size_t i = 0; i < n; ++i) {
        if (field.density[i] < field.capacity[i]) {
            field.density[i] = std::min(field.density[i] + growth,
                                        field.capacity[i]);
        } else if (field.density[i] > field.capacity[i]) {
            field.density[i] = std::max(field.density[i] - decay,
                                        field.capacity[i]);
        }
    }
}

void update_capacity_from_environment(VegetationDensityField& field,
                                      const EnvironmentField& env)
{
    for (uint32_t gz = 0; gz < field.grid_h; ++gz) {
        for (uint32_t gx = 0; gx < field.grid_w; ++gx) {
            float wx = static_cast<float>(gx) * field.cell_size - field.tile_half_x;
            float wz = static_cast<float>(gz) * field.cell_size - field.tile_half_z;
            auto sample = env(wx, wz);

            // Capacity from moisture: peaks at moderate moisture, drops at extremes
            // Trapezoid: 0 below 0.05, ramps to 1 at 0.15, holds to 0.8, ramps to 0 at 1.0
            float m = sample.moisture;
            float cap = 0.0f;
            if (m >= 0.05f && m < 0.15f)
                cap = (m - 0.05f) / 0.10f;
            else if (m >= 0.15f && m <= 0.80f)
                cap = 1.0f;
            else if (m > 0.80f && m <= 1.0f)
                cap = (1.0f - m) / 0.20f;

            size_t idx = static_cast<size_t>(gz) * field.grid_w + gx;
            field.capacity[idx] = cap;
        }
    }
}

float sample_vegetation_fullness(const VegetationDensityField& field,
                                 float wx, float wz)
{
    float fx = (wx + field.tile_half_x) / field.cell_size;
    float fz = (wz + field.tile_half_z) / field.cell_size;

    int x0 = static_cast<int>(std::floor(fx));
    int z0 = static_cast<int>(std::floor(fz));
    int x1 = x0 + 1;
    int z1 = z0 + 1;

    float tx = fx - static_cast<float>(x0);
    float tz = fz - static_cast<float>(z0);

    int w = static_cast<int>(field.grid_w);
    int h = static_cast<int>(field.grid_h);
    x0 = std::clamp(x0, 0, w - 1);
    x1 = std::clamp(x1, 0, w - 1);
    z0 = std::clamp(z0, 0, h - 1);
    z1 = std::clamp(z1, 0, h - 1);

    auto ratio = [&](int gx, int gz) -> float {
        size_t idx = static_cast<size_t>(gz) * field.grid_w + static_cast<size_t>(gx);
        float cap = field.capacity[idx];
        if (cap < 0.001f) return 0.0f;
        return field.density[idx] / cap;
    };

    float r00 = ratio(x0, z0);
    float r10 = ratio(x1, z0);
    float r01 = ratio(x0, z1);
    float r11 = ratio(x1, z1);

    float top    = r00 + (r10 - r00) * tx;
    float bottom = r01 + (r11 - r01) * tx;
    return std::clamp(top + (bottom - top) * tz, 0.0f, 1.0f);
}

} // namespace bestiary

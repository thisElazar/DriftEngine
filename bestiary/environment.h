#pragma once

#include <functional>
#include <cstdint>

namespace bestiary {

struct EnvironmentSample {
    float moisture;
    float temperature;
};

struct EnvironmentField {
    std::function<EnvironmentSample(float x, float z)> sample;

    EnvironmentSample operator()(float x, float z) const {
        if (sample) return sample(x, z);
        return {0.5f, 0.5f};
    }
};

struct NoiseFieldParams {
    float    moisture_freq    = 1.5f;
    int      moisture_octaves = 3;
    float    temp_freq        = 1.0f;
    int      temp_octaves     = 2;
    uint32_t seed             = 0;
};

EnvironmentField make_noise_field(const NoiseFieldParams& params);

} // namespace bestiary

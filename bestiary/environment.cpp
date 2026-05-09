#include "environment.h"

#include <glm/glm.hpp>
#include <algorithm>

namespace bestiary {

static float noise_hash31(glm::vec3 p) {
    p = glm::fract(p * glm::vec3(0.1031f, 0.1030f, 0.0973f));
    p += glm::dot(p, glm::vec3(p.y, p.x, p.z) + 31.32f);
    return glm::fract((p.x + p.y) * p.z);
}

static float noise_gradient_3d(glm::vec3 p) {
    glm::vec3 i = glm::floor(p);
    glm::vec3 f = glm::fract(p);
    glm::vec3 u = f * f * (3.0f - 2.0f * f);

    return glm::mix(
        glm::mix(glm::mix(noise_hash31(i + glm::vec3(0,0,0)), noise_hash31(i + glm::vec3(1,0,0)), u.x),
                 glm::mix(noise_hash31(i + glm::vec3(0,1,0)), noise_hash31(i + glm::vec3(1,1,0)), u.x), u.y),
        glm::mix(glm::mix(noise_hash31(i + glm::vec3(0,0,1)), noise_hash31(i + glm::vec3(1,0,1)), u.x),
                 glm::mix(noise_hash31(i + glm::vec3(0,1,1)), noise_hash31(i + glm::vec3(1,1,1)), u.x), u.y),
        u.z);
}

static float noise_fbm3d(glm::vec3 p, int octaves) {
    float sum = 0.0f, amp = 1.0f, freq = 1.0f, norm = 0.0f;
    for (int i = 0; i < octaves; i++) {
        sum += noise_gradient_3d(p * freq + glm::vec3(42.0f * 0.17f, 0.0f, 0.0f)) * amp;
        norm += amp;
        freq *= 2.0f;
        amp *= 0.5f;
    }
    return sum / norm;
}

EnvironmentField make_noise_field(const NoiseFieldParams& params)
{
    NoiseFieldParams p = params;
    return EnvironmentField{
        [p](float x, float z) -> EnvironmentSample {
            float seed_off = static_cast<float>(p.seed) * 17.31f;

            glm::vec3 mp(x * p.moisture_freq + seed_off, 0.0f,
                         z * p.moisture_freq);
            float moisture = noise_fbm3d(mp, p.moisture_octaves);

            glm::vec3 tp(x * p.temp_freq + seed_off + 100.0f, 0.0f,
                         z * p.temp_freq + 100.0f);
            float temperature = noise_fbm3d(tp, p.temp_octaves);

            return { std::clamp(moisture, 0.0f, 1.0f),
                     std::clamp(temperature, 0.0f, 1.0f) };
        }
    };
}

} // namespace bestiary

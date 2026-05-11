#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace bestiary {

struct AnimalVertex;

struct SkinnedVertex {
    float position[3];
    float normal[3];
    float color[3];
};

void cpu_skin(const std::vector<AnimalVertex>& src,
              const std::vector<glm::mat4>& joint_palette,
              std::vector<SkinnedVertex>& dst);

} // namespace bestiary

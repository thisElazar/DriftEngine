#pragma once

#include "skeleton/joint.h"
#include <vector>

namespace bestiary {

struct Skeleton {
    std::vector<Joint>     joints;
    std::vector<glm::mat4> inverse_bind;
};

} // namespace bestiary

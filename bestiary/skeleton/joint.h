#pragma once

#include <glm/glm.hpp>

namespace bestiary {

struct Joint {
    int       parent     = -1;
    glm::mat4 local_bind = glm::mat4(1.0f);
    char      name[32]   = {};
};

} // namespace bestiary

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

namespace bestiary {

struct JointKeyframe {
    float     phase    = 0.0f;
    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
};

struct WalkCycle {
    float period_seconds  = 1.0f;
    float hip_swing_deg   = 14.0f;
    float stance_fraction = 0.83f;
    float pelvis_drop     = 0.0f;
    std::vector<std::vector<JointKeyframe>> tracks;
};

std::vector<glm::quat> sample_walk(const WalkCycle& cycle, float phase, int joint_count);

} // namespace bestiary

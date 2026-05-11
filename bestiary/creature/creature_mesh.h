#pragma once

#include "agent.h"
#include "../morphology/clump.h"

#include <functional>
#include <vector>

namespace bestiary {

struct HerbivoreProfile;

VegetationMesh generate_creature_meshes(
    std::vector<Agent>& agents,
    const std::vector<HerbivoreProfile>& profiles,
    std::function<float(float x, float z)> terrain_height,
    float dt = 0.016f);

} // namespace bestiary

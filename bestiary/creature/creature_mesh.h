#pragma once

#include "agent.h"
#include "../morphology/clump.h"

#include <functional>
#include <vector>

namespace bestiary {

struct CreatureProfile;

VegetationMesh generate_creature_meshes(
    std::vector<Agent>& agents,
    const std::vector<CreatureProfile>& profiles,
    std::function<float(float x, float z)> terrain_height,
    float dt = 0.016f);

void clear_creature_mesh_cache();

} // namespace bestiary

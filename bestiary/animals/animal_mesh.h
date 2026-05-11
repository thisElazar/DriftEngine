#pragma once

#include "skeleton/skeleton.h"
#include <glm/glm.hpp>
#include <cstdint>
#include <vector>

namespace bestiary {

struct AnimalVertex {
    float   position[3];
    float   normal[3];
    float   color[3];
    uint8_t bone_indices[4];
    uint8_t bone_weights[4];
};

struct AnimalMesh {
    std::vector<AnimalVertex> vertices;
    std::vector<uint32_t>     indices;
    Skeleton                  skeleton;
};

void add_joint(Skeleton& skel, int parent, const glm::vec3& offset, const char* name);
std::vector<glm::mat4> compute_world(const Skeleton& skel);
glm::vec3 jpos(const glm::mat4& m);
void make_frame(const glm::vec3& axis, glm::vec3& right, glm::vec3& up);

void append_tube(AnimalMesh& mesh,
                 const glm::vec3& start, const glm::vec3& end,
                 float r_start, float r_end,
                 const float color[3], uint8_t bone,
                 int slices, int segments);

void append_tube_ellipse(AnimalMesh& mesh,
                         const glm::vec3& start, const glm::vec3& end,
                         float rx_start, float ry_start,
                         float rx_end, float ry_end,
                         const float color[3], uint8_t bone,
                         int slices, int segments);

void append_cap(AnimalMesh& mesh,
                const glm::vec3& center, const glm::vec3& outward,
                float radius, const float color[3], uint8_t bone_index,
                int slices, int stacks);

void append_sphere(AnimalMesh& mesh,
                   const glm::vec3& center, float radius,
                   const float color[3], uint8_t bone_index,
                   int slices = 8, int stacks = 6);

void append_ellipsoid(AnimalMesh& mesh,
                      const glm::vec3& center,
                      float rx, float ry, float rz,
                      const glm::vec3& forward,
                      const float color[3], uint8_t bone_index,
                      int slices = 8, int stacks = 6);

struct PathNode {
    glm::vec3 pos;
    float     radius;
    uint8_t   bone;
};

void append_path_tube(AnimalMesh& mesh,
                      const PathNode* nodes, int count,
                      const float color[3],
                      int slices, int segments_per_span);

} // namespace bestiary

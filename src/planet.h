#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

struct QuadNode {
    uint32_t face;
    uint32_t level;
    uint32_t x, y;

    bool operator==(const QuadNode& o) const {
        return face == o.face && level == o.level && x == o.x && y == o.y;
    }
    bool operator!=(const QuadNode& o) const { return !(*this == o); }
};

glm::vec3  planet_face_uv_to_cube(float u, float v, uint32_t face);
glm::vec3  planet_cube_to_sphere(glm::vec3 p);
glm::dvec3 planet_face_uv_to_cube_d(double u, double v, uint32_t face);
glm::dvec3 planet_cube_to_sphere_d(glm::dvec3 p);

glm::dvec3 planet_tile_center_dir(const QuadNode& node);
glm::dvec3 planet_tile_center_on_sphere(const QuadNode& node, float planet_radius);

struct PlanetTraversalParams {
    glm::dvec3 cam_pos;
    glm::dvec3 cam_forward;
    float      screen_height;
    float      fov_y;
    float      planet_radius;
    float      subdivide_threshold;
    uint32_t   max_level;
    uint32_t   max_tiles;
    double     altitude_above_terrain;
};

std::vector<QuadNode> planet_select_visible_tiles(const PlanetTraversalParams& params);

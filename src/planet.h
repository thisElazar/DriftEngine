#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <cstddef>

struct QuadNode {
    uint32_t face;
    uint32_t level;
    uint32_t x, y;

    bool operator==(const QuadNode& o) const {
        return face == o.face && level == o.level && x == o.x && y == o.y;
    }
    bool operator!=(const QuadNode& o) const { return !(*this == o); }
};

struct QuadNodeHash {
    size_t operator()(const QuadNode& n) const noexcept {
        // Mix face/level/x/y. face<6, level<32, x/y bounded by 1<<level (<=2^14 typical).
        uint64_t k = (uint64_t)n.face
                   | ((uint64_t)n.level << 4)
                   | ((uint64_t)n.x     << 12)
                   | ((uint64_t)n.y     << 36);
        k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return (size_t)k;
    }
};

glm::vec3  planet_face_uv_to_cube(float u, float v, uint32_t face);
glm::vec3  planet_cube_to_sphere(glm::vec3 p);
glm::dvec3 planet_face_uv_to_cube_d(double u, double v, uint32_t face);
glm::dvec3 planet_cube_to_sphere_d(glm::dvec3 p);

// Inverse of cube_to_sphere(face_uv_to_cube): map a unit sphere direction to its
// cube face and face-uv in [-1,1]². Dominant-axis face select + two fixed-point
// refinement steps against cube_to_sphere (same method as planet_pick_tile).
// Used for cross-face neighbor routing on the hydrology grid.
void planet_sphere_to_face_uv(glm::vec3 sphere_dir, uint32_t& out_face,
                              float& out_u, float& out_v);

glm::dvec3 planet_tile_center_dir(const QuadNode& node);
glm::dvec3 planet_tile_center_on_sphere(const QuadNode& node, float planet_radius);

struct PlanetTraversalParams {
    glm::dvec3 cam_pos;
    glm::dvec3 cam_forward;
    float      screen_height;
    float      fov_y;
    float      planet_radius;
    float      max_elevation;
    float      subdivide_threshold;
    uint32_t   max_level;
    uint32_t   max_tiles;
    double     altitude_above_terrain;
};

std::vector<QuadNode> planet_select_visible_tiles(const PlanetTraversalParams& params);

struct PlanetTilePick {
    bool     hit;          // true if a finest-level visible tile contains the projected point
    QuadNode node;         // matching tile (face/level/x/y)
    uint32_t pool_index;   // index into the visible_tiles vector that was passed in
    float    grid_x;       // fractional cell coord within tile, [0, tile_res-1]
    float    grid_y;
    // Per-tile cube-uv span (useful for callers needing tile_dx without recomputing)
    float    u_min, v_min, tile_size;
};

// Find which visible tile a point on the unit sphere falls into and return its
// fractional grid coords. Uses a dominant-axis cube projection refined with two
// fixed-point iterations against cube_to_sphere — accurate enough for brush use.
PlanetTilePick planet_pick_tile(
    glm::vec3 sphere_dir,
    const std::vector<QuadNode>& visible_tiles,
    uint32_t tile_res);

// Same-face same-level neighbor lookup. dir: 0=-u (left), 1=+u (right),
// 2=-v (down), 3=+v (up). Returns valid=false if the neighbor is across a
// cube-face seam — Phase 1 leaves those reflective.
struct QuadNeighbor { bool valid; QuadNode tile; };
QuadNeighbor planet_neighbor_same_face(QuadNode t, int dir);

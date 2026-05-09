#include "planet.h"
#include <algorithm>
#include <cmath>
#include <queue>

glm::vec3 planet_face_uv_to_cube(float u, float v, uint32_t face)
{
    switch (face) {
        case 0: return {1.0f, v, -u};
        case 1: return {-1.0f, v, u};
        case 2: return {u, 1.0f, -v};
        case 3: return {u, -1.0f, v};
        case 4: return {u, v, 1.0f};
        case 5: return {-u, v, -1.0f};
        default: return {0, 0, 0};
    }
}

glm::vec3 planet_cube_to_sphere(glm::vec3 p)
{
    float x2 = p.x*p.x, y2 = p.y*p.y, z2 = p.z*p.z;
    return {
        p.x * std::sqrt(std::max(0.0f, 1.0f - y2*0.5f - z2*0.5f + y2*z2/3.0f)),
        p.y * std::sqrt(std::max(0.0f, 1.0f - x2*0.5f - z2*0.5f + x2*z2/3.0f)),
        p.z * std::sqrt(std::max(0.0f, 1.0f - x2*0.5f - y2*0.5f + x2*y2/3.0f))
    };
}

glm::dvec3 planet_face_uv_to_cube_d(double u, double v, uint32_t face)
{
    switch (face) {
        case 0: return {1.0, v, -u};
        case 1: return {-1.0, v, u};
        case 2: return {u, 1.0, -v};
        case 3: return {u, -1.0, v};
        case 4: return {u, v, 1.0};
        case 5: return {-u, v, -1.0};
        default: return {0, 0, 0};
    }
}

glm::dvec3 planet_cube_to_sphere_d(glm::dvec3 p)
{
    double x2 = p.x*p.x, y2 = p.y*p.y, z2 = p.z*p.z;
    return {
        p.x * std::sqrt(std::max(0.0, 1.0 - y2*0.5 - z2*0.5 + y2*z2/3.0)),
        p.y * std::sqrt(std::max(0.0, 1.0 - x2*0.5 - z2*0.5 + x2*z2/3.0)),
        p.z * std::sqrt(std::max(0.0, 1.0 - x2*0.5 - y2*0.5 + x2*y2/3.0))
    };
}

glm::dvec3 planet_tile_center_dir(const QuadNode& node)
{
    double ts = 2.0 / (1 << node.level);
    double u = -1.0 + (node.x + 0.5) * ts;
    double v = -1.0 + (node.y + 0.5) * ts;
    glm::dvec3 cube_pt = planet_face_uv_to_cube_d(u, v, node.face);
    return glm::normalize(planet_cube_to_sphere_d(cube_pt));
}

glm::dvec3 planet_tile_center_on_sphere(const QuadNode& node, float planet_radius)
{
    return planet_tile_center_dir(node) * static_cast<double>(planet_radius);
}

struct TileCandidate {
    QuadNode node;
    double screen_error;
    bool operator<(const TileCandidate& o) const { return screen_error < o.screen_error; }
};

static double compute_tile_error(const QuadNode& node, const glm::dvec3& cam_pos,
                                 float screen_height, float fov_y, float planet_radius)
{
    glm::dvec3 center = planet_tile_center_on_sphere(node, planet_radius);
    glm::dvec3 rel = center - cam_pos;
    double dist = glm::length(rel);

    double cam_height = glm::length(cam_pos);
    if (cam_height > static_cast<double>(planet_radius) * 1.01) {
        glm::dvec3 cam_dir = glm::normalize(cam_pos);
        glm::dvec3 tile_dir = glm::normalize(center);
        double angle = std::acos(std::clamp(glm::dot(cam_dir, tile_dir), -1.0, 1.0));
        double horizon = std::acos(static_cast<double>(planet_radius) / cam_height);
        double tile_angular_size = 2.0 / (1 << node.level) * 1.5;
        if (angle - tile_angular_size > horizon)
            return -1.0;
    }

    double tile_world_size = (2.0 / (1 << node.level)) * static_cast<double>(planet_radius) * 1.5708;
    return (tile_world_size / std::max(dist, 1.0)) * screen_height / fov_y;
}

std::vector<QuadNode> planet_select_visible_tiles(const PlanetTraversalParams& params)
{
    std::vector<QuadNode> visible_tiles;
    visible_tiles.reserve(params.max_tiles);

    uint32_t effective_max_level = params.max_level;
    if (params.altitude_above_terrain > 100.0) {
        int cap = static_cast<int>(14.0 - std::log2(params.altitude_above_terrain / 100.0));
        effective_max_level = static_cast<uint32_t>(std::clamp(cap, 5, static_cast<int>(params.max_level)));
    }

    std::priority_queue<TileCandidate> pq;

    for (uint32_t f = 0; f < 6; f++) {
        QuadNode root{f, 0, 0, 0};
        double err = compute_tile_error(root, params.cam_pos,
                                        params.screen_height, params.fov_y, params.planet_radius);
        if (err > 0.0) pq.push({root, err});
    }

    while (!pq.empty() && visible_tiles.size() < params.max_tiles) {
        auto top = pq.top();
        pq.pop();

        if (top.screen_error > params.subdivide_threshold && top.node.level < effective_max_level) {
            for (uint32_t cy = 0; cy < 2; cy++)
                for (uint32_t cx = 0; cx < 2; cx++) {
                    QuadNode child{top.node.face, top.node.level + 1,
                                   top.node.x * 2 + cx, top.node.y * 2 + cy};
                    double err = compute_tile_error(child, params.cam_pos,
                                                    params.screen_height, params.fov_y, params.planet_radius);
                    if (err > 0.0) pq.push({child, err});
                }
        } else {
            visible_tiles.push_back(top.node);
        }
    }

    while (!pq.empty() && visible_tiles.size() < params.max_tiles) {
        visible_tiles.push_back(pq.top().node);
        pq.pop();
    }

    return visible_tiles;
}

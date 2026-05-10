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

// --------------------------------------------------------------------------
// planet_pick_tile: sphere_dir → visible tile + fractional grid coords
// --------------------------------------------------------------------------
//
// face_uv_to_cube case table (matches the function above):
//   face 0  (+X): (1,  v, -u)         → on +X face, v = Y, u = -Z
//   face 1  (-X): (-1, v,  u)         → on -X face, v = Y, u =  Z
//   face 2  (+Y): (u,  1, -v)         → on +Y face, u = X, v = -Z
//   face 3  (-Y): (u, -1,  v)         → on -Y face, u = X, v =  Z
//   face 4  (+Z): (u,  v,  1)         → on +Z face, u = X, v =  Y
//   face 5  (-Z): (-u, v, -1)         → on -Z face, u = -X, v =  Y

static void cube_face_to_uv(const glm::vec3& cube, uint32_t face, float& u, float& v)
{
    switch (face) {
        case 0: u = -cube.z; v =  cube.y; break;
        case 1: u =  cube.z; v =  cube.y; break;
        case 2: u =  cube.x; v = -cube.z; break;
        case 3: u =  cube.x; v =  cube.z; break;
        case 4: u =  cube.x; v =  cube.y; break;
        case 5: u = -cube.x; v =  cube.y; break;
        default: u = 0; v = 0; break;
    }
}

PlanetTilePick planet_pick_tile(
    glm::vec3 sphere_dir,
    const std::vector<QuadNode>& visible_tiles,
    uint32_t tile_res)
{
    PlanetTilePick out{};
    out.hit = false;

    if (glm::length(sphere_dir) < 1e-6f) return out;
    sphere_dir = glm::normalize(sphere_dir);

    // 1) Pick face from dominant axis.
    float ax = std::fabs(sphere_dir.x);
    float ay = std::fabs(sphere_dir.y);
    float az = std::fabs(sphere_dir.z);
    uint32_t face;
    if (ax >= ay && ax >= az)      face = (sphere_dir.x > 0.0f) ? 0u : 1u;
    else if (ay >= az)             face = (sphere_dir.y > 0.0f) ? 2u : 3u;
    else                           face = (sphere_dir.z > 0.0f) ? 4u : 5u;

    // 2) Initial cube-point guess: scale sphere_dir so dominant axis is ±1.
    float dom = std::max(ax, std::max(ay, az));
    if (dom < 1e-6f) return out;
    glm::vec3 cube = sphere_dir / dom;

    // 3) Two fixed-point refinement steps against cube_to_sphere.
    //    cube_to_sphere preserves face membership and the dominant-axis sign,
    //    so we keep the dominant axis pinned to ±1 and adjust only the other two.
    auto refine = [&](glm::vec3& q) {
        glm::vec3 s = planet_cube_to_sphere(q);
        glm::vec3 d = sphere_dir - s;
        // Apply correction in cube space; clamp so we don't overshoot the face.
        q += d;
        // Re-pin dominant axis to ±1.
        switch (face) {
            case 0: q.x =  1.0f; break;
            case 1: q.x = -1.0f; break;
            case 2: q.y =  1.0f; break;
            case 3: q.y = -1.0f; break;
            case 4: q.z =  1.0f; break;
            case 5: q.z = -1.0f; break;
        }
        // Clamp the in-face coords to [-1, 1] (sphere_dir near a face edge can otherwise
        // briefly drift out of range during iteration).
        q.x = std::clamp(q.x, -1.0f, 1.0f);
        q.y = std::clamp(q.y, -1.0f, 1.0f);
        q.z = std::clamp(q.z, -1.0f, 1.0f);
    };
    refine(cube);
    refine(cube);

    // 4) Convert cube point on this face → face uv ∈ [-1, 1]².
    float fu, fv;
    cube_face_to_uv(cube, face, fu, fv);

    // 5) Find finest-level visible tile on this face containing (fu, fv).
    uint32_t best_i = 0;
    int best_level = -1;
    float best_u_min = 0, best_v_min = 0, best_ts = 0;
    for (uint32_t i = 0; i < visible_tiles.size(); ++i) {
        const QuadNode& t = visible_tiles[i];
        if (t.face != face) continue;
        float ts = 2.0f / float(1u << t.level);
        float u_min = -1.0f + t.x * ts;
        float v_min = -1.0f + t.y * ts;
        // Inclusive bounds; ties go to the finer tile (last wins on equal level).
        if (fu >= u_min && fu <= u_min + ts &&
            fv >= v_min && fv <= v_min + ts &&
            int(t.level) >= best_level) {
            best_level = int(t.level);
            best_i = i;
            best_u_min = u_min;
            best_v_min = v_min;
            best_ts = ts;
        }
    }
    if (best_level < 0) return out;

    // 6) Fractional grid coords within the tile.
    float lu = (fu - best_u_min) / best_ts;
    float lv = (fv - best_v_min) / best_ts;
    lu = std::clamp(lu, 0.0f, 1.0f);
    lv = std::clamp(lv, 0.0f, 1.0f);

    out.hit        = true;
    out.node       = visible_tiles[best_i];
    out.pool_index = best_i;
    out.grid_x     = lu * float(tile_res - 1);
    out.grid_y     = lv * float(tile_res - 1);
    out.u_min      = best_u_min;
    out.v_min      = best_v_min;
    out.tile_size  = best_ts;
    return out;
}

QuadNeighbor planet_neighbor_same_face(QuadNode t, int dir)
{
    int side = int(1u << t.level);
    int tx = int(t.x), ty = int(t.y);
    switch (dir) {
        case 0: tx -= 1; break;
        case 1: tx += 1; break;
        case 2: ty -= 1; break;
        case 3: ty += 1; break;
        default: return {false, {}};
    }
    if (tx < 0 || tx >= side || ty < 0 || ty >= side) return {false, {}};
    return {true, QuadNode{t.face, t.level, uint32_t(tx), uint32_t(ty)}};
}

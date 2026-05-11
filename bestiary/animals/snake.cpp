#include "animals/snake.h"
#include <cstdio>

namespace bestiary {

// Joint layout (15 joints) — root at 1/3 from head:
//
//  0  anchor (root, ~1/3 from head)
//
// Forward chain → head:
//  1 fwd_4  2 fwd_3  3 fwd_2  4 fwd_1/neck  5 head
//
// Backward chain → tail:
//  6 back_1  7 back_2  8 back_3  9 back_4  10 back_5
// 11 back_6 12 back_7 13 back_8  14 tail_tip
//
// Spatial order (head→tail): 5 4 3 2 1 0 6 7 8 9 10 11 12 13 14

static constexpr int SNAKE_JOINTS = 15;
static constexpr int FWD_COUNT    = 5;
static constexpr int BACK_COUNT   = 9;

// Spatial ordering: maps spatial index (0=head, 14=tail) to joint index
static constexpr int spatial_to_joint[SNAKE_JOINTS] = {
    5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14
};

Skeleton build_snake_skeleton(const SnakeParams& p)
{
    Skeleton skel;
    skel.joints.reserve(SNAKE_JOINTS);

    float seg = p.body_length / static_cast<float>(SNAKE_JOINTS - 1);
    float h   = p.body_thickness;

    // 0: anchor at center of mass
    add_joint(skel, -1, {0, h, 0}, "anchor");

    // Forward chain (1-5): toward head, each step in +Z
    for (int i = 1; i <= FWD_COUNT; ++i) {
        char name[16];
        if (i == FWD_COUNT)
            std::snprintf(name, sizeof(name), "head");
        else if (i == FWD_COUNT - 1)
            std::snprintf(name, sizeof(name), "neck");
        else
            std::snprintf(name, sizeof(name), "fwd_%d", FWD_COUNT - i);
        add_joint(skel, i - 1, {0, 0, seg}, name);
    }

    // Backward chain (6-14): toward tail, each step in -Z
    for (int i = 0; i < BACK_COUNT; ++i) {
        int parent = (i == 0) ? 0 : (FWD_COUNT + i);
        float z = (i == BACK_COUNT - 1) ? -seg * 0.5f : -seg;
        char name[16];
        if (i == BACK_COUNT - 1)
            std::snprintf(name, sizeof(name), "tail_tip");
        else
            std::snprintf(name, sizeof(name), "back_%d", i + 1);
        add_joint(skel, parent, {0, 0, z}, name);
    }

    auto world = compute_world(skel);
    skel.inverse_bind.resize(skel.joints.size());
    for (size_t i = 0; i < skel.joints.size(); ++i)
        skel.inverse_bind[i] = glm::inverse(world[i]);

    return skel;
}

AnimalMesh generate_snake_mesh(const SnakeParams& p)
{
    AnimalMesh mesh;
    mesh.skeleton = build_snake_skeleton(p);

    auto world = compute_world(mesh.skeleton);
    auto jp = [&](int i) { return jpos(world[i]); };

    constexpr int SL = 8;

    // Build path in spatial order (head → tail)
    PathNode nodes[SNAKE_JOINTS];
    for (int si = 0; si < SNAKE_JOINTS; ++si) {
        int ji = spatial_to_joint[si];
        float t = static_cast<float>(si) / static_cast<float>(SNAKE_JOINTS - 1);
        float r;
        if (t < 0.08f) {
            r = glm::mix(p.head_width * 0.8f, p.body_thickness * 0.75f, t / 0.08f);
        } else if (t < 0.3f) {
            r = glm::mix(p.body_thickness * 0.75f, p.body_thickness, (t - 0.08f) / 0.22f);
        } else if (t < 0.65f) {
            r = p.body_thickness;
        } else {
            float tail_t = (t - 0.65f) / 0.35f;
            r = glm::mix(p.body_thickness, p.body_thickness * p.taper_tail, tail_t * tail_t);
        }
        nodes[si] = {jp(ji), r, static_cast<uint8_t>(ji)};
    }

    append_path_tube(mesh, nodes, SNAKE_JOINTS, p.coat_color, SL, 3);

    // Head — wider, flatter ellipsoid
    int head_j = 5;
    glm::vec3 head_dir = glm::normalize(jp(head_j) - jp(4));
    glm::vec3 head_right, head_up;
    make_frame(head_dir, head_right, head_up);

    append_ellipsoid(mesh, jp(head_j),
                     p.head_width, p.head_width * 0.6f, p.head_length * 0.5f,
                     head_dir, p.coat_color, static_cast<uint8_t>(head_j), 8, 6);
    append_cap(mesh, jp(head_j) + head_dir * p.head_length * 0.3f, head_dir,
               p.head_width * 0.5f, p.coat_color, static_cast<uint8_t>(head_j), 6, 3);

    // Eyes
    {
        float eye_color[3] = {0.70f, 0.55f, 0.10f};
        float eye_r = p.head_width * 0.18f;
        glm::vec3 eye_base = jp(head_j) + head_dir * p.head_length * 0.15f;
        glm::vec3 eye_L = eye_base + head_right * p.head_width * 0.7f
                                   + head_up * p.head_width * 0.35f;
        glm::vec3 eye_R = eye_base - head_right * p.head_width * 0.7f
                                   + head_up * p.head_width * 0.35f;
        append_cap(mesh, eye_L,  head_right, eye_r, eye_color,
                   static_cast<uint8_t>(head_j), 6, 3);
        append_cap(mesh, eye_R, -head_right, eye_r, eye_color,
                   static_cast<uint8_t>(head_j), 6, 3);
    }

    // Tail tip cap
    int tail_j = 14;
    glm::vec3 tail_dir = glm::normalize(jp(tail_j) - jp(13));
    append_cap(mesh, jp(tail_j), tail_dir, p.body_thickness * p.taper_tail,
               p.coat_color, static_cast<uint8_t>(tail_j), 6, 2);

    countershade(mesh, p.belly_color);
    return mesh;
}

} // namespace bestiary

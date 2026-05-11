#include "animals/bird.h"

namespace bestiary {

// Joint layout (23 joints):
//  0 pelvis
//  1 spine_1      (mid-back)
//  2 chest        (keel/shoulder area)
//  3 neck_1       (neck base)
//  4 neck_2       (mid-neck)
//  5 head
//  6 wing_L       7 elbow_L      8 wrist_L      9 wingtip_L
// 10 wing_R      11 elbow_R     12 wrist_R     13 wingtip_R
// 14 hip_L       15 upper_L     16 lower_L     17 foot_L
// 18 hip_R       19 upper_R     20 lower_R     21 foot_R
// 22 tail

Skeleton build_bird_skeleton(const BirdParams& p)
{
    Skeleton skel;
    skel.joints.reserve(23);

    float body_half  = p.body_length * 0.5f;
    float neck_seg   = p.neck_length * 0.5f;
    float wing_seg   = p.wing_length / 3.0f;
    float half_girth = p.body_girth * 0.5f;
    float pelvis_h   = p.leg_length_upper + p.leg_length_lower + p.foot_size;

    float neck_dy = neck_seg * 0.766f;
    float neck_dz = neck_seg * 0.643f;

    add_joint(skel, -1, {0, pelvis_h, 0}, "pelvis");                  // 0
    add_joint(skel,  0, {0, 0, body_half}, "spine_1");                 // 1
    add_joint(skel,  1, {0, 0, body_half}, "chest");                   // 2

    add_joint(skel,  2, {0, neck_dy, neck_dz}, "neck_1");              // 3
    add_joint(skel,  3, {0, neck_dy, neck_dz}, "neck_2");              // 4
    add_joint(skel,  4, {0, 0, p.head_size}, "head");                  // 5

    add_joint(skel,  2, { half_girth, 0, 0}, "wing_L");               // 6
    add_joint(skel,  6, { wing_seg, 0, 0}, "elbow_L");                // 7
    add_joint(skel,  7, { wing_seg, 0, 0}, "wrist_L");                // 8
    add_joint(skel,  8, { wing_seg, 0, 0}, "wingtip_L");              // 9

    add_joint(skel,  2, {-half_girth, 0, 0}, "wing_R");               // 10
    add_joint(skel, 10, {-wing_seg, 0, 0}, "elbow_R");                // 11
    add_joint(skel, 11, {-wing_seg, 0, 0}, "wrist_R");                // 12
    add_joint(skel, 12, {-wing_seg, 0, 0}, "wingtip_R");              // 13

    float hip_spread = half_girth * 0.4f;
    add_joint(skel,  1, { hip_spread, 0, 0}, "hip_L");                // 14
    add_joint(skel, 14, {0, -p.leg_length_upper, 0}, "upper_L");      // 15
    add_joint(skel, 15, {0, -p.leg_length_lower, 0}, "lower_L");      // 16
    add_joint(skel, 16, {0, -p.foot_size, 0}, "foot_L");              // 17

    add_joint(skel,  1, {-hip_spread, 0, 0}, "hip_R");                // 18
    add_joint(skel, 18, {0, -p.leg_length_upper, 0}, "upper_R");      // 19
    add_joint(skel, 19, {0, -p.leg_length_lower, 0}, "lower_R");      // 20
    add_joint(skel, 20, {0, -p.foot_size, 0}, "foot_R");              // 21

    add_joint(skel,  0, {0, 0.01f, -p.tail_length}, "tail");          // 22

    auto world = compute_world(skel);
    skel.inverse_bind.resize(skel.joints.size());
    for (size_t i = 0; i < skel.joints.size(); ++i)
        skel.inverse_bind[i] = glm::inverse(world[i]);

    return skel;
}

AnimalMesh generate_bird_mesh(const BirdParams& p)
{
    AnimalMesh mesh;
    mesh.skeleton = build_bird_skeleton(p);

    auto world = compute_world(mesh.skeleton);
    auto jp = [&](int i) { return jpos(world[i]); };

    constexpr int SL = 8;
    float g = p.body_girth;

    // Body — swept path from pelvis through spine to chest
    PathNode body[] = {
        {jp(0), g * 0.42f, 0},
        {jp(1), g * 0.52f, 1},
        {jp(2), g * 0.38f, 2},
    };
    append_path_tube(mesh, body, 3, p.coat_color, SL, 4);
    append_cap(mesh, jp(0), glm::normalize(jp(0) - jp(1)), g * 0.42f, p.coat_color, 0, SL, 3);
    append_cap(mesh, jp(2), glm::normalize(jp(2) - jp(1)), g * 0.38f, p.coat_color, 2, SL, 3);

    // Belly — ellipsoid for roundness
    glm::vec3 belly = glm::mix(jp(0), jp(2), 0.45f);
    belly.y -= g * 0.06f;
    glm::vec3 body_fwd = glm::normalize(jp(2) - jp(0));
    append_ellipsoid(mesh, belly,
                     g * 0.46f, g * 0.40f, p.body_length * 0.42f,
                     body_fwd, p.coat_color, 1, SL, 6);

    // Neck — path tube from chest to head
    float neck_r = g * 0.16f;
    PathNode neck[] = {
        {jp(2), g * 0.28f, 2},
        {jp(3), neck_r, 3},
        {jp(4), neck_r * 0.9f, 4},
        {jp(5), p.head_size * 0.45f, 5},
    };
    append_path_tube(mesh, neck, 4, p.coat_color, SL, 3);

    // Head
    append_sphere(mesh, jp(5), p.head_size * 0.55f, p.coat_color, 5, 8, 6);

    // Beak
    {
        glm::vec3 head_dir = glm::normalize(jp(5) - jp(4));
        glm::vec3 beak_tip = jp(5) + head_dir * p.beak_length;
        float beak_color[3] = {0.75f, 0.62f, 0.20f};
        append_tube(mesh, jp(5), beak_tip,
                    p.head_size * 0.15f, p.head_size * 0.04f,
                    beak_color, 5, 6, 2);
        append_cap(mesh, beak_tip, head_dir, p.head_size * 0.04f,
                   beak_color, 5, 4, 2);
    }

    // Eyes
    {
        glm::vec3 head_dir = glm::normalize(jp(5) - jp(4));
        glm::vec3 head_right, head_up;
        make_frame(head_dir, head_right, head_up);

        float eye_color[3] = {0.08f, 0.06f, 0.04f};
        float eye_r = p.head_size * 0.14f;
        glm::vec3 eye_base = jp(5) - head_dir * p.head_size * 0.05f;
        glm::vec3 eye_L = eye_base + head_right * p.head_size * 0.48f
                                   + head_up * p.head_size * 0.15f;
        glm::vec3 eye_R = eye_base - head_right * p.head_size * 0.48f
                                   + head_up * p.head_size * 0.15f;
        append_cap(mesh, eye_L,  head_right, eye_r, eye_color, 5, 6, 3);
        append_cap(mesh, eye_R, -head_right, eye_r, eye_color, 5, 6, 3);
    }

    // Wings — flattened tubes per segment
    {
        float wc[3] = {p.coat_color[0] * 0.85f,
                        p.coat_color[1] * 0.85f,
                        p.coat_color[2] * 0.85f};

        float ww  = p.wing_width;
        float wt  = p.wing_taper;
        float wd  = ww * 0.35f;
        float w1x = ww;          float w1y = wd;
        float w2x = ww * 0.75f;  float w2y = wd * 0.65f;
        float w3x = ww * 0.45f;  float w3y = wd * 0.35f;
        float w4x = ww * wt;     float w4y = wd * wt * 0.4f;

        auto wing_seg = [&](int from, int to, float rx0, float ry0, float rx1, float ry1) {
            append_tube_ellipse(mesh, jp(from), jp(to),
                                rx0, ry0, rx1, ry1,
                                wc, static_cast<uint8_t>(from), SL, 3);
            append_sphere(mesh, jp(from),
                          (rx0 + ry0) * 0.5f, wc, static_cast<uint8_t>(from), 6, 4);
        };

        wing_seg( 6,  7, w1x, w1y, w2x, w2y);
        wing_seg( 7,  8, w2x, w2y, w3x, w3y);
        wing_seg( 8,  9, w3x, w3y, w4x, w4y);

        wing_seg(10, 11, w1x, w1y, w2x, w2y);
        wing_seg(11, 12, w2x, w2y, w3x, w3y);
        wing_seg(12, 13, w3x, w3y, w4x, w4y);
    }

    // Legs
    {
        float t = p.leg_thickness;
        float leg_color[3] = {0.72f, 0.58f, 0.18f};

        auto leg = [&](int h, int u, int l, int f) {
            append_tube(mesh, jp(h), jp(u), t * 1.0f, t * 0.7f,
                        p.coat_color, static_cast<uint8_t>(h), 6, 2);
            append_tube(mesh, jp(u), jp(l), t * 0.7f, t * 0.4f,
                        leg_color, static_cast<uint8_t>(u), 6, 2);
            append_tube(mesh, jp(l), jp(f), t * 0.4f, t * 0.6f,
                        leg_color, static_cast<uint8_t>(l), 6, 2);

            append_sphere(mesh, jp(h), t * 1.0f, p.coat_color,
                          static_cast<uint8_t>(h), 6, 4);
            append_sphere(mesh, jp(u), t * 0.75f, leg_color,
                          static_cast<uint8_t>(u), 6, 4);
            append_sphere(mesh, jp(l), t * 0.5f, leg_color,
                          static_cast<uint8_t>(l), 6, 4);

            // Toes — 3 forward, 1 rear (anisodactyl)
            float toe_len = p.foot_size * 1.2f;
            float toe_r   = t * 0.22f;
            glm::vec3 fp   = jp(f);
            for (int ti = -1; ti <= 1; ++ti) {
                float spread = static_cast<float>(ti) * 0.01f;
                float zoff = (ti == 0) ? toe_len : toe_len * 0.85f;
                glm::vec3 tip = fp + glm::vec3(spread, 0.0f, zoff);
                append_tube(mesh, fp, tip, toe_r, toe_r * 0.2f,
                            leg_color, static_cast<uint8_t>(f), 4, 2);
            }
            glm::vec3 rear = fp + glm::vec3(0.0f, 0.0f, -toe_len * 0.5f);
            append_tube(mesh, fp, rear, toe_r, toe_r * 0.2f,
                        leg_color, static_cast<uint8_t>(f), 4, 2);
        };

        leg(14, 15, 16, 17);
        leg(18, 19, 20, 21);
    }

    // Tail — flat fan shape
    {
        float tc[3] = {p.coat_color[0] * 0.9f,
                        p.coat_color[1] * 0.9f,
                        p.coat_color[2] * 0.88f};
        append_tube_ellipse(mesh, jp(0), jp(22),
                            g * 0.22f, g * 0.06f,
                            g * 0.14f, g * 0.012f,
                            tc, 22, SL, 3);
    }

    countershade(mesh, p.belly_color);
    return mesh;
}

} // namespace bestiary

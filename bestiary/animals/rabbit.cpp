#include "animals/rabbit.h"

namespace bestiary {

// Joint layout (24 joints):
//  0 pelvis
//  1 spine_1      (mid-back)
//  2 spine_2      (shoulders)
//  3 neck
//  4 head
//  5 ear_L        6 ear_R
//  7 shoulder_L   8 upper_front_L   9 lower_front_L  10 paw_front_L
// 11 shoulder_R  12 upper_front_R  13 lower_front_R  14 paw_front_R
// 15 hip_L       16 upper_back_L   17 lower_back_L   18 paw_back_L
// 19 hip_R       20 upper_back_R   21 lower_back_R   22 paw_back_R
// 23 tail

Skeleton build_rabbit_skeleton(const RabbitParams& p)
{
    Skeleton skel;
    skel.joints.reserve(24);

    float spine_seg    = p.body_length / 2.0f;
    float half_girth   = p.body_girth * 0.5f;
    float half_front   = p.leg_length_front * 0.5f;
    float half_back    = p.leg_length_back * 0.5f;

    // Pelvis at front-leg height so spine is horizontal; back legs bend to reach ground
    float pelvis_h = p.leg_length_front + p.paw_size;

    add_joint(skel, -1, {0, pelvis_h, 0}, "pelvis");                 // 0
    add_joint(skel,  0, {0, 0, spine_seg}, "spine_1");                // 1
    add_joint(skel,  1, {0, 0, spine_seg}, "spine_2");                // 2

    add_joint(skel,  2, {0, p.neck_length*0.5f, p.neck_length*0.866f}, "neck"); // 3
    add_joint(skel,  3, {0, 0, p.head_length}, "head");               // 4

    float ear_spread = p.head_width * 0.35f;
    add_joint(skel,  4, { ear_spread, 0, 0}, "ear_L");                // 5
    add_joint(skel,  4, {-ear_spread, 0, 0}, "ear_R");                // 6

    add_joint(skel,  2, { half_girth, 0, 0},    "shoulder_L");        // 7
    add_joint(skel,  7, {0, -half_front, 0},     "upper_front_L");    // 8
    add_joint(skel,  8, {0, -half_front, 0},     "lower_front_L");    // 9
    add_joint(skel,  9, {0, -p.paw_size, 0},     "paw_front_L");     // 10

    add_joint(skel,  2, {-half_girth, 0, 0},     "shoulder_R");       // 11
    add_joint(skel, 11, {0, -half_front, 0},      "upper_front_R");   // 12
    add_joint(skel, 12, {0, -half_front, 0},      "lower_front_R");   // 13
    add_joint(skel, 13, {0, -p.paw_size, 0},      "paw_front_R");    // 14

    // Back legs: bend so feet reach ground from lowered pelvis
    // cos(θ) = (pelvis_h - paw_size) / leg_length_back
    float back_drop = pelvis_h - p.paw_size;
    float cos_bend  = glm::clamp(back_drop / p.leg_length_back, 0.3f, 1.0f);
    float sin_bend  = std::sqrt(1.0f - cos_bend * cos_bend);
    float ub_dy = -half_back * cos_bend;
    float ub_dz = -half_back * sin_bend;   // upper leg angles backward
    float lb_dy = -half_back * cos_bend;
    float lb_dz =  half_back * sin_bend;   // lower leg angles forward (cancels)

    add_joint(skel,  0, { half_girth, 0, 0},      "hip_L");           // 15
    add_joint(skel, 15, {0, ub_dy, ub_dz},        "upper_back_L");    // 16
    add_joint(skel, 16, {0, lb_dy, lb_dz},         "lower_back_L");    // 17
    add_joint(skel, 17, {0, -p.paw_size, 0},      "paw_back_L");     // 18

    add_joint(skel,  0, {-half_girth, 0, 0},      "hip_R");           // 19
    add_joint(skel, 19, {0, ub_dy, ub_dz},        "upper_back_R");    // 20
    add_joint(skel, 20, {0, lb_dy, lb_dz},         "lower_back_R");    // 21
    add_joint(skel, 21, {0, -p.paw_size, 0},      "paw_back_R");     // 22

    add_joint(skel,  0, {0, 0.01f, -0.02f},       "tail");            // 23

    auto world = compute_world(skel);
    skel.inverse_bind.resize(skel.joints.size());
    for (size_t i = 0; i < skel.joints.size(); ++i)
        skel.inverse_bind[i] = glm::inverse(world[i]);

    return skel;
}

AnimalMesh generate_rabbit_mesh(const RabbitParams& p)
{
    AnimalMesh mesh;
    mesh.skeleton = build_rabbit_skeleton(p);

    auto world = compute_world(mesh.skeleton);
    auto jp = [&](int i) { return jpos(world[i]); };

    constexpr int SL = 8;
    float g = p.body_girth;

    // Body — continuous swept surface
    float hw = p.head_width * 0.5f;
    PathNode body[] = {
        {jp(0), g*0.46f, 0},
        {jp(1), g*0.52f, 1},
        {jp(2), g*0.44f, 2},
        {jp(3), hw*0.92f, 3},
        {jp(4), hw*0.68f, 4},
    };
    append_path_tube(mesh, body, 5, p.coat_color, SL, 4);
    append_cap(mesh, jp(0), glm::normalize(jp(0) - jp(1)), g*0.46f, p.coat_color, 0, SL, 3);
    append_cap(mesh, jp(4), glm::normalize(jp(4) - jp(3)), hw*0.68f, p.coat_color, 4, SL, 3);

    // Head roundness sphere
    glm::vec3 head_center = glm::mix(jp(3), jp(4), 0.4f);
    append_sphere(mesh, head_center, hw*0.95f, p.coat_color, 3, 8, 6);

    // Facial features
    {
        glm::vec3 head_dir = glm::normalize(jp(4) - jp(3));
        glm::vec3 head_right, head_up;
        make_frame(head_dir, head_right, head_up);

        float eye_color[3] = {0.10f, 0.06f, 0.04f};
        float eye_r = hw * 0.22f;
        glm::vec3 eye_base = glm::mix(jp(3), jp(4), 0.45f);
        glm::vec3 eye_L = eye_base + head_right * hw * 0.85f + head_up * hw * 0.2f;
        glm::vec3 eye_R = eye_base - head_right * hw * 0.85f + head_up * hw * 0.2f;
        append_cap(mesh, eye_L,  head_right, eye_r, eye_color, 4, 6, 3);
        append_cap(mesh, eye_R, -head_right, eye_r, eye_color, 4, 6, 3);

        float nose_color[3] = {0.75f, 0.45f, 0.45f};
        glm::vec3 nose_pos = jp(4) + head_dir * hw * 0.1f - head_up * hw * 0.2f;
        append_cap(mesh, nose_pos, head_dir, hw * 0.18f, nose_color, 4, 6, 2);
    }

    // Ears — long, upright
    {
        glm::vec3 head_dir = glm::normalize(jp(4) - jp(3));
        glm::vec3 head_right, head_up;
        make_frame(head_dir, head_right, head_up);

        float ear_r = p.ear_length * 0.12f;
        glm::vec3 ear_L_tip = jp(5) + head_up * p.ear_length + head_right * p.ear_length * 0.08f;
        glm::vec3 ear_R_tip = jp(6) + head_up * p.ear_length - head_right * p.ear_length * 0.08f;

        append_tube(mesh, jp(5), ear_L_tip, ear_r*0.8f, ear_r*0.25f, p.coat_color, 5, 6, 3);
        append_tube(mesh, jp(6), ear_R_tip, ear_r*0.8f, ear_r*0.25f, p.coat_color, 6, 6, 3);
        append_cap(mesh, ear_L_tip, glm::normalize(ear_L_tip - jp(5)), ear_r*0.25f, p.coat_color, 5, 4, 2);
        append_cap(mesh, ear_R_tip, glm::normalize(ear_R_tip - jp(6)), ear_r*0.25f, p.coat_color, 6, 4, 2);

        float inner_color[3] = {0.72f, 0.50f, 0.48f};
        glm::vec3 inner_off = head_dir * (-ear_r * 0.15f);
        append_tube(mesh, jp(5) + inner_off, ear_L_tip + inner_off,
            ear_r*0.5f, ear_r*0.12f, inner_color, 5, 4, 3);
        append_tube(mesh, jp(6) + inner_off, ear_R_tip + inner_off,
            ear_r*0.5f, ear_r*0.12f, inner_color, 6, 4, 3);
    }

    // Legs
    auto leg = [&](int s, int u, int l, int h) {
        float t = p.leg_thickness;
        append_tube(mesh, jp(s), jp(u), t*1.0f, t*0.85f, p.coat_color,
                    static_cast<uint8_t>(s), 6, 2);
        append_tube(mesh, jp(u), jp(l), t*0.85f, t*0.6f, p.coat_color,
                    static_cast<uint8_t>(u), 6, 2);
        append_tube(mesh, jp(l), jp(h), t*0.6f, t*0.45f, p.coat_color,
                    static_cast<uint8_t>(l), 6, 2);
        append_cap(mesh, jp(h), glm::normalize(jp(h) - jp(l)), t*0.45f, p.coat_color,
                   static_cast<uint8_t>(h), 6, 2);
    };
    leg( 7,  8,  9, 10);
    leg(11, 12, 13, 14);
    leg(15, 16, 17, 18);
    leg(19, 20, 21, 22);

    // Tail puff
    append_sphere(mesh, jp(23), p.tail_size, p.coat_color, 23, 6, 4);

    // Leg joint spheres
    auto leg_spheres = [&](int s, int u, int l, int /*h*/) {
        float t = p.leg_thickness;
        append_sphere(mesh, jp(s), t*1.0f,  p.coat_color, static_cast<uint8_t>(s), 6, 4);
        append_sphere(mesh, jp(u), t*0.85f, p.coat_color, static_cast<uint8_t>(u), 6, 4);
        append_sphere(mesh, jp(l), t*0.65f, p.coat_color, static_cast<uint8_t>(l), 6, 4);
    };
    leg_spheres( 7,  8,  9, 10);
    leg_spheres(11, 12, 13, 14);
    leg_spheres(15, 16, 17, 18);
    leg_spheres(19, 20, 21, 22);

    return mesh;
}

} // namespace bestiary

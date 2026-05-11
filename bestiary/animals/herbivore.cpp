#include "animals/herbivore.h"

namespace bestiary {

Skeleton build_herbivore_skeleton(const HerbivoreParams& p)
{
    Skeleton skel;
    skel.joints.reserve(24);

    float spine_seg  = p.torso_length / 3.0f;
    float neck_seg   = p.neck_length / 2.0f;
    float neck_dy    = neck_seg * 0.7071f;
    float neck_dz    = neck_seg * 0.7071f;
    float half_girth = p.torso_girth * 0.5f;
    float half_front = p.leg_length_front * 0.5f;
    float half_back  = p.leg_length_back * 0.5f;

    float spine_dy = (p.leg_length_front - p.leg_length_back) / 3.0f;

    add_joint(skel, -1, {0, p.leg_length_back, 0}, "pelvis");
    add_joint(skel,  0, {0, spine_dy, spine_seg}, "spine_1");
    add_joint(skel,  1, {0, spine_dy, spine_seg}, "spine_2");
    add_joint(skel,  2, {0, spine_dy, spine_seg}, "spine_3");

    add_joint(skel,  3, {0, neck_dy, neck_dz}, "neck_1");
    add_joint(skel,  4, {0, neck_dy, neck_dz}, "neck_2");
    add_joint(skel,  5, {0, 0, p.head_length}, "head");

    add_joint(skel,  3, { half_girth, 0, 0},  "shoulder_L");
    add_joint(skel,  7, {0, -half_front, 0},   "upper_front_L");
    add_joint(skel,  8, {0, -half_front, 0},   "lower_front_L");
    add_joint(skel,  9, {0, -p.hoof_size, 0},  "hoof_front_L");

    add_joint(skel,  3, {-half_girth, 0, 0},   "shoulder_R");
    add_joint(skel, 11, {0, -half_front, 0},    "upper_front_R");
    add_joint(skel, 12, {0, -half_front, 0},    "lower_front_R");
    add_joint(skel, 13, {0, -p.hoof_size, 0},   "hoof_front_R");

    add_joint(skel,  0, { half_girth, 0, 0},    "hip_L");
    add_joint(skel, 15, {0, -half_back, 0},     "upper_back_L");
    add_joint(skel, 16, {0, -half_back, 0},     "lower_back_L");
    add_joint(skel, 17, {0, -p.hoof_size, 0},   "hoof_back_L");

    add_joint(skel,  0, {-half_girth, 0, 0},    "hip_R");
    add_joint(skel, 19, {0, -half_back, 0},     "upper_back_R");
    add_joint(skel, 20, {0, -half_back, 0},     "lower_back_R");
    add_joint(skel, 21, {0, -p.hoof_size, 0},   "hoof_back_R");

    add_joint(skel,  0, {0, 0, -0.15f},         "tail");

    auto world = compute_world(skel);
    skel.inverse_bind.resize(skel.joints.size());
    for (size_t i = 0; i < skel.joints.size(); ++i)
        skel.inverse_bind[i] = glm::inverse(world[i]);

    return skel;
}

AnimalMesh generate_herbivore_mesh(const HerbivoreParams& p)
{
    AnimalMesh mesh;
    mesh.skeleton = build_herbivore_skeleton(p);

    auto world = compute_world(mesh.skeleton);
    auto jp = [&](int i) { return jpos(world[i]); };

    constexpr int SL = 8;
    float g = p.torso_girth;

    // Body — single continuous swept surface from pelvis through head
    PathNode body[] = {
        {jp(0), g*0.35f, 0},   // pelvis
        {jp(1), g*0.50f, 1},   // spine_1
        {jp(2), g*0.50f, 2},   // spine_2
        {jp(3), g*0.40f, 3},   // spine_3 (withers)
        {jp(4), g*0.22f, 4},   // neck_1
        {jp(5), g*0.14f, 5},   // neck_2
        {jp(6), g*0.10f, 6},   // head tip
    };
    append_path_tube(mesh, body, 7, p.coat_color, SL, 6);
    append_cap(mesh, jp(0), glm::normalize(jp(0) - jp(1)), g*0.35f, p.coat_color, 0, SL, 3);
    append_cap(mesh, jp(6), glm::normalize(jp(6) - jp(5)), g*0.10f, p.coat_color, 6, SL, 3);

    // Facial features
    {
        glm::vec3 head_dir = glm::normalize(jp(6) - jp(5));
        glm::vec3 head_right, head_up;
        make_frame(head_dir, head_right, head_up);

        float head_r = g * 0.115f;
        glm::vec3 eye_base = glm::mix(jp(5), jp(6), 0.6f);

        float eye_color[3] = {0.08f, 0.06f, 0.04f};
        float eye_r = head_r * 0.25f;
        glm::vec3 eye_L = eye_base + head_right * head_r * 0.9f + head_up * head_r * 0.25f;
        glm::vec3 eye_R = eye_base - head_right * head_r * 0.9f + head_up * head_r * 0.25f;
        append_cap(mesh, eye_L,  head_right, eye_r, eye_color, 6, 6, 3);
        append_cap(mesh, eye_R, -head_right, eye_r, eye_color, 6, 6, 3);

        float ear_len = g * 0.3f;
        glm::vec3 ear_base_pos = glm::mix(jp(5), jp(6), 0.35f) + head_up * head_r * 0.85f;
        glm::vec3 ear_L_base = ear_base_pos + head_right * head_r * 0.5f;
        glm::vec3 ear_R_base = ear_base_pos - head_right * head_r * 0.5f;
        glm::vec3 ear_L_tip = ear_L_base + head_up * ear_len + head_right * ear_len * 0.3f;
        glm::vec3 ear_R_tip = ear_R_base + head_up * ear_len - head_right * ear_len * 0.3f;
        float ear_r = head_r * 0.2f;
        append_tube(mesh, ear_L_base, ear_L_tip, ear_r, ear_r * 0.3f, p.coat_color, 6, 4, 2);
        append_tube(mesh, ear_R_base, ear_R_tip, ear_r, ear_r * 0.3f, p.coat_color, 6, 4, 2);
        append_cap(mesh, ear_L_tip, glm::normalize(ear_L_tip - ear_L_base), ear_r * 0.3f, p.coat_color, 6, 4, 2);
        append_cap(mesh, ear_R_tip, glm::normalize(ear_R_tip - ear_R_base), ear_r * 0.3f, p.coat_color, 6, 4, 2);

        float nose_color[3] = {0.12f, 0.08f, 0.06f};
        glm::vec3 nose_pos = jp(6) + head_dir * g * 0.02f - head_up * g * 0.03f;
        append_cap(mesh, nose_pos, head_dir, g * 0.035f, nose_color, 6, 6, 2);
    }

    // Legs — tapered segments
    auto leg = [&](int s, int u, int l, int h) {
        float t = p.leg_thickness;
        append_tube(mesh, jp(s), jp(u), t*1.1f, t*0.85f, p.coat_color,
                    static_cast<uint8_t>(s), 6, 3);
        append_tube(mesh, jp(u), jp(l), t*0.85f, t*0.6f, p.coat_color,
                    static_cast<uint8_t>(u), 6, 3);
        append_tube(mesh, jp(l), jp(h), t*0.6f, t*0.45f, p.coat_color,
                    static_cast<uint8_t>(l), 6, 3);
        append_cap(mesh, jp(h), glm::normalize(jp(h) - jp(l)), t*0.45f, p.coat_color,
                   static_cast<uint8_t>(h), 6, 2);
    };
    leg( 7,  8,  9, 10);
    leg(11, 12, 13, 14);
    leg(15, 16, 17, 18);
    leg(19, 20, 21, 22);

    // Tail
    float tt = p.leg_thickness;
    append_tube(mesh, jp(0), jp(23), tt*0.3f, tt*0.1f, p.coat_color, 0, 4, 2);
    append_cap(mesh, jp(23), glm::normalize(jp(23) - jp(0)), tt*0.1f, p.coat_color, 23, 4, 2);

    // Leg joint spheres
    auto leg_spheres = [&](int s, int u, int l, int /*h*/) {
        float t = p.leg_thickness;
        append_sphere(mesh, jp(s), t*1.1f,  p.coat_color, static_cast<uint8_t>(s), 6, 4);
        append_sphere(mesh, jp(u), t*0.90f, p.coat_color, static_cast<uint8_t>(u), 6, 4);
        append_sphere(mesh, jp(l), t*0.65f, p.coat_color, static_cast<uint8_t>(l), 6, 4);
    };
    leg_spheres( 7,  8,  9, 10);
    leg_spheres(11, 12, 13, 14);
    leg_spheres(15, 16, 17, 18);
    leg_spheres(19, 20, 21, 22);

    countershade(mesh, p.belly_color);
    return mesh;
}

} // namespace bestiary

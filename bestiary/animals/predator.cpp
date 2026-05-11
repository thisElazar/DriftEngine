#include "animals/predator.h"

namespace bestiary {

// Joint indices (same count as herbivore for animation system compatibility):
//  0 pelvis
//  1 spine_1      (lumbar — narrow waist)
//  2 spine_2      (thoracic — chest deepens)
//  3 spine_3      (withers — widest chest)
//  4 neck_1
//  5 neck_2
//  6 head          (skull base)
//  7 shoulder_L    8 upper_front_L   9 lower_front_L  10 paw_front_L
// 11 shoulder_R   12 upper_front_R  13 lower_front_R  14 paw_front_R
// 15 hip_L        16 upper_back_L   17 lower_back_L   18 paw_back_L
// 19 hip_R        20 upper_back_R   21 lower_back_R   22 paw_back_R
// 23 tail_base
// 24 tail_mid
// 25 tail_tip

Skeleton build_predator_skeleton(const PredatorParams& p)
{
    Skeleton skel;
    skel.joints.reserve(26);

    float spine_seg    = p.torso_length / 3.0f;
    float neck_seg     = p.neck_length / 2.0f;
    float half_chest   = p.chest_girth * 0.5f;
    float half_waist   = p.waist_girth * 0.5f;
    float half_front   = p.leg_length_front * 0.5f;
    float half_back    = p.leg_length_back * 0.5f;

    float spine_dy = (p.leg_length_front - p.leg_length_back) / 3.0f;

    // 0: pelvis
    add_joint(skel, -1, {0, p.leg_length_back, 0}, "pelvis");
    // 1-3: spine (waist → chest)
    add_joint(skel,  0, {0, spine_dy, spine_seg}, "spine_1");
    add_joint(skel,  1, {0, spine_dy, spine_seg}, "spine_2");
    add_joint(skel,  2, {0, spine_dy, spine_seg}, "spine_3");

    // 4-5: neck — angled forward and slightly up
    float neck_dy = neck_seg * 0.5f;
    float neck_dz = neck_seg * 0.866f;
    add_joint(skel,  3, {0, neck_dy, neck_dz}, "neck_1");
    add_joint(skel,  4, {0, neck_dy, neck_dz}, "neck_2");

    // 6: head/skull — snout extends forward
    add_joint(skel,  5, {0, 0, p.snout_length}, "head");

    // 7-10: front left leg
    add_joint(skel,  3, { half_chest, 0, 0},     "shoulder_L");
    add_joint(skel,  7, {0, -half_front, 0},      "upper_front_L");
    add_joint(skel,  8, {0, -half_front, 0},      "lower_front_L");
    add_joint(skel,  9, {0, -p.paw_size, 0},      "paw_front_L");

    // 11-14: front right leg
    add_joint(skel,  3, {-half_chest, 0, 0},      "shoulder_R");
    add_joint(skel, 11, {0, -half_front, 0},       "upper_front_R");
    add_joint(skel, 12, {0, -half_front, 0},       "lower_front_R");
    add_joint(skel, 13, {0, -p.paw_size, 0},       "paw_front_R");

    // 15-18: back left leg
    add_joint(skel,  0, { half_waist, 0, 0},       "hip_L");
    add_joint(skel, 15, {0, -half_back, 0},        "upper_back_L");
    add_joint(skel, 16, {0, -half_back, 0},        "lower_back_L");
    add_joint(skel, 17, {0, -p.paw_size, 0},       "paw_back_L");

    // 19-22: back right leg
    add_joint(skel,  0, {-half_waist, 0, 0},       "hip_R");
    add_joint(skel, 19, {0, -half_back, 0},        "upper_back_R");
    add_joint(skel, 20, {0, -half_back, 0},        "lower_back_R");
    add_joint(skel, 21, {0, -p.paw_size, 0},       "paw_back_R");

    // 23-25: tail — three segments for expressive motion
    float tail_seg = p.tail_length / 3.0f;
    add_joint(skel,  0, {0, 0.02f, -0.08f},        "tail_base");
    add_joint(skel, 23, {0, -0.02f, -tail_seg},    "tail_mid");
    add_joint(skel, 24, {0, -0.01f, -tail_seg},    "tail_tip");

    auto world = compute_world(skel);
    skel.inverse_bind.resize(skel.joints.size());
    for (size_t i = 0; i < skel.joints.size(); ++i)
        skel.inverse_bind[i] = glm::inverse(world[i]);

    return skel;
}

AnimalMesh generate_predator_mesh(const PredatorParams& p)
{
    AnimalMesh mesh;
    mesh.skeleton = build_predator_skeleton(p);

    auto world = compute_world(mesh.skeleton);
    auto jp = [&](int i) { return jpos(world[i]); };

    constexpr int SL = 8;
    float cg = p.chest_girth;
    float wg = p.waist_girth;

    // Body — continuous swept surface
    float skull_w = p.head_width * 0.5f;
    float snout_w = p.head_width * 0.22f;
    PathNode body[] = {
        {jp(0), wg*0.38f, 0},
        {jp(1), wg*0.45f, 1},
        {jp(2), cg*0.52f, 2},
        {jp(3), cg*0.50f, 3},
        {jp(4), cg*0.28f, 4},
        {jp(5), skull_w*0.85f, 5},
        {jp(6), snout_w*0.8f, 6},
    };
    append_path_tube(mesh, body, 7, p.coat_color, SL, 4);
    append_cap(mesh, jp(0), glm::normalize(jp(0) - jp(1)), wg*0.37f, p.coat_color, 0, SL, 3);
    append_cap(mesh, jp(6), glm::normalize(jp(6) - jp(5)), snout_w*0.6f, p.coat_color, 6, SL, 3);

    // Facial features
    {
        glm::vec3 head_dir = glm::normalize(jp(6) - jp(5));
        glm::vec3 head_right, head_up;
        make_frame(head_dir, head_right, head_up);

        // Eyes — forward-facing (predator vision)
        float eye_color[3] = {0.65f, 0.50f, 0.15f};
        float eye_r = skull_w * 0.18f;
        glm::vec3 eye_base = glm::mix(jp(5), jp(6), 0.35f);
        glm::vec3 eye_L = eye_base + head_right * skull_w * 0.75f + head_up * skull_w * 0.4f + head_dir * skull_w * 0.3f;
        glm::vec3 eye_R = eye_base - head_right * skull_w * 0.75f + head_up * skull_w * 0.4f + head_dir * skull_w * 0.3f;
        glm::vec3 eye_dir_L = glm::normalize(head_dir * 0.7f + head_right * 0.3f);
        glm::vec3 eye_dir_R = glm::normalize(head_dir * 0.7f - head_right * 0.3f);
        append_cap(mesh, eye_L, eye_dir_L, eye_r, eye_color, 6, 6, 3);
        append_cap(mesh, eye_R, eye_dir_R, eye_r, eye_color, 6, 6, 3);

        // Ears — triangular, erect, set further back
        float ear_len = skull_w * 0.9f;
        glm::vec3 ear_base_pos = glm::mix(jp(5), jp(6), 0.15f) + head_up * skull_w * 0.65f;
        glm::vec3 ear_L_base = ear_base_pos + head_right * skull_w * 0.6f;
        glm::vec3 ear_R_base = ear_base_pos - head_right * skull_w * 0.6f;
        glm::vec3 ear_L_tip = ear_L_base + head_up * ear_len + head_right * ear_len * 0.2f;
        glm::vec3 ear_R_tip = ear_R_base + head_up * ear_len - head_right * ear_len * 0.2f;
        float ear_r = skull_w * 0.15f;
        append_tube(mesh, ear_L_base, ear_L_tip, ear_r, ear_r * 0.15f, p.coat_color, 6, 4, 2);
        append_tube(mesh, ear_R_base, ear_R_tip, ear_r, ear_r * 0.15f, p.coat_color, 6, 4, 2);
        append_cap(mesh, ear_L_tip, glm::normalize(ear_L_tip - ear_L_base), ear_r * 0.15f, p.coat_color, 6, 4, 2);
        append_cap(mesh, ear_R_tip, glm::normalize(ear_R_tip - ear_R_base), ear_r * 0.15f, p.coat_color, 6, 4, 2);

        // Nose — dark, at tip of snout
        float nose_color[3] = {0.05f, 0.03f, 0.02f};
        glm::vec3 nose_pos = jp(6) + head_dir * snout_w * 0.3f;
        append_cap(mesh, nose_pos, head_dir, snout_w * 0.45f, nose_color, 6, 6, 2);
    }

    // Legs — lean, longer hind legs
    auto leg = [&](int s, int u, int l, int h) {
        float t = p.leg_thickness;
        append_tube(mesh, jp(s), jp(u), t*1.05f, t*0.80f, p.coat_color,
                    static_cast<uint8_t>(s), 6, 3);
        append_tube(mesh, jp(u), jp(l), t*0.80f, t*0.55f, p.coat_color,
                    static_cast<uint8_t>(u), 6, 3);
        append_tube(mesh, jp(l), jp(h), t*0.55f, t*0.40f, p.coat_color,
                    static_cast<uint8_t>(l), 6, 3);
        append_cap(mesh, jp(h), glm::normalize(jp(h) - jp(l)), t*0.40f, p.coat_color,
                   static_cast<uint8_t>(h), 6, 2);
    };
    leg( 7,  8,  9, 10);
    leg(11, 12, 13, 14);
    leg(15, 16, 17, 18);
    leg(19, 20, 21, 22);

    // Tail — three segments, bushy
    float tt = p.leg_thickness;
    append_tube(mesh, jp(23), jp(24), tt*0.5f, tt*0.55f, p.coat_color, 23, 5, 2);
    append_tube(mesh, jp(24), jp(25), tt*0.55f, tt*0.3f, p.coat_color, 24, 5, 2);
    append_cap(mesh, jp(25), glm::normalize(jp(25) - jp(24)), tt*0.3f, p.coat_color, 25, 5, 2);

    // Leg joint spheres
    auto leg_spheres = [&](int s, int u, int l, int /*h*/) {
        float t = p.leg_thickness;
        append_sphere(mesh, jp(s), t*1.05f, p.coat_color, static_cast<uint8_t>(s), 6, 4);
        append_sphere(mesh, jp(u), t*0.85f, p.coat_color, static_cast<uint8_t>(u), 6, 4);
        append_sphere(mesh, jp(l), t*0.60f, p.coat_color, static_cast<uint8_t>(l), 6, 4);
    };
    leg_spheres( 7,  8,  9, 10);
    leg_spheres(11, 12, 13, 14);
    leg_spheres(15, 16, 17, 18);
    leg_spheres(19, 20, 21, 22);

    // Tail joint spheres
    append_sphere(mesh, jp(23), tt*0.5f, p.coat_color, 23, 4, 3);
    append_sphere(mesh, jp(24), tt*0.55f, p.coat_color, 24, 4, 3);

    return mesh;
}

} // namespace bestiary

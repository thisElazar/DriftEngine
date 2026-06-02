#include "camera.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <algorithm>
#include <cmath>

static glm::quat orientation_from_up(glm::vec3 up)
{
    up = glm::normalize(up);
    glm::vec3 world_up(0.0f, 1.0f, 0.0f);

    float d = glm::dot(up, world_up);
    if (d > 0.9999f)
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    if (d < -0.9999f)
        return glm::angleAxis(glm::pi<float>(), glm::vec3(0.0f, 0.0f, 1.0f));

    glm::vec3 axis = glm::normalize(glm::cross(world_up, up));
    float angle = std::acos(std::clamp(d, -1.0f, 1.0f));
    return glm::angleAxis(angle, axis);
}

// ---------- shared accessors ----------

glm::quat camera_orientation(const Camera& cam)
{
    return cam.mode == CameraMode::Orbital ? cam.orbit.orientation : cam.fp.orientation;
}

glm::dvec3 camera_eye_position(const Camera& cam)
{
    if (cam.mode == CameraMode::FirstPerson)
        return cam.fp.eye;

    glm::vec3 offset = cam.orbit.orientation
                     * glm::vec3(0.0f, 0.0f, static_cast<float>(cam.orbit.arm_length));
    return cam.orbit.pivot + glm::dvec3(offset);
}

glm::vec3 camera_forward(const Camera& cam)
{
    return glm::normalize(camera_orientation(cam) * glm::vec3(0.0f, 0.0f, -1.0f));
}

glm::vec3 camera_up(const Camera& cam)
{
    return glm::normalize(camera_orientation(cam) * glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::vec3 camera_right(const Camera& cam)
{
    return glm::normalize(camera_orientation(cam) * glm::vec3(1.0f, 0.0f, 0.0f));
}

// ---------- orbital ----------

static void orbital_mouse_look(OrbitalState& s, float dx, float dy, float sensitivity)
{
    glm::vec3 radial_up = glm::normalize(glm::vec3(s.pivot));
    glm::vec3 right = glm::normalize(s.orientation * glm::vec3(1.0f, 0.0f, 0.0f));

    glm::quat yaw_rot = glm::angleAxis(-dx * sensitivity, radial_up);
    glm::quat pitch_rot = glm::angleAxis(-dy * sensitivity, right);

    s.orientation = glm::normalize(yaw_rot * pitch_rot * s.orientation);

    glm::vec3 fwd = s.orientation * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 corrected_right = glm::normalize(glm::cross(fwd, radial_up));
    glm::vec3 corrected_up = glm::cross(corrected_right, fwd);
    glm::mat3 m(corrected_right, corrected_up, -fwd);
    s.orientation = glm::normalize(glm::quat_cast(m));
}

static CameraUpdateResult orbital_update(OrbitalState& s, const InputFrame& in, float dt,
                                         float planet_radius,
                                         const std::function<float(glm::vec3)>& height_fn)
{
    glm::dvec3 pivot = s.pivot;
    glm::vec3 pivot_dir = glm::normalize(glm::vec3(pivot));
    double terrain_height = static_cast<double>(height_fn(pivot_dir));
    double terrain_radius = static_cast<double>(planet_radius) + terrain_height;

    glm::vec3 arm_offset(0.0f, 0.0f, static_cast<float>(s.arm_length));
    glm::dvec3 eye = pivot + glm::dvec3(s.orientation * arm_offset);
    double eye_dist = glm::length(eye);
    double altitude = eye_dist - terrain_radius;

    float base_speed = static_cast<float>(std::max(altitude, 10.0)) * 1.5f;
    float speed = base_speed;
    if (in.key_shift) speed *= 3.0f;
    if (in.key_alt)   speed *= 0.2f;

    glm::vec3 radial_up = pivot_dir;
    glm::vec3 fwd = glm::normalize(s.orientation * glm::vec3(0.0f, 0.0f, -1.0f));
    glm::vec3 right = glm::normalize(s.orientation * glm::vec3(1.0f, 0.0f, 0.0f));

    glm::vec3 fwd_tangent = fwd - glm::dot(fwd, radial_up) * radial_up;
    glm::vec3 right_tangent = right - glm::dot(right, radial_up) * radial_up;
    if (glm::dot(fwd_tangent, fwd_tangent) > 1e-6f)   fwd_tangent = glm::normalize(fwd_tangent);
    if (glm::dot(right_tangent, right_tangent) > 1e-6f) right_tangent = glm::normalize(right_tangent);

    glm::vec3 move(0.0f);
    if (in.key_w) move += fwd_tangent;
    if (in.key_s) move -= fwd_tangent;
    if (in.key_d) move += right_tangent;
    if (in.key_a) move -= right_tangent;
    if (in.key_q) move += radial_up;
    if (in.key_e) move -= radial_up;

    if (glm::dot(move, move) > 0.0f) {
        glm::vec3 dir = glm::normalize(move);
        s.pivot += glm::dvec3(dir) * static_cast<double>(speed * dt);
    }

    eye = pivot + glm::dvec3(s.orientation * arm_offset);
    eye_dist = glm::length(eye);
    glm::vec3 eye_dir = glm::normalize(glm::vec3(eye));
    double h_at_eye = static_cast<double>(height_fn(eye_dir));
    double min_eye_radius = static_cast<double>(planet_radius) + h_at_eye + 2.0;
    if (eye_dist < min_eye_radius) {
        glm::dvec3 push_dir = glm::normalize(eye);
        double push = min_eye_radius - eye_dist;
        s.pivot += push_dir * push;
        altitude = 2.0;
    } else {
        altitude = eye_dist - (static_cast<double>(planet_radius) + h_at_eye);
    }

    return {terrain_height, altitude};
}

// ---------- first person ----------

static void fp_mouse_look(FirstPersonState& s, float dx, float dy, float sensitivity)
{
    glm::vec3 radial_up = glm::normalize(glm::vec3(s.eye));
    glm::vec3 right = glm::normalize(s.orientation * glm::vec3(1.0f, 0.0f, 0.0f));

    glm::quat yaw_rot = glm::angleAxis(-dx * sensitivity, radial_up);
    glm::quat pitch_rot = glm::angleAxis(-dy * sensitivity, right);

    s.orientation = glm::normalize(yaw_rot * pitch_rot * s.orientation);

    // Re-anchor "up" to radial — prevents roll drift and keeps pitch clamped naturally
    // by reconstructing the basis. Pitch is limited by clamping the new forward
    // vector against the radial pole.
    glm::vec3 fwd = s.orientation * glm::vec3(0.0f, 0.0f, -1.0f);
    float pole_dot = glm::dot(fwd, radial_up);
    const float pole_limit = 0.985f;
    if (pole_dot >  pole_limit) fwd = glm::normalize(fwd - (pole_dot -  pole_limit) * radial_up);
    if (pole_dot < -pole_limit) fwd = glm::normalize(fwd - (pole_dot - -pole_limit) * radial_up);

    glm::vec3 corrected_right = glm::normalize(glm::cross(fwd, radial_up));
    glm::vec3 corrected_up = glm::cross(corrected_right, fwd);
    glm::mat3 m(corrected_right, corrected_up, -fwd);
    s.orientation = glm::normalize(glm::quat_cast(m));
}

static CameraUpdateResult fp_update(FirstPersonState& s, const InputFrame& in, float dt,
                                    float planet_radius,
                                    const std::function<float(glm::vec3)>& height_fn)
{
    // Fly-to-cursor: parametric great-circle path with sin-bump altitude
    // arc. Slerp the direction from start to target so we follow the curve
    // of the planet (a straight 3D line clips through terrain over distance).
    // Smoothstep ease in/out for natural acceleration. Hard-clamp every step
    // to stay above terrain so we never tunnel.
    if (s.warping) {
        s.warp_elapsed += static_cast<double>(dt);
        double t = std::clamp(s.warp_elapsed / std::max(s.warp_duration, 1e-3), 0.0, 1.0);
        double ts = t * t * (3.0 - 2.0 * t);  // smoothstep ease in/out

        // Slerp direction along the great circle.
        double dot01 = glm::clamp(glm::dot(s.warp_d0, s.warp_d1), -1.0, 1.0);
        double angle = std::acos(dot01);
        glm::dvec3 dir;
        if (angle < 1e-6) {
            dir = glm::normalize(glm::mix(s.warp_d0, s.warp_d1, ts));
        } else {
            double sa = std::sin(angle);
            dir = glm::normalize(
                (std::sin((1.0 - ts) * angle) / sa) * s.warp_d0
              + (std::sin(ts * angle) / sa) * s.warp_d1);
        }

        // Radius: lerp endpoint radii + sin-arc altitude bump (peaks at t=0.5).
        double r_lerp = s.warp_r0 + (s.warp_r1 - s.warp_r0) * ts;
        double r_arc = std::sin(t * 3.14159265358979) * s.warp_arc_h;
        double r = r_lerp + r_arc;

        // Stay above terrain — no clipping at any point along the path.
        glm::vec3 dirf = glm::vec3(dir);
        double h_here = static_cast<double>(height_fn(dirf));
        double min_r = static_cast<double>(planet_radius) + h_here
                     + static_cast<double>(s.eye_height_offset);
        if (r < min_r) r = min_r;

        s.eye = dir * r;

        if (t >= 1.0) {
            // Land — snap to target, restore walking state.
            glm::vec3 land_dir = glm::vec3(s.warp_d1);
            float land_h = height_fn(land_dir);
            double land_r = static_cast<double>(planet_radius) + static_cast<double>(land_h)
                          + static_cast<double>(s.eye_height_offset);
            s.eye = glm::dvec3(land_dir) * land_r;
            s.warping = false;
            s.vertical_velocity = 0.0f;
            s.grounded = true;
        }

        double altitude = glm::length(s.eye)
                        - (static_cast<double>(planet_radius) + h_here);
        return {h_here, altitude};
    }

    glm::vec3 eye_dir = glm::normalize(glm::vec3(s.eye));
    double terrain_height = static_cast<double>(height_fn(eye_dir));
    double terrain_radius = static_cast<double>(planet_radius) + terrain_height;
    double eye_dist = glm::length(s.eye);
    double altitude = eye_dist - terrain_radius;

    bool vertical_input = in.key_q || in.key_e;

    float speed = s.walk_speed;
    if (!s.grounded && !vertical_input)
        speed *= std::clamp(static_cast<float>(altitude) * 0.1f, 1.0f, 20.0f);
    if (in.key_shift) speed *= 4.0f;
    if (in.key_alt)   speed *= 0.25f;

    glm::vec3 radial_up = eye_dir;
    glm::vec3 fwd = glm::normalize(s.orientation * glm::vec3(0.0f, 0.0f, -1.0f));
    glm::vec3 right = glm::normalize(s.orientation * glm::vec3(1.0f, 0.0f, 0.0f));

    glm::vec3 fwd_tangent = fwd - glm::dot(fwd, radial_up) * radial_up;
    glm::vec3 right_tangent = right - glm::dot(right, radial_up) * radial_up;
    if (glm::dot(fwd_tangent, fwd_tangent) > 1e-6f)     fwd_tangent = glm::normalize(fwd_tangent);
    if (glm::dot(right_tangent, right_tangent) > 1e-6f) right_tangent = glm::normalize(right_tangent);

    // Horizontal movement (tangent to sphere)
    glm::vec3 move(0.0f);
    if (in.key_w) move += fwd_tangent;
    if (in.key_s) move -= fwd_tangent;
    if (in.key_d) move += right_tangent;
    if (in.key_a) move -= right_tangent;

    if (glm::dot(move, move) > 0.0f) {
        glm::vec3 dir = glm::normalize(move);
        s.eye += glm::dvec3(dir) * static_cast<double>(speed * dt);
    }

    // Vertical: Q/E give direct thrust, otherwise gravity pulls toward ground
    if (vertical_input) {
        float vert = 0.0f;
        if (in.key_q) vert += 1.0f;
        if (in.key_e) vert -= 1.0f;
        s.vertical_velocity = vert * speed;
        s.grounded = false;
    } else if (!s.grounded) {
        s.vertical_velocity -= 9.81f * dt;
    }

    if (!s.grounded || vertical_input) {
        s.eye += glm::dvec3(radial_up) * static_cast<double>(s.vertical_velocity * dt);
    }

    // Smooth height tracking: lerp toward terrain + eye_height
    glm::vec3 new_dir = glm::normalize(glm::vec3(s.eye));
    double h_here = static_cast<double>(height_fn(new_dir));
    double ground_radius = static_cast<double>(planet_radius) + h_here
                         + static_cast<double>(s.eye_height_offset);
    double cur_dist = glm::length(s.eye);

    // Walking on the ground: always lerp toward target height. Asymmetric
    // rate — fast for stepping up onto rises so we don't clip through them,
    // slower for stepping down so it feels like walking, not falling. Using
    // a single lerp (not snap-vs-lerp) avoids jitter when stamp falloff makes
    // ground_radius oscillate by a few cm across the eye position.
    if (s.grounded && !vertical_input) {
        double diff = ground_radius - cur_dist;
        double rate = (diff > 0.0) ? 30.0 : 12.0;
        double a = std::clamp(static_cast<double>(dt) * rate, 0.0, 1.0);
        double new_r = cur_dist + diff * a;
        // Hard floor — never let the eye dip below the surface. Clipping
        // breaks the ray-pick (sample at ray_origin returns negative and the
        // forward march early-outs).
        double floor_r = static_cast<double>(planet_radius) + h_here
                       + static_cast<double>(s.eye_height_offset) * 0.4;
        if (new_r < floor_r) new_r = floor_r;
        s.eye = glm::dvec3(new_dir) * new_r;
        s.vertical_velocity = 0.0f;
        altitude = new_r - (static_cast<double>(planet_radius) + h_here);
    } else if (cur_dist <= ground_radius) {
        // Falling — landed. Clamp and zero velocity.
        s.eye = glm::dvec3(new_dir) * ground_radius;
        s.vertical_velocity = 0.0f;
        s.grounded = true;
        altitude = static_cast<double>(s.eye_height_offset);
    } else {
        altitude = cur_dist - (static_cast<double>(planet_radius) + h_here);
    }

    return {terrain_height, altitude};
}

// ---------- public dispatch ----------

void camera_apply_mouse_look(Camera& cam, float dx, float dy, float sensitivity)
{
    if (cam.mode == CameraMode::Orbital)
        orbital_mouse_look(cam.orbit, dx, dy, sensitivity);
    else
        fp_mouse_look(cam.fp, dx, dy, sensitivity);
}

void camera_zoom(Camera& cam, double amount)
{
    if (cam.mode == CameraMode::Orbital) {
        double factor = 1.0 - amount * 0.1;
        cam.orbit.arm_length = std::clamp(cam.orbit.arm_length * factor, 10.0, 50000000.0);
    } else {
        // Scroll adjusts walk speed in FP mode.
        float factor = 1.0f - static_cast<float>(amount) * 0.15f;
        cam.fp.walk_speed = std::clamp(cam.fp.walk_speed * factor, 0.5f, 200.0f);
    }
}

void camera_initialize_orientation(Camera& cam)
{
    if (cam.mode == CameraMode::Orbital) {
        glm::vec3 up = glm::normalize(glm::vec3(cam.orbit.pivot));
        cam.orbit.orientation = orientation_from_up(up);
    } else {
        glm::vec3 up = glm::normalize(glm::vec3(cam.fp.eye));
        cam.fp.orientation = orientation_from_up(up);
    }
}

CameraUpdateResult camera_update(Camera& cam, const InputFrame& in, float dt,
                                 float planet_radius,
                                 std::function<float(glm::vec3)> height_fn)
{
    return cam.mode == CameraMode::Orbital
        ? orbital_update(cam.orbit, in, dt, planet_radius, height_fn)
        : fp_update(cam.fp, in, dt, planet_radius, height_fn);
}

glm::mat4 camera_build_view(const Camera& cam)
{
    glm::quat orient = camera_orientation(cam);
    glm::mat3 rot = glm::mat3_cast(orient);
    glm::mat3 inv_rot = glm::transpose(rot);

    // Rotation only — tile positions are already camera-relative (rel_d
    // subtracts the full eye position on the CPU in double precision).
    glm::mat4 view(1.0f);
    view[0] = glm::vec4(inv_rot[0], 0.0f);
    view[1] = glm::vec4(inv_rot[1], 0.0f);
    view[2] = glm::vec4(inv_rot[2], 0.0f);
    view[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return view;
}

glm::mat4 camera_build_proj(const Camera& cam, float aspect)
{
    float f = 1.0f / std::tan(cam.fov_y * 0.5f);
    glm::mat4 proj(0.0f);
    proj[0][0] = f / aspect;
    proj[1][1] = -f;
    proj[2][2] = 0.0f;
    proj[2][3] = -1.0f;
    proj[3][2] = cam.near_plane;
    return proj;
}

void camera_switch_to_first_person(Camera& cam, float planet_radius,
                                   std::function<float(glm::vec3)> height_fn)
{
    if (cam.mode == CameraMode::FirstPerson) return;

    // Place the FP eye at the orbital pivot's surface position + eye_height.
    glm::dvec3 pivot = cam.orbit.pivot;
    glm::vec3 dir = glm::normalize(glm::vec3(pivot));
    double h = static_cast<double>(height_fn(dir));
    double target_radius = static_cast<double>(planet_radius) + h
                         + static_cast<double>(cam.fp.eye_height_offset);

    cam.fp.eye = glm::dvec3(dir) * target_radius;

    // Orbital forward looks back at the pivot — that's straight down once we're
    // standing on it. Re-anchor: project orbital forward onto the tangent plane
    // at the new eye and rebuild orientation looking along the horizon.
    glm::vec3 radial_up = dir;
    glm::vec3 orbit_fwd = glm::normalize(cam.orbit.orientation * glm::vec3(0.0f, 0.0f, -1.0f));
    glm::vec3 tangent_fwd = orbit_fwd - glm::dot(orbit_fwd, radial_up) * radial_up;
    if (glm::dot(tangent_fwd, tangent_fwd) < 1e-6f) {
        // Degenerate (orbital was looking straight down). Pick an arbitrary tangent.
        glm::vec3 alt = std::abs(radial_up.y) > 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        tangent_fwd = glm::normalize(alt - glm::dot(alt, radial_up) * radial_up);
    } else {
        tangent_fwd = glm::normalize(tangent_fwd);
    }
    glm::vec3 right = glm::normalize(glm::cross(tangent_fwd, radial_up));
    glm::mat3 m(right, radial_up, -tangent_fwd);
    cam.fp.orientation = glm::normalize(glm::quat_cast(m));

    cam.mode = CameraMode::FirstPerson;
}

void camera_switch_to_orbital(Camera& cam)
{
    if (cam.mode == CameraMode::Orbital) return;

    // Pivot at the FP eye, short arm — preserves view continuity.
    cam.orbit.pivot = cam.fp.eye;
    cam.orbit.orientation = cam.fp.orientation;
    cam.orbit.arm_length = std::max(cam.orbit.arm_length, 50.0);
    cam.fp.warping = false;
    cam.mode = CameraMode::Orbital;
}

void camera_begin_warp_to(Camera& cam, glm::dvec3 target_eye)
{
    if (cam.mode != CameraMode::FirstPerson) return;

    glm::dvec3 d0 = glm::normalize(cam.fp.eye);
    glm::dvec3 d1 = glm::normalize(target_eye);
    double r0 = glm::length(cam.fp.eye);
    double r1 = glm::length(target_eye);

    double dot01 = glm::clamp(glm::dot(d0, d1), -1.0, 1.0);
    double angle = std::acos(dot01);
    double arc_dist = angle * r0;  // surface-arc distance, approx

    cam.fp.warp_d0 = d0;
    cam.fp.warp_d1 = d1;
    cam.fp.warp_r0 = r0;
    cam.fp.warp_r1 = r1;
    cam.fp.warp_arc_h = std::clamp(arc_dist * 0.1, 50.0, 5000.0);
    cam.fp.warp_duration = std::clamp(arc_dist / 10000.0, 0.4, 3.0);
    cam.fp.warp_elapsed = 0.0;
    cam.fp.warping = true;
    cam.fp.vertical_velocity = 0.0f;
    cam.fp.grounded = false;  // we're in flight
}

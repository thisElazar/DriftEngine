#include "camera.h"
#include <GLFW/glfw3.h>
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

static CameraUpdateResult orbital_update(OrbitalState& s, GLFWwindow* window, float dt,
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
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 3.0f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)   speed *= 0.2f;

    glm::vec3 radial_up = pivot_dir;
    glm::vec3 fwd = glm::normalize(s.orientation * glm::vec3(0.0f, 0.0f, -1.0f));
    glm::vec3 right = glm::normalize(s.orientation * glm::vec3(1.0f, 0.0f, 0.0f));

    glm::vec3 fwd_tangent = fwd - glm::dot(fwd, radial_up) * radial_up;
    glm::vec3 right_tangent = right - glm::dot(right, radial_up) * radial_up;
    if (glm::dot(fwd_tangent, fwd_tangent) > 1e-6f)   fwd_tangent = glm::normalize(fwd_tangent);
    if (glm::dot(right_tangent, right_tangent) > 1e-6f) right_tangent = glm::normalize(right_tangent);

    glm::vec3 move(0.0f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += fwd_tangent;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= fwd_tangent;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += right_tangent;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= right_tangent;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move += radial_up;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move -= radial_up;

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

static CameraUpdateResult fp_update(FirstPersonState& s, GLFWwindow* window, float dt,
                                    float planet_radius,
                                    const std::function<float(glm::vec3)>& height_fn)
{
    glm::vec3 eye_dir = glm::normalize(glm::vec3(s.eye));
    double terrain_height = static_cast<double>(height_fn(eye_dir));
    double terrain_radius = static_cast<double>(planet_radius) + terrain_height;
    double eye_dist = glm::length(s.eye);
    double altitude = eye_dist - terrain_radius;

    float speed = s.walk_speed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 4.0f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT)   == GLFW_PRESS) speed *= 0.25f;

    glm::vec3 radial_up = eye_dir;
    glm::vec3 fwd = glm::normalize(s.orientation * glm::vec3(0.0f, 0.0f, -1.0f));
    glm::vec3 right = glm::normalize(s.orientation * glm::vec3(1.0f, 0.0f, 0.0f));

    glm::vec3 fwd_tangent = fwd - glm::dot(fwd, radial_up) * radial_up;
    glm::vec3 right_tangent = right - glm::dot(right, radial_up) * radial_up;
    if (glm::dot(fwd_tangent, fwd_tangent) > 1e-6f)     fwd_tangent = glm::normalize(fwd_tangent);
    if (glm::dot(right_tangent, right_tangent) > 1e-6f) right_tangent = glm::normalize(right_tangent);

    glm::vec3 move(0.0f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += fwd_tangent;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= fwd_tangent;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += right_tangent;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= right_tangent;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move += radial_up;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move -= radial_up;

    if (glm::dot(move, move) > 0.0f) {
        glm::vec3 dir = glm::normalize(move);
        s.eye += glm::dvec3(dir) * static_cast<double>(speed * dt);
    }

    // Snap eye to terrain + offset (only when not flying upward via Q/E).
    glm::vec3 new_dir = glm::normalize(glm::vec3(s.eye));
    double h_here = static_cast<double>(height_fn(new_dir));
    double target_radius = static_cast<double>(planet_radius) + h_here
                         + static_cast<double>(s.eye_height_offset);
    double cur_dist = glm::length(s.eye);

    bool vertical_input = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS
                       || glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;

    if (!vertical_input) {
        glm::dvec3 d = glm::dvec3(new_dir);
        s.eye = d * target_radius;
        altitude = static_cast<double>(s.eye_height_offset);
    } else {
        double min_radius = static_cast<double>(planet_radius) + h_here
                          + static_cast<double>(s.eye_height_offset) * 0.5;
        if (cur_dist < min_radius) {
            s.eye = glm::dvec3(new_dir) * min_radius;
            altitude = min_radius - (static_cast<double>(planet_radius) + h_here);
        } else {
            altitude = cur_dist - (static_cast<double>(planet_radius) + h_here);
        }
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

CameraUpdateResult camera_update(Camera& cam, GLFWwindow* window, float dt,
                                 float planet_radius,
                                 std::function<float(glm::vec3)> height_fn)
{
    return cam.mode == CameraMode::Orbital
        ? orbital_update(cam.orbit, window, dt, planet_radius, height_fn)
        : fp_update(cam.fp, window, dt, planet_radius, height_fn);
}

glm::mat4 camera_build_view(const Camera& cam)
{
    glm::quat orient = camera_orientation(cam);
    glm::mat3 rot = glm::mat3_cast(orient);
    glm::mat3 inv_rot = glm::transpose(rot);

    if (cam.mode == CameraMode::Orbital) {
        glm::vec3 arm_offset(0.0f, 0.0f, static_cast<float>(cam.orbit.arm_length));
        glm::vec3 eye_local = rot * arm_offset;
        glm::mat4 view(1.0f);
        view[0] = glm::vec4(inv_rot[0], 0.0f);
        view[1] = glm::vec4(inv_rot[1], 0.0f);
        view[2] = glm::vec4(inv_rot[2], 0.0f);
        view[3] = glm::vec4(-(inv_rot * eye_local), 1.0f);
        return view;
    }

    // FP: camera-relative — eye is at origin in its own frame.
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
    cam.mode = CameraMode::Orbital;
}

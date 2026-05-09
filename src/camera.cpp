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

glm::dvec3 camera_eye_position(const Camera& cam)
{
    glm::vec3 offset = cam.orientation * glm::vec3(0.0f, 0.0f, static_cast<float>(cam.arm_length));
    return glm::dvec3(cam.pos_x, cam.pos_y, cam.pos_z) + glm::dvec3(offset);
}

glm::vec3 camera_forward(const Camera& cam)
{
    return glm::normalize(cam.orientation * glm::vec3(0.0f, 0.0f, -1.0f));
}

glm::vec3 camera_up(const Camera& cam)
{
    return glm::normalize(cam.orientation * glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::vec3 camera_right(const Camera& cam)
{
    return glm::normalize(cam.orientation * glm::vec3(1.0f, 0.0f, 0.0f));
}

void camera_apply_mouse_look(Camera& cam, float dx, float dy, float sensitivity)
{
    // Yaw around the radial "up" at the pivot, pitch around camera right.
    // This orbits the camera around the pivot point.
    glm::vec3 radial_up = glm::normalize(glm::vec3(
        static_cast<float>(cam.pos_x),
        static_cast<float>(cam.pos_y),
        static_cast<float>(cam.pos_z)));

    glm::vec3 right = camera_right(cam);

    glm::quat yaw_rot = glm::angleAxis(-dx * sensitivity, radial_up);
    glm::quat pitch_rot = glm::angleAxis(-dy * sensitivity, right);

    cam.orientation = glm::normalize(yaw_rot * pitch_rot * cam.orientation);

    // Prevent roll drift
    glm::vec3 fwd = cam.orientation * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 corrected_right = glm::normalize(glm::cross(fwd, radial_up));
    glm::vec3 corrected_up = glm::cross(corrected_right, fwd);
    glm::mat3 m(corrected_right, corrected_up, -fwd);
    cam.orientation = glm::normalize(glm::quat_cast(m));
}

void camera_zoom(Camera& cam, double amount)
{
    double factor = 1.0 - amount * 0.1;
    cam.arm_length = std::clamp(cam.arm_length * factor, 10.0, 50000000.0);
}

void camera_initialize_orientation(Camera& cam)
{
    glm::vec3 up = glm::normalize(glm::vec3(
        static_cast<float>(cam.pos_x),
        static_cast<float>(cam.pos_y),
        static_cast<float>(cam.pos_z)));

    cam.orientation = orientation_from_up(up);
}

CameraUpdateResult camera_update(Camera& cam, GLFWwindow* window, float dt,
                                 float planet_radius,
                                 std::function<float(glm::vec3)> height_fn)
{
    // Pivot is at pos_x/y/z. Compute altitude at pivot.
    glm::dvec3 pivot(cam.pos_x, cam.pos_y, cam.pos_z);
    glm::vec3 pivot_dir = glm::normalize(glm::vec3(pivot));
    double terrain_height = static_cast<double>(height_fn(pivot_dir));
    double terrain_radius = static_cast<double>(planet_radius) + terrain_height;

    // Eye position for altitude-based speed
    glm::dvec3 eye = camera_eye_position(cam);
    double eye_dist = glm::length(eye);
    double altitude = eye_dist - terrain_radius;

    float base_speed = static_cast<float>(std::max(altitude, 10.0)) * 1.5f;
    float speed = base_speed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        speed *= 3.0f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
        speed *= 0.2f;

    // WASD moves the pivot along the surface tangent plane, Q/E moves radially
    glm::vec3 radial_up = pivot_dir;
    glm::vec3 fwd = camera_forward(cam);
    glm::vec3 right = camera_right(cam);

    glm::vec3 fwd_tangent = fwd - glm::dot(fwd, radial_up) * radial_up;
    glm::vec3 right_tangent = right - glm::dot(right, radial_up) * radial_up;
    if (glm::dot(fwd_tangent, fwd_tangent) > 1e-6f)
        fwd_tangent = glm::normalize(fwd_tangent);
    if (glm::dot(right_tangent, right_tangent) > 1e-6f)
        right_tangent = glm::normalize(right_tangent);

    glm::vec3 move(0.0f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += fwd_tangent;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= fwd_tangent;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += right_tangent;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= right_tangent;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move += radial_up;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move -= radial_up;

    if (glm::dot(move, move) > 0.0f) {
        glm::vec3 dir = glm::normalize(move);
        cam.pos_x += static_cast<double>(dir.x) * speed * dt;
        cam.pos_y += static_cast<double>(dir.y) * speed * dt;
        cam.pos_z += static_cast<double>(dir.z) * speed * dt;
    }

    // Clamp pivot so eye doesn't go below terrain
    eye = camera_eye_position(cam);
    eye_dist = glm::length(eye);
    glm::vec3 eye_dir = glm::normalize(glm::vec3(eye));
    double h_at_eye = static_cast<double>(height_fn(eye_dir));
    double min_eye_radius = static_cast<double>(planet_radius) + h_at_eye + 2.0;
    if (eye_dist < min_eye_radius) {
        glm::dvec3 push_dir = glm::normalize(eye);
        double push = min_eye_radius - eye_dist;
        cam.pos_x += push_dir.x * push;
        cam.pos_y += push_dir.y * push;
        cam.pos_z += push_dir.z * push;
        altitude = 2.0;
    } else {
        altitude = eye_dist - (static_cast<double>(planet_radius) + h_at_eye);
    }

    return {terrain_height, altitude};
}

glm::mat4 camera_build_view(const Camera& cam)
{
    glm::mat3 rot = glm::mat3_cast(cam.orientation);
    glm::mat3 inv_rot = glm::transpose(rot);
    // Camera-relative: translate by arm length along local Z
    glm::vec3 arm_offset(0.0f, 0.0f, static_cast<float>(cam.arm_length));
    glm::vec3 eye_local = rot * arm_offset;
    glm::mat4 view(1.0f);
    view[0] = glm::vec4(inv_rot[0], 0.0f);
    view[1] = glm::vec4(inv_rot[1], 0.0f);
    view[2] = glm::vec4(inv_rot[2], 0.0f);
    view[3] = glm::vec4(-(inv_rot * eye_local), 1.0f);
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

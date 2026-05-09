#include "camera.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>

glm::vec3 camera_forward(const Camera& cam)
{
    float cy = std::cos(cam.yaw);
    float sy = std::sin(cam.yaw);
    float cp = std::cos(cam.pitch);
    float sp = std::sin(cam.pitch);
    return glm::vec3(cy * cp, sp, sy * cp);
}

void camera_apply_mouse_look(Camera& cam, float dx, float dy, float sensitivity)
{
    cam.yaw   += dx * sensitivity;
    cam.pitch -= dy * sensitivity;
    cam.pitch = std::clamp(cam.pitch, -1.5f, 1.5f);
}

CameraUpdateResult camera_update(Camera& cam, GLFWwindow* window, float dt,
                                 float planet_radius,
                                 std::function<float(glm::vec3)> height_fn)
{
    glm::dvec3 cam_pos_d(cam.pos_x, cam.pos_y, cam.pos_z);
    double cam_dist = glm::length(cam_pos_d);
    glm::vec3 cam_dir_f = glm::normalize(glm::vec3(cam_pos_d));
    double terrain_height = static_cast<double>(height_fn(cam_dir_f));
    double terrain_radius = static_cast<double>(planet_radius) + terrain_height;
    double altitude = cam_dist - terrain_radius;

    float base_speed = static_cast<float>(std::max(altitude, 10.0)) * 1.5f;
    float speed = base_speed;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        speed *= 3.0f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
        speed *= 0.2f;

    glm::vec3 fwd = camera_forward(cam);
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
    glm::vec3 up(0, 1, 0);

    glm::vec3 move(0.0f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) move += fwd;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) move -= fwd;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) move += right;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) move -= right;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) move += up;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) move -= up;

    if (glm::dot(move, move) > 0.0f) {
        glm::vec3 dir = glm::normalize(move);
        cam.pos_x += static_cast<double>(dir.x) * speed * dt;
        cam.pos_y += static_cast<double>(dir.y) * speed * dt;
        cam.pos_z += static_cast<double>(dir.z) * speed * dt;
    }

    cam_pos_d = glm::dvec3(cam.pos_x, cam.pos_y, cam.pos_z);
    cam_dist = glm::length(cam_pos_d);
    cam_dir_f = glm::normalize(glm::vec3(cam_pos_d));
    double h_at_new_pos = static_cast<double>(height_fn(cam_dir_f));
    double min_radius = static_cast<double>(planet_radius) + h_at_new_pos + 2.0;
    if (cam_dist < min_radius) {
        glm::dvec3 cam_dir_d = cam_pos_d / cam_dist;
        cam.pos_x = cam_dir_d.x * min_radius;
        cam.pos_y = cam_dir_d.y * min_radius;
        cam.pos_z = cam_dir_d.z * min_radius;
        altitude = 2.0;
    }

    return {terrain_height, altitude};
}

glm::mat4 camera_build_view(const Camera& cam)
{
    glm::vec3 fwd = camera_forward(cam);
    return glm::lookAtRH(glm::vec3(0.0f), fwd, glm::vec3(0.0f, 1.0f, 0.0f));
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

#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <functional>

struct GLFWwindow;

struct Camera {
    double pos_x = (6371000.0 + 20000.0) * 0.7071;
    double pos_y = (6371000.0 + 20000.0) * 0.7071;
    double pos_z = 0.0;
    float yaw   = 0.0f;
    float pitch = -0.3f;
    float fov_y = glm::radians(60.0f);
    float near_plane = 0.5f;
    float far_plane  = 100000000.0f;
};

struct CameraUpdateResult {
    double terrain_height_at_cam;
    double altitude_above_terrain;
};

glm::vec3 camera_forward(const Camera& cam);

void camera_apply_mouse_look(Camera& cam, float dx, float dy,
                             float sensitivity = 0.002f);

CameraUpdateResult camera_update(Camera& cam, GLFWwindow* window, float dt,
                                 float planet_radius,
                                 std::function<float(glm::vec3)> height_fn);

glm::mat4 camera_build_view(const Camera& cam);

glm::mat4 camera_build_proj(const Camera& cam, float aspect);

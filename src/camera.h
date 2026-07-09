#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <functional>

#include "input_frame.h"

enum class CameraMode { Orbital, FirstPerson };

// Orbits an external pivot at arm_length offset.
struct OrbitalState {
    glm::dvec3 pivot{(6371000.0 + 20000.0) * 0.7071,
                     (6371000.0 + 20000.0) * 0.7071,
                     0.0};
    glm::quat  orientation{1.0f, 0.0f, 0.0f, 0.0f};
    double     arm_length = 5000.0;

    // Smoothing state (gives mouse-look and movement an ease-in/out, weighty
    // feel and keeps motion surface-locked — see orbital_update).
    glm::vec2  look_pending{0.0f, 0.0f};  // unconsumed mouse delta (px), drained with easing
    glm::dvec3 move_vel{0.0, 0.0, 0.0};   // smoothed surface-tangential velocity (m/s, world)
    double     radial_vel = 0.0;          // smoothed radial (Q/E) velocity (m/s)
};

// Eye is the camera position; mouse-look rotates the eye in place.
struct FirstPersonState {
    glm::dvec3 eye{0.0, 0.0, 0.0};
    glm::quat  orientation{1.0f, 0.0f, 0.0f, 0.0f};
    float      eye_height_offset = 1.7f;
    float      walk_speed = 4.0f;
    float      vertical_velocity = 0.0f;
    bool       grounded = true;

    // Fly-to-cursor: parametric great-circle path with altitude arc.
    glm::dvec3 warp_d0{0.0, 0.0, 1.0};      // start direction (unit)
    glm::dvec3 warp_d1{0.0, 0.0, 1.0};      // end direction (unit)
    double     warp_r0 = 0.0;               // start radius from planet center
    glm::quat  warp_o0{1.0f, 0.0f, 0.0f, 0.0f};  // orientation at start
    glm::quat  warp_o1{1.0f, 0.0f, 0.0f, 0.0f};  // orientation at landing (horizon-level)
    double     warp_arc_h = 0.0;            // peak altitude bump (meters)
    double     warp_duration = 0.0;         // seconds total
    double     warp_elapsed = 0.0;
    bool       warping = false;
};

struct Camera {
    CameraMode       mode = CameraMode::Orbital;
    OrbitalState     orbit;
    FirstPersonState fp;

    float fov_y = glm::radians(60.0f);
    float near_plane = 0.5f;
};

struct CameraUpdateResult {
    double terrain_height_at_cam;
    double altitude_above_terrain;
};

struct CameraData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 sun_dir;   float _pad0;
    glm::vec3 sun_color; float time;   // _pad1 repurposed: seconds, for water animation
    glm::vec3 cam_pos;   float _pad2;
    glm::vec4 brush_world;
    glm::vec4 brush_color;
    glm::mat4 inv_view_proj;
};

glm::vec3  camera_forward(const Camera& cam);
glm::vec3  camera_up(const Camera& cam);
glm::vec3  camera_right(const Camera& cam);
glm::dvec3 camera_eye_position(const Camera& cam);
glm::quat  camera_orientation(const Camera& cam);

void camera_initialize_orientation(Camera& cam);

void camera_apply_mouse_look(Camera& cam, float dx, float dy,
                             float sensitivity = 0.002f);

void camera_zoom(Camera& cam, double amount);

// Movement (WASD/QE tangent walk + Q/E vertical, shift/alt speed mods) is read
// from the per-frame InputFrame. Mouse-look and zoom are applied separately by
// the caller via camera_apply_mouse_look / camera_zoom.
CameraUpdateResult camera_update(Camera& cam, const InputFrame& in, float dt,
                                 float planet_radius,
                                 std::function<float(glm::vec3)> height_fn);

glm::mat4 camera_build_view(const Camera& cam);

glm::mat4 camera_build_proj(const Camera& cam, float aspect);

void camera_switch_to_first_person(Camera& cam, float planet_radius,
                                   std::function<float(glm::vec3)> height_fn);
void camera_switch_to_orbital(Camera& cam);

// Begin a smoothed fly-to point. In FP, target_world is the desired eye
// position (already including eye_height_offset above the surface). No-op if
// the camera is not in FirstPerson mode.
void camera_begin_warp_to(Camera& cam, glm::dvec3 target_eye);

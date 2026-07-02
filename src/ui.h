#pragma once

#include <cstdint>

struct UIState {
    bool show_menu = true;

    // Embedding (set by caller): when true, the lab is hosted by the launcher,
    // so ui_draw shows a "< Back" button and ESC requests a return to the menu
    // instead of closing the window. wants_back is set by ui_draw / the input
    // callback and consumed by the caller.
    bool embedded   = false;
    bool wants_back = false;

    // Performance display (set by caller before ui_draw)
    double cpu_avg_ms = 0;
    double gpu_avg_ms = 0;
    double altitude_above_terrain = 0;
    double terrain_height_at_cam = 0;
    uint32_t visible_tile_count = 0;
    uint32_t stamp_count = 0;
    uint32_t max_stamps = 0;

    // Atmosphere debug (set by caller after readback)
    float pressure_min = 0.0f;
    float pressure_max = 0.0f;
    float pressure_mean = 0.0f;
    float wind_speed_max = 0.0f;

    // Water physics — defaults tuned for visible flow at planet scale.
    // (At level-8 cells of ~800 m, c = sqrt(g*h) ≈ 10 m/s, so flow takes
    // tens of seconds to traverse a tile; lower friction lets it actually move.)
    float gravity = 9.81f;
    float friction = 0.002f;
    float damping = 0.0002f;
    float time_scale = 1.0f;
    bool request_water_reset = false;

    // Brushes
    float brush_radius_grid = 30.0f;
    float brush_strength = 1.5f;
    float terrain_strength = 50.0f;
    float stamp_angular_scale = 0.0001f;
    float pulse_amount = 50.0f;
    float pulse_radius_cells = 30.0f;

    // Stamp actions (set by ui_draw, consumed by caller)
    bool undo_stamp = false;
    bool clear_stamps = false;

    // Erosion
    bool erosion_enabled = false;
    float k_erosion = 0.0005f;
    float k_deposit = 0.001f;
    float k_capacity = 0.01f;
    float min_slope = 0.005f;
    float max_change_m = 5.0f;
    float min_erosion_depth = 0.01f;
    float max_sediment = 0.5f;
    float mud_visibility = 5.0f;
    bool request_sediment_reset = false;

    // Atmosphere
    bool atmosphere_enabled = false;
    float cloud_opacity = 1.0f;
    float cloud_altitude = 0.0f;
    float orographic_lift = 0.5f;
    float adiabatic_cooling = 0.0065f;
    float rain_shadow = 0.7f;
    float k_pressure = 0.15f;
    bool request_atmo_reset = false;

    // Sand (particle system — consumer of wind field)
    bool sand_enabled = false;
    float sand_loft_threshold = 1.5f;
    float sand_loft_rate = 0.5f;
    float sand_streak = 0.05f;
    float sand_alpha = 0.8f;
    float sand_bounce = 0.3f;
    float sand_gravity = 9.81f;

    // Ocean
    bool ocean_enabled = true;
    float sea_level = 800.0f;

    // Sky / atmosphere rendering (M1b: scattering + aerial + tonemap)
    bool  sky_enabled   = true;
    float atmo_density  = 1.0f;   // scales Rayleigh+Mie optical depth
    float sun_intensity = 22.0f;  // sun radiance feeding the scattering
    float exposure      = 1.0f;   // tonemap exposure

    // Scene
    bool request_basin_reset = false;

    // Camera (read-only display)
    bool  first_person_mode  = false;
    float move_speed_mps     = 0.0f;
    float walk_speed_setting = 0.0f;
};

void ui_draw(UIState& state);

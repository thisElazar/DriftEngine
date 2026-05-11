#include "ui.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <algorithm>
#include <cmath>

void ui_draw(UIState& s)
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (s.show_menu) {
        ImGui::Begin("drift_engine", &s.show_menu);

        if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("CPU: %.2f ms (%.1f fps)", s.cpu_avg_ms, 1000.0 / std::max(s.cpu_avg_ms, 0.001));
            ImGui::Text("GPU SWE: %.2f ms", s.gpu_avg_ms);
            ImGui::Text("Altitude AGL: %.1f m", s.altitude_above_terrain);
            ImGui::Text("Terrain height: %.1f m", s.terrain_height_at_cam);
            ImGui::Text("Tiles: %u", s.visible_tile_count);
            ImGui::Text("Camera: %s  [F to toggle]", s.first_person_mode ? "First-person" : "Orbital");
            if (s.first_person_mode) {
                ImGui::Text("Speed: %.2f m/s   (walk %.1f)", s.move_speed_mps, s.walk_speed_setting);
                ImGui::TextDisabled("Scroll = walk speed   C = fly to cursor");
            } else {
                ImGui::Text("Speed: %.0f m/s", s.move_speed_mps);
            }
        }

        if (ImGui::CollapsingHeader("Water Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Gravity", &s.gravity, 0.5f, 30.0f);
            ImGui::SliderFloat("Friction", &s.friction, 0.0f, 0.05f, "%.4f");
            ImGui::SliderFloat("Damping", &s.damping, 0.0f, 0.05f, "%.4f");
            ImGui::SliderFloat("Time scale", &s.time_scale, 0.0f, 5.0f);
            if (ImGui::Button("Reset water")) s.request_water_reset = true;
        }

        if (ImGui::CollapsingHeader("Brushes", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Brush radius (cells)", &s.brush_radius_grid, 2.0f, 300.0f);
            ImGui::SliderFloat("Water strength", &s.brush_strength, 0.0f, 20.0f);
            ImGui::SliderFloat("Terrain strength m/s", &s.terrain_strength, 1.0f, 500.0f);
            ImGui::SliderFloat("Stamp angular scale", &s.stamp_angular_scale, 0.00001f, 0.01f, "%.5f");
            ImGui::SliderFloat("Pulse amount (m)", &s.pulse_amount, 1.0f, 200.0f);
            ImGui::SliderFloat("Pulse radius (cells)", &s.pulse_radius_cells, 5.0f, 200.0f);
            ImGui::Text("Stamps: %u / %u", s.stamp_count, s.max_stamps);
            if (ImGui::Button("Undo stamp")) s.undo_stamp = true;
            ImGui::SameLine();
            if (ImGui::Button("Clear stamps")) s.clear_stamps = true;
        }

        if (ImGui::CollapsingHeader("Erosion", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Erosion enabled", &s.erosion_enabled);
            if (s.erosion_enabled) {
                ImGui::SliderFloat("k_erosion", &s.k_erosion, 0.0f, 0.01f, "%.5f");
                ImGui::SliderFloat("k_deposit", &s.k_deposit, 0.0f, 0.01f, "%.5f");
                ImGui::SliderFloat("k_capacity", &s.k_capacity, 0.0f, 0.1f, "%.4f");
                ImGui::SliderFloat("min_slope", &s.min_slope, 0.0001f, 0.05f, "%.4f");
                ImGui::SliderFloat("max_change m/s", &s.max_change_m, 0.1f, 20.0f, "%.1f");
                ImGui::SliderFloat("min depth", &s.min_erosion_depth, 0.001f, 0.5f, "%.3f");
                ImGui::SliderFloat("max sediment", &s.max_sediment, 0.05f, 5.0f, "%.2f");
            }
            ImGui::SliderFloat("Mud visibility", &s.mud_visibility, 0.0f, 20.0f);
            if (ImGui::Button("Reset sediment")) s.request_sediment_reset = true;
        }

        if (ImGui::CollapsingHeader("Atmosphere")) {
            ImGui::Checkbox("Atmosphere enabled", &s.atmosphere_enabled);
            if (s.atmosphere_enabled) {
                ImGui::SliderFloat("Cloud opacity", &s.cloud_opacity, 0.0f, 3.0f);
                ImGui::SliderFloat("Cloud altitude", &s.cloud_altitude, 0.0f, 2000.0f, "%.0f m");
                ImGui::SliderFloat("Orographic lift", &s.orographic_lift, 0.0f, 2.0f);
                ImGui::SliderFloat("Adiabatic cooling", &s.adiabatic_cooling, 0.0f, 0.02f, "%.5f");
                ImGui::SliderFloat("Rain shadow", &s.rain_shadow, 0.0f, 1.0f);
                ImGui::SliderFloat("Pressure force", &s.k_pressure, 0.0f, 1.0f, "%.3f");
                ImGui::Separator();
                ImGui::Checkbox("Sand enabled", &s.sand_enabled);
                if (s.sand_enabled) {
                    ImGui::SliderFloat("Loft threshold", &s.sand_loft_threshold, 0.5f, 4.0f);
                    ImGui::SliderFloat("Loft rate", &s.sand_loft_rate, 0.1f, 3.0f);
                    ImGui::SliderFloat("Streak length", &s.sand_streak, 0.01f, 0.3f, "%.3f");
                    ImGui::SliderFloat("Particle alpha", &s.sand_alpha, 0.1f, 1.0f);
                    ImGui::SliderFloat("Bounce energy", &s.sand_bounce, 0.0f, 0.8f);
                    ImGui::SliderFloat("Gravity", &s.sand_gravity, 1.0f, 20.0f);
                }
                if (ImGui::Button("Reset atmosphere")) s.request_atmo_reset = true;
                ImGui::Separator();
                ImGui::Text("Pressure: %.1f / %.1f / %.1f hPa",
                            s.pressure_min, s.pressure_mean, s.pressure_max);
                ImGui::Text("Max wind: %.2f m/s", s.wind_speed_max);
            }
        }

        if (ImGui::CollapsingHeader("Ocean", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Ocean enabled", &s.ocean_enabled);
            if (s.ocean_enabled) {
                ImGui::SliderFloat("Sea level (m)", &s.sea_level, 0.0f, 4000.0f, "%.0f m");
            }
        }

        if (ImGui::CollapsingHeader("Scene")) {
            if (ImGui::Button("Regenerate basin")) s.request_basin_reset = true;
            ImGui::TextDisabled("Pulse: SPACE | Modes: 1=raise 2=lower 3=water 4=sand");
            ImGui::TextDisabled("Look: RMB+drag | Move: WASD+QE | Menu: `");
            ImGui::TextDisabled("F5: hot-reload shaders");
        }
        ImGui::End();
    }

    ImGui::Render();
}

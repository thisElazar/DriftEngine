// Drift Launcher — unified single-window hub for the Drift ecosystem engine.

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "renderer.h"
#include "species_file.h"
#include "shared/lab_common.h"

#include "plant_lab/plant_lab.h"
#include "animals_lab/animals_lab.h"
#include "world_lab/world_lab.h"
#include "globe/globe.h"

#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

// ---------------------------------------------------------------------------
// Species library scanner
// ---------------------------------------------------------------------------
struct SpeciesEntry {
    std::string name;
    std::string kind;
};

static std::vector<SpeciesEntry> scan_species()
{
    std::vector<SpeciesEntry> entries;
    auto& dir = species_dir();
    if (!std::filesystem::exists(dir)) return entries;

    for (auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".toml") continue;
        auto kind = bestiary::detect_species_kind(entry.path());
        if (kind.empty()) continue;
        entries.push_back({entry.path().stem().string(), kind});
    }

    std::sort(entries.begin(), entries.end(),
        [](const SpeciesEntry& a, const SpeciesEntry& b) {
            if (a.kind != b.kind) return a.kind < b.kind;
            return a.name < b.name;
        });

    return entries;
}

// ---------------------------------------------------------------------------
// Menu-only rendering (clear + ImGui)
// ---------------------------------------------------------------------------
static void render_menu_frame(Renderer& r, FrameData& frame,
                              uint32_t image_index, VkExtent2D extent)
{
    VkCommandBuffer cmd = frame.cmd;

    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.image         = r.swapchain_images[image_index];
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    {
        VkRenderingAttachmentInfo color{};
        color.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView   = r.swapchain_views[image_index];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{0.06f, 0.06f, 0.08f, 1.0f}};

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea           = {{0, 0}, extent};
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &color;

        vkCmdBeginRendering(cmd, &ri);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRendering(cmd);
    }

    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_NONE;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.image         = r.swapchain_images[image_index];
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
enum class AppMode { Menu, PlantLab, AnimalsLab, WorldLab, Globe };

int main()
{
    Renderer renderer{};
    renderer_init(renderer, 1280, 800, "Drift Engine");

    glfwSetScrollCallback(renderer.window, lab_scroll_cb);
    ImGui_ImplGlfw_InstallCallbacks(renderer.window);

    AppMode mode = AppMode::Menu;

    PlantLabState   plant_state{};
    AnimalsLabState animals_state{};
    WorldLabState   world_state{};
    GlobeState      globe_state{};

    // Preview world (attractor scene behind menu)
    WorldLabState preview_state{};
    preview_state.embedded = true;
    preview_state.preview_mode = true;
    world_lab_init(preview_state, renderer);
    preview_state.ui_creatures_enabled = true;
    preview_state.ui_show_plants = true;
    preview_state.ui_autorun = true;
    preview_state.ui_spring_enabled = true;
    preview_state.ui_atmo_enabled = true;
    preview_state.camera.distance = 180.0f;
    preview_state.camera.pitch    = 0.55f;
    preview_state.camera.yaw      = 0.0f;
    bool preview_alive = true;

    auto species = scan_species();
    int selected_species = -1;
    float refresh_timer = 0.0f;

    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(renderer.window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        glfwPollEvents();

        // --- Tick active mode ---
        bool back_pressed = false;

        if (mode != AppMode::Menu &&
            glfwGetKey(renderer.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            back_pressed = true;
        }

        if (mode == AppMode::Menu) {
            if (preview_alive)
                world_lab_tick(preview_state, renderer, dt);

            refresh_timer += dt;
            if (refresh_timer > 3.0f) {
                refresh_timer = 0.0f;
                species = scan_species();
            }

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.06f, 0.06f, 0.08f, 0.82f));

            ImVec2 win_size = ImGui::GetIO().DisplaySize;
            float panel_w = std::min(400.0f, win_size.x * 0.35f);
            ImGui::SetNextWindowPos(ImVec2(20, 20));
            ImGui::SetNextWindowSize(ImVec2(panel_w, win_size.y - 40));
            ImGui::Begin("##main", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

            ImGui::TextUnformatted("DRIFT ENGINE");
            ImGui::Separator();
            ImGui::Spacing();

            float btn_w = ImGui::GetContentRegionAvail().x;
            float btn_half = (btn_w - ImGui::GetStyle().ItemSpacing.x) * 0.5f;

            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Design");
            if (ImGui::Button("Plant Lab", ImVec2(btn_half, 32))) {
                plant_state = {};
                plant_state.embedded = true;
                plant_lab_init(plant_state, renderer);
                mode = AppMode::PlantLab;
                glfwSetWindowTitle(renderer.window, "Drift Engine - Plant Lab");
            }
            ImGui::SameLine();
            if (ImGui::Button("Animal Lab", ImVec2(btn_half, 32))) {
                animals_state = {};
                animals_state.embedded = true;
                animals_lab_init(animals_state, renderer);
                mode = AppMode::AnimalsLab;
                glfwSetWindowTitle(renderer.window, "Drift Engine - Animals Lab");
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "World");
            if (ImGui::Button("World Lab", ImVec2(btn_half, 32))) {
                world_state = {};
                world_state.embedded = true;
                world_lab_init(world_state, renderer);
                mode = AppMode::WorldLab;
                glfwSetWindowTitle(renderer.window, "Drift Engine - World Lab");
            }
            ImGui::SameLine();
            if (ImGui::Button("Globe", ImVec2(btn_half, 32))) {
                globe_state = {};
                globe_state.embedded = true;
                globe_init(globe_state, renderer);
                mode = AppMode::Globe;
                glfwSetWindowTitle(renderer.window, "Drift Engine - Globe");
            }

            ImGui::Spacing();
            ImGui::Separator();

            char lib_label[64];
            std::snprintf(lib_label, sizeof(lib_label), "Species Library (%d)",
                          static_cast<int>(species.size()));

            if (ImGui::CollapsingHeader(lib_label)) {

            if (species.empty()) {
                ImGui::TextDisabled("No species files found in species/");
                ImGui::TextDisabled("Use Plant Lab or Animal Lab to create some.");
            } else {
                float list_height = win_size.y - ImGui::GetCursorPosY() - 30;
                ImGui::BeginChild("##species_list", ImVec2(0, list_height), true);

                std::string current_kind;
                for (int i = 0; i < static_cast<int>(species.size()); ++i) {
                    auto& sp = species[i];
                    if (sp.kind != current_kind) {
                        if (!current_kind.empty()) ImGui::Spacing();
                        current_kind = sp.kind;

                        const char* label = current_kind.c_str();
                        if (sp.kind == "grass_clump")     label = "Grass";
                        else if (sp.kind == "bush")       label = "Bushes";
                        else if (sp.kind == "tree")       label = "Trees";
                        else if (sp.kind == "lplant")     label = "L-Plants";
                        else if (sp.kind == "reed")       label = "Reeds";
                        else if (sp.kind == "wildflower") label = "Wildflowers";
                        else if (sp.kind == "herbivore")  label = "Herbivores";
                        else if (sp.kind == "predator")   label = "Predators";
                        else if (sp.kind == "rabbit")     label = "Rabbits";
                        else if (sp.kind == "bird")       label = "Birds";
                        else if (sp.kind == "raptor")     label = "Raptors";
                        else if (sp.kind == "snake")      label = "Snakes";

                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%s", label);
                        ImGui::Separator();
                    }

                    bool is_selected = (selected_species == i);
                    if (ImGui::Selectable(sp.name.c_str(), is_selected,
                                          ImGuiSelectableFlags_AllowDoubleClick)) {
                        selected_species = i;
                        if (ImGui::IsMouseDoubleClicked(0)) {
                            auto file_path = (species_dir() / (sp.name + ".toml")).string();
                            bool is_plant = (sp.kind == "grass_clump" || sp.kind == "bush"
                                          || sp.kind == "tree" || sp.kind == "lplant"
                                          || sp.kind == "reed" || sp.kind == "wildflower");
                            bool is_creature = (sp.kind == "herbivore" || sp.kind == "predator"
                                             || sp.kind == "rabbit" || sp.kind == "bird"
                                             || sp.kind == "raptor" || sp.kind == "snake");
                            if (is_plant) {
                                plant_state = {};
                                plant_state.embedded = true;
                                plant_state.pending_file = file_path;
                                plant_lab_init(plant_state, renderer);
                                mode = AppMode::PlantLab;
                                glfwSetWindowTitle(renderer.window, "Drift Engine - Plant Lab");
                            } else if (is_creature) {
                                animals_state = {};
                                animals_state.embedded = true;
                                animals_state.pending_file = file_path;
                                animals_lab_init(animals_state, renderer);
                                mode = AppMode::AnimalsLab;
                                glfwSetWindowTitle(renderer.window, "Drift Engine - Animals Lab");
                            }
                        }
                    }
                }
                ImGui::EndChild();
            }
            } // CollapsingHeader

            ImGui::End();
            ImGui::PopStyleColor();
            ImGui::Render();

        } else if (mode == AppMode::PlantLab) {
            if (!plant_lab_tick(plant_state, renderer, dt)) back_pressed = true;
        } else if (mode == AppMode::AnimalsLab) {
            if (!animals_lab_tick(animals_state, renderer, dt)) back_pressed = true;
        } else if (mode == AppMode::WorldLab) {
            if (!world_lab_tick(world_state, renderer, dt)) back_pressed = true;
        } else if (mode == AppMode::Globe) {
            if (!globe_tick(globe_state, renderer, dt)) back_pressed = true;
            // Globe's input callback sets WindowShouldClose on ESC —
            // undo that, the launcher handles ESC as back-to-menu.
            if (glfwWindowShouldClose(renderer.window)) {
                glfwSetWindowShouldClose(renderer.window, GLFW_FALSE);
                back_pressed = true;
            }
        }

        // --- Handle back-to-menu ---
        if (back_pressed) {
            vkDeviceWaitIdle(renderer.device);
            if (mode == AppMode::PlantLab)   plant_lab_shutdown(plant_state, renderer);
            if (mode == AppMode::AnimalsLab)  animals_lab_shutdown(animals_state, renderer);
            if (mode == AppMode::WorldLab)    world_lab_shutdown(world_state, renderer);
            if (mode == AppMode::Globe)  {
                globe_shutdown(globe_state, renderer);
            }
            mode = AppMode::Menu;
            glfwSetWindowTitle(renderer.window, "Drift Engine");
            species = scan_species();
            continue;
        }

        // --- Render ---
        FrameData* frame = nullptr;
        uint32_t image_index = 0;
        VkExtent2D extent{};
        if (!renderer_begin_frame(renderer, frame, image_index, extent)) continue;

        if (mode == AppMode::Menu) {
            if (preview_alive) {
                world_lab_render(preview_state, renderer, *frame, image_index, extent);
                // Draw menu ImGui on top of the preview scene
                VkCommandBuffer cmd = frame->cmd;
                VkRenderingAttachmentInfo ig_color{};
                ig_color.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                ig_color.imageView   = renderer.swapchain_views[image_index];
                ig_color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                ig_color.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                ig_color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo ig_ri{};
                ig_ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
                ig_ri.renderArea           = {{0, 0}, extent};
                ig_ri.layerCount           = 1;
                ig_ri.colorAttachmentCount = 1;
                ig_ri.pColorAttachments    = &ig_color;
                vkCmdBeginRendering(cmd, &ig_ri);
                ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
                vkCmdEndRendering(cmd);
                image_barrier(cmd, renderer.swapchain_images[image_index],
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
            } else {
                render_menu_frame(renderer, *frame, image_index, extent);
            }
        } else if (mode == AppMode::PlantLab) {
            plant_lab_render(plant_state, renderer, *frame, image_index, extent);
        } else if (mode == AppMode::AnimalsLab) {
            animals_lab_render(animals_state, renderer, *frame, image_index, extent);
        } else if (mode == AppMode::WorldLab) {
            world_lab_render(world_state, renderer, *frame, image_index, extent);
        } else if (mode == AppMode::Globe) {
            globe_render(globe_state, renderer, *frame, image_index, extent);
        }

        renderer_end_frame(renderer, *frame, image_index);
    }

    // Shutdown active lab and preview
    vkDeviceWaitIdle(renderer.device);
    if (mode == AppMode::PlantLab)   plant_lab_shutdown(plant_state, renderer);
    if (mode == AppMode::AnimalsLab)  animals_lab_shutdown(animals_state, renderer);
    if (mode == AppMode::WorldLab)    world_lab_shutdown(world_state, renderer);
    if (mode == AppMode::Globe)       globe_shutdown(globe_state, renderer);
    if (preview_alive)               world_lab_shutdown(preview_state, renderer);

    renderer_shutdown(renderer);
    return 0;
}

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
#include <array>
#include <functional>

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
// Re-establish the launcher's baseline input handling.
//
// Labs that install their own GLFW callbacks (Globe, via input_install_callbacks)
// restore only ImGui's callbacks on shutdown, dropping lab_scroll_cb — which
// kills scroll-zoom in the menu preview and every lab opened afterward. Calling
// this after any lab exits puts the launcher's scroll handler back.
// ---------------------------------------------------------------------------
static void install_launcher_input(Renderer& r)
{
    ImGui_ImplGlfw_RestoreCallbacks(r.window);
    glfwSetScrollCallback(r.window, lab_scroll_cb);
    ImGui_ImplGlfw_InstallCallbacks(r.window);
}

// ---------------------------------------------------------------------------
// Poll all user input into one immutable snapshot for this frame. This is the
// single input source the labs consume via their tick (see input_frame.h /
// docs/INPUT_UNIFICATION.md). Mouse/scroll deltas are computed against prev;
// the wheel accumulator (fed by lab_scroll_cb) is drained here.
// ---------------------------------------------------------------------------
static InputFrame poll_input(Renderer& r, const InputFrame& prev)
{
    InputFrame in;
    GLFWwindow* w = r.window;

    glfwGetCursorPos(w, &in.mouse_x, &in.mouse_y);
    in.mouse_dx = static_cast<float>(in.mouse_x - prev.mouse_x);
    in.mouse_dy = static_cast<float>(in.mouse_y - prev.mouse_y);

    in.scroll = g_scroll_accum;
    g_scroll_accum = 0.0f;

    in.lmb = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
    in.rmb = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
    in.mmb = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    in.lmb_pressed = in.lmb && !prev.lmb;
    in.rmb_pressed = in.rmb && !prev.rmb;

    in.key_shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS
                || glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    in.key_r     = glfwGetKey(w, GLFW_KEY_R)     == GLFW_PRESS;
    in.key_space = glfwGetKey(w, GLFW_KEY_SPACE) == GLFW_PRESS;
    in.key_f     = glfwGetKey(w, GLFW_KEY_F)     == GLFW_PRESS;
    in.key_c     = glfwGetKey(w, GLFW_KEY_C)     == GLFW_PRESS;
    in.key_f5    = glfwGetKey(w, GLFW_KEY_F5)    == GLFW_PRESS;
    in.esc_pressed = glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    if      (glfwGetKey(w, GLFW_KEY_1) == GLFW_PRESS) in.brush_digit = 1;
    else if (glfwGetKey(w, GLFW_KEY_2) == GLFW_PRESS) in.brush_digit = 2;
    else if (glfwGetKey(w, GLFW_KEY_3) == GLFW_PRESS) in.brush_digit = 3;
    else if (glfwGetKey(w, GLFW_KEY_4) == GLFW_PRESS) in.brush_digit = 4;

    if (ImGui::GetCurrentContext()) {
        in.ui_wants_mouse    = ImGui::GetIO().WantCaptureMouse;
        in.ui_wants_keyboard = ImGui::GetIO().WantCaptureKeyboard;
    }

    glfwGetWindowSize(w, &in.win_w, &in.win_h);
    glfwGetFramebufferSize(w, &in.fb_w, &in.fb_h);
    return in;
}

// ---------------------------------------------------------------------------
// Lab vtable — every lab exposes the same init/tick/render/shutdown contract
// over its own state struct. Binding those free functions to a state instance
// behind std::function lets the main loop drive any lab uniformly, so adding a
// lab is one table row instead of edits to four parallel if/else chains.
//
//   enter(file): reset state, mark embedded, optionally queue a species file,
//                then init. file is "" for the menu buttons and a .toml path
//                for species double-click (ignored by labs without pending_file).
//   tick(dt):    returns false to request return-to-menu.
// ---------------------------------------------------------------------------
enum class AppMode { Menu, PlantLab, AnimalsLab, WorldLab, Globe };
constexpr int LAB_COUNT = 4;  // PlantLab..Globe; Menu is not a lab

struct Lab {
    const char* title;
    std::function<void(const std::string& file)>                  enter;
    std::function<bool(const InputFrame&, float dt)>              tick;
    std::function<void(FrameData&, uint32_t, VkExtent2D)>         render;
    std::function<void()>                                        shutdown;
};

// AppMode -> labs[] index (Menu has no entry).
static int lab_index(AppMode m) { return static_cast<int>(m) - 1; }

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

    // Lab table, indexed by lab_index(mode). Order must match AppMode:
    // PlantLab, AnimalsLab, WorldLab, Globe.
    std::array<Lab, LAB_COUNT> labs = {{
        Lab{
            "Drift Engine - Plant Lab",
            [&](const std::string& f) {
                plant_state = {};
                plant_state.embedded = true;
                if (!f.empty()) plant_state.pending_file = f;
                plant_lab_init(plant_state, renderer);
            },
            [&](const InputFrame& in, float dt) { return plant_lab_tick(plant_state, renderer, in, dt); },
            [&](FrameData& fr, uint32_t ii, VkExtent2D ex) {
                plant_lab_render(plant_state, renderer, fr, ii, ex);
            },
            [&]() { plant_lab_shutdown(plant_state, renderer); },
        },
        Lab{
            "Drift Engine - Animals Lab",
            [&](const std::string& f) {
                animals_state = {};
                animals_state.embedded = true;
                if (!f.empty()) animals_state.pending_file = f;
                animals_lab_init(animals_state, renderer);
            },
            [&](const InputFrame& in, float dt) { return animals_lab_tick(animals_state, renderer, in, dt); },
            [&](FrameData& fr, uint32_t ii, VkExtent2D ex) {
                animals_lab_render(animals_state, renderer, fr, ii, ex);
            },
            [&]() { animals_lab_shutdown(animals_state, renderer); },
        },
        Lab{
            "Drift Engine - World Lab",
            [&](const std::string&) {
                world_state = {};
                world_state.embedded = true;
                world_lab_init(world_state, renderer);
            },
            [&](const InputFrame& in, float dt) { return world_lab_tick(world_state, renderer, in, dt); },
            [&](FrameData& fr, uint32_t ii, VkExtent2D ex) {
                world_lab_render(world_state, renderer, fr, ii, ex);
            },
            [&]() { world_lab_shutdown(world_state, renderer); },
        },
        Lab{
            "Drift Engine - Globe",
            [&](const std::string&) {
                globe_state = {};
                globe_state.embedded = true;
                globe_init(globe_state, renderer);
            },
            [&](const InputFrame& in, float dt) { return globe_tick(globe_state, renderer, in, dt); },
            [&](FrameData& fr, uint32_t ii, VkExtent2D ex) {
                globe_render(globe_state, renderer, fr, ii, ex);
            },
            [&]() { globe_shutdown(globe_state, renderer); },
        },
    }};

    // Enter a lab from the menu: build its scene and switch the window title.
    auto enter_lab = [&](AppMode m, const std::string& file = "") {
        Lab& lab = labs[lab_index(m)];
        lab.enter(file);
        mode = m;
        glfwSetWindowTitle(renderer.window, lab.title);
    };

    auto species = scan_species();
    int selected_species = -1;
    float refresh_timer = 0.0f;

    InputFrame input_prev{};
    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(renderer.window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        glfwPollEvents();

        InputFrame in = poll_input(renderer, input_prev);
        input_prev = in;

        // --- Tick active mode ---
        bool back_pressed = false;

        if (mode != AppMode::Menu &&
            glfwGetKey(renderer.window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            back_pressed = true;
        }

        if (mode == AppMode::Menu) {
            if (preview_alive)
                world_lab_tick(preview_state, renderer, in, dt);

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
            if (ImGui::Button("Plant Lab", ImVec2(btn_half, 32)))
                enter_lab(AppMode::PlantLab);
            ImGui::SameLine();
            if (ImGui::Button("Animal Lab", ImVec2(btn_half, 32)))
                enter_lab(AppMode::AnimalsLab);

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "World");
            if (ImGui::Button("World Lab", ImVec2(btn_half, 32)))
                enter_lab(AppMode::WorldLab);
            ImGui::SameLine();
            if (ImGui::Button("Globe", ImVec2(btn_half, 32)))
                enter_lab(AppMode::Globe);

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
                            if (is_plant)
                                enter_lab(AppMode::PlantLab, file_path);
                            else if (is_creature)
                                enter_lab(AppMode::AnimalsLab, file_path);
                        }
                    }
                }
                ImGui::EndChild();
            }
            } // CollapsingHeader

            ImGui::End();
            ImGui::PopStyleColor();
            ImGui::Render();

        } else {
            if (!labs[lab_index(mode)].tick(in, dt)) back_pressed = true;
        }

        // --- Handle back-to-menu ---
        if (back_pressed) {
            vkDeviceWaitIdle(renderer.device);
            labs[lab_index(mode)].shutdown();
            install_launcher_input(renderer);
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
        } else {
            labs[lab_index(mode)].render(*frame, image_index, extent);
        }

        renderer_end_frame(renderer, *frame, image_index);
    }

    // Shutdown active lab and preview
    vkDeviceWaitIdle(renderer.device);
    if (mode != AppMode::Menu)  labs[lab_index(mode)].shutdown();
    if (preview_alive)          world_lab_shutdown(preview_state, renderer);

    renderer_shutdown(renderer);
    return 0;
}

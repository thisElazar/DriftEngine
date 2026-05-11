// animals_lab.h — embeddable Animals Lab module.
//
// Provides init/tick/render/shutdown so the lab can be driven by either
// a standalone executable or a unified launcher.
#pragma once

#include "renderer.h"
#include "shared/lab_common.h"

#include "animals/herbivore.h"
#include "animals/predator.h"
#include "animals/rabbit.h"
#include "animals/bird.h"
#include "animals/snake.h"
#include "skeleton/animation.h"
#include "skeleton/skinning.h"
#include "species_file.h"

#include <glm/gtc/quaternion.hpp>

#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Pipeline types
// ---------------------------------------------------------------------------
struct MeshPipeline {
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkShaderModule   vs       = VK_NULL_HANDLE;
    VkShaderModule   fs       = VK_NULL_HANDLE;
};

struct SkeletonPipeline {
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout      layout      = VK_NULL_HANDLE;
    VkPipeline            sphere_pipe = VK_NULL_HANDLE;
    VkPipeline            line_pipe   = VK_NULL_HANDLE;
    VkShaderModule        sphere_vs   = VK_NULL_HANDLE;
    VkShaderModule        sphere_fs   = VK_NULL_HANDLE;
    VkShaderModule        line_vs     = VK_NULL_HANDLE;
    VkShaderModule        line_fs     = VK_NULL_HANDLE;
};

struct SkeletonPC {
    glm::mat4 mvp;
    float     joint_radius;
};

// ---------------------------------------------------------------------------
// Sphere vertex (for skeleton overlay)
// ---------------------------------------------------------------------------
struct SphereVertex {
    float position[3];
    float normal[3];
};

// ---------------------------------------------------------------------------
// Ground plane
// ---------------------------------------------------------------------------
struct GroundPlane {
    GpuBuffer vb{};
    GpuBuffer ib{};
    uint32_t  index_count = 0;
};

// ---------------------------------------------------------------------------
// File I/O (animal-specific: filters by creature kinds)
// ---------------------------------------------------------------------------
struct AnimalFileIO {
    char name_buf[128] = "default";
    char status_buf[256] = "";
    std::vector<std::string> file_list;
    int  selected_file = -1;
    float status_timer = 0.0f;

    void set_status(const char* msg) {
        std::snprintf(status_buf, sizeof(status_buf), "%s", msg);
        status_timer = 3.0f;
    }

    void refresh_files() {
        file_list.clear();
        selected_file = -1;
        const auto& dir = species_dir();
        if (!std::filesystem::exists(dir)) return;
        for (auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.path().extension() != ".toml") continue;
            auto kind = bestiary::detect_species_kind(entry.path());
            if (kind == "herbivore" || kind == "predator" || kind == "rabbit" || kind == "bird" || kind == "raptor" || kind == "snake")
                file_list.push_back(entry.path().stem().string());
        }
        std::sort(file_list.begin(), file_list.end());
    }
};

// ---------------------------------------------------------------------------
// Creature type enum
// ---------------------------------------------------------------------------
enum CreatureType {
    CREATURE_HERBIVORE = 0,
    CREATURE_PREDATOR  = 1,
    CREATURE_RABBIT    = 2,
    CREATURE_BIRD      = 3,
    CREATURE_RAPTOR    = 4,
    CREATURE_SNAKE     = 5,
};

// ---------------------------------------------------------------------------
// All animals-lab state, bundled into a single struct.
// ---------------------------------------------------------------------------
struct AnimalsLabState {
    bool embedded    = false;   // true when driven by launcher (shows Back button)
    bool initialized = false;

    // Pipelines
    MeshPipeline     mesh_pipe{};
    SkeletonPipeline skel_pipe{};

    // Sphere mesh (for skeleton overlay)
    GpuBuffer sphere_vb{};
    GpuBuffer sphere_ib{};
    uint32_t  sphere_index_count = 0;

    // Joint transform SSBO + descriptors
    GpuBuffer       joint_ssbo{};
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet  desc_set  = VK_NULL_HANDLE;

    // Bone line buffer
    GpuBuffer bone_vb{};
    uint32_t  bone_vertex_count = 0;

    // Skinned mesh GPU buffers
    GpuBuffer mesh_vb{};
    GpuBuffer mesh_ib{};
    uint32_t  mesh_index_count = 0;

    // Ground plane
    GroundPlane ground{};

    // Creature params
    int creature_type      = CREATURE_HERBIVORE;
    int last_creature_type = -1;

    bestiary::HerbivoreParams herb_params{};
    bestiary::PredatorParams  pred_params{};
    bestiary::RabbitParams    rabbit_params{};
    bestiary::BirdParams      bird_params{};
    bestiary::BirdParams      raptor_params{};
    bestiary::SnakeParams     snake_params{};

    // Dirty-check copies
    bestiary::HerbivoreParams last_herb_params{};
    bestiary::PredatorParams  last_pred_params{};
    bestiary::RabbitParams    last_rabbit_params{};
    bestiary::BirdParams      last_bird_params{};
    bestiary::BirdParams      last_raptor_params{};
    bestiary::SnakeParams     last_snake_params{};

    // Animal mesh + skeleton
    bestiary::AnimalMesh animal_mesh{};
    bestiary::WalkCycle  gaits[8]{};
    int                  joint_count = 0;

    std::vector<glm::mat4>                joint_palette;
    std::vector<bestiary::SkinnedVertex>  skinned_verts;

    // Animation state
    bool  anim_playing = false;
    float anim_phase   = 0.0f;
    float anim_speed   = 1.0f;
    float last_phase   = -1.0f;
    bool  root_motion  = true;
    float root_offset  = 0.0f;
    float gait_blend   = 0.0f;
    int   anim_mode    = 3;     // 0=walk 1=trot 2=run 3=move 4=idle 5=graze/stalk
    float feed_height  = 0.0f;
    float last_feed_h  = -1.0f;

    // Display
    bool show_skeleton = true;

    // Camera + file I/O
    OrbitCamera  camera{};
    AnimalFileIO fio{};
};

// Module API ---------------------------------------------------------------

/// Create pipelines, sphere mesh, initial creature mesh, ground plane, SSBO, descriptors.
void animals_lab_init(AnimalsLabState& s, Renderer& r);

/// Poll camera, dirty-check -> regen mesh/skeleton/gaits, animation update, ImGui panel.
/// Returns false if the user pressed "Back" (only possible when s.embedded).
bool animals_lab_tick(AnimalsLabState& s, Renderer& r, float dt);

/// Record the command buffer: barriers, mesh draw, skeleton overlay, ImGui draw, present barrier.
void animals_lab_render(AnimalsLabState& s, Renderer& r,
                        FrameData& frame, uint32_t image_index, VkExtent2D extent);

/// Destroy pipelines, buffers, descriptors.  Must be called before renderer_shutdown.
void animals_lab_shutdown(AnimalsLabState& s, Renderer& r);

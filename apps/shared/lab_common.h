// lab_common.h — shared types used across all Drift lab applications.
#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <imgui.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "vk_util.h"           // from drift_engine_core
#include "input_frame.h"       // from drift_engine_core — unified per-frame input
#include "morphology/clump.h"  // from bestiary — VegetationVertex, VegetationMesh
#include "species_file.h"      // from bestiary — detect_species_kind

#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <initializer_list>
#include <cstring>
#include <cstdio>

// ---------------------------------------------------------------------------
// Scroll accumulator (global, set by GLFW callback)
// ---------------------------------------------------------------------------
inline float g_scroll_accum = 0.0f;

inline void lab_scroll_cb(GLFWwindow*, double, double yoffset)
{
    g_scroll_accum += static_cast<float>(yoffset);
}

// ---------------------------------------------------------------------------
// Orbit camera
// ---------------------------------------------------------------------------
struct OrbitCamera {
    float     yaw      = 0.0f;
    float     pitch    = 0.3f;
    float     distance = 1.5f;
    glm::vec3 target   = {0.0f, 0.2f, 0.0f};
    double    last_mx  = 0.0;
    double    last_my  = 0.0;
    bool      dragging = false;
};

// Orbit camera update driven by the unified per-frame InputFrame. RMB-drag
// rotates; wheel zooms. ImGui mouse capture is respected via in.ui_wants_mouse.
// (The launcher drains the scroll accumulator into in.scroll once per frame.)
inline void update_orbit(OrbitCamera& cam, const InputFrame& in, float max_dist = 5.0f)
{
    if (in.rmb && !in.ui_wants_mouse) {
        if (cam.dragging) {
            cam.yaw   -= static_cast<float>(in.mouse_x - cam.last_mx) * 0.005f;
            cam.pitch += static_cast<float>(in.mouse_y - cam.last_my) * 0.005f;
            cam.pitch  = glm::clamp(cam.pitch, -1.5f, 1.5f);
        }
        cam.dragging = true;
    } else {
        cam.dragging = false;
    }
    cam.last_mx = in.mouse_x;
    cam.last_my = in.mouse_y;

    if (!in.ui_wants_mouse) {
        cam.distance -= in.scroll * 0.1f;
        cam.distance  = glm::clamp(cam.distance, 0.2f, max_dist);
    }
}

inline glm::mat4 orbit_view(const OrbitCamera& cam)
{
    float cx = std::cos(cam.pitch) * std::sin(cam.yaw);
    float cy = std::sin(cam.pitch);
    float cz = std::cos(cam.pitch) * std::cos(cam.yaw);
    glm::vec3 eye = cam.target + glm::vec3(cx, cy, cz) * cam.distance;
    return glm::lookAt(eye, cam.target, glm::vec3(0.0f, 1.0f, 0.0f));
}

// ---------------------------------------------------------------------------
// Clump rendering pipeline (shared by plant_lab and world_lab)
// ---------------------------------------------------------------------------
struct ClumpPipeline {
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkShaderModule   vs       = VK_NULL_HANDLE;
    VkShaderModule   fs       = VK_NULL_HANDLE;
};

// Defined in the translation unit that includes pipeline.h (which provides ClumpPC).
ClumpPipeline create_clump_pipeline(VkDevice device);
void          destroy_clump_pipeline(VkDevice device, ClumpPipeline& p);

// ---------------------------------------------------------------------------
// Mesh state
// ---------------------------------------------------------------------------
struct MeshState {
    GpuBuffer vertex_buf{};
    GpuBuffer index_buf{};
    uint32_t  index_count  = 0;
    uint32_t  vertex_count = 0;
};

inline void upload_mesh(VmaAllocator allocator, MeshState& ms,
                        const bestiary::VegetationMesh& mesh)
{
    ms.vertex_count = static_cast<uint32_t>(mesh.vertices.size());
    ms.index_count  = static_cast<uint32_t>(mesh.indices.size());

    VkDeviceSize vb_size = mesh.vertices.size() * sizeof(bestiary::VegetationVertex);
    VkDeviceSize ib_size = mesh.indices.size()  * sizeof(uint32_t);

    ms.vertex_buf = create_host_buffer(allocator, vb_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    ms.index_buf  = create_host_buffer(allocator, ib_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    void* mapped = nullptr;
    VK_CHECK(vmaMapMemory(allocator, ms.vertex_buf.allocation, &mapped));
    std::memcpy(mapped, mesh.vertices.data(), vb_size);
    vmaUnmapMemory(allocator, ms.vertex_buf.allocation);

    VK_CHECK(vmaMapMemory(allocator, ms.index_buf.allocation, &mapped));
    std::memcpy(mapped, mesh.indices.data(), ib_size);
    vmaUnmapMemory(allocator, ms.index_buf.allocation);
}

inline void destroy_mesh(VmaAllocator allocator, MeshState& ms)
{
    destroy_buffer(allocator, ms.vertex_buf);
    destroy_buffer(allocator, ms.index_buf);
    ms.vertex_count = 0;
    ms.index_count  = 0;
}

// ---------------------------------------------------------------------------
// File I/O state
// ---------------------------------------------------------------------------
inline const std::filesystem::path& species_dir() {
    static const std::filesystem::path dir{"species"};
    return dir;
}

struct FileIOState {
    char name_buf[128]   = "default";
    char status_buf[256] = "";
    std::vector<std::string> file_list;
    int  selected_file   = -1;
    float status_timer   = 0.0f;

    void set_status(const char* msg) {
        std::snprintf(status_buf, sizeof(status_buf), "%s", msg);
        status_timer = 3.0f;
    }

    std::vector<std::string> kind_filter;

    void refresh_files() {
        file_list.clear();
        selected_file = -1;
        if (!std::filesystem::exists(species_dir())) return;
        for (auto& entry : std::filesystem::directory_iterator(species_dir())) {
            if (entry.path().extension() != ".toml") continue;
            if (!kind_filter.empty()) {
                auto kind = bestiary::detect_species_kind(entry.path());
                bool match = false;
                for (auto& k : kind_filter)
                    if (k == kind) { match = true; break; }
                if (!match) continue;
            }
            file_list.push_back(entry.path().stem().string());
        }
        std::sort(file_list.begin(), file_list.end());
    }
};

// ---------------------------------------------------------------------------
// Wind state
// ---------------------------------------------------------------------------
struct WindState {
    float angle = 0.0f;
    float speed = 0.3f;
};

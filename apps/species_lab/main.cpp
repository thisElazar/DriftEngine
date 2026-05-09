// Species Lab — v0.0.5

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "renderer.h"

#include "morphology/clump.h"
#include "morphology/bush.h"
#include "species_file.h"

#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// SPIR-V loading (same as engine's pipeline.cpp)
// ---------------------------------------------------------------------------
std::vector<uint32_t> load_spirv(const char* path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "Failed to open SPIR-V file: %s\n", path);
        std::abort();
    }
    auto size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));
    return buffer;
}

// ---------------------------------------------------------------------------
// GPU buffer helpers
// ---------------------------------------------------------------------------
struct GpuBuffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

GpuBuffer create_buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size  = size;
    ci.usage = usage;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    GpuBuffer buf;
    VK_CHECK(vmaCreateBuffer(allocator, &ci, &alloc_ci, &buf.buffer, &buf.allocation, nullptr));
    return buf;
}

void destroy_buffer(VmaAllocator allocator, GpuBuffer& buf)
{
    if (buf.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buf.buffer, buf.allocation);
        buf.buffer     = VK_NULL_HANDLE;
        buf.allocation = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// Orbit camera
// ---------------------------------------------------------------------------
float g_scroll_accum = 0.0f;

void scroll_cb(GLFWwindow*, double, double yoffset)
{
    g_scroll_accum += static_cast<float>(yoffset);
}

struct OrbitCamera {
    float     yaw      = 0.0f;
    float     pitch    = 0.3f;
    float     distance = 1.5f;
    glm::vec3 target   = {0.0f, 0.2f, 0.0f};
    double    last_mx  = 0.0;
    double    last_my  = 0.0;
    bool      dragging = false;
};

void update_orbit(OrbitCamera& cam, GLFWwindow* window, float max_dist = 5.0f)
{
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);

    bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (rmb && !ImGui::GetIO().WantCaptureMouse) {
        if (cam.dragging) {
            cam.yaw   -= static_cast<float>(mx - cam.last_mx) * 0.005f;
            cam.pitch += static_cast<float>(my - cam.last_my) * 0.005f;
            cam.pitch  = glm::clamp(cam.pitch, -1.5f, 1.5f);
        }
        cam.dragging = true;
    } else {
        cam.dragging = false;
    }
    cam.last_mx = mx;
    cam.last_my = my;

    if (!ImGui::GetIO().WantCaptureMouse) {
        cam.distance -= g_scroll_accum * 0.1f;
        cam.distance  = glm::clamp(cam.distance, 0.2f, max_dist);
    }
    g_scroll_accum = 0.0f;
}

glm::mat4 orbit_view(const OrbitCamera& cam)
{
    float cx = std::cos(cam.pitch) * std::sin(cam.yaw);
    float cy = std::sin(cam.pitch);
    float cz = std::cos(cam.pitch) * std::cos(cam.yaw);
    glm::vec3 eye = cam.target + glm::vec3(cx, cy, cz) * cam.distance;
    return glm::lookAt(eye, cam.target, glm::vec3(0.0f, 1.0f, 0.0f));
}

// ---------------------------------------------------------------------------
// Clump pipeline
// ---------------------------------------------------------------------------
struct ClumpPipeline {
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkShaderModule   vs       = VK_NULL_HANDLE;
    VkShaderModule   fs       = VK_NULL_HANDLE;
};

struct ClumpPC {
    glm::mat4 mvp;
    float wind_dir[2];
    float wind_speed;
    float time;
};

ClumpPipeline create_clump_pipeline(VkDevice device)
{
    ClumpPipeline p{};

    auto vs_spv = load_spirv("shaders/clump_vs.spv");
    auto fs_spv = load_spirv("shaders/clump_fs.spv");

    VkShaderModuleCreateInfo vs_ci{};
    vs_ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vs_ci.codeSize = vs_spv.size() * sizeof(uint32_t);
    vs_ci.pCode    = vs_spv.data();
    VK_CHECK(vkCreateShaderModule(device, &vs_ci, nullptr, &p.vs));

    VkShaderModuleCreateInfo fs_ci{};
    fs_ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fs_ci.codeSize = fs_spv.size() * sizeof(uint32_t);
    fs_ci.pCode    = fs_spv.data();
    VK_CHECK(vkCreateShaderModule(device, &fs_ci, nullptr, &p.fs));

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.offset     = 0;
    push.size       = sizeof(ClumpPC);

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &push;
    VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.layout));

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = p.vs;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = p.fs;
    stages[1].pName  = "main";

    VkVertexInputBindingDescription vb{};
    vb.binding   = 0;
    vb.stride    = sizeof(bestiary::ClumpVertex);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[4]{};
    attrs[0].location = 0;  attrs[0].binding = 0;
    attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;  attrs[0].offset = offsetof(bestiary::ClumpVertex, position);
    attrs[1].location = 1;  attrs[1].binding = 0;
    attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;  attrs[1].offset = offsetof(bestiary::ClumpVertex, normal);
    attrs[2].location = 2;  attrs[2].binding = 0;
    attrs[2].format   = VK_FORMAT_R32G32B32_SFLOAT;  attrs[2].offset = offsetof(bestiary::ClumpVertex, color);
    attrs[3].location = 3;  attrs[3].binding = 0;
    attrs[3].format   = VK_FORMAT_R32_SFLOAT;         attrs[3].offset = offsetof(bestiary::ClumpVertex, height_t);

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vb;
    vi.vertexAttributeDescriptionCount = 4;
    vi.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_NONE;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blend_att{};
    blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                             | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &blend_att;

    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dyn_states;

    VkFormat color_fmt = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depth_fmt = VK_FORMAT_D32_SFLOAT;

    VkPipelineRenderingCreateInfo rendering_ci{};
    rendering_ci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_ci.colorAttachmentCount    = 1;
    rendering_ci.pColorAttachmentFormats = &color_fmt;
    rendering_ci.depthAttachmentFormat   = depth_fmt;

    VkGraphicsPipelineCreateInfo gp_ci{};
    gp_ci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp_ci.pNext               = &rendering_ci;
    gp_ci.stageCount          = 2;
    gp_ci.pStages             = stages;
    gp_ci.pVertexInputState   = &vi;
    gp_ci.pInputAssemblyState = &ia;
    gp_ci.pViewportState      = &vp;
    gp_ci.pRasterizationState = &rs;
    gp_ci.pMultisampleState   = &ms;
    gp_ci.pDepthStencilState  = &ds;
    gp_ci.pColorBlendState    = &cb;
    gp_ci.pDynamicState       = &dyn;
    gp_ci.layout              = p.layout;

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp_ci, nullptr, &p.pipeline));

    return p;
}

void destroy_clump_pipeline(VkDevice device, ClumpPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
}

// ---------------------------------------------------------------------------
// Mesh state
// ---------------------------------------------------------------------------
struct MeshState {
    GpuBuffer vertex_buf{};
    GpuBuffer index_buf{};
    uint32_t  index_count  = 0;
    uint32_t  vertex_count = 0;
};

void upload_mesh(VmaAllocator allocator, MeshState& ms, const bestiary::ClumpMesh& mesh)
{
    ms.vertex_count = static_cast<uint32_t>(mesh.vertices.size());
    ms.index_count  = static_cast<uint32_t>(mesh.indices.size());

    VkDeviceSize vb_size = mesh.vertices.size() * sizeof(bestiary::ClumpVertex);
    VkDeviceSize ib_size = mesh.indices.size()  * sizeof(uint32_t);

    ms.vertex_buf = create_buffer(allocator, vb_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    ms.index_buf  = create_buffer(allocator, ib_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    void* mapped = nullptr;
    VK_CHECK(vmaMapMemory(allocator, ms.vertex_buf.allocation, &mapped));
    std::memcpy(mapped, mesh.vertices.data(), vb_size);
    vmaUnmapMemory(allocator, ms.vertex_buf.allocation);

    VK_CHECK(vmaMapMemory(allocator, ms.index_buf.allocation, &mapped));
    std::memcpy(mapped, mesh.indices.data(), ib_size);
    vmaUnmapMemory(allocator, ms.index_buf.allocation);
}

void destroy_mesh(VmaAllocator allocator, MeshState& ms)
{
    destroy_buffer(allocator, ms.vertex_buf);
    destroy_buffer(allocator, ms.index_buf);
    ms.vertex_count = 0;
    ms.index_count  = 0;
}

// ---------------------------------------------------------------------------
// File I/O state
// ---------------------------------------------------------------------------
const std::filesystem::path SPECIES_DIR = "../species";

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

    void refresh_files() {
        file_list.clear();
        selected_file = -1;
        if (!std::filesystem::exists(SPECIES_DIR)) return;
        for (auto& entry : std::filesystem::directory_iterator(SPECIES_DIR)) {
            if (entry.path().extension() == ".toml")
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

// ---------------------------------------------------------------------------
// Dirty-check snapshot
// ---------------------------------------------------------------------------
struct LabSnapshot {
    int                        species_kind;
    int                        view_mode;
    bestiary::ClumpParams      clump_params;
    bestiary::ClumpExpression  clump_expr;
    bestiary::BushParams       bush_params;
    bestiary::BushExpression   bush_expr;
    bestiary::FieldParams      field;
};

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
void draw_lab_panel(int& species_kind, int& view_mode,
                    bestiary::ClumpParams& clump_params, bestiary::ClumpExpression& clump_expr,
                    bestiary::BushParams& bush_params, bestiary::BushExpression& bush_expr,
                    bestiary::FieldParams& field_params,
                    const MeshState& ms, FileIOState& fio, WindState& wind, float dt)
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 780), ImGuiCond_FirstUseEver);
    ImGui::Begin("Species Lab v0.0.5");

    const char* kinds[] = {"Grass Clump", "Bush"};
    ImGui::Combo("species", &species_kind, kinds, 2);

    const char* modes[] = {"Single", "Field"};
    ImGui::Combo("mode", &view_mode, modes, 2);

    ImGui::Spacing();
    ImGui::TextUnformatted("Morphology (base)");
    ImGui::Separator();

    if (species_kind == 0) {
        ImGui::SliderInt  ("blade_count",  &clump_params.blade_count,  1,     60);
        ImGui::SliderFloat("blade_height", &clump_params.blade_height, 0.05f, 2.0f,  "%.3f m");
        ImGui::SliderFloat("blade_width",  &clump_params.blade_width,  0.005f,0.05f, "%.4f m");
        ImGui::SliderFloat("splay_angle",  &clump_params.splay_angle,  0.0f,  80.0f, "%.1f deg");
        ImGui::SliderFloat("clump_radius", &clump_params.clump_radius, 0.0f,  0.5f,  "%.3f m");

        ImGui::Spacing();
        ImGui::TextUnformatted("Appearance");
        ImGui::Separator();
        ImGui::ColorEdit3("base_color", clump_params.base_color);
    } else {
        ImGui::SliderInt  ("leaf_count",  &bush_params.leaf_count,  10,    80);
        ImGui::SliderFloat("leaf_length", &bush_params.leaf_length, 0.05f, 0.3f,  "%.3f m");
        ImGui::SliderFloat("leaf_width",  &bush_params.leaf_width,  0.02f, 0.15f, "%.3f m");
        ImGui::SliderFloat("bush_height", &bush_params.bush_height, 0.1f,  1.5f,  "%.2f m");
        ImGui::SliderFloat("bush_radius", &bush_params.bush_radius, 0.05f, 0.5f,  "%.3f m");
        ImGui::SliderFloat("stem_height", &bush_params.stem_height, 0.0f,  0.3f,  "%.3f m");

        ImGui::Spacing();
        ImGui::TextUnformatted("Appearance");
        ImGui::Separator();
        ImGui::ColorEdit3("base_color", bush_params.base_color);
    }

    auto range_ui = [](const char* label, bestiary::ParamRange& r,
                       float lo, float hi, const char* fmt) {
        ImGui::PushID(label);
        ImGui::Checkbox("##enabled", &r.enabled);
        ImGui::SameLine();
        ImGui::TextUnformatted(label);
        if (r.enabled) {
            ImGui::Indent();
            ImGui::SliderFloat("low",  &r.low,  lo, hi, fmt);
            ImGui::SliderFloat("high", &r.high, lo, hi, fmt);
            ImGui::Unindent();
        }
        ImGui::PopID();
    };

    if (view_mode == 1) {
        ImGui::Spacing();
        ImGui::TextUnformatted("Phenotype Expression (moisture)");
        ImGui::Separator();
        ImGui::TextDisabled("dry (left) -> wet (right)");

        if (species_kind == 0) {
            range_ui("blade_count",  clump_expr.blade_count,  1.0f,  60.0f, "%.0f");
            range_ui("blade_height", clump_expr.blade_height, 0.05f, 2.0f,  "%.3f m");
            range_ui("blade_width",  clump_expr.blade_width,  0.005f,0.05f, "%.4f m");
            range_ui("splay_angle",  clump_expr.splay_angle,  0.0f,  80.0f, "%.1f deg");
            range_ui("clump_radius", clump_expr.clump_radius, 0.0f,  0.5f,  "%.3f m");

            ImGui::Spacing();
            ImGui::Checkbox("vary_color", &clump_expr.vary_color);
            if (clump_expr.vary_color) {
                ImGui::Indent();
                ImGui::ColorEdit3("dry_color", clump_expr.dry_color);
                ImGui::ColorEdit3("wet_color", clump_expr.wet_color);
                ImGui::Unindent();
            }
        } else {
            range_ui("leaf_count",  bush_expr.leaf_count,  10.0f, 80.0f, "%.0f");
            range_ui("leaf_length", bush_expr.leaf_length, 0.05f, 0.3f,  "%.3f m");
            range_ui("leaf_width",  bush_expr.leaf_width,  0.02f, 0.15f, "%.3f m");
            range_ui("bush_height", bush_expr.bush_height, 0.1f,  1.5f,  "%.2f m");
            range_ui("bush_radius", bush_expr.bush_radius, 0.05f, 0.5f,  "%.3f m");
            range_ui("stem_height", bush_expr.stem_height, 0.0f,  0.3f,  "%.3f m");

            ImGui::Spacing();
            ImGui::Checkbox("vary_color", &bush_expr.vary_color);
            if (bush_expr.vary_color) {
                ImGui::Indent();
                ImGui::ColorEdit3("dry_color", bush_expr.dry_color);
                ImGui::ColorEdit3("wet_color", bush_expr.wet_color);
                ImGui::Unindent();
            }
        }

        ImGui::Spacing();
        ImGui::TextUnformatted("Field Settings");
        ImGui::Separator();
        ImGui::SliderInt("grid_n",    &field_params.grid_n,  2,    16);
        ImGui::SliderFloat("spacing", &field_params.spacing, 0.1f, 1.0f, "%.2f m");
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Wind");
    ImGui::Separator();
    ImGui::SliderFloat("direction", &wind.angle, 0.0f, 360.0f, "%.0f deg");
    ImGui::SliderFloat("speed",     &wind.speed, 0.0f, 2.0f,   "%.2f");

    ImGui::Spacing();
    ImGui::TextUnformatted("File");
    ImGui::Separator();

    ImGui::InputText("name", fio.name_buf, sizeof(fio.name_buf));

    if (ImGui::Button("Save")) {
        std::filesystem::create_directories(SPECIES_DIR);
        auto path = SPECIES_DIR / (std::string(fio.name_buf) + ".toml");
        bool ok = false;
        if (species_kind == 0)
            ok = bestiary::save_clump(path, clump_params, fio.name_buf, &clump_expr);
        else
            ok = bestiary::save_bush(path, bush_params, fio.name_buf, &bush_expr);
        fio.set_status(ok ? "Saved." : "Save failed.");
        if (ok) fio.refresh_files();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load") && fio.selected_file >= 0) {
        auto path = SPECIES_DIR / (fio.file_list[static_cast<size_t>(fio.selected_file)] + ".toml");
        std::string kind = bestiary::detect_species_kind(path);
        std::string loaded_name;
        bool ok = false;
        if (kind == "grass_clump") {
            ok = bestiary::load_clump(path, clump_params, loaded_name, &clump_expr);
            if (ok) species_kind = 0;
        } else if (kind == "bush") {
            ok = bestiary::load_bush(path, bush_params, loaded_name, &bush_expr);
            if (ok) species_kind = 1;
        }
        if (ok) {
            std::snprintf(fio.name_buf, sizeof(fio.name_buf), "%s", loaded_name.c_str());
            fio.set_status("Loaded.");
        } else {
            fio.set_status("Load failed.");
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) {
        fio.refresh_files();
    }

    if (fio.status_timer > 0.0f) {
        fio.status_timer -= dt;
        ImGui::TextUnformatted(fio.status_buf);
    }

    if (!fio.file_list.empty()) {
        ImGui::BeginChild("file_list", ImVec2(0, 120), ImGuiChildFlags_Borders);
        for (int i = 0; i < static_cast<int>(fio.file_list.size()); ++i) {
            bool selected = (i == fio.selected_file);
            if (ImGui::Selectable(fio.file_list[static_cast<size_t>(i)].c_str(), selected))
                fio.selected_file = i;
        }
        ImGui::EndChild();
    }

    ImGui::Spacing();
    ImGui::Separator();
    if (view_mode == 0) {
        ImGui::Text("[INFO] mesh: %u verts, %u tris",
                    ms.vertex_count, ms.index_count / 3);
    } else {
        ImGui::Text("[INFO] field: %dx%d, mesh: %u verts, %u tris",
                    field_params.grid_n, field_params.grid_n,
                    ms.vertex_count, ms.index_count / 3);
    }
    ImGui::TextUnformatted("RMB drag to orbit, scroll to zoom.");

    ImGui::End();
    ImGui::Render();
}

// ---------------------------------------------------------------------------
// Frame recording
// ---------------------------------------------------------------------------
void record_frame(Renderer& r, FrameData& frame, uint32_t image_index, VkExtent2D extent,
                  const ClumpPipeline& cpipe, const MeshState& mesh, const ClumpPC& pc)
{
    VkCommandBuffer cmd = frame.cmd;

    // Transition swapchain and depth to attachment layouts.
    {
        VkImageMemoryBarrier2 barriers[2]{};

        barriers[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_NONE;
        barriers[0].dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].image         = r.swapchain_images[image_index];
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        barriers[1].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[1].srcAccessMask = VK_ACCESS_2_NONE;
        barriers[1].dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barriers[1].image         = r.depth_buffer.image;
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount  = 2;
        dep.pImageMemoryBarriers     = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Main pass: clear + draw clump.
    {
        VkRenderingAttachmentInfo color{};
        color.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView   = r.swapchain_views[image_index];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{0.07f, 0.08f, 0.10f, 1.0f}};

        VkRenderingAttachmentInfo depth{};
        depth.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView   = r.depth_buffer.view;
        depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea           = {{0, 0}, extent};
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &color;
        ri.pDepthAttachment     = &depth;

        vkCmdBeginRendering(cmd, &ri);

        if (mesh.index_count > 0) {
            VkViewport viewport{};
            viewport.x        = 0.0f;
            viewport.y        = 0.0f;
            viewport.width    = static_cast<float>(extent.width);
            viewport.height   = static_cast<float>(extent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{{0, 0}, extent};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, cpipe.pipeline);

            vkCmdPushConstants(cmd, cpipe.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertex_buf.buffer, &offset);
            vkCmdBindIndexBuffer(cmd, mesh.index_buf.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
        }

        vkCmdEndRendering(cmd);
    }

    // ImGui pass (color load, no depth).
    {
        VkRenderingAttachmentInfo color{};
        color.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView   = r.swapchain_views[image_index];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
        color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

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

    // Transition swapchain to present.
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
        dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount  = 1;
        dep.pImageMemoryBarriers     = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

} // namespace

int main()
{
    Renderer renderer{};
    renderer_init(renderer, 1280, 800, "Species Lab");

    glfwSetScrollCallback(renderer.window, scroll_cb);
    ImGui_ImplGlfw_InstallCallbacks(renderer.window);

    ClumpPipeline cpipe = create_clump_pipeline(renderer.device);

    bestiary::ClumpParams      clump_params{};
    bestiary::ClumpExpression  clump_expr{};
    bestiary::BushParams       bush_params{};
    bestiary::BushExpression   bush_expr{};
    bestiary::FieldParams      field_params{};
    int species_kind = 0;
    int view_mode = 0;
    int last_view_mode = -1;

    LabSnapshot last_snapshot{};
    std::memset(&last_snapshot, 0xFF, sizeof(last_snapshot));

    MeshState mesh{};
    OrbitCamera camera{};
    FileIOState fio{};
    fio.refresh_files();
    WindState wind{};

    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(renderer.window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        glfwPollEvents();
        float max_zoom = (view_mode == 1) ? 20.0f : 5.0f;
        update_orbit(camera, renderer.window, max_zoom);

        if (view_mode != last_view_mode) {
            if (view_mode == 1) {
                float field_extent = static_cast<float>(field_params.grid_n - 1) * field_params.spacing;
                camera.distance = field_extent * 1.2f;
                camera.target   = {0.0f, 0.1f, 0.0f};
                camera.pitch    = 0.6f;
            } else {
                camera.distance = 1.5f;
                camera.target   = {0.0f, 0.2f, 0.0f};
                camera.pitch    = 0.3f;
            }
            last_view_mode = view_mode;
        }

        LabSnapshot current{};
        std::memset(&current, 0, sizeof(current));
        current.species_kind = species_kind;
        current.view_mode    = view_mode;
        current.clump_params = clump_params;
        current.clump_expr   = clump_expr;
        current.bush_params  = bush_params;
        current.bush_expr    = bush_expr;
        current.field        = field_params;

        if (std::memcmp(&current, &last_snapshot, sizeof(LabSnapshot)) != 0) {
            vkDeviceWaitIdle(renderer.device);
            destroy_mesh(renderer.allocator, mesh);

            if (species_kind == 0) {
                if (view_mode == 0) {
                    auto m = bestiary::generate_clump(clump_params);
                    upload_mesh(renderer.allocator, mesh, m);
                } else {
                    auto m = bestiary::generate_field(clump_params, clump_expr, field_params);
                    upload_mesh(renderer.allocator, mesh, m);
                }
            } else {
                if (view_mode == 0) {
                    auto m = bestiary::generate_bush(bush_params);
                    upload_mesh(renderer.allocator, mesh, m);
                } else {
                    auto m = bestiary::generate_bush_field(bush_params, bush_expr, field_params);
                    upload_mesh(renderer.allocator, mesh, m);
                }
            }
            last_snapshot = current;
        }

        draw_lab_panel(species_kind, view_mode,
                       clump_params, clump_expr, bush_params, bush_expr,
                       field_params, mesh, fio, wind, dt);

        FrameData* frame = nullptr;
        uint32_t   image_index = 0;
        VkExtent2D extent{};
        if (!renderer_begin_frame(renderer, frame, image_index, extent))
            continue;

        float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
        proj[1][1] *= -1.0f;
        glm::mat4 view = orbit_view(camera);

        float wind_rad = glm::radians(wind.angle);
        ClumpPC pc{};
        pc.mvp        = proj * view;
        pc.wind_dir[0] = std::cos(wind_rad);
        pc.wind_dir[1] = std::sin(wind_rad);
        pc.wind_speed  = wind.speed;
        pc.time        = static_cast<float>(now);

        record_frame(renderer, *frame, image_index, extent, cpipe, mesh, pc);
        renderer_end_frame(renderer, *frame, image_index);
    }

    vkDeviceWaitIdle(renderer.device);
    destroy_mesh(renderer.allocator, mesh);
    destroy_clump_pipeline(renderer.device, cpipe);
    renderer_shutdown(renderer);
    return 0;
}

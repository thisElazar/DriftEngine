// Sim Couple Testing v0.1 — simulation coupling test harness.
//
// Single-tile sandbox: terrain brush + SWE water + atmosphere 3D + erosion.
// Tests the full authority chain: brush -> SWE -> atmosphere -> erosion -> render.
// No bestiary, no planet mode, no stamps.

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
#include "resources.h"
#include "vk_util.h"
#include "grid_util.h"
#include "pipeline.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Tile constants
// ---------------------------------------------------------------------------
constexpr uint32_t GRID_W   = 256;
constexpr uint32_t GRID_H   = 256;
constexpr float    DX       = 1.0f;        // 1 m per cell -> tile is 256m x 256m
constexpr float    TILE_HALF_X = GRID_W * DX * 0.5f;
constexpr float    TILE_HALF_Z = GRID_H * DX * 0.5f;

// Atmosphere volume dimensions
constexpr uint32_t ATMO_W = GRID_W;
constexpr uint32_t ATMO_H = GRID_H;
constexpr uint32_t ATMO_D = 32;
// ATMO_LAYER_HEIGHT controlled via ui_layer_height slider

struct CameraUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec3 sun_dir;   float _p0;
    glm::vec3 sun_color; float _p1;
    glm::vec3 cam_pos;   float _p2;
    glm::vec4 brush_world;
    glm::vec4 brush_color;
    glm::mat4 inv_view_proj;
};

struct WorldTerrainPC {
    glm::mat4 mvp;
    float     grid_w_f;
    float     grid_h_f;
    float     cell_size;
    float     sea_level;
    float     brush_x;
    float     brush_y;
    float     brush_radius;
    float     brush_active;
    float     moisture_overlay;
    float     _pad0;
    float     _pad1;
    float     _pad2;
};

namespace {

// Volume image barrier — all layers/mips
void volume_barrier(VkCommandBuffer cmd, VkImage img,
                    VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                    VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                    VkImageLayout old_layout, VkImageLayout new_layout)
{
    VkImageMemoryBarrier2 b{};
    b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask  = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask  = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout     = old_layout;
    b.newLayout     = new_layout;
    b.image         = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

// Compute->graphics full barrier
void compute_to_graphics_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                     | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                     | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

// ---------------------------------------------------------------------------
// 3D volume image with TRANSFER_SRC for readback
// ---------------------------------------------------------------------------
SweImage create_volume_image_readback(VkDevice device, VmaAllocator allocator,
                                      uint32_t w, uint32_t h, uint32_t d, VkFormat format)
{
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_3D;
    img_ci.format = format;
    img_ci.extent = {w, h, d};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                 | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    SweImage img{};
    VK_CHECK(vmaCreateImage(allocator, &img_ci, &alloc_ci, &img.image, &img.allocation, nullptr));

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = img.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view_ci.format = format;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &view_ci, nullptr, &img.view));

    return img;
}

// 2D shadow image (R16F, storage + sampled)
SweImage create_shadow_image(VkDevice device, VmaAllocator allocator,
                             uint32_t w, uint32_t h)
{
    VkImageCreateInfo img_ci{};
    img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType = VK_IMAGE_TYPE_2D;
    img_ci.format = VK_FORMAT_R16_SFLOAT;
    img_ci.extent = {w, h, 1};
    img_ci.mipLevels = 1;
    img_ci.arrayLayers = 1;
    img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    SweImage img{};
    VK_CHECK(vmaCreateImage(allocator, &img_ci, &alloc_ci, &img.image, &img.allocation, nullptr));

    VkImageViewCreateInfo view_ci{};
    view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image = img.image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_R16_SFLOAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &view_ci, nullptr, &img.view));

    return img;
}

// ---------------------------------------------------------------------------
// Graphics pipeline — terrain
// ---------------------------------------------------------------------------
struct TerrainPipeline {
    VkShaderModule        vs       = VK_NULL_HANDLE;
    VkShaderModule        fs       = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl      = VK_NULL_HANDLE;
    VkPipelineLayout      layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
};

TerrainPipeline create_terrain_pipeline(VkDevice device)
{
    TerrainPipeline p{};
    p.vs = make_shader(device, "shaders/world_terrain_vs.spv");
    p.fs = make_shader(device, "shaders/world_terrain_fs.spv");

    // Bindings: 0=heightmap, 1=water output, 2=moisture (debug overlay).
    VkDescriptorSetLayoutBinding b[3]{};
    for (int i = 0; i < 3; ++i) {
        b[i].binding         = static_cast<uint32_t>(i);
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings    = b;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &p.dsl));

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push.size       = sizeof(WorldTerrainPC);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &p.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &push;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &p.layout));

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
    vb.stride    = sizeof(float) * 2;
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attr{};
    attr.location = 0;
    attr.binding  = 0;
    attr.format   = VK_FORMAT_R32G32_SFLOAT;
    attr.offset   = 0;

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vb;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions    = &attr;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vpstate{};
    vpstate.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpstate.viewportCount = 1;
    vpstate.scissorCount  = 1;

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

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                      | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &ba;

    VkDynamicState dyns[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dyns;

    VkFormat color_fmt = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depth_fmt = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rci{};
    rci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rci.colorAttachmentCount    = 1;
    rci.pColorAttachmentFormats = &color_fmt;
    rci.depthAttachmentFormat   = depth_fmt;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.pNext               = &rci;
    gp.stageCount          = 2;
    gp.pStages             = stages;
    gp.pVertexInputState   = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState      = &vpstate;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState   = &ms;
    gp.pDepthStencilState  = &ds;
    gp.pColorBlendState    = &cb;
    gp.pDynamicState       = &dyn;
    gp.layout              = p.layout;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &p.pipeline));
    return p;
}

void destroy_terrain_pipeline(VkDevice device, TerrainPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.dsl, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
}

// ---------------------------------------------------------------------------
// Cloud raymarch pipeline (fullscreen, alpha-blended, no depth, no vertex input)
// ---------------------------------------------------------------------------
struct CloudPipeline {
    VkShaderModule        vs       = VK_NULL_HANDLE;
    VkShaderModule        fs       = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl      = VK_NULL_HANDLE;
    VkPipelineLayout      layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
};

CloudPipeline create_cloud_pipeline(VkDevice device)
{
    CloudPipeline p{};
    p.vs = make_shader(device, "shaders/cloud_raymarch_vs.spv");
    p.fs = make_shader(device, "shaders/cloud_raymarch_fs.spv");

    VkDescriptorSetLayoutBinding b[3]{};
    b[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    b[2] = {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings    = b;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &p.dsl));

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push.size       = sizeof(RaymarchPC);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &p.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &push;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &p.layout));

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = p.vs;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = p.fs;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vpstate{};
    vpstate.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpstate.viewportCount = 1;
    vpstate.scissorCount  = 1;

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
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState ba{};
    ba.blendEnable         = VK_TRUE;
    ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.colorBlendOp        = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.alphaBlendOp        = VK_BLEND_OP_ADD;
    ba.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                           | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &ba;

    VkDynamicState dyns[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dyns;

    VkFormat color_fmt = VK_FORMAT_B8G8R8A8_UNORM;
    VkPipelineRenderingCreateInfo rci{};
    rci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rci.colorAttachmentCount    = 1;
    rci.pColorAttachmentFormats = &color_fmt;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.pNext               = &rci;
    gp.stageCount          = 2;
    gp.pStages             = stages;
    gp.pVertexInputState   = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState      = &vpstate;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState   = &ms;
    gp.pDepthStencilState  = &ds;
    gp.pColorBlendState    = &cb;
    gp.pDynamicState       = &dyn;
    gp.layout              = p.layout;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &p.pipeline));
    return p;
}

void destroy_cloud_pipeline(VkDevice device, CloudPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.dsl, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
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
    float yaw = 0.6f, pitch = 0.7f;
    float distance = 220.0f;
    glm::vec3 target = {0.0f, 0.0f, 0.0f};
    double last_mx = 0.0, last_my = 0.0;
    bool   dragging = false;
};

void update_orbit(OrbitCamera& cam, GLFWwindow* window)
{
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    bool capture = ImGui::GetIO().WantCaptureMouse;
    if (rmb && !capture) {
        if (cam.dragging) {
            cam.yaw   -= static_cast<float>(mx - cam.last_mx) * 0.005f;
            cam.pitch += static_cast<float>(my - cam.last_my) * 0.005f;
            cam.pitch  = glm::clamp(cam.pitch, 0.05f, 1.4f);
        }
        cam.dragging = true;
    } else {
        cam.dragging = false;
    }
    cam.last_mx = mx; cam.last_my = my;
    if (!capture) {
        cam.distance *= std::pow(0.9f, g_scroll_accum);
        cam.distance = glm::clamp(cam.distance, 30.0f, 800.0f);
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
// Ray vs Y=0 plane (in tile world space), returns grid coords in [0..GRID_W).
// ---------------------------------------------------------------------------
bool pick_grid_cell(const glm::mat4& view, const glm::mat4& proj,
                    glm::vec2 cursor_ndc,
                    float& out_gx, float& out_gy)
{
    glm::mat4 inv = glm::inverse(proj * view);
    glm::vec4 near_h = inv * glm::vec4(cursor_ndc.x, cursor_ndc.y, 0.0f, 1.0f);
    glm::vec4 far_h  = inv * glm::vec4(cursor_ndc.x, cursor_ndc.y, 1.0f, 1.0f);
    glm::vec3 near_p = glm::vec3(near_h) / near_h.w;
    glm::vec3 far_p  = glm::vec3(far_h)  / far_h.w;
    glm::vec3 dir    = far_p - near_p;
    if (std::abs(dir.y) < 1e-6f) return false;
    float t = -near_p.y / dir.y;
    if (t < 0.0f) return false;
    glm::vec3 hit = near_p + t * dir;
    out_gx = (hit.x + TILE_HALF_X) / DX;
    out_gy = (hit.z + TILE_HALF_Z) / DX;
    return out_gx >= 0.0f && out_gx < GRID_W && out_gy >= 0.0f && out_gy < GRID_H;
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    Renderer renderer{};
    renderer_init(renderer, 1280, 800, "Sim Couple Testing");
    glfwSetScrollCallback(renderer.window, scroll_cb);
    ImGui_ImplGlfw_InstallCallbacks(renderer.window);

    VkDevice device   = renderer.device;
    VmaAllocator alloc = renderer.allocator;
    VkQueue gqueue    = renderer.graphics_queue;
    uint32_t gfamily  = renderer.gfx_family;

    // ---- Heightmap (initial: a bowl / valley for water to collect) ---------
    HeightmapData hm_data{};
    hm_data.width = GRID_W; hm_data.height = GRID_H;
    hm_data.values.assign(GRID_W * GRID_H, 0.0f);
    {
        float cx = GRID_W * 0.5f, cy = GRID_H * 0.5f;
        for (uint32_t y = 0; y < GRID_H; ++y) {
            for (uint32_t x = 0; x < GRID_W; ++x) {
                float dx_ = static_cast<float>(x) - cx;
                float dy_ = static_cast<float>(y) - cy;
                float r  = std::sqrt(dx_ * dx_ + dy_ * dy_);
                // Bowl: deep center, rising rim. ~20m total relief.
                float h = -8.0f + 0.0008f * r * r;
                hm_data.values[y * GRID_W + x] = h;
            }
        }
    }
    HeightmapGPU hm_gpu = upload_heightmap(device, alloc, gqueue, gfamily, hm_data);
    std::vector<float> hm_cpu = hm_data.values;

    // Moisture debug texture — same R32F shape as heightmap, starts at zero.
    HeightmapData moist_data{};
    moist_data.width = GRID_W; moist_data.height = GRID_H;
    moist_data.values.assign(GRID_W * GRID_H, 0.0f);
    HeightmapGPU moist_gpu = upload_heightmap(device, alloc, gqueue, gfamily, moist_data);

    // ---- SWE images -------------------------------------------------------
    SweImage state_a = create_swe_image(device, alloc, GRID_W, GRID_H);
    SweImage state_b = create_swe_image(device, alloc, GRID_W, GRID_H);
    SweImage water_out = create_swe_image(device, alloc, GRID_W, GRID_H);

    // Transition water images to GENERAL and clear to zero
    {
        OneShot s = oneshot_begin(device, gqueue, gfamily);
        for (VkImage img : {state_a.image, state_b.image, water_out.image}) {
            image_barrier(s.cmd, img,
                          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        VkClearColorValue clr{}; clr.float32[0] = 0.0f;
        VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(s.cmd, state_a.image,   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
        vkCmdClearColorImage(s.cmd, state_b.image,   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
        vkCmdClearColorImage(s.cmd, water_out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
        for (VkImage img : {state_a.image, state_b.image, water_out.image}) {
            image_barrier(s.cmd, img,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                            | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }
        oneshot_end(s);
    }

    // ---- Atmosphere 3D images -----------------------------------------------
    // State volume: RGBA16F (temperature, humidity, cloud_density, pressure)
    SweImage atmo_state[2] = {
        create_volume_image(device, alloc, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT),
        create_volume_image(device, alloc, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT),
    };
    // Wind volume: RGBA16F (vx, vy, vz, pressure) — readback-capable for debug
    SweImage wind_field[2] = {
        create_volume_image_readback(device, alloc, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT),
        create_volume_image_readback(device, alloc, ATMO_W, ATMO_H, ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT),
    };
    // Shadow map: R16F 2D
    SweImage atmo_shadow = create_shadow_image(device, alloc, ATMO_W, ATMO_H);

    // Ground conditions: 2D projections of ground-level atmosphere state
    SweImage ground_cond = create_swe_image(device, alloc, ATMO_W, ATMO_H);   // RGBA16F
    SweImage ground_wind = create_wind_image(device, alloc, ATMO_W, ATMO_H);  // RG16F

    // Transition atmosphere images to GENERAL
    {
        OneShot s = oneshot_begin(device, gqueue, gfamily);
        VkImage atmo_imgs[] = {
            atmo_state[0].image, atmo_state[1].image,
            wind_field[0].image, wind_field[1].image,
            atmo_shadow.image,
            ground_cond.image, ground_wind.image
        };
        for (VkImage img : atmo_imgs) {
            volume_barrier(s.cmd, img,
                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                             | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        }
        oneshot_end(s);
    }

    // ---- Sediment ping-pong images for erosion (256x256, R16F) ---------------
    SweImage sediment[2] = {
        create_sediment_image(device, alloc, GRID_W, GRID_H),
        create_sediment_image(device, alloc, GRID_W, GRID_H),
    };
    // Transition to GENERAL and clear
    {
        OneShot s = oneshot_begin(device, gqueue, gfamily);
        for (int i = 0; i < 2; ++i) {
            image_barrier(s.cmd, sediment[i].image,
                          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        VkClearColorValue clr{}; clr.float32[0] = 0.0f;
        VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (int i = 0; i < 2; ++i) {
            vkCmdClearColorImage(s.cmd, sediment[i].image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
            image_barrier(s.cmd, sediment[i].image,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                            | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }
        oneshot_end(s);
    }

    // ---- Wind readback staging buffers (per frame in flight) ----------------
    // Bottom layer of wind_field: ATMO_W * ATMO_H * 4 channels * 2 bytes = 128KB
    constexpr VkDeviceSize WIND_READBACK_SIZE = VkDeviceSize{ATMO_W} * ATMO_H * 4 * sizeof(uint16_t);
    constexpr VkDeviceSize GCOND_READBACK_SIZE = VkDeviceSize{ATMO_W} * ATMO_H * 4 * sizeof(uint16_t);
    GpuBuffer wind_readback[FRAMES_IN_FLIGHT];
    GpuBuffer gcond_readback[FRAMES_IN_FLIGHT];
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        wind_readback[i] = create_readback_buffer(alloc, WIND_READBACK_SIZE);
        gcond_readback[i] = create_readback_buffer(alloc, GCOND_READBACK_SIZE);
    }

    // ---- Sampler (for graphics) ------------------------------------------
    VkSampler sampler = VK_NULL_HANDLE;
    {
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device, &sci, nullptr, &sampler));
    }

    // ---- Compute pipelines ------------------------------------------------
    ComputePipeline pipe_swe_init = make_compute_pipeline(device, "shaders/swe_init.spv",
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        sizeof(SweInitPC));

    ComputePipeline pipe_swe_step = make_compute_pipeline(device, "shaders/swe_step.spv",
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,           // 0: terrain
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,            // 1: state_read
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 2: state_write
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 3: output
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},  // 4: ground_cond
        sizeof(SweStepPC));

    ComputePipeline pipe_brush = make_compute_pipeline(device, "shaders/terrain_brush.spv",
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        sizeof(TerrainBrushPC));

    // Atmosphere 3D compute pipeline
    // binding 0: terrain heightmap (combined image sampler)
    // binding 1: state_read (Texture3D)
    // binding 2: state_write (RWTexture3D)
    // binding 3: wind_read (Texture3D)
    // binding 4: wind_write (RWTexture3D)
    // binding 5: shadow_out (RWTexture2D)
    // binding 6: ground_cond_out (RWTexture2D<float4>)
    // binding 7: ground_wind_out (RWTexture2D<float2>)
    ComputePipeline pipe_atmo = make_compute_pipeline(device, "shaders/atmosphere3d_cs.spv",
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  // 0: terrain heightmap
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,            // 1: state_read
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 2: state_write
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,            // 3: wind_read
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 4: wind_write
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 5: shadow_out
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 6: ground_cond_out
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 7: ground_wind_out
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},  // 8: swe_output
        sizeof(Atmo3DPC));

    // Erosion compute pipeline
    // binding 0: terrain (RWTexture2D<float>)
    // binding 1: swe_state (Texture2D<float4>)
    // binding 2: sediment_in (Texture2D<float>)
    // binding 3: sediment_out (RWTexture2D<float>)
    ComputePipeline pipe_erosion = make_compute_pipeline(device, "shaders/erosion.spv",
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 0: terrain (heightmap RW)
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,            // 1: swe_state
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,            // 2: sediment_in
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,            // 3: sediment_out
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,   // 4: ground_cond
         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},  // 5: ground_wind
        sizeof(ErosionPC));

    // ---- Descriptor pool --------------------------------------------------
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,         32},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         32},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = 32;
        pci.poolSizeCount = 3;
        pci.pPoolSizes    = pool_sizes;
        VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &desc_pool));
    }

    auto alloc_set = [&](VkDescriptorSetLayout dsl) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = desc_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &dsl;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(device, &ai, &ds));
        return ds;
    };

    auto write_image = [&](VkDescriptorSet set, uint32_t binding,
                           VkDescriptorType type, VkImageView view, VkSampler sm) {
        VkDescriptorImageInfo info{};
        info.imageView   = view;
        info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        info.sampler     = sm;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = set;
        w.dstBinding      = binding;
        w.descriptorCount = 1;
        w.descriptorType  = type;
        w.pImageInfo      = &info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    };

    // --- SWE descriptor sets ---
    VkDescriptorSet ds_swe_init = alloc_set(pipe_swe_init.dsl);
    VkDescriptorSet ds_swe_step[2] = {
        alloc_set(pipe_swe_step.dsl),
        alloc_set(pipe_swe_step.dsl)
    };
    VkDescriptorSet ds_brush = alloc_set(pipe_brush.dsl);

    write_image(ds_swe_init, 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, hm_gpu.view,    VK_NULL_HANDLE);
    write_image(ds_swe_init, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, state_a.view,   VK_NULL_HANDLE);

    // step set 0: read A -> write B
    write_image(ds_swe_step[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, hm_gpu.view,  VK_NULL_HANDLE);
    write_image(ds_swe_step[0], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, state_a.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, state_b.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, water_out.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[0], 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ground_cond.view, sampler);

    // step set 1: read B -> write A
    write_image(ds_swe_step[1], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, hm_gpu.view,  VK_NULL_HANDLE);
    write_image(ds_swe_step[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, state_b.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, state_a.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, water_out.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[1], 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ground_cond.view, sampler);

    write_image(ds_brush, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, hm_gpu.view, VK_NULL_HANDLE);

    // --- Atmosphere descriptor sets (ping-ponged) ---
    VkDescriptorSet ds_atmo[2] = {
        alloc_set(pipe_atmo.dsl),
        alloc_set(pipe_atmo.dsl)
    };
    // Set 0: read state[0]/wind[0], write state[1]/wind[1]
    write_image(ds_atmo[0], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hm_gpu.view,         sampler);
    write_image(ds_atmo[0], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          atmo_state[0].view,   VK_NULL_HANDLE);
    write_image(ds_atmo[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          atmo_state[1].view,   VK_NULL_HANDLE);
    write_image(ds_atmo[0], 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          wind_field[0].view,   VK_NULL_HANDLE);
    write_image(ds_atmo[0], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          wind_field[1].view,   VK_NULL_HANDLE);
    write_image(ds_atmo[0], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          atmo_shadow.view,     VK_NULL_HANDLE);
    write_image(ds_atmo[0], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          ground_cond.view,     VK_NULL_HANDLE);
    write_image(ds_atmo[0], 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          ground_wind.view,     VK_NULL_HANDLE);
    write_image(ds_atmo[0], 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, water_out.view,       sampler);

    // Set 1: read state[1]/wind[1], write state[0]/wind[0]
    write_image(ds_atmo[1], 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hm_gpu.view,         sampler);
    write_image(ds_atmo[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          atmo_state[1].view,   VK_NULL_HANDLE);
    write_image(ds_atmo[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          atmo_state[0].view,   VK_NULL_HANDLE);
    write_image(ds_atmo[1], 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          wind_field[1].view,   VK_NULL_HANDLE);
    write_image(ds_atmo[1], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          wind_field[0].view,   VK_NULL_HANDLE);
    write_image(ds_atmo[1], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          atmo_shadow.view,     VK_NULL_HANDLE);
    write_image(ds_atmo[1], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          ground_cond.view,     VK_NULL_HANDLE);
    write_image(ds_atmo[1], 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          ground_wind.view,     VK_NULL_HANDLE);
    write_image(ds_atmo[1], 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, water_out.view,       sampler);

    // --- Erosion descriptor sets (ping-ponged on sediment) ---
    VkDescriptorSet ds_erosion[2] = {
        alloc_set(pipe_erosion.dsl),
        alloc_set(pipe_erosion.dsl)
    };
    // Set 0: sediment[0] -> sediment[1], reads current SWE state A
    write_image(ds_erosion[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  hm_gpu.view,       VK_NULL_HANDLE);
    write_image(ds_erosion[0], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  state_a.view,      VK_NULL_HANDLE);
    write_image(ds_erosion[0], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  sediment[0].view,  VK_NULL_HANDLE);
    write_image(ds_erosion[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  sediment[1].view,  VK_NULL_HANDLE);
    write_image(ds_erosion[0], 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ground_cond.view, sampler);
    write_image(ds_erosion[0], 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ground_wind.view, sampler);

    // Set 1: sediment[1] -> sediment[0], reads current SWE state B
    write_image(ds_erosion[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  hm_gpu.view,       VK_NULL_HANDLE);
    write_image(ds_erosion[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  state_b.view,      VK_NULL_HANDLE);
    write_image(ds_erosion[1], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  sediment[1].view,  VK_NULL_HANDLE);
    write_image(ds_erosion[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  sediment[0].view,  VK_NULL_HANDLE);
    write_image(ds_erosion[1], 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ground_cond.view, sampler);
    write_image(ds_erosion[1], 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ground_wind.view, sampler);

    // ---- Graphics pipelines ---------------------------------------------
    TerrainPipeline pipe_terrain = create_terrain_pipeline(device);

    VkDescriptorSet ds_terrain = alloc_set(pipe_terrain.dsl);
    write_image(ds_terrain, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hm_gpu.view,    sampler);
    write_image(ds_terrain, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, water_out.view, sampler);
    write_image(ds_terrain, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, moist_gpu.view, sampler);

    // ---- Cloud pipeline + camera UBO ----------------------------------------
    CloudPipeline pipe_cloud = create_cloud_pipeline(device);

    VkBuffer camera_ubo = VK_NULL_HANDLE;
    VmaAllocation camera_ubo_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo camera_ubo_info{};
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = sizeof(CameraUBO);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VK_CHECK(vmaCreateBuffer(alloc, &bci, &ai,
            &camera_ubo, &camera_ubo_alloc, &camera_ubo_info));
    }

    VkDescriptorSet ds_cloud[2] = {alloc_set(pipe_cloud.dsl), alloc_set(pipe_cloud.dsl)};
    for (int s = 0; s < 2; ++s) {
        VkDescriptorBufferInfo buf_info{};
        buf_info.buffer = camera_ubo;
        buf_info.offset = 0;
        buf_info.range  = sizeof(CameraUBO);
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = ds_cloud[s];
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo     = &buf_info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);

        write_image(ds_cloud[s], 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    hm_gpu.view, sampler);
        write_image(ds_cloud[s], 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    atmo_state[s].view, sampler);
    }

    StaticGridMesh terrain_mesh = build_grid_mesh(alloc, GRID_W, GRID_H);

    // ---- Run swe_init once -----------------------------------------------
    {
        OneShot s = oneshot_begin(device, gqueue, gfamily);
        vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_swe_init.pipeline);
        vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_swe_init.layout, 0, 1, &ds_swe_init, 0, nullptr);
        SweInitPC pc{GRID_W, GRID_H, -10000.0f, 0.0f}; // sea level very low -> no static water
        vkCmdPushConstants(s.cmd, pipe_swe_init.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(s.cmd, (GRID_W + 7) / 8, (GRID_H + 7) / 8, 1);
        oneshot_end(s);
    }

    // ---- App state --------------------------------------------------------
    OrbitCamera camera;

    // Water sim params
    float ui_gravity    = 9.81f;
    float ui_friction   = 0.05f;
    float ui_damping    = 0.0f;
    float ui_time_scale = 1.0f;
    int   ui_swe_substeps = 4;
    float ui_swe_dt     = 0.05f;
    float ui_k_rain     = 0.5f;

    // Brush params
    int   ui_brush_radius_cells  = 12;
    float ui_water_strength      = 0.5f;
    float ui_terrain_strength    = 1.5f;

    // Atmosphere params
    bool  ui_atmo_enabled        = true;
    float ui_cloud_opacity       = 1.0f;
    float ui_orographic_lift     = 0.15f;
    float ui_adiabatic_cooling   = 0.0065f;
    float ui_rain_shadow         = 0.3f;
    float ui_k_pressure          = 1.0f;
    float ui_wind_strength       = 1.0f;
    float ui_k_evaporation       = 0.3f;
    float ui_cloud_base          = -10.0f;
    float ui_layer_height        = 10.0f;
    bool  ui_atmo_reset          = false;
    bool  ui_terrain_reset       = false;
    bool  ui_water_reset         = false;

    // Erosion params
    bool  ui_erosion_enabled     = true;
    float ui_k_erosion           = 0.01f;
    float ui_k_deposit           = 0.02f;
    float ui_k_capacity          = 0.5f;
    float ui_min_slope           = 0.001f;
    // Spring (persistent water source)
    bool  spring_active          = false;
    float spring_x               = 0.0f;
    float spring_y               = 0.0f;
    float ui_spring_rate         = 0.3f;
    float ui_spring_radius       = 5.0f;

    float ui_k_wind              = 0.05f;
    float ui_k_thermal           = 0.02f;
    float ui_wind_threshold      = 1.0f;

    // Debug readback state
    float debug_pressure_min  = 0.0f;
    float debug_pressure_mean = 0.0f;
    float debug_pressure_max  = 0.0f;
    float debug_wind_speed_max = 0.0f;
    float debug_precip_max    = 0.0f;
    float debug_precip_mean   = 0.0f;
    float debug_temp_mean     = 0.0f;
    float debug_humidity_mean = 0.0f;
    float debug_gc_wind_mean  = 0.0f;
    float debug_gc_wind_max   = 0.0f;

    // Dispatch tracking
    bool ran_brush   = false;
    bool ran_swe     = false;
    bool ran_atmo    = false;
    bool ran_erosion = false;

    int  swe_ping_pong  = 0;
    int  atmo_ping_pong = 0;
    int  ero_ping_pong  = 0;
    bool atmo_force_init = true; // first frame forces init

    double last_time = glfwGetTime();
    float  accumulated_time = 0.0f;

    // ---- Frame loop -------------------------------------------------------
    while (!glfwWindowShouldClose(renderer.window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;
        accumulated_time += dt;

        glfwPollEvents();
        update_orbit(camera, renderer.window);

        // --- Debug readback: process PREVIOUS frame's readback buffer ---
        // Use the OTHER frame's buffer (the one that finished rendering)
        {
            uint32_t prev_frame = (renderer.current_frame + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT;
            VmaAllocationInfo rb_info{};
            vmaGetAllocationInfo(alloc, wind_readback[prev_frame].allocation, &rb_info);
            if (rb_info.pMappedData) {
                const uint16_t* hp = static_cast<const uint16_t*>(rb_info.pMappedData);
                float p_min = 1e30f, p_max = -1e30f, p_sum = 0.0f;
                float ws_max = 0.0f;
                uint32_t count = ATMO_W * ATMO_H;
                for (uint32_t i = 0; i < count; ++i) {
                    float vx = half_to_float(hp[i * 4 + 0]);
                    float vy = half_to_float(hp[i * 4 + 1]);
                    float pressure = half_to_float(hp[i * 4 + 3]); // .a channel
                    float speed = std::sqrt(vx * vx + vy * vy);
                    if (pressure < p_min) p_min = pressure;
                    if (pressure > p_max) p_max = pressure;
                    p_sum += pressure;
                    if (speed > ws_max) ws_max = speed;
                }
                debug_pressure_min  = p_min;
                debug_pressure_max  = p_max;
                debug_pressure_mean = p_sum / static_cast<float>(count);
                debug_wind_speed_max = ws_max;
            }

            // Ground conditions readback
            VmaAllocationInfo gc_info{};
            vmaGetAllocationInfo(alloc, gcond_readback[prev_frame].allocation, &gc_info);
            if (gc_info.pMappedData) {
                const uint16_t* gp = static_cast<const uint16_t*>(gc_info.pMappedData);
                float pr_max = 0.0f, pr_sum = 0.0f, t_sum = 0.0f, h_sum = 0.0f;
                float gw_max = 0.0f, gw_sum = 0.0f;
                uint32_t count = ATMO_W * ATMO_H;
                for (uint32_t i = 0; i < count; ++i) {
                    float precip   = half_to_float(gp[i * 4 + 0]);
                    float temp_k   = half_to_float(gp[i * 4 + 1]);
                    float humidity = half_to_float(gp[i * 4 + 2]);
                    float gc_wind  = half_to_float(gp[i * 4 + 3]);
                    pr_sum += precip;
                    if (precip > pr_max) pr_max = precip;
                    t_sum += temp_k;
                    h_sum += humidity;
                    gw_sum += gc_wind;
                    if (gc_wind > gw_max) gw_max = gc_wind;
                }
                debug_precip_max    = pr_max;
                debug_precip_mean   = pr_sum / static_cast<float>(count);
                debug_temp_mean     = t_sum / static_cast<float>(count);
                debug_humidity_mean = h_sum / static_cast<float>(count);
                debug_gc_wind_mean  = gw_sum / static_cast<float>(count);
                debug_gc_wind_max   = gw_max;
            }
        }

        // Cursor pick
        int win_w, win_h;
        glfwGetWindowSize(renderer.window, &win_w, &win_h);
        double mx, my;
        glfwGetCursorPos(renderer.window, &mx, &my);
        float aspect = float(win_w) / float(win_h);
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.5f, 1500.0f);
        proj[1][1] *= -1.0f;
        glm::mat4 view = orbit_view(camera);

        glm::vec2 ndc = glm::vec2(
            float(mx) / win_w * 2.0f - 1.0f,
            float(my) / win_h * 2.0f - 1.0f
        );

        float pick_gx = -1.0f, pick_gy = -1.0f;
        bool have_pick = pick_grid_cell(view, proj, ndc, pick_gx, pick_gy);

        bool capture_mouse = ImGui::GetIO().WantCaptureMouse;
        bool lmb = !capture_mouse && glfwGetMouseButton(renderer.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool shift = glfwGetKey(renderer.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS
                  || glfwGetKey(renderer.window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        bool rain_key = glfwGetKey(renderer.window, GLFW_KEY_R) == GLFW_PRESS;
        bool spring_key = glfwGetKey(renderer.window, GLFW_KEY_S) == GLFW_PRESS;
        if (spring_key && have_pick && !capture_mouse) {
            spring_active = true;
            spring_x = pick_gx;
            spring_y = pick_gy;
        }

        // ---- ImGui --------------------------------------------------------
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(380, 720), ImGuiCond_FirstUseEver);
        ImGui::Begin("Sim Couple Testing");
        ImGui::Text("LMB: raise terrain   Shift+LMB: lower");
        ImGui::Text("R: rain pulse  S: place spring  RMB: orbit  Scroll: zoom");

        // -- Water Physics --
        ImGui::Separator();
        ImGui::TextUnformatted("Water Physics");
        ImGui::SliderFloat("gravity",     &ui_gravity,    1.0f, 20.0f, "%.2f m/s2");
        ImGui::SliderFloat("friction",    &ui_friction,   0.0f, 0.5f,  "%.3f");
        ImGui::SliderFloat("damping",     &ui_damping,    0.0f, 0.1f,  "%.4f");
        ImGui::SliderFloat("time scale",  &ui_time_scale, 0.0f, 4.0f,  "%.2f");
        ImGui::SliderInt  ("substeps",    &ui_swe_substeps, 1, 10);
        ImGui::SliderFloat("dt (s)",      &ui_swe_dt,     0.005f, 0.1f, "%.3f");
        ImGui::SliderFloat("rain coeff",  &ui_k_rain,     0.0f, 2.0f, "%.3f");
        if (ImGui::Button("Reset water")) ui_water_reset = true;

        // -- Brushes --
        ImGui::Separator();
        ImGui::TextUnformatted("Brushes");
        ImGui::SliderInt  ("radius (cells)",    &ui_brush_radius_cells, 2, 60);
        ImGui::SliderFloat("water strength",    &ui_water_strength,     0.05f, 5.0f, "%.2f");
        ImGui::SliderFloat("terrain strength",  &ui_terrain_strength,   0.05f, 5.0f, "%.2f m");
        ImGui::Separator();
        ImGui::TextUnformatted("Spring (S key to place)");
        if (spring_active) {
            ImGui::Text("  at (%.0f, %.0f)", spring_x, spring_y);
            ImGui::SliderFloat("flow rate", &ui_spring_rate, 0.01f, 2.0f, "%.2f");
            ImGui::SliderFloat("radius",    &ui_spring_radius, 1.0f, 20.0f, "%.0f");
            if (ImGui::Button("Remove spring")) spring_active = false;
        } else {
            ImGui::TextDisabled("  none placed");
        }

        // -- Atmosphere --
        ImGui::Separator();
        ImGui::TextUnformatted("Atmosphere");
        ImGui::Checkbox  ("enable atmosphere", &ui_atmo_enabled);
        ImGui::SliderFloat("cloud opacity",    &ui_cloud_opacity,     0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("orographic lift",  &ui_orographic_lift,   0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("adiabatic cooling", &ui_adiabatic_cooling, 0.0f, 0.02f, "%.4f");
        ImGui::SliderFloat("rain shadow",      &ui_rain_shadow,       0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("k_pressure",       &ui_k_pressure,        0.0f, 5.0f, "%.2f");
        ImGui::SliderFloat("wind strength",    &ui_wind_strength,     0.0f, 10.0f, "%.1f");
        ImGui::SliderFloat("evaporation",     &ui_k_evaporation,    0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("cloud base (m)",   &ui_cloud_base,        -20.0f, 100.0f, "%.0f");
        ImGui::SliderFloat("layer height (m)", &ui_layer_height,      1.0f, 50.0f, "%.1f");
        if (ImGui::Button("Reset atmosphere")) {
            ui_atmo_reset = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset terrain")) {
            ui_terrain_reset = true;
        }

        // -- Erosion --
        ImGui::Separator();
        ImGui::TextUnformatted("Erosion");
        ImGui::Checkbox  ("enable erosion", &ui_erosion_enabled);
        ImGui::SliderFloat("k_erosion",  &ui_k_erosion,  0.0f, 0.1f,  "%.4f");
        ImGui::SliderFloat("k_deposit",  &ui_k_deposit,  0.0f, 0.1f,  "%.4f");
        ImGui::SliderFloat("k_capacity", &ui_k_capacity, 0.0f, 2.0f,  "%.3f");
        ImGui::SliderFloat("min slope",  &ui_min_slope,  0.0f, 0.01f, "%.4f");
        ImGui::SliderFloat("k_wind",     &ui_k_wind,     0.0f, 0.5f,  "%.3f");
        ImGui::SliderFloat("wind thresh", &ui_wind_threshold, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("k_thermal",  &ui_k_thermal,  0.0f, 0.2f,  "%.3f");

        // -- Debug Readback --
        ImGui::Separator();
        ImGui::TextUnformatted("Debug Readback (wind bottom layer)");
        ImGui::Text("pressure  min=%.3f  mean=%.3f  max=%.3f",
                    static_cast<double>(debug_pressure_min),
                    static_cast<double>(debug_pressure_mean),
                    static_cast<double>(debug_pressure_max));
        ImGui::Text("wind speed max=%.3f", static_cast<double>(debug_wind_speed_max));
        ImGui::Separator();
        ImGui::TextUnformatted("Ground Conditions");
        ImGui::Text("precip  mean=%.4f  max=%.4f",
                    static_cast<double>(debug_precip_mean), static_cast<double>(debug_precip_max));
        ImGui::Text("temp    mean=%.1f K (%.1f C)",
                    static_cast<double>(debug_temp_mean), static_cast<double>(debug_temp_mean) - 273.15);
        ImGui::Text("humidity mean=%.3f", static_cast<double>(debug_humidity_mean));
        ImGui::Text("gc wind  mean=%.3f  max=%.3f",
                    static_cast<double>(debug_gc_wind_mean), static_cast<double>(debug_gc_wind_max));

        // -- Dispatch Order --
        ImGui::Separator();
        ImGui::TextUnformatted("Dispatch Order (this frame)");
        ImGui::Text(" 1. Terrain brush:  %s", ran_brush   ? "RAN" : "skip");
        ImGui::Text(" 2. SWE step:       %s", ran_swe     ? "RAN" : "skip");
        ImGui::Text(" 3. Atmosphere 3D:  %s", ran_atmo    ? "RAN" : "skip");
        ImGui::Text(" 4. Erosion:        %s", ran_erosion ? "RAN" : "skip");

        if (have_pick)
            ImGui::Text("cursor: (%.1f, %.1f) cells", pick_gx, pick_gy);

        ImGui::End();
        ImGui::Render();

        // ---- Frame command recording -------------------------------------
        FrameData* frame = nullptr;
        uint32_t image_index = 0;
        VkExtent2D extent{};
        if (!renderer_begin_frame(renderer, frame, image_index, extent))
            continue;

        VkCommandBuffer cmd = frame->cmd;
        uint32_t cur_frame_idx = renderer.current_frame;

        // Reset dispatch tracking (after ImGui reads, before dispatches set them)
        ran_brush = ran_swe = ran_atmo = ran_erosion = false;

        // ---- Terrain reset (re-upload bowl heightmap) ----
        if (ui_terrain_reset) {
            for (uint32_t y = 0; y < GRID_H; ++y) {
                for (uint32_t x = 0; x < GRID_W; ++x) {
                    float dx_ = static_cast<float>(x) - GRID_W * 0.5f;
                    float dy_ = static_cast<float>(y) - GRID_H * 0.5f;
                    float r = std::sqrt(dx_ * dx_ + dy_ * dy_);
                    hm_cpu[y * GRID_W + x] = -8.0f + 0.0008f * r * r;
                }
            }
            // Re-upload via staging buffer
            VkDeviceSize hm_bytes = VkDeviceSize{GRID_W} * GRID_H * sizeof(float);
            GpuBuffer staging = create_host_buffer(alloc, hm_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            void* mapped = nullptr;
            vmaMapMemory(alloc, staging.allocation, &mapped);
            std::memcpy(mapped, hm_cpu.data(), hm_bytes);
            vmaUnmapMemory(alloc, staging.allocation);
            vmaFlushAllocation(alloc, staging.allocation, 0, hm_bytes);
            {
                OneShot s = oneshot_begin(device, gqueue, gfamily);
                VkBufferImageCopy2 region{};
                region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {GRID_W, GRID_H, 1};
                VkCopyBufferToImageInfo2 ci{};
                ci.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
                ci.srcBuffer = staging.buffer;
                ci.dstImage = hm_gpu.image;
                ci.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
                ci.regionCount = 1;
                ci.pRegions = &region;
                vkCmdCopyBufferToImage2(s.cmd, &ci);
                oneshot_end(s);
            }
            destroy_buffer(alloc, staging);
            ui_terrain_reset = false;
            ui_water_reset = true;
            ui_atmo_reset = true;
        }

        // ---- Water reset (re-run SWE init) ----
        if (ui_water_reset) {
            OneShot s = oneshot_begin(device, gqueue, gfamily);
            vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_swe_init.pipeline);
            vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipe_swe_init.layout, 0, 1, &ds_swe_init, 0, nullptr);
            SweInitPC pc{GRID_W, GRID_H, -10000.0f, 0.0f};
            vkCmdPushConstants(s.cmd, pipe_swe_init.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);
            vkCmdDispatch(s.cmd, (GRID_W + 7) / 8, (GRID_H + 7) / 8, 1);
            oneshot_end(s);
            swe_ping_pong = 0;
            ui_water_reset = false;
        }

        // ================================================================
        // DISPATCH ORDER: brush -> SWE -> atmosphere -> erosion -> render
        // ================================================================

        // 1. Terrain brush dispatch + CPU mirror
        bool brush_apply_now = lmb && have_pick && !rain_key;
        if (brush_apply_now) {
            ran_brush = true;
            float amount = (shift ? -1.0f : 1.0f) * ui_terrain_strength;
            TerrainBrushPC bpc{};
            bpc.brush_x = pick_gx;
            bpc.brush_y = pick_gy;
            bpc.brush_radius = static_cast<float>(ui_brush_radius_cells);
            bpc.brush_amount = amount * dt * 4.0f;
            bpc.grid_w = GRID_W; bpc.grid_h = GRID_H;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_brush.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipe_brush.layout, 0, 1, &ds_brush, 0, nullptr);
            vkCmdPushConstants(cmd, pipe_brush.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(bpc), &bpc);
            vkCmdDispatch(cmd, (GRID_W + 15) / 16, (GRID_H + 15) / 16, 1);
            compute_memory_barrier(cmd);

            // CPU mirror
            cpu_apply_brush(hm_cpu, GRID_W, GRID_H, pick_gx, pick_gy, bpc.brush_radius, bpc.brush_amount);
        }

        // 2. SWE step(s)
        {
            ran_swe = true;
            float scaled_dt = ui_swe_dt * ui_time_scale;
            for (int sub = 0; sub < ui_swe_substeps; ++sub) {
                int cur = swe_ping_pong & 1;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_swe_step.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipe_swe_step.layout, 0, 1, &ds_swe_step[cur], 0, nullptr);
                SweStepPC pc{};
                pc.time     = accumulated_time;
                pc.dt       = scaled_dt;
                pc.gravity  = ui_gravity;
                pc.friction = ui_friction;
                pc.dx       = DX;
                pc.sea_level = -10000.0f;
                pc.damping  = ui_damping;
                pc.k_rain   = ui_k_rain;
                pc.grid_w   = GRID_W;
                pc.grid_h   = GRID_H;
                if (sub == 0) {
                    if (rain_key && have_pick) {
                        pc.pulse_x      = pick_gx;
                        pc.pulse_y      = pick_gy;
                        pc.pulse_radius = static_cast<float>(ui_brush_radius_cells);
                        pc.pulse_amount = ui_water_strength;
                    } else if (spring_active) {
                        pc.pulse_x      = spring_x;
                        pc.pulse_y      = spring_y;
                        pc.pulse_radius = ui_spring_radius;
                        pc.pulse_amount = ui_spring_rate * scaled_dt;
                    }
                }
                vkCmdPushConstants(cmd, pipe_swe_step.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, (GRID_W + 7) / 8, (GRID_H + 7) / 8, 1);
                compute_memory_barrier(cmd);
                ++swe_ping_pong;
            }
        }

        // 3. Atmosphere 3D dispatch
        if (ui_atmo_enabled) {
            ran_atmo = true;
            int acur = atmo_ping_pong & 1;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_atmo.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipe_atmo.layout, 0, 1, &ds_atmo[acur], 0, nullptr);
            Atmo3DPC apc{};
            apc.dt                    = dt * ui_time_scale;
            apc.accumulated_time      = accumulated_time;
            apc.grid_w                = ATMO_W;
            apc.grid_h                = ATMO_H;
            apc.grid_d                = ATMO_D;
            apc.terrain_scale         = static_cast<float>(GRID_W) * DX; // tile width in metres
            apc.layer_height          = ui_layer_height;
            apc.max_elevation         = 50.0f;  // max terrain height
            apc.orographic_lift_coeff = ui_orographic_lift;
            apc.adiabatic_cooling_rate = ui_adiabatic_cooling;
            apc.rain_shadow_intensity = ui_rain_shadow;
            apc.force_init            = (ui_atmo_reset || atmo_force_init) ? 1u : 0u;
            apc.k_pressure            = ui_k_pressure;
            apc.wind_strength         = ui_wind_strength;
            apc.k_evaporation         = ui_k_evaporation;
            vkCmdPushConstants(cmd, pipe_atmo.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(apc), &apc);
            vkCmdDispatch(cmd, (ATMO_W + 3) / 4, (ATMO_H + 3) / 4, (ATMO_D + 3) / 4);
            compute_memory_barrier(cmd);
            ++atmo_ping_pong;
            atmo_force_init = false;
            ui_atmo_reset = false;

            // --- Wind readback: copy bottom layer of the WRITE wind volume ---
            // The write target this frame is wind_field[acur ^ 1] (we just wrote it)
            int wind_write_idx = acur ^ 1;
            VkImage wind_src = wind_field[wind_write_idx].image;

            // Barrier: compute write -> transfer read
            volume_barrier(cmd, wind_src,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT,
                           VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            VkBufferImageCopy copy{};
            copy.bufferOffset = 0;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageOffset = {0, 0, 0};
            copy.imageExtent = {ATMO_W, ATMO_H, 1}; // just the bottom layer
            vkCmdCopyImageToBuffer(cmd, wind_src,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   wind_readback[cur_frame_idx].buffer, 1, &copy);

            // Barrier: transfer read -> compute (back to GENERAL)
            volume_barrier(cmd, wind_src,
                           VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                             | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_IMAGE_LAYOUT_GENERAL);

            // --- Ground conditions readback ---
            image_barrier(cmd, ground_cond.image,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COPY_BIT,
                          VK_ACCESS_2_TRANSFER_READ_BIT,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            VkBufferImageCopy gc_copy{};
            gc_copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            gc_copy.imageExtent = {ATMO_W, ATMO_H, 1};
            vkCmdCopyImageToBuffer(cmd, ground_cond.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   gcond_readback[cur_frame_idx].buffer, 1, &gc_copy);

            image_barrier(cmd, ground_cond.image,
                          VK_PIPELINE_STAGE_2_COPY_BIT,
                          VK_ACCESS_2_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                            | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL);
        }

        // 4. Erosion dispatch
        if (ui_erosion_enabled) {
            ran_erosion = true;
            // Use the correct erosion descriptor set based on which SWE state is current
            // and which sediment buffer is current
            int ero_cur = ero_ping_pong & 1;

            // We need to select erosion set that reads the current SWE state.
            // ds_erosion[0] reads state_a, ds_erosion[1] reads state_b.
            // For sediment ping-pong: ero_cur==0 reads sediment[0] writes sediment[1],
            //                         ero_cur==1 reads sediment[1] writes sediment[0].
            // We'll use the simpler approach: always use ds_erosion[ero_cur] and
            // accept that we might read one-step-stale SWE. This is fine for erosion
            // which operates on longer timescales.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_erosion.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipe_erosion.layout, 0, 1, &ds_erosion[ero_cur], 0, nullptr);
            ErosionPC epc{};
            epc.dt          = dt * ui_time_scale;
            epc.dx          = DX;
            epc.grid_w      = GRID_W;
            epc.grid_h      = GRID_H;
            epc.k_erosion   = ui_k_erosion;
            epc.k_deposit   = ui_k_deposit;
            epc.k_capacity  = ui_k_capacity;
            epc.min_slope   = ui_min_slope;
            epc.min_depth       = 0.001f;
            epc.max_change      = 0.1f;
            epc.max_sediment    = 1.0f;
            epc.k_wind          = ui_k_wind;
            epc.k_thermal       = ui_k_thermal;
            epc.wind_threshold  = ui_wind_threshold;
            vkCmdPushConstants(cmd, pipe_erosion.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(epc), &epc);
            vkCmdDispatch(cmd, (GRID_W + 15) / 16, (GRID_H + 15) / 16, 1);
            ++ero_ping_pong;
        }

        // 5. Compute -> graphics barrier
        compute_to_graphics_barrier(cmd);

        // ---- Color attachment + depth barriers ---------------------------
        image_barrier(cmd, renderer.swapchain_images[image_index],
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        {
            VkImageMemoryBarrier2 db{};
            db.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            db.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            db.srcAccessMask = VK_ACCESS_2_NONE;
            db.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
            db.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            db.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            db.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            db.image         = renderer.depth_buffer.image;
            db.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            VkDependencyInfo di{};
            di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.imageMemoryBarrierCount = 1;
            di.pImageMemoryBarriers = &db;
            vkCmdPipelineBarrier2(cmd, &di);
        }

        // ---- 6. Render terrain + water -----------------------------------
        glm::mat4 mvp = proj * view;

        VkRenderingAttachmentInfo color{};
        color.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView   = renderer.swapchain_views[image_index];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{0.45f, 0.55f, 0.70f, 1.0f}};

        VkRenderingAttachmentInfo depth{};
        depth.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView   = renderer.depth_buffer.view;
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
        VkViewport vp{0, 0, float(extent.width), float(extent.height), 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0, 0}, extent};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // Update camera UBO for cloud raymarch
        {
            CameraUBO cam{};
            cam.view = view;
            cam.proj = proj;
            cam.sun_dir = glm::normalize(glm::vec3(0.4f, 0.7f, -0.3f));
            cam.sun_color = glm::vec3(1.0f, 0.95f, 0.85f);
            cam.cam_pos = glm::vec3(0.0f);
            cam.inv_view_proj = glm::inverse(proj * view);
            std::memcpy(camera_ubo_info.pMappedData, &cam, sizeof(cam));
            vmaFlushAllocation(alloc, camera_ubo_alloc, 0, VK_WHOLE_SIZE);
        }

        // Terrain
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_terrain.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipe_terrain.layout, 0, 1, &ds_terrain, 0, nullptr);
        WorldTerrainPC tpc{};
        tpc.mvp        = mvp;
        tpc.grid_w_f   = float(GRID_W);
        tpc.grid_h_f   = float(GRID_H);
        tpc.cell_size  = DX;
        tpc.sea_level  = -10000.0f;
        tpc.brush_x    = pick_gx;
        tpc.brush_y    = pick_gy;
        tpc.brush_radius = float(ui_brush_radius_cells);
        tpc.brush_active = (have_pick && !capture_mouse) ? 1.0f : 0.0f;
        tpc.moisture_overlay = 0.0f;
        vkCmdPushConstants(cmd, pipe_terrain.layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(tpc), &tpc);
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &terrain_mesh.vbo.buffer, &zero);
        vkCmdBindIndexBuffer(cmd, terrain_mesh.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, terrain_mesh.index_count, 1, 0, 0, 0);

        vkCmdEndRendering(cmd);

        // ---- Cloud raymarch pass (alpha-blended over terrain) -------------
        if (ui_atmo_enabled) {
            VkRenderingAttachmentInfo cloud_color = color;
            cloud_color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            VkRenderingInfo cloud_ri{};
            cloud_ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
            cloud_ri.renderArea           = {{0, 0}, extent};
            cloud_ri.layerCount           = 1;
            cloud_ri.colorAttachmentCount = 1;
            cloud_ri.pColorAttachments    = &cloud_color;

            vkCmdBeginRendering(cmd, &cloud_ri);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);

            int cloud_set_idx = atmo_ping_pong & 1;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_cloud.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipe_cloud.layout, 0, 1, &ds_cloud[cloud_set_idx], 0, nullptr);

            float terrain_size = float(GRID_W) * DX;
            RaymarchPC rpc{};
            rpc.terrain_size  = terrain_size;
            rpc.max_elevation = 2000.0f;
            rpc.cloud_opacity = ui_cloud_opacity;
            rpc.cloud_base    = ui_cloud_base;
            rpc.vol_w         = ATMO_W;
            rpc.vol_h         = ATMO_H;
            rpc.vol_d         = ATMO_D;
            rpc.layer_height  = ui_layer_height;
            vkCmdPushConstants(cmd, pipe_cloud.layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(rpc), &rpc);
            vkCmdDraw(cmd, 3, 1, 0, 0);

            vkCmdEndRendering(cmd);
        }

        // ---- ImGui pass --------------------------------------------------
        VkRenderingAttachmentInfo ig_color = color;
        ig_color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
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
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        renderer_end_frame(renderer, *frame, image_index);
    }

    // ---- Cleanup ----------------------------------------------------------
    vkDeviceWaitIdle(device);

    destroy_buffer(alloc, terrain_mesh.vbo);
    destroy_buffer(alloc, terrain_mesh.ibo);

    destroy_cloud_pipeline(device, pipe_cloud);
    vmaDestroyBuffer(alloc, camera_ubo, camera_ubo_alloc);
    destroy_terrain_pipeline(device, pipe_terrain);
    destroy_compute_pipeline(device, pipe_erosion);
    destroy_compute_pipeline(device, pipe_atmo);
    destroy_compute_pipeline(device, pipe_brush);
    destroy_compute_pipeline(device, pipe_swe_step);
    destroy_compute_pipeline(device, pipe_swe_init);

    vkDestroyDescriptorPool(device, desc_pool, nullptr);
    vkDestroySampler(device, sampler, nullptr);

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        destroy_buffer(alloc, wind_readback[i]);
        destroy_buffer(alloc, gcond_readback[i]);
    }

    for (int i = 0; i < 2; ++i) {
        destroy_swe_image(device, alloc, sediment[i]);
        destroy_swe_image(device, alloc, wind_field[i]);
        destroy_swe_image(device, alloc, atmo_state[i]);
    }
    destroy_swe_image(device, alloc, atmo_shadow);
    destroy_swe_image(device, alloc, ground_cond);
    destroy_swe_image(device, alloc, ground_wind);

    destroy_swe_image(device, alloc, water_out);
    destroy_swe_image(device, alloc, state_b);
    destroy_swe_image(device, alloc, state_a);

    vkDestroyImageView(device, moist_gpu.view, nullptr);
    vmaDestroyImage(alloc, moist_gpu.image, moist_gpu.allocation);
    vkDestroyImageView(device, hm_gpu.view, nullptr);
    vmaDestroyImage(alloc, hm_gpu.image, hm_gpu.allocation);

    renderer_shutdown(renderer);
    return 0;
}

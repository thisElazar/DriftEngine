// world_lab.cpp — World Lab module implementation.

#include "world_lab.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "species_file.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers (file-local)
// ---------------------------------------------------------------------------
static float wl_world_to_gx(float wx) { return (wx + WL_TILE_HALF_X) / WL_DX - 0.5f; }
static float wl_world_to_gy(float wz) { return (wz + WL_TILE_HALF_Z) / WL_DX - 0.5f; }

static void volume_barrier(VkCommandBuffer cmd, VkImage img,
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

static void compute_to_graphics_barrier(VkCommandBuffer cmd)
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

static SweImage create_volume_image_readback(VkDevice device, VmaAllocator allocator,
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

static SweImage create_shadow_image(VkDevice device, VmaAllocator allocator,
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
static WL_TerrainPipeline create_terrain_pipeline(VkDevice device)
{
    WL_TerrainPipeline p{};
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
    gp.pViewportState      = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState   = &ms;
    gp.pDepthStencilState  = &ds;
    gp.pColorBlendState    = &cb;
    gp.pDynamicState       = &dyn;
    gp.layout              = p.layout;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &p.pipeline));
    return p;
}

static void destroy_terrain_pipeline(VkDevice device, WL_TerrainPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.dsl, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
}

// ---------------------------------------------------------------------------
// Graphics pipeline — clump (plants)
// ---------------------------------------------------------------------------
static WL_ClumpPipeline create_wl_clump_pipeline(VkDevice device)
{
    WL_ClumpPipeline p{};
    p.vs = make_shader(device, "shaders/world_clump_vs.spv");
    p.fs = make_shader(device, "shaders/world_clump_fs.spv");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.size       = sizeof(ClumpPC);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
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

    VkVertexInputBindingDescription vbs[2]{};
    vbs[0].binding   = 0;
    vbs[0].stride    = sizeof(bestiary::VegetationVertex);
    vbs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    vbs[1].binding   = 1;
    vbs[1].stride    = sizeof(bestiary::PlantGPUInstance);
    vbs[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription attrs[6]{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex,  position)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex,  normal)};
    attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex,  color)};
    attrs[3] = {3, 0, VK_FORMAT_R32_SFLOAT,        offsetof(bestiary::VegetationVertex,  height_t)};
    attrs[4] = {4, 1, VK_FORMAT_R32G32B32_SFLOAT,  offsetof(bestiary::PlantGPUInstance,  x)};  // x,y,z
    attrs[5] = {5, 1, VK_FORMAT_R32_SFLOAT,         offsetof(bestiary::PlantGPUInstance,  health)};
    // inst_seed omitted — uint vertex attrs need DXC; add back when jitter lands

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 2;
    vi.pVertexBindingDescriptions      = vbs;
    vi.vertexAttributeDescriptionCount = 6;
    vi.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

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
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &ba;

    VkDynamicState dyns[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dyns;

    VkFormat color_fmt = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depth_fmt = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rci{};
    rci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &color_fmt;
    rci.depthAttachmentFormat = depth_fmt;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.pNext               = &rci;
    gp.stageCount          = 2;
    gp.pStages             = stages;
    gp.pVertexInputState   = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState      = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState   = &ms;
    gp.pDepthStencilState  = &ds;
    gp.pColorBlendState    = &cb;
    gp.pDynamicState       = &dyn;
    gp.layout              = p.layout;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &p.pipeline));
    return p;
}

static void destroy_wl_clump_pipeline(VkDevice device, WL_ClumpPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
}

// ---------------------------------------------------------------------------
// Creature pipeline (non-instanced, pre-baked world-space vertices)
// ---------------------------------------------------------------------------
static WL_ClumpPipeline create_wl_creature_pipeline(VkDevice device)
{
    WL_ClumpPipeline p{};
    p.vs = make_shader(device, "shaders/world_creature_vs.spv");
    p.fs = make_shader(device, "shaders/world_clump_fs.spv");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.size       = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
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
    vb.stride    = sizeof(bestiary::VegetationVertex);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[3]{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, position)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, normal)};
    attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, color)};

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vb;
    vi.vertexAttributeDescriptionCount = 3;
    vi.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

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
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &ba;

    VkDynamicState dyns[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dyns;

    VkFormat color_fmt = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depth_fmt = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rci{};
    rci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rci.colorAttachmentCount = 1;
    rci.pColorAttachmentFormats = &color_fmt;
    rci.depthAttachmentFormat = depth_fmt;

    VkGraphicsPipelineCreateInfo gp{};
    gp.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gp.pNext               = &rci;
    gp.stageCount          = 2;
    gp.pStages             = stages;
    gp.pVertexInputState   = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState      = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState   = &ms;
    gp.pDepthStencilState  = &ds;
    gp.pColorBlendState    = &cb;
    gp.pDynamicState       = &dyn;
    gp.layout              = p.layout;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &p.pipeline));
    return p;
}

static void destroy_wl_creature_pipeline(VkDevice device, WL_ClumpPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
}

// ---------------------------------------------------------------------------
// Cloud raymarch pipeline
// ---------------------------------------------------------------------------
static WL_CloudPipeline create_cloud_pipeline(VkDevice device)
{
    WL_CloudPipeline p{};
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

static void destroy_cloud_pipeline(VkDevice device, WL_CloudPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.dsl, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
}

// ---------------------------------------------------------------------------
// Plant mesh helpers
// ---------------------------------------------------------------------------
static void destroy_plant_mesh(VmaAllocator alloc, WL_PlantMesh& m)
{
    destroy_buffer(alloc, m.vbo);
    destroy_buffer(alloc, m.ibo);
    m = {};
}

static void upload_plant_mesh(VmaAllocator alloc, WL_PlantMesh& dst, const bestiary::VegetationMesh& src)
{
    if (src.vertices.empty() || src.indices.empty()) {
        dst = {};
        return;
    }
    dst.vertex_count = static_cast<uint32_t>(src.vertices.size());
    dst.index_count  = static_cast<uint32_t>(src.indices.size());
    VkDeviceSize vbs = src.vertices.size() * sizeof(bestiary::VegetationVertex);
    VkDeviceSize ibs = src.indices.size()  * sizeof(uint32_t);
    dst.vbo = create_host_buffer(alloc, vbs, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    dst.ibo = create_host_buffer(alloc, ibs, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    void* mapped = nullptr;
    VK_CHECK(vmaMapMemory(alloc, dst.vbo.allocation, &mapped));
    std::memcpy(mapped, src.vertices.data(), vbs);
    vmaUnmapMemory(alloc, dst.vbo.allocation);
    VK_CHECK(vmaMapMemory(alloc, dst.ibo.allocation, &mapped));
    std::memcpy(mapped, src.indices.data(), ibs);
    vmaUnmapMemory(alloc, dst.ibo.allocation);
}

// ---------------------------------------------------------------------------
// Ray vs Y=0 plane — returns grid coords in [0..GRID_W).
// ---------------------------------------------------------------------------
static bool pick_grid_cell(const glm::mat4& view, const glm::mat4& proj,
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
    out_gx = (hit.x + WL_TILE_HALF_X) / WL_DX;
    out_gy = (hit.z + WL_TILE_HALF_Z) / WL_DX;
    return out_gx >= 0.0f && out_gx < WL_GRID_W && out_gy >= 0.0f && out_gy < WL_GRID_H;
}

// ---------------------------------------------------------------------------
// Descriptor helpers (file-local)
// ---------------------------------------------------------------------------
static VkDescriptorSet alloc_set(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout dsl)
{
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &dsl;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &ai, &ds));
    return ds;
}

static void write_image(VkDevice device, VkDescriptorSet set, uint32_t binding,
                         VkDescriptorType type, VkImageView view, VkSampler sm)
{
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
}

// ---------------------------------------------------------------------------
// Build one canonical mesh per plant kind (called once at init / on species change)
// ---------------------------------------------------------------------------
static void build_canonical_meshes(WorldLabState& s, VmaAllocator alloc)
{
    using namespace bestiary;
    constexpr float mid = 0.5f;

    auto upload = [&](int kind, const VegetationMesh& m) {
        destroy_plant_mesh(alloc, s.plant_canonical[kind]);
        upload_plant_mesh(alloc, s.plant_canonical[kind], m);
    };

    upload(PLANT_GRASS,
           generate_clump(evaluate_expression(s.clump_params, s.clump_expr, mid),
                          0, false, 0.0f, 0.0f));
    upload(PLANT_BUSH,
           generate_bush(evaluate_bush_expression(s.bush_params, s.bush_expr, mid),
                         0, false, 0.0f, 0.0f));
    upload(PLANT_TREE,
           generate_tree(evaluate_tree_expression(s.tree_params, s.tree_expr, mid),
                         0, false, 0.0f, 0.0f));
    upload(PLANT_REED,
           generate_clump(evaluate_expression(s.reed_params, s.reed_expr, mid),
                          0, false, 0.0f, 0.0f));
    upload(PLANT_WILDFLOWER,
           generate_wildflower(evaluate_wildflower_expression(
                                   s.wildflower_params, s.wildflower_expr, mid),
                               0, false, 0.0f, 0.0f));
}

// ---------------------------------------------------------------------------
// Upload per-kind instance buffers from current plant population
// ---------------------------------------------------------------------------
static void upload_plant_instances(WorldLabState& s, VkDevice device, VmaAllocator alloc)
{
    constexpr float plant_lift = 0.01f;
    std::vector<bestiary::PlantGPUInstance> buf;
    vkDeviceWaitIdle(device);
    for (int k = 0; k < bestiary::PLANT_KIND_COUNT; ++k) {
        buf.clear();
        for (const auto& p : s.plant_population) {
            if (p.kind != k || p.health <= 0.0f) continue;
            float terrain_y = sample_hm_bilinear(s.hm_cpu, WL_GRID_W, WL_GRID_H,
                                                  wl_world_to_gx(p.x), wl_world_to_gy(p.z));
            buf.push_back({p.x, terrain_y + plant_lift, p.z, p.health, p.seed});
        }
        destroy_buffer(alloc, s.plant_inst[k]);
        s.plant_inst_count[k] = 0;
        if (buf.empty()) continue;
        VkDeviceSize sz = buf.size() * sizeof(bestiary::PlantGPUInstance);
        s.plant_inst[k] = create_host_buffer(alloc, sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        void* mapped = nullptr;
        vmaMapMemory(alloc, s.plant_inst[k].allocation, &mapped);
        std::memcpy(mapped, buf.data(), sz);
        vmaUnmapMemory(alloc, s.plant_inst[k].allocation);
        s.plant_inst_count[k] = static_cast<uint32_t>(buf.size());
    }
}

// ---------------------------------------------------------------------------
// Refresh environment from current water state
// ---------------------------------------------------------------------------
static void refresh_env(WorldLabState& s, VkDevice device, VmaAllocator alloc,
                        VkQueue gqueue, uint32_t gfamily)
{
    VkImage fresh = (s.swe_ping_pong & 1) ? s.state_b.image : s.state_a.image;
    s.persistent_water_depth = readback_water_depth(
        device, alloc, gqueue, gfamily, fresh, WL_GRID_W, WL_GRID_H);

    std::vector<float> moisture = build_moisture_grid(
        s.persistent_water_depth, WL_GRID_W, WL_GRID_H,
        s.ui_capillary_depth, s.ui_capillary_blur);

    update_r32_image(device, alloc, gqueue, gfamily,
                     s.moist_gpu.image, moisture, WL_GRID_W, WL_GRID_H);

    bestiary::EnvironmentField env;
    env.sample = [moisture, &s](float x, float z) -> bestiary::EnvironmentSample {
        float gx = (x + WL_TILE_HALF_X) / WL_DX;
        float gy = (z + WL_TILE_HALF_Z) / WL_DX;
        int ix = std::clamp((int)std::floor(gx), 0, (int)WL_GRID_W - 1);
        int iy = std::clamp((int)std::floor(gy), 0, (int)WL_GRID_H - 1);
        float m = moisture[iy * WL_GRID_W + ix];
        float h = s.hm_cpu[iy * WL_GRID_W + ix];
        float t = std::clamp(0.5f + 0.05f * h, 0.0f, 1.0f);
        return {m, t};
    };
    s.persistent_env = env;
}

// ---------------------------------------------------------------------------
// Full replant: re-seed plant population from scratch
// ---------------------------------------------------------------------------
static void run_replant(WorldLabState& s, VkDevice device, VmaAllocator alloc,
                        VkQueue gqueue, uint32_t gfamily)
{
    refresh_env(s, device, alloc, gqueue, gfamily);
    s.plant_population = bestiary::place_ecosystem(s.eco_params, s.persistent_env);
    upload_plant_instances(s, device, alloc);
    s.veg_rebuild_timer = 0.0f;
}

// ---------------------------------------------------------------------------
// Generate default heightmap data
// ---------------------------------------------------------------------------
static void generate_heightmap(std::vector<float>& values)
{
    float cx = WL_GRID_W * 0.5f, cy = WL_GRID_H * 0.5f;
    for (uint32_t y = 0; y < WL_GRID_H; ++y) {
        for (uint32_t x = 0; x < WL_GRID_W; ++x) {
            float fx = static_cast<float>(x);
            float fy = static_cast<float>(y);
            float dx_ = fx - cx;
            float dy_ = fy - cy;
            float r = std::sqrt(dx_ * dx_ + dy_ * dy_);

            // base: rim rises at edges (bowl)
            float h = -4.0f + 0.0005f * r * r;

            // ridge running NW-SE — creates two drainage basins
            float ridge_dist = std::abs((fx - cx) * 0.7f + (fy - cy) * 0.7f);
            h += 3.0f * std::exp(-ridge_dist * ridge_dist / 1800.0f);

            // hillside near spring (raised area NW)
            float sp_dx = fx - WL_SPRING_GX;
            float sp_dy = fy - WL_SPRING_GY;
            float sp_r = std::sqrt(sp_dx * sp_dx + sp_dy * sp_dy);
            h += 6.0f * std::exp(-sp_r * sp_r / 2000.0f);

            // depression at watering hole (SE area)
            float pool_dx = fx - WL_POOL_GX;
            float pool_dy = fy - WL_POOL_GY;
            float pool_r = std::sqrt(pool_dx * pool_dx + pool_dy * pool_dy);
            h -= 3.0f * std::exp(-pool_r * pool_r / 800.0f);

            values[y * WL_GRID_W + x] = h;
        }
    }
}

// ---------------------------------------------------------------------------
// Species file I/O
// ---------------------------------------------------------------------------
static const char* wl_species_dir()
{
    return "species";
}

static const char* wl_profile_names[] = {
    "sprinter", "grazer", "browser", "wolf", "rabbit",
    "songbird", "raptor", "snake"
};

static void save_creature_profiles(const WorldLabState& s)
{
    std::filesystem::create_directories(wl_species_dir());
    for (size_t i = 0; i < s.creature_profiles.size(); ++i) {
        std::filesystem::path p = std::filesystem::path(wl_species_dir())
            / (std::string(wl_profile_names[i]) + ".toml");
        bestiary::save_creature(p, s.creature_profiles[i], wl_profile_names[i]);
    }
    std::fprintf(stderr, "Saved %zu creature profiles to %s/\n",
                 s.creature_profiles.size(), wl_species_dir());
}

static bool load_creature_profiles(WorldLabState& s)
{
    auto loaded = bestiary::load_creature_dir(wl_species_dir());
    if (loaded.empty()) return false;

    s.creature_profiles.clear();

    for (size_t i = 0; i < 8; ++i) {
        const char* want = wl_profile_names[i];
        bool found = false;
        for (auto& ncp : loaded) {
            if (ncp.name == want) {
                s.creature_profiles.push_back(ncp.profile);
                found = true;
                break;
            }
        }
        if (!found) {
            std::fprintf(stderr, "Warning: species '%s' not found on disk, using default\n", want);
            return false;
        }
    }

    std::fprintf(stderr, "Loaded %zu creature profiles from %s/\n",
                 s.creature_profiles.size(), wl_species_dir());
    return true;
}

// ---------------------------------------------------------------------------
// Plant species file I/O
// ---------------------------------------------------------------------------
static void save_plant_profiles(const WorldLabState& s)
{
    std::filesystem::create_directories(wl_species_dir());
    auto dir = std::filesystem::path(wl_species_dir());

    bestiary::save_clump(dir / "world_grass.toml", s.clump_params, "world_grass",
                         &s.clump_expr);
    bestiary::save_clump(dir / "world_reed.toml", s.reed_params, "world_reed",
                         &s.reed_expr, "reed");
    bestiary::save_bush(dir / "world_bush.toml", s.bush_params, "world_bush",
                        &s.bush_expr);
    bestiary::save_tree(dir / "world_tree.toml", s.tree_params, "world_tree",
                        &s.tree_expr);
    bestiary::save_wildflower(dir / "world_wildflower.toml", s.wildflower_params,
                              "world_wildflower", &s.wildflower_expr);

    std::fprintf(stderr, "Saved 5 plant profiles to %s/\n", wl_species_dir());
}

static bool load_plant_profiles(WorldLabState& s)
{
    auto dir = std::filesystem::path(wl_species_dir());
    bool any = false;

    {
        std::string name;
        auto p = dir / "world_grass.toml";
        if (std::filesystem::exists(p) &&
            bestiary::load_clump(p, s.clump_params, name, &s.clump_expr)) {
            any = true;
            std::fprintf(stderr, "Loaded plant: %s\n", name.c_str());
        }
    }
    {
        std::string name;
        auto p = dir / "world_reed.toml";
        if (std::filesystem::exists(p) &&
            bestiary::load_clump(p, s.reed_params, name, &s.reed_expr)) {
            any = true;
            std::fprintf(stderr, "Loaded plant: %s\n", name.c_str());
        }
    }
    {
        std::string name;
        auto p = dir / "world_bush.toml";
        if (std::filesystem::exists(p) &&
            bestiary::load_bush(p, s.bush_params, name, &s.bush_expr)) {
            any = true;
            std::fprintf(stderr, "Loaded plant: %s\n", name.c_str());
        }
    }
    {
        std::string name;
        auto p = dir / "world_tree.toml";
        if (std::filesystem::exists(p) &&
            bestiary::load_tree(p, s.tree_params, name, &s.tree_expr)) {
            any = true;
            std::fprintf(stderr, "Loaded plant: %s\n", name.c_str());
        }
    }
    {
        std::string name;
        auto p = dir / "world_wildflower.toml";
        if (std::filesystem::exists(p) &&
            bestiary::load_wildflower(p, s.wildflower_params, name, &s.wildflower_expr)) {
            any = true;
            std::fprintf(stderr, "Loaded plant: %s\n", name.c_str());
        }
    }

    return any;
}

// ---------------------------------------------------------------------------
// Spawn creatures across all species
// ---------------------------------------------------------------------------
static void spawn_all_creatures(WorldLabState& s)
{
    s.agents.clear();
    int total = s.ui_creature_count;
    int wolf_count   = std::max(2, total / 10);
    int rabbit_count = total / 5;
    int bird_count   = total / 6;
    int raptor_count = std::max(1, total / 20);
    int snake_count  = std::max(1, total / 15);
    int herb_count   = total - wolf_count - rabbit_count - bird_count - raptor_count - snake_count;
    herb_count = std::max(0, herb_count);
    int per_herb = herb_count / 3;
    int herb_extra = herb_count - per_herb * 3;
    bestiary::spawn_creatures(s.agents, 0, per_herb + (herb_extra > 0 ? 1 : 0),
        s.persistent_env, WL_TILE_HALF_X, WL_TILE_HALF_Z, 42u);
    bestiary::spawn_creatures(s.agents, 1, per_herb + (herb_extra > 1 ? 1 : 0),
        s.persistent_env, WL_TILE_HALF_X, WL_TILE_HALF_Z, 137u);
    bestiary::spawn_creatures(s.agents, 2, per_herb,
        s.persistent_env, WL_TILE_HALF_X, WL_TILE_HALF_Z, 271u);
    bestiary::spawn_creatures(s.agents, 3, wolf_count,
        s.persistent_env, WL_TILE_HALF_X, WL_TILE_HALF_Z, 404u);
    bestiary::spawn_creatures(s.agents, 4, rabbit_count,
        s.persistent_env, WL_TILE_HALF_X, WL_TILE_HALF_Z, 555u);
    bestiary::spawn_creatures(s.agents, 5, bird_count,
        s.persistent_env, WL_TILE_HALF_X, WL_TILE_HALF_Z, 666u);
    bestiary::spawn_creatures(s.agents, 6, raptor_count,
        s.persistent_env, WL_TILE_HALF_X, WL_TILE_HALF_Z, 777u);
    bestiary::spawn_creatures(s.agents, 7, snake_count,
        s.persistent_env, WL_TILE_HALF_X, WL_TILE_HALF_Z, 888u);
}

// ===========================================================================
// world_lab_init
// ===========================================================================
void world_lab_init(WorldLabState& s, Renderer& r)
{
    VkDevice device   = r.device;
    VmaAllocator alloc = r.allocator;
    VkQueue gqueue    = r.graphics_queue;
    uint32_t gfamily  = r.gfx_family;

    // ---- Camera defaults for world-lab (different from plant_lab) -----------
    s.camera.yaw      = 0.6f;
    s.camera.pitch    = 0.7f;
    s.camera.distance = 220.0f;
    s.camera.target   = {0.0f, 0.0f, 0.0f};

    // ---- Heightmap (watershed: ridges, valley, watering hole) --------------
    s.hm_data.width = WL_GRID_W;
    s.hm_data.height = WL_GRID_H;
    s.hm_data.values.assign(WL_GRID_W * WL_GRID_H, 0.0f);
    generate_heightmap(s.hm_data.values);
    s.hm_gpu = upload_heightmap(device, alloc, gqueue, gfamily, s.hm_data);
    s.hm_cpu = s.hm_data.values;

    // Moisture debug texture
    s.moist_data.width = WL_GRID_W;
    s.moist_data.height = WL_GRID_H;
    s.moist_data.values.assign(WL_GRID_W * WL_GRID_H, 0.0f);
    s.moist_gpu = upload_heightmap(device, alloc, gqueue, gfamily, s.moist_data);

    // ---- SWE images -------------------------------------------------------
    s.state_a  = create_swe_image(device, alloc, WL_GRID_W, WL_GRID_H);
    s.state_b  = create_swe_image(device, alloc, WL_GRID_W, WL_GRID_H);
    s.water_out = create_swe_image(device, alloc, WL_GRID_W, WL_GRID_H);

    // Transition water images to GENERAL and clear to zero
    {
        OneShot os = oneshot_begin(device, gqueue, gfamily);
        for (VkImage img : {s.state_a.image, s.state_b.image, s.water_out.image}) {
            image_barrier(os.cmd, img,
                          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        VkClearColorValue clr{}; clr.float32[0] = 0.0f;
        VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(os.cmd, s.state_a.image,   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
        vkCmdClearColorImage(os.cmd, s.state_b.image,   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
        vkCmdClearColorImage(os.cmd, s.water_out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
        for (VkImage img : {s.state_a.image, s.state_b.image, s.water_out.image}) {
            image_barrier(os.cmd, img,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                            | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }
        oneshot_end(os);
    }

    // ---- Atmosphere 3D images -----------------------------------------------
    s.atmo_state[0] = create_volume_image(device, alloc, WL_ATMO_W, WL_ATMO_H, WL_ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    s.atmo_state[1] = create_volume_image(device, alloc, WL_ATMO_W, WL_ATMO_H, WL_ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    s.wind_field[0] = create_volume_image_readback(device, alloc, WL_ATMO_W, WL_ATMO_H, WL_ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    s.wind_field[1] = create_volume_image_readback(device, alloc, WL_ATMO_W, WL_ATMO_H, WL_ATMO_D, VK_FORMAT_R16G16B16A16_SFLOAT);
    s.atmo_shadow = create_shadow_image(device, alloc, WL_ATMO_W, WL_ATMO_H);
    s.ground_cond = create_swe_image(device, alloc, WL_ATMO_W, WL_ATMO_H);
    s.ground_wind = create_wind_image(device, alloc, WL_ATMO_W, WL_ATMO_H);

    {
        OneShot os = oneshot_begin(device, gqueue, gfamily);
        VkImage atmo_imgs[] = {
            s.atmo_state[0].image, s.atmo_state[1].image,
            s.wind_field[0].image, s.wind_field[1].image,
            s.atmo_shadow.image,
            s.ground_cond.image, s.ground_wind.image
        };
        for (VkImage img : atmo_imgs) {
            volume_barrier(os.cmd, img,
                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                             | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        }
        oneshot_end(os);
    }

    // ---- Sediment images for erosion ----------------------------------------
    s.sediment[0] = create_sediment_image(device, alloc, WL_GRID_W, WL_GRID_H);
    s.sediment[1] = create_sediment_image(device, alloc, WL_GRID_W, WL_GRID_H);
    {
        OneShot os = oneshot_begin(device, gqueue, gfamily);
        for (int i = 0; i < 2; ++i) {
            image_barrier(os.cmd, s.sediment[i].image,
                          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }
        VkClearColorValue sed_clr{}; sed_clr.float32[0] = 0.0f;
        VkImageSubresourceRange sed_rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (int i = 0; i < 2; ++i) {
            vkCmdClearColorImage(os.cmd, s.sediment[i].image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &sed_clr, 1, &sed_rng);
            image_barrier(os.cmd, s.sediment[i].image,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                            | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }
        oneshot_end(os);
    }

    // ---- Wind/ground condition readback staging buffers ---------------------
    constexpr VkDeviceSize WIND_READBACK_SIZE = VkDeviceSize{WL_ATMO_W} * WL_ATMO_H * 4 * sizeof(uint16_t);
    constexpr VkDeviceSize GCOND_READBACK_SIZE = VkDeviceSize{WL_ATMO_W} * WL_ATMO_H * 4 * sizeof(uint16_t);
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        s.wind_readback[i] = create_readback_buffer(alloc, WIND_READBACK_SIZE);
        s.gcond_readback[i] = create_readback_buffer(alloc, GCOND_READBACK_SIZE);
    }

    // ---- Sampler -----------------------------------------------------------
    {
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device, &sci, nullptr, &s.sampler));
    }

    // ---- Compute pipelines ------------------------------------------------
    s.pipe_swe_init = make_compute_pipeline(device, "shaders/swe_init.spv",
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        sizeof(SweInitPC));

    s.pipe_swe_step = make_compute_pipeline(device, "shaders/swe_step.spv",
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 0: terrain
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 1: state_read
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 2: state_write
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 3: output
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 4: ground_cond
         VK_DESCRIPTOR_TYPE_SAMPLER},       // 5: ground_cond_sampler
        sizeof(SweStepPC));

    s.pipe_brush = make_compute_pipeline(device, "shaders/terrain_brush.spv",
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        sizeof(TerrainBrushPC));

    s.pipe_atmo = make_compute_pipeline(device, "shaders/atmosphere3d_cs.spv",
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 0: terrain heightmap
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 1: state_read
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 2: state_write
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 3: wind_read
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 4: wind_write
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 5: shadow_out
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 6: ground_cond_out
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 7: ground_wind_out
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 8: swe_output
         VK_DESCRIPTOR_TYPE_SAMPLER,        // 9: terrain_sampler
         VK_DESCRIPTOR_TYPE_SAMPLER},       // 10: swe_sampler
        sizeof(Atmo3DPC));

    s.pipe_erosion = make_compute_pipeline(device, "shaders/erosion.spv",
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 0: terrain (heightmap RW)
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 1: swe_state
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 2: sediment_in
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 3: sediment_out
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 4: ground_cond
         VK_DESCRIPTOR_TYPE_SAMPLER,        // 5: ground_cond_sampler
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  // 6: ground_wind
         VK_DESCRIPTOR_TYPE_SAMPLER},       // 7: ground_wind_sampler
        sizeof(ErosionPC));

    // ---- Descriptor pool --------------------------------------------------
    {
        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,         48},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         32},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          4},
            {VK_DESCRIPTOR_TYPE_SAMPLER,                24},
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = 32;
        pci.poolSizeCount = 5;
        pci.pPoolSizes    = pool_sizes;
        VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &s.desc_pool));
    }

    s.ds_swe_init = alloc_set(device, s.desc_pool, s.pipe_swe_init.dsl);
    s.ds_swe_step[0] = alloc_set(device, s.desc_pool, s.pipe_swe_step.dsl);
    s.ds_swe_step[1] = alloc_set(device, s.desc_pool, s.pipe_swe_step.dsl);
    s.ds_brush = alloc_set(device, s.desc_pool, s.pipe_brush.dsl);

    write_image(device, s.ds_swe_init, 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.hm_gpu.view,    VK_NULL_HANDLE);
    write_image(device, s.ds_swe_init, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.state_a.view,   VK_NULL_HANDLE);

    // step set 0: read A -> write B
    write_image(device, s.ds_swe_step[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.hm_gpu.view,  VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[0], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.state_a.view, VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.state_b.view, VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.water_out.view, VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[0], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.ground_cond.view, VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[0], 5, VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, s.sampler);

    // step set 1: read B -> write A
    write_image(device, s.ds_swe_step[1], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.hm_gpu.view,  VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.state_b.view, VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.state_a.view, VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.water_out.view, VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[1], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.ground_cond.view, VK_NULL_HANDLE);
    write_image(device, s.ds_swe_step[1], 5, VK_DESCRIPTOR_TYPE_SAMPLER, VK_NULL_HANDLE, s.sampler);

    write_image(device, s.ds_brush, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.hm_gpu.view, VK_NULL_HANDLE);

    // ---- Atmosphere descriptor sets (ping-ponged) -----------------------
    s.ds_atmo[0] = alloc_set(device, s.desc_pool, s.pipe_atmo.dsl);
    s.ds_atmo[1] = alloc_set(device, s.desc_pool, s.pipe_atmo.dsl);
    write_image(device, s.ds_atmo[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.hm_gpu.view,         VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.atmo_state[0].view,   VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.atmo_state[1].view,   VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.wind_field[0].view,   VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.wind_field[1].view,   VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.atmo_shadow.view,     VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.ground_cond.view,     VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.ground_wind.view,     VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.water_out.view,       VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[0], 9, VK_DESCRIPTOR_TYPE_SAMPLER,       VK_NULL_HANDLE,         s.sampler);
    write_image(device, s.ds_atmo[0],10, VK_DESCRIPTOR_TYPE_SAMPLER,       VK_NULL_HANDLE,         s.sampler);

    write_image(device, s.ds_atmo[1], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.hm_gpu.view,         VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.atmo_state[1].view,   VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.atmo_state[0].view,   VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.wind_field[1].view,   VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.wind_field[0].view,   VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.atmo_shadow.view,     VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.ground_cond.view,     VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.ground_wind.view,     VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.water_out.view,       VK_NULL_HANDLE);
    write_image(device, s.ds_atmo[1], 9, VK_DESCRIPTOR_TYPE_SAMPLER,       VK_NULL_HANDLE,         s.sampler);
    write_image(device, s.ds_atmo[1],10, VK_DESCRIPTOR_TYPE_SAMPLER,       VK_NULL_HANDLE,         s.sampler);

    // ---- Erosion descriptor sets (ping-ponged on sediment) --------------
    s.ds_erosion[0] = alloc_set(device, s.desc_pool, s.pipe_erosion.dsl);
    s.ds_erosion[1] = alloc_set(device, s.desc_pool, s.pipe_erosion.dsl);
    write_image(device, s.ds_erosion[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  s.hm_gpu.view,       VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[0], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.state_a.view,      VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[0], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.sediment[0].view,  VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  s.sediment[1].view,  VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[0], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.ground_cond.view,  VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[0], 5, VK_DESCRIPTOR_TYPE_SAMPLER,        VK_NULL_HANDLE,      s.sampler);
    write_image(device, s.ds_erosion[0], 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.ground_wind.view,  VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[0], 7, VK_DESCRIPTOR_TYPE_SAMPLER,        VK_NULL_HANDLE,      s.sampler);

    write_image(device, s.ds_erosion[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  s.hm_gpu.view,       VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.state_b.view,      VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[1], 2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.sediment[1].view,  VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  s.sediment[0].view,  VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[1], 4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.ground_cond.view,  VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[1], 5, VK_DESCRIPTOR_TYPE_SAMPLER,        VK_NULL_HANDLE,      s.sampler);
    write_image(device, s.ds_erosion[1], 6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,  s.ground_wind.view,  VK_NULL_HANDLE);
    write_image(device, s.ds_erosion[1], 7, VK_DESCRIPTOR_TYPE_SAMPLER,        VK_NULL_HANDLE,      s.sampler);

    // ---- Graphics pipelines ---------------------------------------------
    s.pipe_terrain = create_terrain_pipeline(device);
    s.pipe_clump    = create_wl_clump_pipeline(device);
    s.pipe_creature = create_wl_creature_pipeline(device);

    s.ds_terrain = alloc_set(device, s.desc_pool, s.pipe_terrain.dsl);
    write_image(device, s.ds_terrain, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s.hm_gpu.view,    s.sampler);
    write_image(device, s.ds_terrain, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s.water_out.view, s.sampler);
    write_image(device, s.ds_terrain, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s.moist_gpu.view, s.sampler);

    // ---- Cloud pipeline + camera UBO ----------------------------------------
    s.pipe_cloud = create_cloud_pipeline(device);

    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = sizeof(WL_CameraUBO);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VK_CHECK(vmaCreateBuffer(alloc, &bci, &ai,
            &s.camera_ubo, &s.camera_ubo_alloc, &s.camera_ubo_info));
    }

    s.ds_cloud[0] = alloc_set(device, s.desc_pool, s.pipe_cloud.dsl);
    s.ds_cloud[1] = alloc_set(device, s.desc_pool, s.pipe_cloud.dsl);
    for (int cs = 0; cs < 2; ++cs) {
        VkDescriptorBufferInfo buf_info{};
        buf_info.buffer = s.camera_ubo;
        buf_info.offset = 0;
        buf_info.range  = sizeof(WL_CameraUBO);
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = s.ds_cloud[cs];
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo     = &buf_info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);

        write_image(device, s.ds_cloud[cs], 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    s.hm_gpu.view, s.sampler);
        write_image(device, s.ds_cloud[cs], 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    s.atmo_state[cs].view, s.sampler);
    }

    s.terrain_mesh = build_grid_mesh(alloc, WL_GRID_W, WL_GRID_H);

    // ---- Run swe_init once -----------------------------------------------
    {
        OneShot os = oneshot_begin(device, gqueue, gfamily);
        vkCmdBindPipeline(os.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe_swe_init.pipeline);
        vkCmdBindDescriptorSets(os.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s.pipe_swe_init.layout, 0, 1, &s.ds_swe_init, 0, nullptr);
        SweInitPC pc{WL_GRID_W, WL_GRID_H, -10000.0f, 0.0f};
        vkCmdPushConstants(os.cmd, s.pipe_swe_init.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(os.cmd, (WL_GRID_W + 7) / 8, (WL_GRID_H + 7) / 8, 1);
        oneshot_end(os);
    }

    // ---- Plant + ecosystem default params ---------------------------------
    load_plant_profiles(s);
    build_canonical_meshes(s, alloc);

    s.eco_params.region_size   = static_cast<float>(WL_GRID_W) * WL_DX;
    s.eco_params.density_scale = 1.0f;
    s.eco_params.r_min         = 1.5f;
    s.eco_params.r_max         = 6.0f;
    s.eco_params.grass_suit = {0.0f, 0.05f, 0.95f, 1.0f,
                               0.0f, 0.0f, 1.0f, 1.0f, 4.0f};

    s.clump_expr.vary_color = true;
    s.clump_expr.dry_color[0] = 0.62f; s.clump_expr.dry_color[1] = 0.55f; s.clump_expr.dry_color[2] = 0.25f;
    s.clump_expr.wet_color[0] = 0.20f; s.clump_expr.wet_color[1] = 0.55f; s.clump_expr.wet_color[2] = 0.18f;
    s.clump_expr.blade_height.enabled = true;
    s.clump_expr.blade_height.low  = 0.20f;
    s.clump_expr.blade_height.high = 0.80f;
    s.clump_expr.blade_count.enabled = true;
    s.clump_expr.blade_count.low  = 4.0f;
    s.clump_expr.blade_count.high = 30.0f;

    // Reed params
    s.reed_params.blade_count  = 8;
    s.reed_params.blade_height = 0.80f;
    s.reed_params.blade_width  = 0.008f;
    s.reed_params.splay_angle  = 8.0f;
    s.reed_params.clump_radius = 0.04f;
    s.reed_params.base_color[0] = 0.40f;
    s.reed_params.base_color[1] = 0.48f;
    s.reed_params.base_color[2] = 0.22f;

    s.reed_expr.vary_color = true;
    s.reed_expr.dry_color[0] = 0.50f; s.reed_expr.dry_color[1] = 0.45f; s.reed_expr.dry_color[2] = 0.25f;
    s.reed_expr.wet_color[0] = 0.30f; s.reed_expr.wet_color[1] = 0.50f; s.reed_expr.wet_color[2] = 0.20f;
    s.reed_expr.blade_height.enabled = true;
    s.reed_expr.blade_height.low  = 0.50f;
    s.reed_expr.blade_height.high = 1.20f;
    s.reed_expr.blade_count.enabled = true;
    s.reed_expr.blade_count.low  = 4.0f;
    s.reed_expr.blade_count.high = 14.0f;

    // Wildflower params — scaled up for world-lab visibility (256m tile viewed from ~220m)
    s.wildflower_params.stem_height  = 0.50f;
    s.wildflower_params.stem_width   = 0.012f;
    s.wildflower_params.petal_radius = 0.08f;
    s.wildflower_params.flower_count = 6;
    s.wildflower_params.clump_radius = 0.15f;

    s.wildflower_expr.vary_color = true;
    s.wildflower_expr.dry_color[0] = 0.90f; s.wildflower_expr.dry_color[1] = 0.80f; s.wildflower_expr.dry_color[2] = 0.20f;
    s.wildflower_expr.wet_color[0] = 0.65f; s.wildflower_expr.wet_color[1] = 0.30f; s.wildflower_expr.wet_color[2] = 0.75f;
    s.wildflower_expr.stem_height.enabled = true;
    s.wildflower_expr.stem_height.low  = 0.35f;
    s.wildflower_expr.stem_height.high = 0.70f;
    s.wildflower_expr.petal_radius.enabled = true;
    s.wildflower_expr.petal_radius.low  = 0.06f;
    s.wildflower_expr.petal_radius.high = 0.12f;
    s.wildflower_expr.flower_count.enabled = true;
    s.wildflower_expr.flower_count.low  = 4.0f;
    s.wildflower_expr.flower_count.high = 10.0f;

    // ---- Creature profiles --------------------------------------------------
    if (!load_creature_profiles(s))
    {
        // Species 0: Sprinter
        bestiary::CreatureProfile sprinter{};
        sprinter.archetype = bestiary::Archetype::Herbivore;
        sprinter.body_length = 0.6f;
        sprinter.body_height = 0.45f;
        sprinter.body_color[0] = 0.72f; sprinter.body_color[1] = 0.58f; sprinter.body_color[2] = 0.38f;
        sprinter.move_speed = 2.8f;
        sprinter.run_speed  = 8.0f;
        sprinter.turn_rate  = 4.5f;
        sprinter.body_mass = 12.0f;
        sprinter.basal_rate = 0.0018f;
        sprinter.locomotion_cost = 0.0005f;
        sprinter.hunger_threshold = 0.4f;
        sprinter.graze_consume = 0.20f;
        sprinter.graze_duration = 3.0f;
        sprinter.graze_radius = 3.0f;
        sprinter.trophic_efficiency = 0.35f;
        sprinter.grass_caloric_value      = 0.8f;
        sprinter.bush_caloric_value       = 0.5f;
        sprinter.tree_caloric_value       = 0.1f;
        sprinter.reed_caloric_value       = 0.6f;
        sprinter.wildflower_caloric_value = 0.9f;
        sprinter.herd_weight = 0.5f;
        sprinter.separation_weight = 0.6f;
        sprinter.separation_radius = 1.5f;
        sprinter.reproduce_threshold = 0.65f;
        sprinter.reproduce_cost = 0.20f;
        sprinter.reproduce_cooldown = 20.0f;
        sprinter.offspring_energy = 0.35f;
        sprinter.water_avoidance = 10.0f;
        sprinter.max_slope = 0.9f;
        sprinter.slope_cost_factor = 1.5f;
        sprinter.max_age = 600.0f;
        s.creature_profiles.push_back(sprinter);

        // Species 1: Grazer
        bestiary::CreatureProfile grazer{};
        grazer.archetype = bestiary::Archetype::Herbivore;
        grazer.body_length = 1.0f;
        grazer.body_height = 0.7f;
        grazer.body_color[0] = 0.55f; grazer.body_color[1] = 0.40f; grazer.body_color[2] = 0.25f;
        grazer.move_speed = 1.8f;
        grazer.run_speed  = 6.0f;
        grazer.body_mass = 30.0f;
        grazer.basal_rate = 0.0012f;
        grazer.locomotion_cost = 0.0007f;
        grazer.hunger_threshold = 0.4f;
        grazer.graze_consume = 0.25f;
        grazer.graze_duration = 5.0f;
        grazer.graze_radius = 3.5f;
        grazer.trophic_efficiency = 0.30f;
        grazer.herd_weight = 0.4f;
        grazer.separation_weight = 0.5f;
        grazer.reproduce_threshold = 0.75f;
        grazer.reproduce_cost = 0.25f;
        grazer.reproduce_cooldown = 30.0f;
        grazer.offspring_energy = 0.40f;
        grazer.water_avoidance = 8.0f;
        grazer.max_slope = 0.7f;
        grazer.slope_cost_factor = 2.0f;
        grazer.max_age = 900.0f;
        s.creature_profiles.push_back(grazer);

        // Species 2: Browser
        bestiary::CreatureProfile browser{};
        browser.archetype = bestiary::Archetype::Herbivore;
        browser.body_length = 1.6f;
        browser.body_height = 1.1f;
        browser.body_color[0] = 0.38f; browser.body_color[1] = 0.32f; browser.body_color[2] = 0.22f;
        browser.move_speed = 1.2f;
        browser.run_speed  = 4.0f;
        browser.turn_rate  = 2.0f;
        browser.body_mass = 80.0f;
        browser.basal_rate = 0.0010f;
        browser.locomotion_cost = 0.0010f;
        browser.hunger_threshold = 0.4f;
        browser.graze_consume = 0.30f;
        browser.graze_duration = 6.0f;
        browser.graze_radius = 4.5f;
        browser.trophic_efficiency = 0.25f;
        browser.grass_caloric_value      = 0.3f;
        browser.bush_caloric_value       = 1.2f;
        browser.tree_caloric_value       = 0.8f;
        browser.reed_caloric_value       = 0.2f;
        browser.wildflower_caloric_value = 0.4f;
        browser.herd_weight = 0.2f;
        browser.separation_weight = 0.4f;
        browser.separation_radius = 3.0f;
        browser.herd_radius = 25.0f;
        browser.reproduce_threshold = 0.80f;
        browser.reproduce_cost = 0.30f;
        browser.reproduce_cooldown = 45.0f;
        browser.offspring_energy = 0.45f;
        browser.water_avoidance = 6.0f;
        browser.max_slope = 0.5f;
        browser.slope_cost_factor = 3.0f;
        browser.max_age = 1200.0f;
        s.creature_profiles.push_back(browser);

        // Species 3: Wolf
        bestiary::CreatureProfile wolf{};
        wolf.archetype = bestiary::Archetype::Predator;
        wolf.body_length = 1.0f;
        wolf.body_height = 0.75f;
        wolf.body_color[0] = 0.45f; wolf.body_color[1] = 0.42f; wolf.body_color[2] = 0.38f;
        wolf.move_speed = 2.5f;
        wolf.run_speed  = 7.0f;
        wolf.turn_rate  = 4.0f;
        wolf.body_mass  = 35.0f;
        wolf.basal_rate = 0.0010f;
        wolf.locomotion_cost = 0.0003f;
        wolf.hunger_threshold = 0.5f;
        wolf.drink_rate = 0.015f;
        wolf.hunt_radius = 30.0f;
        wolf.chase_speed = 9.0f;
        wolf.attack_range = 1.2f;
        wolf.kill_energy_gain = 0.80f;
        wolf.stalk_speed = 1.5f;
        wolf.consume_duration = 3.0f;
        wolf.max_prey_mass = 200.0f;
        wolf.herd_radius = 15.0f;
        wolf.herd_weight = 0.2f;
        wolf.separation_radius = 3.0f;
        wolf.reproduce_threshold = 0.80f;
        wolf.reproduce_cost = 0.30f;
        wolf.reproduce_cooldown = 60.0f;
        wolf.offspring_energy = 0.45f;
        wolf.water_avoidance = 4.0f;
        wolf.max_slope = 0.9f;
        wolf.slope_cost_factor = 1.0f;
        wolf.max_age = 1200.0f;
        s.creature_profiles.push_back(wolf);

        // Species 4: Cottontail
        bestiary::CreatureProfile rabbit{};
        rabbit.archetype = bestiary::Archetype::Rabbit;
        rabbit.body_length = 0.30f;
        rabbit.body_height = 0.20f;
        rabbit.body_color[0] = 0.60f; rabbit.body_color[1] = 0.52f; rabbit.body_color[2] = 0.42f;
        rabbit.move_speed = 2.0f;
        rabbit.run_speed  = 10.0f;
        rabbit.turn_rate  = 6.0f;
        rabbit.body_mass  = 2.0f;
        rabbit.basal_rate = 0.0020f;
        rabbit.locomotion_cost = 0.0003f;
        rabbit.hunger_threshold = 0.35f;
        rabbit.graze_consume = 0.12f;
        rabbit.graze_duration = 2.0f;
        rabbit.graze_radius = 2.5f;
        rabbit.trophic_efficiency = 0.45f;
        rabbit.grass_caloric_value      = 1.0f;
        rabbit.bush_caloric_value       = 0.3f;
        rabbit.tree_caloric_value       = 0.1f;
        rabbit.reed_caloric_value       = 0.7f;
        rabbit.wildflower_caloric_value = 1.2f;
        rabbit.herd_radius = 10.0f;
        rabbit.herd_weight = 0.1f;
        rabbit.separation_radius = 0.8f;
        rabbit.reproduce_threshold = 0.55f;
        rabbit.reproduce_cost = 0.12f;
        rabbit.reproduce_cooldown = 10.0f;
        rabbit.offspring_energy = 0.30f;
        rabbit.flee_radius = 20.0f;
        rabbit.flee_duration = 3.0f;
        rabbit.water_avoidance = 10.0f;
        rabbit.max_slope = 1.0f;
        rabbit.slope_cost_factor = 1.0f;
        rabbit.max_age = 480.0f;
        s.creature_profiles.push_back(rabbit);

        // Species 5: Songbird
        bestiary::CreatureProfile bird{};
        bird.archetype = bestiary::Archetype::Bird;
        bird.body_length = 0.25f;
        bird.body_height = 0.18f;
        bird.body_color[0] = 0.35f; bird.body_color[1] = 0.32f; bird.body_color[2] = 0.30f;
        bird.move_speed = 3.0f;
        bird.run_speed  = 5.0f;
        bird.turn_rate  = 6.0f;
        bird.body_mass  = 0.5f;
        bird.basal_rate = 0.0025f;
        bird.locomotion_cost = 0.0002f;
        bird.hunger_threshold = 0.35f;
        bird.graze_consume = 0.08f;
        bird.graze_duration = 1.5f;
        bird.graze_radius = 2.0f;
        bird.trophic_efficiency = 0.50f;
        bird.grass_caloric_value      = 0.5f;
        bird.bush_caloric_value       = 0.3f;
        bird.tree_caloric_value       = 0.1f;
        bird.reed_caloric_value       = 0.4f;
        bird.wildflower_caloric_value = 1.2f;
        bird.herd_radius = 15.0f;
        bird.herd_weight = 0.6f;
        bird.separation_radius = 1.0f;
        bird.reproduce_threshold = 0.60f;
        bird.reproduce_cost = 0.15f;
        bird.reproduce_cooldown = 15.0f;
        bird.offspring_energy = 0.30f;
        bird.water_avoidance = 0.0f;
        bird.max_slope = 2.0f;
        bird.slope_cost_factor = 0.0f;
        bird.max_age = 400.0f;
        bird.can_fly = true;
        bird.fly_altitude = 8.0f;
        bird.takeoff_speed = 3.0f;
        bird.landing_speed = 2.0f;
        bird.altitude_wander_amp = 2.0f;
        bird.seed_disperser = true;
        bird.seed_drop_probability = 0.015f;
        bird.seed_kind = 4;
        s.creature_profiles.push_back(bird);

        // Species 6: Raptor
        bestiary::CreatureProfile raptor{};
        raptor.archetype = bestiary::Archetype::Raptor;
        raptor.body_length = 0.38f;
        raptor.body_height = 0.25f;
        raptor.body_color[0] = 0.45f; raptor.body_color[1] = 0.35f; raptor.body_color[2] = 0.25f;
        raptor.move_speed = 2.5f;
        raptor.run_speed  = 8.0f;
        raptor.turn_rate  = 5.0f;
        raptor.body_mass  = 2.0f;
        raptor.basal_rate = 0.0015f;
        raptor.locomotion_cost = 0.0002f;
        raptor.hunger_threshold = 0.45f;
        raptor.hunt_radius = 40.0f;
        raptor.chase_speed = 12.0f;
        raptor.attack_range = 1.5f;
        raptor.kill_energy_gain = 0.70f;
        raptor.stalk_speed = 2.0f;
        raptor.consume_duration = 4.0f;
        raptor.max_prey_mass = 5.0f;
        raptor.herd_radius = 0.0f;
        raptor.herd_weight = 0.0f;
        raptor.reproduce_threshold = 0.80f;
        raptor.reproduce_cost = 0.30f;
        raptor.reproduce_cooldown = 90.0f;
        raptor.offspring_energy = 0.40f;
        raptor.water_avoidance = 0.0f;
        raptor.max_slope = 2.0f;
        raptor.slope_cost_factor = 0.0f;
        raptor.max_age = 1500.0f;
        raptor.can_fly = true;
        raptor.fly_altitude = 25.0f;
        raptor.takeoff_speed = 4.0f;
        raptor.landing_speed = 2.0f;
        raptor.altitude_wander_amp = 5.0f;
        s.creature_profiles.push_back(raptor);

        // Species 7: Snake
        bestiary::CreatureProfile snake{};
        snake.archetype = bestiary::Archetype::Snake;
        snake.body_length = 1.0f;
        snake.body_height = 0.05f;
        snake.body_color[0] = 0.40f; snake.body_color[1] = 0.35f; snake.body_color[2] = 0.20f;
        snake.move_speed = 1.0f;
        snake.run_speed  = 3.0f;
        snake.turn_rate  = 4.0f;
        snake.body_mass  = 1.5f;
        snake.basal_rate = 0.0008f;
        snake.locomotion_cost = 0.0004f;
        snake.hunger_threshold = 0.40f;
        snake.hunt_radius = 8.0f;
        snake.chase_speed = 4.0f;
        snake.attack_range = 0.8f;
        snake.kill_energy_gain = 0.90f;
        snake.stalk_speed = 0.0f;
        snake.consume_duration = 6.0f;
        snake.max_prey_mass = 5.0f;
        snake.herd_radius = 0.0f;
        snake.herd_weight = 0.0f;
        snake.reproduce_threshold = 0.85f;
        snake.reproduce_cost = 0.25f;
        snake.reproduce_cooldown = 120.0f;
        snake.offspring_energy = 0.35f;
        snake.water_avoidance = 3.0f;
        snake.max_slope = 1.2f;
        snake.slope_cost_factor = 0.5f;
        snake.max_age = 1800.0f;
        s.creature_profiles.push_back(snake);
    }

    s.persistent_water_depth.assign(WL_GRID_W * WL_GRID_H, 0.0f);

    // ---- Initial replant ---------------------------------------------------
    run_replant(s, device, alloc, gqueue, gfamily);

    s.accumulated_time = 0.0f;
    s.initialized = true;
}

// ===========================================================================
// world_lab_tick
// ===========================================================================
bool world_lab_tick(WorldLabState& s, Renderer& r, const InputFrame& in, float dt)
{
    constexpr float AUTO_REPLANT_INTERVAL = 8.0f;

    VkDevice device   = r.device;
    VmaAllocator alloc = r.allocator;
    VkQueue gqueue    = r.graphics_queue;
    uint32_t gfamily  = r.gfx_family;

    const float sim_dt = s.ui_paused ? 0.0f : dt * s.ui_sim_speed;
    s.accumulated_time += sim_dt;
    s.last_dt = sim_dt;

    // ---- Debug readback from previous frame --------------------------
    if (s.ui_atmo_enabled) {
        uint32_t prev_frame = (r.current_frame + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT;
        VmaAllocationInfo rb_info{};
        vmaGetAllocationInfo(alloc, s.wind_readback[prev_frame].allocation, &rb_info);
        if (rb_info.pMappedData) {
            const uint16_t* hp = static_cast<const uint16_t*>(rb_info.pMappedData);
            float p_min = 1e30f, p_max = -1e30f, p_sum = 0.0f, ws_max = 0.0f;
            uint32_t count = WL_ATMO_W * WL_ATMO_H;
            for (uint32_t i = 0; i < count; ++i) {
                float vx = half_to_float(hp[i * 4 + 0]);
                float vy = half_to_float(hp[i * 4 + 1]);
                float pressure = half_to_float(hp[i * 4 + 3]);
                float speed = std::sqrt(vx * vx + vy * vy);
                if (pressure < p_min) p_min = pressure;
                if (pressure > p_max) p_max = pressure;
                p_sum += pressure;
                if (speed > ws_max) ws_max = speed;
            }
            s.debug_pressure_min  = p_min;
            s.debug_pressure_max  = p_max;
            s.debug_pressure_mean = p_sum / static_cast<float>(count);
            s.debug_wind_speed_max = ws_max;
        }
        VmaAllocationInfo gc_info{};
        vmaGetAllocationInfo(alloc, s.gcond_readback[prev_frame].allocation, &gc_info);
        if (gc_info.pMappedData) {
            const uint16_t* gp = static_cast<const uint16_t*>(gc_info.pMappedData);
            float pr_max = 0, pr_sum = 0, t_sum = 0, h_sum = 0, gw_max = 0, gw_sum = 0;
            uint32_t count = WL_ATMO_W * WL_ATMO_H;
            for (uint32_t i = 0; i < count; ++i) {
                float precip   = half_to_float(gp[i * 4 + 0]);
                float temp_k   = half_to_float(gp[i * 4 + 1]);
                float humidity = half_to_float(gp[i * 4 + 2]);
                float gc_wind  = half_to_float(gp[i * 4 + 3]);
                pr_sum += precip; if (precip > pr_max) pr_max = precip;
                t_sum += temp_k; h_sum += humidity;
                gw_sum += gc_wind; if (gc_wind > gw_max) gw_max = gc_wind;
            }
            s.debug_precip_max    = pr_max;
            s.debug_precip_mean   = pr_sum / static_cast<float>(count);
            s.debug_temp_mean     = t_sum / static_cast<float>(count);
            s.debug_humidity_mean = h_sum / static_cast<float>(count);
            s.debug_gc_wind_mean  = gw_sum / static_cast<float>(count);
            s.debug_gc_wind_max   = gw_max;
        }
    }

    // Camera — orbit with optional agent-follow mode
    {
        // If following, validate index and lock target to agent position
        if (s.ui_follow_agent >= 0) {
            // Advance past dead/invalid agents
            int n = static_cast<int>(s.agents.size());
            if (n == 0) {
                s.ui_follow_agent = -1;
            } else {
                s.ui_follow_agent = s.ui_follow_agent % n;
                if (!s.agents[s.ui_follow_agent].alive) {
                    // Find next alive
                    bool found = false;
                    for (int i = 1; i < n; ++i) {
                        int idx = (s.ui_follow_agent + i) % n;
                        if (s.agents[idx].alive) {
                            s.ui_follow_agent = idx;
                            found = true;
                            break;
                        }
                    }
                    if (!found) s.ui_follow_agent = -1;
                }
            }

            if (s.ui_follow_agent >= 0) {
                const auto& a = s.agents[s.ui_follow_agent];
                float gy = sample_hm_bilinear(s.hm_cpu, WL_GRID_W, WL_GRID_H,
                                              wl_world_to_gx(a.pos[0]),
                                              wl_world_to_gy(a.pos[1]));
                s.camera.target = {a.pos[0], gy, a.pos[1]};
            }
        }

        bool rmb = in.rmb;
        bool capture = in.ui_wants_mouse;
        if (rmb && !capture) {
            if (s.camera.dragging) {
                s.camera.yaw   -= static_cast<float>(in.mouse_x - s.camera.last_mx) * 0.005f;
                s.camera.pitch += static_cast<float>(in.mouse_y - s.camera.last_my) * 0.005f;
                s.camera.pitch  = glm::clamp(s.camera.pitch, 0.05f, 1.4f);
            }
            s.camera.dragging = true;
        } else {
            s.camera.dragging = false;
        }
        s.camera.last_mx = in.mouse_x; s.camera.last_my = in.mouse_y;
        if (!capture) {
            s.camera.distance *= std::pow(0.9f, in.scroll);
            float dist_min = (s.ui_follow_agent >= 0) ?  2.0f : 30.0f;
            float dist_max = (s.ui_follow_agent >= 0) ? 40.0f : 800.0f;
            s.camera.distance = glm::clamp(s.camera.distance, dist_min, dist_max);
        }
    }

    // Cursor pick
    int win_w = in.win_w, win_h = in.win_h;
    double mx = in.mouse_x, my = in.mouse_y;
    float aspect = float(win_w) / float(win_h);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.5f, 1500.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 view = orbit_view(s.camera);

    glm::vec2 ndc = glm::vec2(
        float(mx) / win_w * 2.0f - 1.0f,
        float(my) / win_h * 2.0f - 1.0f
    );

    float pick_gx = -1.0f, pick_gy = -1.0f;
    bool have_pick = pick_grid_cell(view, proj, ndc, pick_gx, pick_gy);

    bool capture_mouse = in.ui_wants_mouse;
    bool lmb = !capture_mouse && in.lmb;
    bool rain_key = in.key_r;

    if (s.preview_mode) {
        s.camera.yaw += 0.15f * sim_dt;
        lmb = false;
        rain_key = false;
        return true;
    }

    // ---- ImGui --------------------------------------------------------
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool wants_back = false;

    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 700), ImGuiCond_FirstUseEver);
    ImGui::Begin("World Lab");

    if (s.embedded) {
        if (ImGui::Button("< Back"))
            wants_back = true;
        ImGui::Separator();
    }

    ImGui::TextDisabled("LMB: raise  Shift+LMB: lower  R: rain  RMB: orbit");
    if (have_pick)
        ImGui::Text("cursor: (%.1f, %.1f) cells", pick_gx, pick_gy);

    ImGui::Spacing();
    if (ImGui::Button(s.ui_paused ? "Resume" : " Pause ", ImVec2(80, 0)))
        s.ui_paused = !s.ui_paused;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    ImGui::SliderFloat("##sim_speed", &s.ui_sim_speed, 0.1f, 5.0f, "speed %.1fx");
    ImGui::Spacing();

    // ---- Terrain -----------------------------------------------------------
    if (ImGui::CollapsingHeader("Terrain", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt  ("brush radius", &s.ui_brush_radius_cells, 2, 60);
        ImGui::SliderFloat("brush amount", &s.ui_brush_amount, 0.05f, 5.0f, "%.2f m");
        if (ImGui::Button("Reset terrain")) {
            generate_heightmap(s.hm_cpu);
            update_r32_image(device, alloc, gqueue, gfamily,
                             s.hm_gpu.image, s.hm_cpu, WL_GRID_W, WL_GRID_H);
            {
                OneShot os = oneshot_begin(device, gqueue, gfamily);
                vkCmdBindPipeline(os.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe_swe_init.pipeline);
                vkCmdBindDescriptorSets(os.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        s.pipe_swe_init.layout, 0, 1, &s.ds_swe_init, 0, nullptr);
                SweInitPC pc{WL_GRID_W, WL_GRID_H, -10000.0f, 0.0f};
                vkCmdPushConstants(os.cmd, s.pipe_swe_init.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc), &pc);
                vkCmdDispatch(os.cmd, (WL_GRID_W + 7) / 8, (WL_GRID_H + 7) / 8, 1);
                oneshot_end(os);
            }
            s.swe_ping_pong = 0;
            s.agents.clear();
            s.replant_pending = 1;
        }
        if (ImGui::CollapsingHeader("Erosion")) {
            ImGui::Checkbox("enable erosion", &s.ui_erosion_enabled);
            ImGui::SliderFloat("k_erosion",   &s.ui_k_erosion,     0.0f, 0.1f);
            ImGui::SliderFloat("k_deposit",   &s.ui_k_deposit,     0.0f, 0.1f);
            ImGui::SliderFloat("k_capacity",  &s.ui_k_capacity,    0.0f, 2.0f);
            ImGui::SliderFloat("min_slope",   &s.ui_min_slope,     0.0f, 0.01f);
            ImGui::SliderFloat("k_wind ero",  &s.ui_k_wind,        0.0f, 0.5f);
            ImGui::SliderFloat("wind thresh", &s.ui_wind_threshold, 0.0f, 3.0f);
            ImGui::SliderFloat("k_thermal",   &s.ui_k_thermal,     0.0f, 0.2f);
        }
    }

    // ---- Water -------------------------------------------------------------
    if (ImGui::CollapsingHeader("Water", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox  ("enable SWE",  &s.ui_swe_enabled);
        ImGui::SliderInt ("substeps",    &s.ui_swe_substeps, 1, 10);
        ImGui::SliderFloat("dt (s)",     &s.ui_swe_dt,       0.005f, 0.1f, "%.3f");
        ImGui::SliderFloat("gravity",    &s.ui_gravity,      0.1f,  20.0f, "%.2f m/s²");
        ImGui::SliderFloat("friction",   &s.ui_friction,     0.0f,   0.5f, "%.3f");
        ImGui::SliderFloat("damping",    &s.ui_damping,      0.0f,   0.5f, "%.3f");
        ImGui::SliderFloat("k_rain",     &s.ui_k_rain, 0.0f, 2.0f);
        ImGui::Separator();
        ImGui::TextUnformatted("Moisture");
        ImGui::SliderFloat("capillary depth", &s.ui_capillary_depth, 0.005f, 1.0f, "%.3f");
        ImGui::SliderInt  ("capillary blur",  &s.ui_capillary_blur,  0, 12);
        ImGui::Checkbox   ("show moisture (debug)", &s.ui_show_moisture);
        if (ImGui::CollapsingHeader("Atmosphere")) {
            ImGui::Checkbox("enable atmosphere", &s.ui_atmo_enabled);
            ImGui::SliderFloat("cloud opacity",    &s.ui_cloud_opacity,     0.0f, 2.0f);
            ImGui::SliderFloat("orographic lift",  &s.ui_orographic_lift,   0.0f, 1.0f);
            ImGui::SliderFloat("adiabatic cool",   &s.ui_adiabatic_cooling, 0.0f, 0.02f);
            ImGui::SliderFloat("rain shadow",      &s.ui_rain_shadow,       0.0f, 1.0f);
            ImGui::SliderFloat("k_pressure",       &s.ui_k_pressure,        0.0f, 5.0f);
            ImGui::SliderFloat("wind strength",    &s.ui_wind_strength,     0.0f, 10.0f);
            ImGui::SliderFloat("k_evaporation",    &s.ui_k_evaporation,     0.0f, 2.0f);
            ImGui::SliderFloat("cloud base (m)",   &s.ui_cloud_base,        -20.0f, 100.0f);
            ImGui::SliderFloat("layer height (m)", &s.ui_layer_height,      1.0f, 50.0f);
            if (ImGui::Button("Reset atmosphere")) s.ui_atmo_reset = true;
        }
        if (s.ui_atmo_enabled && ImGui::CollapsingHeader("Atmo debug")) {
            ImGui::Text("pressure — min:%.2f mean:%.2f max:%.2f",
                        (double)s.debug_pressure_min, (double)s.debug_pressure_mean, (double)s.debug_pressure_max);
            ImGui::Text("wind max: %.2f  precip mean: %.4f", (double)s.debug_wind_speed_max, (double)s.debug_precip_mean);
            ImGui::Text("temp: %.1f C  humidity: %.3f",
                        (double)(s.debug_temp_mean - 273.15f), (double)s.debug_humidity_mean);
        }
    }

    // ---- Vegetation --------------------------------------------------------
    if (ImGui::CollapsingHeader("Vegetation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("show plants", &s.ui_show_plants);
        ImGui::SliderFloat("density",  &s.eco_params.density_scale, 0.1f, 3.0f, "%.2f");
        ImGui::SliderFloat("r_min",    &s.eco_params.r_min, 0.5f, 4.0f, "%.2f m");
        ImGui::SliderFloat("r_max",    &s.eco_params.r_max, 1.0f, 10.0f, "%.2f m");
        ImGui::SliderFloat("growth",   &s.plant_growth_rate, 0.0f, 0.3f, "%.3f");
        ImGui::SliderFloat("decay",    &s.plant_decay_rate, 0.0f, 0.2f, "%.3f");
        ImGui::Separator();
        ImGui::Checkbox("autorun", &s.ui_autorun);
        ImGui::Checkbox("spring",  &s.ui_spring_enabled);
        ImGui::SliderFloat("spring rate", &s.ui_spring_rate, 0.01f, 1.0f, "%.2f");
        ImGui::Text("t: %.0f s  sprout in: %.1f s",
                    static_cast<double>(s.accumulated_time),
                    static_cast<double>(AUTO_REPLANT_INTERVAL - s.auto_replant_timer));
        if (ImGui::Button("Replant now")) s.replant_pending = 1;
        ImGui::SameLine();
        if (ImGui::Button("Save Plants")) save_plant_profiles(s);
        ImGui::SameLine();
        if (ImGui::Button("Load Plants")) {
            if (load_plant_profiles(s)) {
                build_canonical_meshes(s, alloc);
                s.replant_pending = 1;
            }
        }
        ImGui::Separator();

        // Per-species plant count
        int n_grass = 0, n_bush = 0, n_tree = 0, n_reed = 0, n_wf = 0;
        for (const auto& p : s.plant_population) {
            switch (p.kind) {
            case bestiary::PLANT_GRASS:      ++n_grass; break;
            case bestiary::PLANT_BUSH:       ++n_bush;  break;
            case bestiary::PLANT_TREE:       ++n_tree;  break;
            case bestiary::PLANT_REED:       ++n_reed;  break;
            case bestiary::PLANT_WILDFLOWER: ++n_wf;    break;
            }
        }
        int n_plants = static_cast<int>(s.plant_population.size());
        ImGui::Text("plants: %d total", n_plants);
        ImGui::Text("  grass:%d  bush:%d  tree:%d", n_grass, n_bush, n_tree);
        ImGui::Text("  reed:%d  wildflower:%d", n_reed, n_wf);
        uint32_t total_inst = 0;
        for (int k = 0; k < bestiary::PLANT_KIND_COUNT; ++k) total_inst += s.plant_inst_count[k];
        ImGui::Text("instances: %u  canonical verts/kind: %u/%u/%u/%u/%u",
                    total_inst,
                    s.plant_canonical[0].vertex_count, s.plant_canonical[1].vertex_count,
                    s.plant_canonical[2].vertex_count, s.plant_canonical[3].vertex_count,
                    s.plant_canonical[4].vertex_count);
    }

    // ---- Creatures ---------------------------------------------------------
    if (ImGui::CollapsingHeader("Creatures", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("enable", &s.ui_creatures_enabled);
        ImGui::SliderInt  ("count",      &s.ui_creature_count, 1, 100);
        ImGui::SliderFloat("speed mult", &s.ui_creature_speed, 0.0f, 5.0f, "%.1f");

        if (ImGui::Button("Spawn All")) spawn_all_creatures(s);
        ImGui::SameLine();
        if (ImGui::Button("Clear")) { s.agents.clear(); s.ui_follow_agent = -1; }

        if (ImGui::Button("Save Species")) save_creature_profiles(s);
        ImGui::SameLine();
        if (ImGui::Button("Load Species")) {
            if (load_creature_profiles(s)) {
                bestiary::clear_creature_mesh_cache();
                spawn_all_creatures(s);
            }
        }

        // Follow camera controls
        {
            bool following = s.ui_follow_agent >= 0;
            if (ImGui::Button(following ? "Unfollow" : "Follow")) {
                if (following) {
                    s.ui_follow_agent = -1;
                    // Restore wide view
                    s.camera.distance = 220.0f;
                    s.camera.target   = {0.0f, 0.0f, 0.0f};
                } else {
                    // Find first alive agent
                    for (int i = 0; i < static_cast<int>(s.agents.size()); ++i) {
                        if (s.agents[i].alive) {
                            s.ui_follow_agent = i;
                            s.camera.distance = 8.0f;
                            s.camera.pitch    = 0.4f;
                            break;
                        }
                    }
                }
            }
            if (following) {
                ImGui::SameLine();
                if (ImGui::Button("Next")) {
                    int n = static_cast<int>(s.agents.size());
                    for (int i = 1; i < n; ++i) {
                        int idx = (s.ui_follow_agent + i) % n;
                        if (s.agents[idx].alive) {
                            s.ui_follow_agent = idx;
                            break;
                        }
                    }
                }
                ImGui::SameLine();
                const char* sp_names[] = {"Sprinter","Grazer","Browser","Wolf","Rabbit","Bird","Raptor","Snake"};
                int sp = s.agents[s.ui_follow_agent].species_id;
                ImGui::TextDisabled("#%d %s", s.ui_follow_agent,
                    (sp >= 0 && sp < 8) ? sp_names[sp] : "?");
            }
        }

        int alive = bestiary::count_alive(s.agents);
        int s0 = bestiary::count_alive_species(s.agents, 0);
        int s1 = bestiary::count_alive_species(s.agents, 1);
        int s2 = bestiary::count_alive_species(s.agents, 2);
        int s3 = bestiary::count_alive_species(s.agents, 3);
        int s4 = bestiary::count_alive_species(s.agents, 4);
        int s5 = bestiary::count_alive_species(s.agents, 5);
        int s6 = bestiary::count_alive_species(s.agents, 6);
        int s7 = bestiary::count_alive_species(s.agents, 7);
        ImGui::Text("alive: %d", alive);
        ImGui::Text("  S:%d G:%d B:%d W:%d R:%d Bi:%d Ra:%d Sn:%d",
                    s0, s1, s2, s3, s4, s5, s6, s7);
        float cur_avg_energy = bestiary::avg_energy(s.agents);
        float cur_min_energy = bestiary::min_energy(s.agents);
        float cur_max_energy = bestiary::max_energy(s.agents);
        ImGui::Text("energy avg:%.2f min:%.2f max:%.2f",
                    static_cast<double>(cur_avg_energy),
                    static_cast<double>(cur_min_energy),
                    static_cast<double>(cur_max_energy));

        // ---- Sample into ring buffer ----
        s.graph.sample_timer += dt;
        if (s.graph.sample_timer >= WL_GRAPH_SAMPLE_INTERVAL) {
            s.graph.sample_timer -= WL_GRAPH_SAMPLE_INTERVAL;
            s.graph.pop_sprinter[s.graph.write_idx] = static_cast<float>(s0);
            s.graph.pop_grazer[s.graph.write_idx]   = static_cast<float>(s1);
            s.graph.pop_browser[s.graph.write_idx]  = static_cast<float>(s2);
            s.graph.pop_wolf[s.graph.write_idx]     = static_cast<float>(s3);
            s.graph.pop_rabbit[s.graph.write_idx]   = static_cast<float>(s4);
            s.graph.pop_bird[s.graph.write_idx]     = static_cast<float>(s5);
            s.graph.pop_raptor[s.graph.write_idx]   = static_cast<float>(s6);
            s.graph.pop_snake[s.graph.write_idx]    = static_cast<float>(s7);
            s.graph.pop_total[s.graph.write_idx]    = static_cast<float>(alive);
            s.graph.energy_avg[s.graph.write_idx]   = cur_avg_energy;
            s.graph.energy_min[s.graph.write_idx]   = cur_min_energy;
            s.graph.energy_max[s.graph.write_idx]   = cur_max_energy;
            s.graph.write_idx = (s.graph.write_idx + 1) % WL_GRAPH_HISTORY;
            if (s.graph.count < WL_GRAPH_HISTORY) ++s.graph.count;
        }

        if (ImGui::CollapsingHeader("Population Graph")) {
            auto plot_ring = [&](const char* label, const float* buf, ImVec4 color,
                                 float scale_min, float scale_max) {
                ImGui::PushStyleColor(ImGuiCol_PlotLines, color);
                int offset = (s.graph.count < WL_GRAPH_HISTORY) ? 0 : s.graph.write_idx;
                ImGui::PlotLines(label, buf, s.graph.count, offset,
                                 nullptr, scale_min, scale_max, ImVec2(0, 40));
                ImGui::PopStyleColor();
            };
            float pop_max = 1.0f;
            for (int k = 0; k < s.graph.count; ++k)
                if (s.graph.pop_total[k] > pop_max) pop_max = s.graph.pop_total[k];
            pop_max = std::ceil(pop_max / 10.0f) * 10.0f;
            plot_ring("Sprinter", s.graph.pop_sprinter, ImVec4(0.9f, 0.7f, 0.3f, 1.0f), 0.0f, pop_max);
            plot_ring("Grazer",   s.graph.pop_grazer,   ImVec4(0.5f, 0.8f, 0.3f, 1.0f), 0.0f, pop_max);
            plot_ring("Browser",  s.graph.pop_browser,  ImVec4(0.4f, 0.3f, 0.2f, 1.0f), 0.0f, pop_max);
            plot_ring("Wolf",     s.graph.pop_wolf,     ImVec4(0.8f, 0.2f, 0.2f, 1.0f), 0.0f, pop_max);
            plot_ring("Rabbit",   s.graph.pop_rabbit,   ImVec4(0.7f, 0.6f, 0.9f, 1.0f), 0.0f, pop_max);
            plot_ring("Bird",     s.graph.pop_bird,     ImVec4(0.3f, 0.7f, 0.9f, 1.0f), 0.0f, pop_max);
            plot_ring("Raptor",   s.graph.pop_raptor,   ImVec4(0.9f, 0.5f, 0.1f, 1.0f), 0.0f, pop_max);
            plot_ring("Snake",    s.graph.pop_snake,    ImVec4(0.2f, 0.5f, 0.2f, 1.0f), 0.0f, pop_max);
            plot_ring("Total",    s.graph.pop_total,    ImVec4(1.0f, 1.0f, 1.0f, 1.0f), 0.0f, pop_max);
        }

        if (ImGui::CollapsingHeader("Energy Graph")) {
            auto plot_ring = [&](const char* label, const float* buf, ImVec4 color) {
                ImGui::PushStyleColor(ImGuiCol_PlotLines, color);
                int offset = (s.graph.count < WL_GRAPH_HISTORY) ? 0 : s.graph.write_idx;
                ImGui::PlotLines(label, buf, s.graph.count, offset,
                                 nullptr, 0.0f, 1.0f, ImVec2(0, 40));
                ImGui::PopStyleColor();
            };
            plot_ring("Avg", s.graph.energy_avg, ImVec4(0.3f, 0.9f, 0.3f, 1.0f));
            plot_ring("Min", s.graph.energy_min, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
            plot_ring("Max", s.graph.energy_max, ImVec4(0.3f, 0.6f, 0.9f, 1.0f));
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Species");
        const char* species_names[] = {"Sprinter", "Grazer", "Browser", "Wolf", "Rabbit", "Bird", "Raptor", "Snake"};
        ImGui::Combo("##species", &s.ui_species_sel, species_names, 8);
        auto& sp = s.creature_profiles[static_cast<size_t>(s.ui_species_sel)];

        ImGui::SliderFloat("body mass",       &sp.body_mass,            1.0f, 200.0f, "%.0f kg");
        ImGui::SliderFloat("basal rate",      &sp.basal_rate,           0.001f, 0.05f, "%.3f");
        ImGui::SliderFloat("locomotion cost", &sp.locomotion_cost,      0.001f, 0.02f, "%.4f");
        bool sp_is_predator = (sp.archetype == bestiary::Archetype::Predator
                            || sp.archetype == bestiary::Archetype::Raptor
                            || sp.archetype == bestiary::Archetype::Snake);
        if (!sp_is_predator) {
            ImGui::SliderFloat("trophic eff",  &sp.trophic_efficiency,  0.01f, 0.50f, "%.2f");
            ImGui::SliderFloat("graze consume",&sp.graze_consume,        0.01f, 0.50f, "%.2f");
        }
        ImGui::SliderFloat("reproduce at", &sp.reproduce_threshold, 0.40f, 0.95f, "%.2f");
        ImGui::SliderFloat("max age (s)",  &sp.max_age,             30.0f, 1800.0f, "%.0f");
        if (sp_is_predator) {
            ImGui::SliderFloat("hunt radius",  &sp.hunt_radius,        5.0f, 60.0f, "%.0f m");
            ImGui::SliderFloat("chase speed",  &sp.chase_speed,        3.0f, 15.0f, "%.1f m/s");
            ImGui::SliderFloat("attack range", &sp.attack_range,       0.5f, 3.0f,  "%.1f m");
            ImGui::SliderFloat("kill energy",  &sp.kill_energy_gain,   0.1f, 1.0f,  "%.2f");
            ImGui::SliderFloat("stalk speed",  &sp.stalk_speed,        0.5f, 3.0f,  "%.1f m/s");
        }
        if (sp.can_fly) {
            ImGui::SliderFloat("fly altitude", &sp.fly_altitude,       2.0f, 40.0f, "%.0f m");
        }
        ImGui::SliderFloat("water avoid", &sp.water_avoidance,    0.0f, 20.0f, "%.1f");
        ImGui::SliderFloat("max slope",   &sp.max_slope,          0.2f, 1.5f,  "%.2f");
        ImGui::SliderFloat("slope cost",  &sp.slope_cost_factor,  0.0f, 5.0f,  "%.1f");
    }

    ImGui::End();
    ImGui::Render();

    // Track brush stroke end
    bool brush_apply_now_tick = lmb && have_pick;
    if (s.prev_lmb && !lmb && s.brushed_this_stroke) {
        s.brushed_this_stroke = false;
    }
    s.prev_lmb     = lmb;
    s.prev_rain_key = rain_key;

    // ---- Plant population tick (always, independent of creatures) --------
    bestiary::tick_plant_population(s.plant_population, s.persistent_env,
                                    s.eco_params, sim_dt,
                                    s.plant_growth_rate, s.plant_decay_rate);

    constexpr float VEG_REBUILD_INTERVAL = 0.5f;
    s.veg_rebuild_timer += dt;
    if (s.veg_rebuild_timer >= VEG_REBUILD_INTERVAL) {
        s.veg_rebuild_timer = 0.0f;
        upload_plant_instances(s, device, alloc);
    }

    // ---- Creature tick ---------------------------------------------------
    if (s.ui_creatures_enabled && !s.agents.empty()) {
        bestiary::CreatureWorldView world_view{};
        world_view.plant_population = &s.plant_population;
        world_view.env_field    = &s.persistent_env;
        world_view.terrain_height = [&s](float x, float z) {
            return sample_hm_bilinear(s.hm_cpu, WL_GRID_W, WL_GRID_H,
                                      wl_world_to_gx(x), wl_world_to_gy(z));
        };
        world_view.water_depth = [&s](float x, float z) {
            float gx = (x + WL_TILE_HALF_X) / WL_DX;
            float gz = (z + WL_TILE_HALF_Z) / WL_DX;
            int ix = std::clamp(static_cast<int>(std::floor(gx)), 0, static_cast<int>(WL_GRID_W) - 1);
            int iz = std::clamp(static_cast<int>(std::floor(gz)), 0, static_cast<int>(WL_GRID_H) - 1);
            return s.persistent_water_depth[static_cast<size_t>(iz * WL_GRID_W + ix)];
        };
        world_view.tile_half_x  = WL_TILE_HALF_X;
        world_view.tile_half_z  = WL_TILE_HALF_Z;
        world_view.has_threat   = brush_apply_now_tick;
        world_view.threat_pos[0] = pick_gx * WL_DX - WL_TILE_HALF_X;
        world_view.threat_pos[1] = pick_gy * WL_DX - WL_TILE_HALF_Z;
        world_view.threat_radius = static_cast<float>(s.ui_brush_radius_cells) * WL_DX + 10.0f;

        float creature_dt = sim_dt * s.ui_creature_speed;
        bestiary::update_creatures(s.agents, s.creature_profiles,
                                   world_view,
                                   creature_dt, s.creature_tick);
        ++s.creature_tick;

        // Rebuild creature meshes every frame (cheap for <100 agents)
        auto cm = bestiary::generate_creature_meshes(
            s.agents, s.creature_profiles,
            [&s](float x, float z) {
                return sample_hm_bilinear(s.hm_cpu, WL_GRID_W, WL_GRID_H,
                                          wl_world_to_gx(x), wl_world_to_gy(z));
            }, dt);

        vkDeviceWaitIdle(device);
        destroy_plant_mesh(alloc, s.creature_mesh_gpu);
        if (!cm.vertices.empty()) {
            upload_plant_mesh(alloc, s.creature_mesh_gpu, cm);
        }
    }

    if (s.replant_pending > 0) {
        --s.replant_pending;
        if (s.replant_pending == 0) {
            run_replant(s, device, alloc, gqueue, gfamily);
        }
    }

    // ---- Autorun: refresh environment + sprout + creature spawn ---------
    if (s.ui_autorun) {
        s.auto_replant_timer += sim_dt;
        if (s.auto_replant_timer >= AUTO_REPLANT_INTERVAL) {
            s.auto_replant_timer = 0.0f;

            refresh_env(s, device, alloc, gqueue, gfamily);

            // Sprout new plants where conditions are favorable
            bestiary::sprout_plants(s.plant_population, s.eco_params,
                                     s.persistent_env, s.sprout_seed);
            ++s.sprout_seed;
        }

        if (!s.initial_spawn_done && s.accumulated_time > 3.0f) {
            s.initial_spawn_done = true;
            spawn_all_creatures(s);
        }
    }

    return !wants_back;
}

// ===========================================================================
// world_lab_render
// ===========================================================================
void world_lab_render(WorldLabState& s, Renderer& r,
                      FrameData& frame, uint32_t image_index, VkExtent2D extent)
{
    VkDevice device   = r.device;
    VmaAllocator alloc = r.allocator;
    VkCommandBuffer cmd = frame.cmd;

    // Recompute view/proj (same as in tick)
    int win_w, win_h;
    glfwGetWindowSize(r.window, &win_w, &win_h);
    double mx_d, my_d;
    glfwGetCursorPos(r.window, &mx_d, &my_d);
    float aspect = float(win_w) / float(win_h);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.5f, 1500.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 view = orbit_view(s.camera);

    glm::vec2 ndc = glm::vec2(
        float(mx_d) / win_w * 2.0f - 1.0f,
        float(my_d) / win_h * 2.0f - 1.0f
    );

    float pick_gx = -1.0f, pick_gy = -1.0f;
    bool have_pick = pick_grid_cell(view, proj, ndc, pick_gx, pick_gy);

    bool capture_mouse = ImGui::GetIO().WantCaptureMouse;
    bool lmb = !capture_mouse && glfwGetMouseButton(r.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    bool shift = glfwGetKey(r.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS
              || glfwGetKey(r.window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    bool rain_key = glfwGetKey(r.window, GLFW_KEY_R) == GLFW_PRESS;

    // Brush dispatch (terrain) + CPU mirror update
    bool brush_apply_now = lmb && have_pick;
    if (brush_apply_now) {
        float amount = (shift ? -1.0f : 1.0f) * s.ui_brush_amount;
        // GPU
        TerrainBrushPC bpc{};
        bpc.brush_x = pick_gx;
        bpc.brush_y = pick_gy;
        bpc.brush_radius = static_cast<float>(s.ui_brush_radius_cells);
        bpc.brush_amount = amount * s.last_dt * 4.0f;
        bpc.grid_w = WL_GRID_W; bpc.grid_h = WL_GRID_H;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe_brush.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s.pipe_brush.layout, 0, 1, &s.ds_brush, 0, nullptr);
        vkCmdPushConstants(cmd, s.pipe_brush.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(bpc), &bpc);
        vkCmdDispatch(cmd, (WL_GRID_W + 15) / 16, (WL_GRID_H + 15) / 16, 1);
        compute_memory_barrier(cmd);

        // CPU mirror (replay same falloff)
        cpu_apply_brush(s.hm_cpu, WL_GRID_W, WL_GRID_H, pick_gx, pick_gy, bpc.brush_radius, bpc.brush_amount);
        s.brushed_this_stroke = true;
    }

    // SWE step(s)
    if (s.ui_swe_enabled && !s.ui_paused) {
        for (int sub = 0; sub < s.ui_swe_substeps; ++sub) {
            int cur = s.swe_ping_pong & 1;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe_swe_step.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    s.pipe_swe_step.layout, 0, 1, &s.ds_swe_step[cur], 0, nullptr);
            SweStepPC pc{};
            pc.time     = s.accumulated_time;
            pc.dt       = s.ui_swe_dt;
            pc.gravity  = s.ui_gravity;
            pc.friction = s.ui_friction;
            pc.damping  = s.ui_damping;
            pc.dx       = WL_DX;
            pc.sea_level = -10000.0f;
            pc.k_rain   = s.ui_k_rain;
            pc.grid_w   = WL_GRID_W;
            pc.grid_h   = WL_GRID_H;
            if (rain_key && have_pick) {
                pc.pulse_x      = pick_gx;
                pc.pulse_y      = pick_gy;
                pc.pulse_radius = static_cast<float>(s.ui_brush_radius_cells);
                pc.pulse_amount = 0.5f;
                s.wt_water_pulse_active = 1;
            } else if (s.ui_spring_enabled) {
                pc.pulse_x      = WL_SPRING_GX;
                pc.pulse_y      = WL_SPRING_GY;
                pc.pulse_radius = 6.0f;
                pc.pulse_amount = s.ui_spring_rate;
            }
            vkCmdPushConstants(cmd, s.pipe_swe_step.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (WL_GRID_W + 7) / 8, (WL_GRID_H + 7) / 8, 1);
            compute_memory_barrier(cmd);
            ++s.swe_ping_pong;
        }
    }

    // ---- Atmosphere 3D dispatch ----------------------------------------
    if (s.ui_atmo_enabled) {
        int acur = s.atmo_ping_pong & 1;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe_atmo.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s.pipe_atmo.layout, 0, 1, &s.ds_atmo[acur], 0, nullptr);
        Atmo3DPC apc{};
        apc.dt                    = s.last_dt;
        apc.accumulated_time      = s.accumulated_time;
        apc.grid_w                = WL_ATMO_W;
        apc.grid_h                = WL_ATMO_H;
        apc.grid_d                = WL_ATMO_D;
        apc.terrain_scale         = static_cast<float>(WL_GRID_W) * WL_DX;
        apc.layer_height          = s.ui_layer_height;
        apc.max_elevation         = 50.0f;
        apc.orographic_lift_coeff = s.ui_orographic_lift;
        apc.adiabatic_cooling_rate = s.ui_adiabatic_cooling;
        apc.rain_shadow_intensity = s.ui_rain_shadow;
        apc.force_init            = (s.ui_atmo_reset || s.atmo_force_init) ? 1u : 0u;
        apc.k_pressure            = s.ui_k_pressure;
        apc.wind_strength         = s.ui_wind_strength;
        apc.k_evaporation         = s.ui_k_evaporation;
        vkCmdPushConstants(cmd, s.pipe_atmo.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(apc), &apc);
        vkCmdDispatch(cmd, (WL_ATMO_W + 3) / 4, (WL_ATMO_H + 3) / 4, (WL_ATMO_D + 3) / 4);
        compute_memory_barrier(cmd);
        ++s.atmo_ping_pong;
        s.atmo_force_init = false;
        s.ui_atmo_reset = false;

        int wind_write_idx = acur ^ 1;
        VkImage wind_src = s.wind_field[wind_write_idx].image;
        uint32_t cur_frame_idx = r.current_frame;

        volume_barrier(cmd, wind_src,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkBufferImageCopy wind_copy{};
        wind_copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        wind_copy.imageExtent = {WL_ATMO_W, WL_ATMO_H, 1};
        vkCmdCopyImageToBuffer(cmd, wind_src,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               s.wind_readback[cur_frame_idx].buffer, 1, &wind_copy);

        volume_barrier(cmd, wind_src,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                         | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_GENERAL);

        image_barrier(cmd, s.ground_cond.image,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT,
                      VK_IMAGE_LAYOUT_GENERAL,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkBufferImageCopy gc_copy{};
        gc_copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        gc_copy.imageExtent = {WL_ATMO_W, WL_ATMO_H, 1};
        vkCmdCopyImageToBuffer(cmd, s.ground_cond.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               s.gcond_readback[cur_frame_idx].buffer, 1, &gc_copy);

        image_barrier(cmd, s.ground_cond.image,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                        | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_IMAGE_LAYOUT_GENERAL);
    }

    // ---- Erosion dispatch -----------------------------------------------
    if (s.ui_erosion_enabled) {
        int ero_cur = s.ero_ping_pong & 1;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe_erosion.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s.pipe_erosion.layout, 0, 1, &s.ds_erosion[ero_cur], 0, nullptr);
        ErosionPC epc{};
        epc.dt          = s.last_dt;
        epc.dx          = WL_DX;
        epc.grid_w      = WL_GRID_W;
        epc.grid_h      = WL_GRID_H;
        epc.k_erosion   = s.ui_k_erosion;
        epc.k_deposit   = s.ui_k_deposit;
        epc.k_capacity  = s.ui_k_capacity;
        epc.min_slope   = s.ui_min_slope;
        epc.min_depth   = 0.001f;
        epc.max_change  = 0.1f;
        epc.max_sediment = 1.0f;
        epc.k_wind      = s.ui_k_wind;
        epc.k_thermal   = s.ui_k_thermal;
        epc.wind_threshold = s.ui_wind_threshold;
        vkCmdPushConstants(cmd, s.pipe_erosion.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(epc), &epc);
        vkCmdDispatch(cmd, (WL_GRID_W + 15) / 16, (WL_GRID_H + 15) / 16, 1);
        ++s.ero_ping_pong;
    }

    compute_to_graphics_barrier(cmd);

    // ---- Color attachment + depth barriers ---------------------------
    image_barrier(cmd, r.swapchain_images[image_index],
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
        db.image         = r.depth_buffer.image;
        db.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &db;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    // ---- Main pass: terrain + plants ---------------------------------
    glm::mat4 mvp = proj * view;

    VkRenderingAttachmentInfo color{};
    color.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView   = r.swapchain_views[image_index];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.45f, 0.55f, 0.70f, 1.0f}}; // sky

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
    VkViewport vp{0, 0, float(extent.width), float(extent.height), 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // Terrain
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipe_terrain.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            s.pipe_terrain.layout, 0, 1, &s.ds_terrain, 0, nullptr);
    WorldTerrainPC tpc{};
    tpc.mvp        = mvp;
    tpc.grid_w_f   = float(WL_GRID_W);
    tpc.grid_h_f   = float(WL_GRID_H);
    tpc.cell_size  = WL_DX;
    tpc.sea_level  = -10000.0f;
    tpc.brush_x    = pick_gx;
    tpc.brush_y    = pick_gy;
    tpc.brush_radius = float(s.ui_brush_radius_cells);
    tpc.brush_active = (have_pick && !capture_mouse) ? 1.0f : 0.0f;
    tpc.moisture_overlay = s.ui_show_moisture ? 1.0f : 0.0f;
    vkCmdPushConstants(cmd, s.pipe_terrain.layout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(tpc), &tpc);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &s.terrain_mesh.vbo.buffer, &zero);
    vkCmdBindIndexBuffer(cmd, s.terrain_mesh.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, s.terrain_mesh.index_count, 1, 0, 0, 0);

    // Plants (instanced, one draw per kind)
    if (s.ui_show_plants) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipe_clump.pipeline);
        ClumpPC cpc{};
        cpc.mvp = mvp;
        cpc.wind_dir[0] = 0.6f;
        cpc.wind_dir[1] = 0.4f;
        cpc.wind_speed  = 0.5f;
        cpc.time = s.accumulated_time;
        vkCmdPushConstants(cmd, s.pipe_clump.layout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(cpc), &cpc);
        for (int k = 0; k < bestiary::PLANT_KIND_COUNT; ++k) {
            if (s.plant_inst_count[k] == 0 || s.plant_canonical[k].index_count == 0) continue;
            VkBuffer     vbufs[2]   = {s.plant_canonical[k].vbo.buffer, s.plant_inst[k].buffer};
            VkDeviceSize offsets[2] = {0, 0};
            vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, offsets);
            vkCmdBindIndexBuffer(cmd, s.plant_canonical[k].ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, s.plant_canonical[k].index_count,
                             s.plant_inst_count[k], 0, 0, 0);
        }
    }

    // Creatures (dedicated pipeline — pre-baked world-space vertices, no instancing)
    if (s.ui_creatures_enabled && s.creature_mesh_gpu.index_count > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipe_creature.pipeline);
        vkCmdPushConstants(cmd, s.pipe_creature.layout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(glm::mat4), &mvp);
        vkCmdBindVertexBuffers(cmd, 0, 1, &s.creature_mesh_gpu.vbo.buffer, &zero);
        vkCmdBindIndexBuffer(cmd, s.creature_mesh_gpu.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, s.creature_mesh_gpu.index_count, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);

    // ---- Cloud raymarch pass (alpha-blended, no depth) ---------------
    if (s.ui_atmo_enabled && s.ui_cloud_opacity > 0.0f) {
        WL_CameraUBO cam_ubo{};
        cam_ubo.view      = view;
        cam_ubo.proj      = proj;
        cam_ubo.sun_dir   = glm::normalize(glm::vec3(0.4f, 0.7f, -0.3f));
        cam_ubo.sun_color = glm::vec3(1.0f, 0.95f, 0.85f);
        cam_ubo.cam_pos   = glm::vec3(0.0f);
        cam_ubo.inv_view_proj = glm::inverse(proj * view);
        std::memcpy(s.camera_ubo_info.pMappedData, &cam_ubo, sizeof(cam_ubo));
        vmaFlushAllocation(alloc, s.camera_ubo_alloc, 0, VK_WHOLE_SIZE);

        VkRenderingAttachmentInfo cloud_color = color;
        cloud_color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkRenderingInfo cloud_ri{};
        cloud_ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        cloud_ri.renderArea           = {{0, 0}, extent};
        cloud_ri.layerCount           = 1;
        cloud_ri.colorAttachmentCount = 1;
        cloud_ri.pColorAttachments    = &cloud_color;
        vkCmdBeginRendering(cmd, &cloud_ri);
        VkViewport cloud_vp{0, 0, float(extent.width), float(extent.height), 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &cloud_vp);
        VkRect2D cloud_sc{{0, 0}, extent};
        vkCmdSetScissor(cmd, 0, 1, &cloud_sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipe_cloud.pipeline);
        int cloud_set_idx = s.atmo_ping_pong & 1;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                s.pipe_cloud.layout, 0, 1, &s.ds_cloud[cloud_set_idx], 0, nullptr);
        RaymarchPC rpc{};
        rpc.terrain_size  = static_cast<float>(WL_GRID_W) * WL_DX;
        rpc.max_elevation = 50.0f;
        rpc.cloud_opacity = s.ui_cloud_opacity;
        rpc.cloud_base    = s.ui_cloud_base;
        rpc.vol_w         = WL_ATMO_W;
        rpc.vol_h         = WL_ATMO_H;
        rpc.vol_d         = WL_ATMO_D;
        rpc.layer_height  = s.ui_layer_height;
        vkCmdPushConstants(cmd, s.pipe_cloud.layout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(rpc), &rpc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }

    // ---- ImGui pass (skipped in preview mode — caller draws its own) ---
    if (!s.preview_mode) {
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
    }

    if (!s.preview_mode) {
        image_barrier(cmd, r.swapchain_images[image_index],
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }

    (void)s.brushing;
    (void)s.wt_water_pulse_active;
}

// ===========================================================================
// world_lab_shutdown
// ===========================================================================
void world_lab_shutdown(WorldLabState& s, Renderer& r)
{
    VkDevice device    = r.device;
    VmaAllocator alloc = r.allocator;

    vkDeviceWaitIdle(device);

    for (int k = 0; k < bestiary::PLANT_KIND_COUNT; ++k) {
        destroy_plant_mesh(alloc, s.plant_canonical[k]);
        destroy_buffer(alloc, s.plant_inst[k]);
    }
    destroy_plant_mesh(alloc, s.creature_mesh_gpu);
    destroy_buffer(alloc, s.terrain_mesh.vbo);
    destroy_buffer(alloc, s.terrain_mesh.ibo);

    destroy_cloud_pipeline(device, s.pipe_cloud);
    vmaDestroyBuffer(alloc, s.camera_ubo, s.camera_ubo_alloc);
    destroy_wl_creature_pipeline(device, s.pipe_creature);
    destroy_wl_clump_pipeline(device, s.pipe_clump);
    destroy_terrain_pipeline(device, s.pipe_terrain);
    destroy_compute_pipeline(device, s.pipe_erosion);
    destroy_compute_pipeline(device, s.pipe_atmo);
    destroy_compute_pipeline(device, s.pipe_brush);
    destroy_compute_pipeline(device, s.pipe_swe_step);
    destroy_compute_pipeline(device, s.pipe_swe_init);

    vkDestroyDescriptorPool(device, s.desc_pool, nullptr);
    vkDestroySampler(device, s.sampler, nullptr);

    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        destroy_buffer(alloc, s.wind_readback[i]);
        destroy_buffer(alloc, s.gcond_readback[i]);
    }
    for (int i = 0; i < 2; ++i) {
        destroy_swe_image(device, alloc, s.sediment[i]);
        destroy_swe_image(device, alloc, s.wind_field[i]);
        destroy_swe_image(device, alloc, s.atmo_state[i]);
    }
    destroy_swe_image(device, alloc, s.atmo_shadow);
    destroy_swe_image(device, alloc, s.ground_cond);
    destroy_swe_image(device, alloc, s.ground_wind);

    destroy_swe_image(device, alloc, s.water_out);
    destroy_swe_image(device, alloc, s.state_b);
    destroy_swe_image(device, alloc, s.state_a);

    vkDestroyImageView(device, s.moist_gpu.view, nullptr);
    vmaDestroyImage(alloc, s.moist_gpu.image, s.moist_gpu.allocation);
    vkDestroyImageView(device, s.hm_gpu.view, nullptr);
    vmaDestroyImage(alloc, s.hm_gpu.image, s.hm_gpu.allocation);

    s.initialized = false;
}

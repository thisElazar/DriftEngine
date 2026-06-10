// planet_dev.cpp — PLANET (DEV) module implementation.
//
// 2x2 World-Lab-scale tiles with cross-tile water (the planet's own
// planet_swe_step compute shader, flat) and a single world-spanning plant
// population. See planet_dev.h for the architecture notes.

#include "planet_dev.h"

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
// World-space coordinate helpers
// ---------------------------------------------------------------------------
static float pd_world_to_local_gx(float wx, uint32_t t) { return (wx - pd_tile_origin_x(t)) / PD_DX - 0.5f; }
static float pd_world_to_local_gy(float wz, uint32_t t) { return (wz - pd_tile_origin_z(t)) / PD_DX - 0.5f; }

// ---------------------------------------------------------------------------
// Continuous analytic terrain over WORLD coordinates. Evaluated per tile into
// the heightmap pool; because it is a pure function of (wx, wz) the tiles
// agree exactly at the seams. Features deliberately span tile boundaries:
// the central depression sits on the 4-corner point at (0,0).
// ---------------------------------------------------------------------------
static float pd_height(float wx, float wz)
{
    float r = std::sqrt(wx * wx + wz * wz);

    // World rim rises at the edges (one big bowl across all 4 tiles).
    float h = -4.0f + 0.00008f * r * r;

    // Ridge running NW-SE through the world, crossing both seams.
    float ridge_dist = std::abs(wx * 0.7f + wz * 0.7f);
    h += 5.0f * std::exp(-ridge_dist * ridge_dist / 7000.0f);

    // Central depression at the 4-corner point — the canonical seam test:
    // water poured anywhere nearby should pool symmetrically across tiles.
    float pool_r2 = wx * wx + wz * wz;
    h -= 4.0f * std::exp(-pool_r2 / 3000.0f);

    // Spring hill in the NW tile.
    float sp_dx = wx - (-120.0f);
    float sp_dz = wz - (-90.0f);
    float sp_r2 = sp_dx * sp_dx + sp_dz * sp_dz;
    h += 6.0f * std::exp(-sp_r2 / 4000.0f);

    // Gentle large-scale variation so the tiles aren't featureless.
    h += 1.2f * std::sin(wx * 0.020f) * std::cos(wz * 0.017f);
    h += 0.5f * std::sin(wx * 0.061f + 1.7f) * std::sin(wz * 0.053f + 0.6f);

    return h;
}

// ---------------------------------------------------------------------------
// GPU helpers (file-local)
// ---------------------------------------------------------------------------
static PD_ArrayImage create_array_image(VkDevice device, VmaAllocator alloc,
                                        VkFormat format, VkImageUsageFlags usage)
{
    VkImageCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {PD_GRID, PD_GRID, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = PD_TILE_COUNT;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    PD_ArrayImage img{};
    VK_CHECK(vmaCreateImage(alloc, &ci, &ai, &img.image, &img.alloc, nullptr));

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = img.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PD_TILE_COUNT};
    VK_CHECK(vkCreateImageView(device, &vci, nullptr, &img.view));
    return img;
}

static void destroy_array_image(VkDevice device, VmaAllocator alloc, PD_ArrayImage& img)
{
    vkDestroyImageView(device, img.view, nullptr);
    vmaDestroyImage(alloc, img.image, img.alloc);
    img = {};
}

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

static void write_buffer(VkDevice device, VkDescriptorSet set, uint32_t binding,
                         VkBuffer buf, VkDeviceSize range)
{
    VkDescriptorBufferInfo info{};
    info.buffer = buf;
    info.range  = range;
    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = set;
    w.dstBinding      = binding;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo     = &info;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
}

static void compute_to_graphics_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                     | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

// ---------------------------------------------------------------------------
// Graphics pipeline — tiled terrain (clone of world_lab's, array-sampling
// shaders + PlanetDevTerrainPC)
// ---------------------------------------------------------------------------
static PD_TerrainPipeline create_pd_terrain_pipeline(VkDevice device)
{
    PD_TerrainPipeline p{};
    p.vs = make_shader(device, "shaders/planet_dev_terrain_vs.spv");
    p.fs = make_shader(device, "shaders/planet_dev_terrain_fs.spv");

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
    push.size       = sizeof(PlanetDevTerrainPC);

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

static void destroy_pd_terrain_pipeline(VkDevice device, PD_TerrainPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.dsl, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
}

// ---------------------------------------------------------------------------
// Graphics pipeline — clump (verbatim world_lab pattern; instances are world
// coords so the existing world_clump shaders work unchanged)
// ---------------------------------------------------------------------------
static PD_ClumpPipeline create_pd_clump_pipeline(VkDevice device)
{
    PD_ClumpPipeline p{};
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
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, position)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, normal)};
    attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, color)};
    attrs[3] = {3, 0, VK_FORMAT_R32_SFLOAT,       offsetof(bestiary::VegetationVertex, height_t)};
    attrs[4] = {4, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::PlantGPUInstance, x)};
    attrs[5] = {5, 1, VK_FORMAT_R32_SFLOAT,       offsetof(bestiary::PlantGPUInstance, health)};

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

static void destroy_pd_clump_pipeline(VkDevice device, PD_ClumpPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
}

// ---------------------------------------------------------------------------
// Graphics pipeline — creatures (non-instanced, pre-baked world-space
// vertices; clone of world_lab's creature pipeline)
// ---------------------------------------------------------------------------
static PD_ClumpPipeline create_pd_creature_pipeline(VkDevice device)
{
    PD_ClumpPipeline p{};
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
    vi.pVertexAttributeDescriptions    = &attrs[0];

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

// ---------------------------------------------------------------------------
// Plant roster (copied from world_lab; extraction to apps/shared is a
// follow-up once the seam settles with two consumers)
// ---------------------------------------------------------------------------
static const char* pd_species_dir() { return "species"; }

static bestiary::SpeciesSuitability suit_for_kind(int kind,
                                                  const bestiary::EcosystemParams& e)
{
    switch (kind) {
    case bestiary::PLANT_BUSH:       return e.bush_suit;
    case bestiary::PLANT_TREE:       return e.tree_suit;
    case bestiary::PLANT_REED:       return e.reed_suit;
    case bestiary::PLANT_WILDFLOWER: return e.wildflower_suit;
    case bestiary::PLANT_LPLANT:     return e.tree_suit;
    default:                         return e.grass_suit;
    }
}

static void load_plant_profiles(PlanetDevState& s)
{
    namespace fs = std::filesystem;
    auto dir = fs::path(pd_species_dir());
    std::string name;
    auto p = dir / "world_grass.toml";
    if (fs::exists(p)) bestiary::load_clump(p, s.clump_params, name, &s.clump_expr);
    p = dir / "world_reed.toml";
    if (fs::exists(p)) bestiary::load_clump(p, s.reed_params, name, &s.reed_expr);
    p = dir / "world_bush.toml";
    if (fs::exists(p)) bestiary::load_bush(p, s.bush_params, name, &s.bush_expr);
    p = dir / "world_tree.toml";
    if (fs::exists(p)) bestiary::load_tree(p, s.tree_params, name, &s.tree_expr);
    p = dir / "world_wildflower.toml";
    if (fs::exists(p)) bestiary::load_wildflower(p, s.wildflower_params, name, &s.wildflower_expr);
    p = dir / "world_lplant.toml";
    if (fs::exists(p)) bestiary::load_lplant(p, s.lplant_params, name, &s.lplant_expr);
}

static void build_plant_roster(PlanetDevState& s)
{
    using namespace bestiary;
    s.plant_roster.clear();

    auto add = [&](std::string name, int kind,
                   std::function<VegetationMesh(float, uint32_t, float, float)> gen) {
        PlantSpecies sp;
        sp.name = std::move(name);
        sp.kind = kind;
        sp.suit = suit_for_kind(kind, s.eco_params);
        sp.gen  = std::move(gen);
        s.plant_roster.push_back(std::move(sp));
    };

    add("world_grass", PLANT_GRASS,
        [cp = s.clump_params, ce = s.clump_expr](float m, uint32_t sd, float x, float z) {
            return generate_clump(evaluate_expression(cp, ce, m), sd, false, x, z); });
    add("world_bush", PLANT_BUSH,
        [bp = s.bush_params, be = s.bush_expr](float m, uint32_t sd, float x, float z) {
            return generate_bush(evaluate_bush_expression(bp, be, m), sd, false, x, z); });
    add("world_tree", PLANT_TREE,
        [tp = s.tree_params, te = s.tree_expr](float m, uint32_t sd, float x, float z) {
            return generate_tree(evaluate_tree_expression(tp, te, m), sd, false, x, z); });
    add("world_reed", PLANT_REED,
        [rp = s.reed_params, re = s.reed_expr](float m, uint32_t sd, float x, float z) {
            return generate_clump(evaluate_expression(rp, re, m), sd, false, x, z); });
    add("world_wildflower", PLANT_WILDFLOWER,
        [wp = s.wildflower_params, we = s.wildflower_expr](float m, uint32_t sd, float x, float z) {
            return generate_wildflower(evaluate_wildflower_expression(wp, we, m), sd, false, x, z); });
    add("world_lplant", PLANT_LPLANT,
        [lp = s.lplant_params, le = s.lplant_expr](float m, uint32_t sd, float x, float z) {
            return generate_lplant(evaluate_lplant_expression(lp, le, m), sd, false, x, z); });

    static const char* house[] = {"world_grass", "world_bush", "world_tree",
                                  "world_reed", "world_wildflower", "world_lplant"};
    auto is_house = [&](const std::string& stem) {
        for (const char* h : house) if (stem == h) return true;
        return false;
    };

    namespace fs = std::filesystem;
    auto dir = fs::path(pd_species_dir());
    if (!fs::exists(dir)) return;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".toml") continue;
        std::string stem = entry.path().stem().string();
        if (is_house(stem)) continue;
        std::string kind = detect_species_kind(entry.path());
        std::string nm;
        if (kind == "grass_clump") {
            ClumpParams p; ClumpExpression ex;
            if (load_clump(entry.path(), p, nm, &ex))
                add(stem, PLANT_GRASS, [p, ex](float m, uint32_t sd, float x, float z) {
                    return generate_clump(evaluate_expression(p, ex, m), sd, false, x, z); });
        } else if (kind == "reed") {
            ClumpParams p; ClumpExpression ex;
            if (load_clump(entry.path(), p, nm, &ex))
                add(stem, PLANT_REED, [p, ex](float m, uint32_t sd, float x, float z) {
                    return generate_clump(evaluate_expression(p, ex, m), sd, false, x, z); });
        } else if (kind == "bush") {
            BushParams p; BushExpression ex;
            if (load_bush(entry.path(), p, nm, &ex))
                add(stem, PLANT_BUSH, [p, ex](float m, uint32_t sd, float x, float z) {
                    return generate_bush(evaluate_bush_expression(p, ex, m), sd, false, x, z); });
        } else if (kind == "tree") {
            TreeParams p; TreeExpression ex;
            if (load_tree(entry.path(), p, nm, &ex))
                add(stem, PLANT_TREE, [p, ex](float m, uint32_t sd, float x, float z) {
                    return generate_tree(evaluate_tree_expression(p, ex, m), sd, false, x, z); });
        } else if (kind == "wildflower") {
            WildflowerParams p; WildflowerExpression ex;
            if (load_wildflower(entry.path(), p, nm, &ex))
                add(stem, PLANT_WILDFLOWER, [p, ex](float m, uint32_t sd, float x, float z) {
                    return generate_wildflower(evaluate_wildflower_expression(p, ex, m), sd, false, x, z); });
        } else if (kind == "lplant") {
            LPlantParams p; LPlantExpression ex;
            if (load_lplant(entry.path(), p, nm, &ex))
                add(stem, PLANT_LPLANT, [p, ex](float m, uint32_t sd, float x, float z) {
                    return generate_lplant(evaluate_lplant_expression(p, ex, m), sd, false, x, z); });
        }
    }

    std::fprintf(stderr, "[planet_dev] Plant roster: %zu species\n", s.plant_roster.size());
}

static void destroy_pd_plant_mesh(VmaAllocator alloc, PD_PlantMesh& m)
{
    destroy_buffer(alloc, m.vbo);
    destroy_buffer(alloc, m.ibo);
    m = {};
}

static void upload_pd_mesh(VmaAllocator alloc, PD_PlantMesh& dst,
                           const bestiary::VegetationMesh& src)
{
    VkDeviceSize vbs = src.vertices.size() * sizeof(bestiary::VegetationVertex);
    VkDeviceSize ibs = src.indices.size() * sizeof(uint32_t);
    dst.vbo = create_host_buffer(alloc, vbs, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    dst.ibo = create_host_buffer(alloc, ibs, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    dst.index_count = static_cast<uint32_t>(src.indices.size());

    void* mapped = nullptr;
    VK_CHECK(vmaMapMemory(alloc, dst.vbo.allocation, &mapped));
    std::memcpy(mapped, src.vertices.data(), vbs);
    vmaUnmapMemory(alloc, dst.vbo.allocation);
    VK_CHECK(vmaMapMemory(alloc, dst.ibo.allocation, &mapped));
    std::memcpy(mapped, src.indices.data(), ibs);
    vmaUnmapMemory(alloc, dst.ibo.allocation);
}

// World-routed terrain height (creatures and their meshes query world coords).
static float pd_terrain_height_world(const PlanetDevState& s, float x, float z)
{
    uint32_t t = pd_tile_of(x, z);
    return sample_hm_bilinear(s.hm_cpu[t], PD_GRID, PD_GRID,
                              pd_world_to_local_gx(x, t),
                              pd_world_to_local_gy(z, t));
}

// ---------------------------------------------------------------------------
// Creatures: whole species library as the roster, spawned across the WHOLE
// 2x2 world — agents wander over tile seams because, in world coordinates,
// there is nothing there.
// ---------------------------------------------------------------------------
static void load_creature_profiles(PlanetDevState& s)
{
    auto loaded = bestiary::load_creature_dir(pd_species_dir());
    std::sort(loaded.begin(), loaded.end(),
              [](const bestiary::NamedCreatureProfile& a,
                 const bestiary::NamedCreatureProfile& b) { return a.name < b.name; });
    s.creature_profiles.clear();
    s.creature_names.clear();
    for (auto& ncp : loaded) {
        s.creature_profiles.push_back(ncp.profile);
        s.creature_names.push_back(ncp.name);
    }
    std::fprintf(stderr, "[planet_dev] Creature roster: %zu species\n",
                 s.creature_profiles.size());
}

static void spawn_all_creatures(PlanetDevState& s)
{
    s.agents.clear();
    if (s.creature_profiles.empty()) return;

    auto archetype_weight = [](bestiary::Archetype a) -> float {
        switch (a) {
        case bestiary::Archetype::Herbivore: return 3.0f;
        case bestiary::Archetype::Rabbit:    return 3.0f;
        case bestiary::Archetype::Bird:      return 2.0f;
        case bestiary::Archetype::Predator:  return 1.0f;
        case bestiary::Archetype::Raptor:    return 1.0f;
        case bestiary::Archetype::Snake:     return 1.0f;
        }
        return 1.0f;
    };

    float wsum = 0.0f;
    for (const auto& p : s.creature_profiles) wsum += archetype_weight(p.archetype);
    if (wsum < 1e-4f) wsum = 1.0f;

    int total = s.ui_creature_count;
    for (size_t i = 0; i < s.creature_profiles.size(); ++i) {
        float w = archetype_weight(s.creature_profiles[i].archetype);
        int count = static_cast<int>(std::round(static_cast<float>(total) * w / wsum));
        if (count < 1) count = 1;
        bestiary::spawn_creatures(s.agents, static_cast<uint16_t>(i), count,
            s.persistent_env, PD_WORLD_HALF, PD_WORLD_HALF,
            42u + static_cast<uint32_t>(i) * 101u);
    }
}

static void build_canonical_meshes(PlanetDevState& s, VmaAllocator alloc)
{
    for (auto& m : s.plant_canonical) destroy_pd_plant_mesh(alloc, m);
    s.plant_canonical.assign(s.plant_roster.size(), {});
    for (size_t i = 0; i < s.plant_roster.size(); ++i) {
        if (!s.plant_roster[i].gen) continue;
        bestiary::VegetationMesh mesh = s.plant_roster[i].gen(0.5f, 0, 0.0f, 0.0f);
        if (mesh.vertices.empty()) continue;

        PD_PlantMesh& dst = s.plant_canonical[i];
        VkDeviceSize vbs = mesh.vertices.size() * sizeof(bestiary::VegetationVertex);
        VkDeviceSize ibs = mesh.indices.size() * sizeof(uint32_t);
        dst.vbo = create_host_buffer(alloc, vbs, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        dst.ibo = create_host_buffer(alloc, ibs, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        dst.index_count = static_cast<uint32_t>(mesh.indices.size());

        void* mapped = nullptr;
        VK_CHECK(vmaMapMemory(alloc, dst.vbo.allocation, &mapped));
        std::memcpy(mapped, mesh.vertices.data(), vbs);
        vmaUnmapMemory(alloc, dst.vbo.allocation);
        VK_CHECK(vmaMapMemory(alloc, dst.ibo.allocation, &mapped));
        std::memcpy(mapped, mesh.indices.data(), ibs);
        vmaUnmapMemory(alloc, dst.ibo.allocation);
    }
}

// ---------------------------------------------------------------------------
// Plant instance upload (world coords; terrain height routed by tile)
// ---------------------------------------------------------------------------
static void upload_plant_instances(PlanetDevState& s, VkDevice device, VmaAllocator alloc)
{
    constexpr float plant_lift = 0.01f;
    std::vector<bestiary::PlantGPUInstance> buf;
    vkDeviceWaitIdle(device);   // world_lab wart, inherited knowingly (0.5 s cadence)

    for (auto& b : s.plant_inst) destroy_buffer(alloc, b);
    s.plant_inst.assign(s.plant_roster.size(), GpuBuffer{});
    s.plant_inst_count.assign(s.plant_roster.size(), 0u);

    for (size_t k = 0; k < s.plant_roster.size(); ++k) {
        buf.clear();
        for (const auto& p : s.plant_population) {
            if (p.species != static_cast<int>(k) || p.health <= 0.0f) continue;
            uint32_t t = pd_tile_of(p.x, p.z);
            float terrain_y = sample_hm_bilinear(s.hm_cpu[t], PD_GRID, PD_GRID,
                                                 pd_world_to_local_gx(p.x, t),
                                                 pd_world_to_local_gy(p.z, t));
            buf.push_back({p.x, terrain_y + plant_lift, p.z, p.health, p.seed});
        }
        if (buf.empty()) continue;
        VkDeviceSize sz = buf.size() * sizeof(bestiary::PlantGPUInstance);
        s.plant_inst[k] = create_host_buffer(alloc, sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        void* mapped = nullptr;
        VK_CHECK(vmaMapMemory(alloc, s.plant_inst[k].allocation, &mapped));
        std::memcpy(mapped, buf.data(), sz);
        vmaUnmapMemory(alloc, s.plant_inst[k].allocation);
        s.plant_inst_count[k] = static_cast<uint32_t>(buf.size());
    }
}

// ---------------------------------------------------------------------------
// Environment refresh: stitch the per-tile water readbacks into ONE world
// grid, blur globally (so moisture crosses tile seams exactly like the water
// does), then slice back per tile for the GPU overlay.
// ---------------------------------------------------------------------------
constexpr uint32_t PD_WORLD_GRID = PD_GRID * PD_TILES_X;   // 512

static size_t pd_world_cell(float wx, float wz)
{
    int wi = std::clamp(static_cast<int>(wx + PD_WORLD_HALF), 0, static_cast<int>(PD_WORLD_GRID) - 1);
    int wj = std::clamp(static_cast<int>(wz + PD_WORLD_HALF), 0, static_cast<int>(PD_WORLD_GRID) - 1);
    return static_cast<size_t>(wj) * PD_WORLD_GRID + wi;
}

static void refresh_env(PlanetDevState& s, VkDevice device, VmaAllocator alloc,
                        VkQueue gqueue, uint32_t gfamily)
{
    VkImage fresh = (s.swe_ping_pong & 1) ? s.swe_state_b.image : s.swe_state_a.image;

    s.water_world.assign(static_cast<size_t>(PD_WORLD_GRID) * PD_WORLD_GRID, 0.0f);
    for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
        std::vector<float> tile = readback_water_depth(device, alloc, gqueue, gfamily,
                                                       fresh, PD_GRID, PD_GRID, t);
        size_t oxg = (t % PD_TILES_X) * PD_GRID;
        size_t ozg = (t / PD_TILES_X) * PD_GRID;
        for (uint32_t j = 0; j < PD_GRID; ++j)
            std::memcpy(&s.water_world[(ozg + j) * PD_WORLD_GRID + oxg],
                        &tile[static_cast<size_t>(j) * PD_GRID],
                        PD_GRID * sizeof(float));
    }

    s.moisture_world = build_moisture_grid(s.water_world, PD_WORLD_GRID, PD_WORLD_GRID,
                                           s.ui_capillary_depth, s.ui_capillary_blur);

    for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
        size_t oxg = (t % PD_TILES_X) * PD_GRID;
        size_t ozg = (t / PD_TILES_X) * PD_GRID;
        s.moisture_cpu[t].resize(static_cast<size_t>(PD_GRID) * PD_GRID);
        for (uint32_t j = 0; j < PD_GRID; ++j)
            std::memcpy(&s.moisture_cpu[t][static_cast<size_t>(j) * PD_GRID],
                        &s.moisture_world[(ozg + j) * PD_WORLD_GRID + oxg],
                        PD_GRID * sizeof(float));
        update_r32_image(device, alloc, gqueue, gfamily,
                         s.moisture_pool.image, s.moisture_cpu[t], PD_GRID, PD_GRID, t);
    }

    // World-coordinate env field straight off the world grids.
    s.persistent_env.sample = [&s](float x, float z) -> bestiary::EnvironmentSample {
        float m = s.moisture_world.empty() ? 0.0f : s.moisture_world[pd_world_cell(x, z)];
        uint32_t t = pd_tile_of(x, z);
        int ix = std::clamp(static_cast<int>(x - pd_tile_origin_x(t)), 0, static_cast<int>(PD_GRID) - 1);
        int iz = std::clamp(static_cast<int>(z - pd_tile_origin_z(t)), 0, static_cast<int>(PD_GRID) - 1);
        float h = s.hm_cpu[t][static_cast<size_t>(iz) * PD_GRID + ix];
        float temp = std::clamp(0.5f + 0.05f * h, 0.0f, 1.0f);
        return {m, temp};
    };
}

// Per-tile env wrapper: place_ecosystem/sprout_plants generate coords in a
// region centered at the ORIGIN, so per-tile calls sample through this offset
// and the returned instances get the tile center added back.
static bestiary::EnvironmentField tile_env(const PlanetDevState& s, uint32_t t)
{
    float cx = pd_tile_origin_x(t) + PD_TILE_SIZE * 0.5f;
    float cz = pd_tile_origin_z(t) + PD_TILE_SIZE * 0.5f;
    bestiary::EnvironmentField env;
    env.sample = [&s, cx, cz](float x, float z) {
        return s.persistent_env(x + cx, z + cz);
    };
    return env;
}

static void run_replant(PlanetDevState& s, VkDevice device, VmaAllocator alloc,
                        VkQueue gqueue, uint32_t gfamily)
{
    refresh_env(s, device, alloc, gqueue, gfamily);

    s.plant_population.clear();
    for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
        bestiary::EcosystemParams eco = s.eco_params;
        eco.region_size = PD_TILE_SIZE;
        eco.seed = s.eco_params.seed + t * 7919u;
        auto placed = bestiary::place_ecosystem(s.plant_roster, eco, tile_env(s, t));
        float cx = pd_tile_origin_x(t) + PD_TILE_SIZE * 0.5f;
        float cz = pd_tile_origin_z(t) + PD_TILE_SIZE * 0.5f;
        for (auto& p : placed) { p.x += cx; p.z += cz; }
        s.plant_population.insert(s.plant_population.end(), placed.begin(), placed.end());
    }

    upload_plant_instances(s, device, alloc);
    s.veg_rebuild_timer = 0.0f;
}

// ---------------------------------------------------------------------------
// Ray vs y=0 plane -> world coords
// ---------------------------------------------------------------------------
static bool pick_world(const glm::mat4& view, const glm::mat4& proj,
                       glm::vec2 cursor_ndc, float& out_wx, float& out_wz)
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
    out_wx = hit.x;
    out_wz = hit.z;
    return out_wx > -PD_WORLD_HALF && out_wx < PD_WORLD_HALF
        && out_wz > -PD_WORLD_HALF && out_wz < PD_WORLD_HALF;
}

// ===========================================================================
// planet_dev_init
// ===========================================================================
void planet_dev_init(PlanetDevState& s, Renderer& r)
{
    VkDevice device    = r.device;
    VmaAllocator alloc = r.allocator;
    VkQueue gqueue     = r.graphics_queue;
    uint32_t gfamily   = r.gfx_family;

    s.camera.yaw      = 0.6f;
    s.camera.pitch    = 0.7f;
    s.camera.distance = 380.0f;
    s.camera.target   = {0.0f, 0.0f, 0.0f};

    // ---- Pools (one array layer per tile) -----------------------------------
    s.terrain_pool = create_array_image(device, alloc, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
        | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    s.swe_state_a = create_array_image(device, alloc, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    s.swe_state_b = create_array_image(device, alloc, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    s.swe_output = create_array_image(device, alloc, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    s.moisture_pool = create_array_image(device, alloc, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    // Transition everything to GENERAL, clearing the water/moisture pools.
    // (No SWE init pipeline needed: the planet init shader's no-stamp path
    // reduces to zero water with the ocean off, which a clear already is.)
    {
        OneShot os = oneshot_begin(device, gqueue, gfamily);
        VkImage clear_imgs[] = {s.swe_state_a.image, s.swe_state_b.image,
                                s.swe_output.image, s.moisture_pool.image};
        for (VkImage img : clear_imgs)
            image_barrier(os.cmd, img,
                          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          0, PD_TILE_COUNT);
        VkClearColorValue clr{};
        VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, PD_TILE_COUNT};
        for (VkImage img : clear_imgs)
            vkCmdClearColorImage(os.cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clr, 1, &rng);
        for (VkImage img : clear_imgs)
            image_barrier(os.cmd, img,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                          0, PD_TILE_COUNT);
        // Terrain pool straight to GENERAL; update_r32_image uploads per layer.
        image_barrier(os.cmd, s.terrain_pool.image,
                      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                      0, PD_TILE_COUNT);
        oneshot_end(os);
    }

    // ---- Heightmaps: one continuous analytic function, four tiles -----------
    for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
        s.hm_cpu[t].assign(static_cast<size_t>(PD_GRID) * PD_GRID, 0.0f);
        float ox = pd_tile_origin_x(t), oz = pd_tile_origin_z(t);
        for (uint32_t gy = 0; gy < PD_GRID; ++gy)
            for (uint32_t gx = 0; gx < PD_GRID; ++gx)
                s.hm_cpu[t][static_cast<size_t>(gy) * PD_GRID + gx] =
                    pd_height(ox + (gx + 0.5f) * PD_DX, oz + (gy + 0.5f) * PD_DX);
        update_r32_image(device, alloc, gqueue, gfamily,
                         s.terrain_pool.image, s.hm_cpu[t], PD_GRID, PD_GRID, t);
    }

    // ---- Edge flags buffer (debug readout) ----------------------------------
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = PD_TILE_COUNT * sizeof(uint32_t);
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                 | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VK_CHECK(vmaCreateBuffer(alloc, &bci, &ai,
            &s.edge_flags_buf, &s.edge_flags_alloc, &s.edge_flags_info));
    }

    // ---- Sampler ------------------------------------------------------------
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

    // ---- Compute pipeline: the planet's own SWE step, reused flat -----------
    s.pipe_swe_step = make_compute_pipeline(device, "shaders/planet_swe_step_cs.spv",
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,    // 0: terrain pool (array)
         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,    // 1: state in (array)
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,    // 2: state out (array)
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,    // 3: render output (array)
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},  // 4: edge flags
        sizeof(PlanetSweStepPC));

    s.pipe_brush = make_compute_pipeline(device, "shaders/planet_dev_brush_cs.spv",
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        sizeof(PD_BrushPC));

    // ---- Descriptor pool + sets ----------------------------------------------
    {
        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          8},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          8},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         4},
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = 8;
        pci.poolSizeCount = 4;
        pci.pPoolSizes    = pool_sizes;
        VK_CHECK(vkCreateDescriptorPool(device, &pci, nullptr, &s.desc_pool));
    }

    s.ds_step[0] = alloc_set(device, s.desc_pool, s.pipe_swe_step.dsl);
    s.ds_step[1] = alloc_set(device, s.desc_pool, s.pipe_swe_step.dsl);
    s.ds_brush   = alloc_set(device, s.desc_pool, s.pipe_brush.dsl);
    write_image(device, s.ds_brush, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.terrain_pool.view, VK_NULL_HANDLE);

    // set 0: read A -> write B; set 1: read B -> write A
    for (int i = 0; i < 2; ++i) {
        VkImageView in_view  = (i == 0) ? s.swe_state_a.view : s.swe_state_b.view;
        VkImageView out_view = (i == 0) ? s.swe_state_b.view : s.swe_state_a.view;
        write_image(device, s.ds_step[i], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, s.terrain_pool.view, VK_NULL_HANDLE);
        write_image(device, s.ds_step[i], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, in_view,  VK_NULL_HANDLE);
        write_image(device, s.ds_step[i], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, out_view, VK_NULL_HANDLE);
        write_image(device, s.ds_step[i], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s.swe_output.view, VK_NULL_HANDLE);
        write_buffer(device, s.ds_step[i], 4, s.edge_flags_buf, PD_TILE_COUNT * sizeof(uint32_t));
    }

    // ---- Graphics -------------------------------------------------------------
    s.pipe_terrain = create_pd_terrain_pipeline(device);
    s.pipe_clump   = create_pd_clump_pipeline(device);

    s.ds_terrain = alloc_set(device, s.desc_pool, s.pipe_terrain.dsl);
    write_image(device, s.ds_terrain, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s.terrain_pool.view,  s.sampler);
    write_image(device, s.ds_terrain, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s.swe_output.view,    s.sampler);
    write_image(device, s.ds_terrain, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, s.moisture_pool.view, s.sampler);

    s.terrain_mesh = build_grid_mesh(alloc, PD_GRID, PD_GRID);

    // ---- Plants ----------------------------------------------------------------
    s.eco_params.region_size = PD_TILE_SIZE;
    load_plant_profiles(s);
    build_plant_roster(s);
    build_canonical_meshes(s, alloc);
    run_replant(s, device, alloc, gqueue, gfamily);

    // ---- Creatures ---------------------------------------------------------------
    s.pipe_creature = create_pd_creature_pipeline(device);
    load_creature_profiles(s);
    spawn_all_creatures(s);

    s.initialized = true;
}

// ===========================================================================
// planet_dev_tick
// ===========================================================================
bool planet_dev_tick(PlanetDevState& s, Renderer& r, const InputFrame& in, float dt)
{
    VkDevice device    = r.device;
    VmaAllocator alloc = r.allocator;
    VkQueue gqueue     = r.graphics_queue;
    uint32_t gfamily   = r.gfx_family;

    s.last_dt = dt;
    float sim_dt = s.ui_paused ? 0.0f : dt;
    s.accumulated_time += sim_dt;

    // ---- Camera: orbit + WASD pans the target across the world --------------
    update_orbit(s.camera, in, 900.0f);
    s.camera.distance = std::max(s.camera.distance, 20.0f);
    if (!in.ui_wants_keyboard) {
        glm::vec3 fwd(-std::sin(s.camera.yaw), 0.0f, -std::cos(s.camera.yaw));
        glm::vec3 right(std::cos(s.camera.yaw), 0.0f, -std::sin(s.camera.yaw));
        float speed = 0.5f * s.camera.distance * dt;
        glm::vec3 move(0.0f);
        if (in.key_w) move += fwd;
        if (in.key_s) move -= fwd;
        if (in.key_d) move += right;
        if (in.key_a) move -= right;
        s.camera.target += move * speed;
        float margin = 32.0f;
        s.camera.target.x = std::clamp(s.camera.target.x, -PD_WORLD_HALF - margin, PD_WORLD_HALF + margin);
        s.camera.target.z = std::clamp(s.camera.target.z, -PD_WORLD_HALF - margin, PD_WORLD_HALF + margin);
    }

    // ---- Cursor pick ----------------------------------------------------------
    float aspect = (in.win_h > 0) ? static_cast<float>(in.win_w) / in.win_h : 1.0f;
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.5f, 3000.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 view = orbit_view(s.camera);
    glm::vec2 ndc(static_cast<float>(in.mouse_x) / in.win_w * 2.0f - 1.0f,
                  static_cast<float>(in.mouse_y) / in.win_h * 2.0f - 1.0f);
    float wx = 0.0f, wz = 0.0f;
    s.cursor_valid = pick_world(view, proj, ndc, wx, wz);
    if (s.cursor_valid) {
        s.cursor_wx = wx;
        s.cursor_wz = wz;
        s.cursor_tile = pd_tile_of(wx, wz);
        s.cursor_gx = (wx - pd_tile_origin_x(s.cursor_tile)) / PD_DX;
        s.cursor_gy = (wz - pd_tile_origin_z(s.cursor_tile)) / PD_DX;
    }
    s.brushing = s.cursor_valid && in.lmb && !in.ui_wants_mouse;

    // Globe-style brush mode keys: 1 = raise, 2 = lower, 3 = water.
    if (!in.ui_wants_keyboard) {
        if (in.brush_digit == 1) s.brush_mode = PD_BrushMode::Raise;
        if (in.brush_digit == 2) s.brush_mode = PD_BrushMode::Lower;
        if (in.brush_digit == 3) s.brush_mode = PD_BrushMode::Water;
    }

    // ---- Latch edge flags (the frame about to record will clear them) -------
    {
        const uint32_t* flags = static_cast<const uint32_t*>(s.edge_flags_info.pMappedData);
        for (uint32_t t = 0; t < PD_TILE_COUNT; ++t)
            s.edge_flags_ui[t] = flags[t];
    }

    // ---- Ecosystem cadence (the World Lab loop, on world coords) ------------
    if (sim_dt > 0.0f && !s.plant_roster.empty()) {
        bestiary::tick_plant_population(s.plant_population, s.plant_roster,
                                        s.persistent_env, sim_dt,
                                        s.plant_growth_rate, s.plant_decay_rate);

        s.veg_rebuild_timer += sim_dt;
        if (s.veg_rebuild_timer >= 0.5f) {
            s.veg_rebuild_timer = 0.0f;
            upload_plant_instances(s, device, alloc);
        }

        s.auto_replant_timer += sim_dt;
        if (s.auto_replant_timer >= 8.0f) {
            s.auto_replant_timer = 0.0f;
            refresh_env(s, device, alloc, gqueue, gfamily);
            for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
                bestiary::EcosystemParams eco = s.eco_params;
                eco.region_size = PD_TILE_SIZE;
                size_t before = s.plant_population.size();
                bestiary::sprout_plants(s.plant_population, s.plant_roster, eco,
                                        tile_env(s, t), s.sprout_seed++);
                float cx = pd_tile_origin_x(t) + PD_TILE_SIZE * 0.5f;
                float cz = pd_tile_origin_z(t) + PD_TILE_SIZE * 0.5f;
                for (size_t i = before; i < s.plant_population.size(); ++i) {
                    s.plant_population[i].x += cx;
                    s.plant_population[i].z += cz;
                }
            }
        }
    }

    if (s.replant_pending) {
        s.replant_pending = 0;
        run_replant(s, device, alloc, gqueue, gfamily);
    }
    if (s.respawn_pending) {
        s.respawn_pending = 0;
        spawn_all_creatures(s);
    }

    // ---- Creatures: world-spanning update + per-frame mesh rebuild -----------
    if (s.ui_creatures_enabled && !s.agents.empty() && sim_dt > 0.0f) {
        bestiary::CreatureWorldView world_view{};
        world_view.plant_population = &s.plant_population;
        world_view.env_field = &s.persistent_env;
        world_view.terrain_height = [&s](float x, float z) {
            return pd_terrain_height_world(s, x, z);
        };
        world_view.water_depth = [&s](float x, float z) {
            return s.water_world.empty() ? 0.0f : s.water_world[pd_world_cell(x, z)];
        };
        // The whole 2x2 world is one roaming range — tile seams don't exist
        // for agents; this is the cross-tile movement test.
        world_view.tile_half_x = PD_WORLD_HALF;
        world_view.tile_half_z = PD_WORLD_HALF;
        world_view.has_threat = s.brushing;
        world_view.threat_pos[0] = s.cursor_wx;
        world_view.threat_pos[1] = s.cursor_wz;
        world_view.threat_radius = static_cast<float>(s.ui_brush_radius) * PD_DX + 10.0f;

        bestiary::update_creatures(s.agents, s.creature_profiles, world_view,
                                   sim_dt * s.ui_creature_speed, s.creature_tick);
        ++s.creature_tick;

        auto cm = bestiary::generate_creature_meshes(
            s.agents, s.creature_profiles,
            [&s](float x, float z) { return pd_terrain_height_world(s, x, z); }, dt);

        vkDeviceWaitIdle(device);   // world_lab wart, inherited knowingly
        destroy_pd_plant_mesh(alloc, s.creature_mesh_gpu);
        if (!cm.vertices.empty())
            upload_pd_mesh(alloc, s.creature_mesh_gpu, cm);
    }

    // ---- UI -------------------------------------------------------------------
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool back = false;
    ImGui::Begin("PLANET (DEV)");
    if (s.embedded) {
        if (ImGui::Button("< Back")) back = true;
        ImGui::Separator();
    }
    ImGui::TextDisabled("2x2 tiles | planet SWE, flat | LMB = apply brush");
    ImGui::TextDisabled("WASD pan, RMB orbit, wheel zoom | 1=raise 2=lower 3=water");

    if (ImGui::CollapsingHeader("Brush", ImGuiTreeNodeFlags_DefaultOpen)) {
        int mode = static_cast<int>(s.brush_mode);
        ImGui::RadioButton("Raise", &mode, 0); ImGui::SameLine();
        ImGui::RadioButton("Lower", &mode, 1); ImGui::SameLine();
        ImGui::RadioButton("Water", &mode, 2);
        s.brush_mode = static_cast<PD_BrushMode>(mode);
        ImGui::SliderInt("Radius (cells)", &s.ui_brush_radius, 2, 40);
        ImGui::SliderFloat("Water strength", &s.ui_brush_strength, 0.1f, 10.0f);
        ImGui::SliderFloat("Terrain strength", &s.ui_terrain_strength, 0.5f, 20.0f);
    }

    if (ImGui::CollapsingHeader("Water", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Paused", &s.ui_paused);
        ImGui::SliderInt("Substeps", &s.ui_swe_substeps, 1, 8);
        ImGui::SliderFloat("dt", &s.ui_swe_dt, 0.01f, 0.1f);
        ImGui::SliderFloat("Gravity", &s.ui_gravity, 0.5f, 30.0f);
        ImGui::SliderFloat("Friction", &s.ui_friction, 0.0f, 0.2f);
        ImGui::SliderFloat("Damping", &s.ui_damping, 0.0f, 0.1f);
    }

    if (ImGui::CollapsingHeader("Plants", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show plants", &s.ui_show_plants);
        ImGui::Checkbox("Moisture overlay", &s.ui_show_moisture);
        ImGui::SliderFloat("Density", &s.eco_params.density_scale, 0.1f, 3.0f);
        ImGui::Text("Population: %zu", s.plant_population.size());
        if (ImGui::Button("Replant")) s.replant_pending = 1;
    }

    if (ImGui::CollapsingHeader("Creatures", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Creatures enabled", &s.ui_creatures_enabled);
        ImGui::SliderInt("Count", &s.ui_creature_count, 4, 100);
        ImGui::SliderFloat("Speed", &s.ui_creature_speed, 0.1f, 4.0f);
        ImGui::Text("Alive: %d / %zu agents (%zu species)",
                    bestiary::count_alive(s.agents), s.agents.size(),
                    s.creature_profiles.size());
        if (ImGui::Button("Respawn")) s.respawn_pending = 1;
    }

    if (ImGui::CollapsingHeader("Tiles (debug)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Seam highlight", &s.ui_seam_highlight);
        if (s.cursor_valid)
            ImGui::Text("Cursor: (%.0f, %.0f) -> tile %u cell (%.0f, %.0f)",
                        s.cursor_wx, s.cursor_wz, s.cursor_tile, s.cursor_gx, s.cursor_gy);
        else
            ImGui::TextDisabled("Cursor off-world");
        for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
            uint32_t f = s.edge_flags_ui[t];
            ImGui::Text("Tile %u edges: %s%s%s%s", t,
                        (f & 1u) ? "L " : "- ", (f & 2u) ? "R " : "- ",
                        (f & 4u) ? "D " : "- ", (f & 8u) ? "U" : "-");
        }
        ImGui::Text("Frame: %.2f ms", static_cast<double>(dt) * 1000.0);
    }
    ImGui::End();

    ImGui::Render();
    return !back;
}

// ===========================================================================
// planet_dev_render
// ===========================================================================
void planet_dev_render(PlanetDevState& s, Renderer& r,
                       FrameData& frame, uint32_t image_index, VkExtent2D extent)
{
    VkCommandBuffer cmd = frame.cmd;

    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.5f, 3000.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 view = orbit_view(s.camera);
    glm::mat4 mvp = proj * view;

    // ---- Terrain brush (raise/lower): world-space, applied to EVERY tile ----
    // with the brush expressed in tile-local coords; tiles the falloff misses
    // do no work, and edits straddling a seam stay seamless. CPU heightmaps
    // mirror the same falloff so plants/creatures/picking track the edit.
    bool terrain_brushing = s.brushing && s.brush_mode != PD_BrushMode::Water;
    if (terrain_brushing) {
        float sign = (s.brush_mode == PD_BrushMode::Raise) ? 1.0f : -1.0f;
        float amount = sign * s.ui_terrain_strength * std::min(s.last_dt, 0.033f);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe_brush.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                s.pipe_brush.layout, 0, 1, &s.ds_brush, 0, nullptr);
        for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
            PD_BrushPC bpc{};
            bpc.brush_x      = (s.cursor_wx - pd_tile_origin_x(t)) / PD_DX;
            bpc.brush_y      = (s.cursor_wz - pd_tile_origin_z(t)) / PD_DX;
            bpc.brush_radius = static_cast<float>(s.ui_brush_radius);
            bpc.brush_amount = amount;
            bpc.grid_w       = PD_GRID;
            bpc.grid_h       = PD_GRID;
            bpc.layer        = t;
            vkCmdPushConstants(cmd, s.pipe_brush.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(bpc), &bpc);
            vkCmdDispatch(cmd, (PD_GRID + 15) / 16, (PD_GRID + 15) / 16, 1);

            cpu_apply_brush(s.hm_cpu[t], PD_GRID, PD_GRID,
                            bpc.brush_x, bpc.brush_y, bpc.brush_radius, bpc.brush_amount);
        }
        compute_memory_barrier(cmd);
    }

    // ---- Clear edge flags, then run the SWE substeps over all 4 tiles -------
    vkCmdFillBuffer(cmd, s.edge_flags_buf, 0, VK_WHOLE_SIZE, 0u);
    {
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                         | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    if (!s.ui_paused) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe_swe_step.pipeline);
        for (int sub = 0; sub < s.ui_swe_substeps; ++sub) {
            int cur = s.swe_ping_pong & 1;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    s.pipe_swe_step.layout, 0, 1, &s.ds_step[cur], 0, nullptr);

            // Tiles within a substep are independent (they read the previous
            // substep's state, including neighbors'), so: dispatch all 4, then
            // one barrier before the next substep.
            for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
                PlanetSweStepPC pc{};
                pc.time       = s.accumulated_time;
                pc.dt         = s.ui_swe_dt;
                pc.gravity    = s.ui_gravity;
                pc.friction   = s.ui_friction;
                pc.dx         = PD_DX;
                pc.sea_level  = -10000.0f;     // static water off (world_lab convention)
                pc.damping    = s.ui_damping;
                pc.pool_index = t;
                pc.grid_w     = PD_GRID;
                pc.grid_h     = PD_GRID;
                // Water brush: same world-space treatment as the terrain brush
                // — every tile gets the pulse in its own local frame, so pours
                // straddling a seam fill both sides symmetrically.
                if (s.brushing && s.brush_mode == PD_BrushMode::Water) {
                    pc.pulse_x      = (s.cursor_wx - pd_tile_origin_x(t)) / PD_DX;
                    pc.pulse_y      = (s.cursor_wz - pd_tile_origin_z(t)) / PD_DX;
                    pc.pulse_radius = static_cast<float>(s.ui_brush_radius);
                    pc.pulse_amount = s.ui_brush_strength * s.ui_swe_dt;
                }
                pc.neighbor_left  = PD_NEIGHBORS[t][0];
                pc.neighbor_right = PD_NEIGHBORS[t][1];
                pc.neighbor_down  = PD_NEIGHBORS[t][2];
                pc.neighbor_up    = PD_NEIGHBORS[t][3];

                vkCmdPushConstants(cmd, s.pipe_swe_step.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, (PD_GRID + 7) / 8, (PD_GRID + 7) / 8, 1);
            }
            compute_memory_barrier(cmd);
            ++s.swe_ping_pong;
        }
    }

    // Edge flags -> host (read by next tick's latch)
    {
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
        mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    compute_to_graphics_barrier(cmd);

    // ---- Color + depth barriers, main pass -----------------------------------
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

    VkRenderingAttachmentInfo color{};
    color.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView   = r.swapchain_views[image_index];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{0.45f, 0.55f, 0.70f, 1.0f}};

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
    VkViewport vp{0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{{0, 0}, extent};
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // Terrain: the same grid mesh drawn once per tile.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipe_terrain.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            s.pipe_terrain.layout, 0, 1, &s.ds_terrain, 0, nullptr);
    VkDeviceSize zero = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &s.terrain_mesh.vbo.buffer, &zero);
    vkCmdBindIndexBuffer(cmd, s.terrain_mesh.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);

    for (uint32_t t = 0; t < PD_TILE_COUNT; ++t) {
        PlanetDevTerrainPC tpc{};
        tpc.mvp           = mvp;
        tpc.grid_w_f      = static_cast<float>(PD_GRID);
        tpc.grid_h_f      = static_cast<float>(PD_GRID);
        tpc.cell_size     = PD_DX;
        tpc.sea_level     = -10000.0f;
        tpc.brush_x       = s.cursor_gx;
        tpc.brush_y       = s.cursor_gy;
        tpc.brush_radius  = static_cast<float>(s.ui_brush_radius);
        tpc.brush_active  = (s.cursor_valid && t == s.cursor_tile) ? 1.0f : 0.0f;
        tpc.moisture_overlay = s.ui_show_moisture ? 1.0f : 0.0f;
        tpc.tile_origin_x = pd_tile_origin_x(t);
        tpc.tile_origin_z = pd_tile_origin_z(t);
        tpc.layer         = static_cast<float>(t);
        tpc.seam_highlight = s.ui_seam_highlight ? 1.0f : 0.0f;
        vkCmdPushConstants(cmd, s.pipe_terrain.layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(tpc), &tpc);
        vkCmdDrawIndexed(cmd, s.terrain_mesh.index_count, 1, 0, 0, 0);
    }

    // Plants (instanced, world coords — continuous across seams by construction)
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
        for (size_t k = 0; k < s.plant_inst.size(); ++k) {
            if (k >= s.plant_canonical.size()) break;
            if (s.plant_inst_count[k] == 0 || s.plant_canonical[k].index_count == 0) continue;
            VkBuffer     vbufs[2]   = {s.plant_canonical[k].vbo.buffer, s.plant_inst[k].buffer};
            VkDeviceSize offsets[2] = {0, 0};
            vkCmdBindVertexBuffers(cmd, 0, 2, vbufs, offsets);
            vkCmdBindIndexBuffer(cmd, s.plant_canonical[k].ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, s.plant_canonical[k].index_count,
                             s.plant_inst_count[k], 0, 0, 0);
        }
    }

    // Creatures (pre-baked world-space mesh, one draw)
    if (s.ui_creatures_enabled && s.creature_mesh_gpu.index_count > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipe_creature.pipeline);
        vkCmdPushConstants(cmd, s.pipe_creature.layout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(glm::mat4), &mvp);
        vkCmdBindVertexBuffers(cmd, 0, 1, &s.creature_mesh_gpu.vbo.buffer, &zero);
        vkCmdBindIndexBuffer(cmd, s.creature_mesh_gpu.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, s.creature_mesh_gpu.index_count, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);

    // ---- ImGui pass ------------------------------------------------------------
    {
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

    image_barrier(cmd, r.swapchain_images[image_index],
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

// ===========================================================================
// planet_dev_shutdown
// ===========================================================================
void planet_dev_shutdown(PlanetDevState& s, Renderer& r)
{
    VkDevice device    = r.device;
    VmaAllocator alloc = r.allocator;

    vkDeviceWaitIdle(device);

    for (auto& m : s.plant_canonical) destroy_pd_plant_mesh(alloc, m);
    for (auto& b : s.plant_inst)      destroy_buffer(alloc, b);
    s.plant_canonical.clear();
    s.plant_inst.clear();
    s.plant_inst_count.clear();
    destroy_pd_plant_mesh(alloc, s.creature_mesh_gpu);
    destroy_buffer(alloc, s.terrain_mesh.vbo);
    destroy_buffer(alloc, s.terrain_mesh.ibo);

    destroy_pd_clump_pipeline(device, s.pipe_creature);
    destroy_pd_clump_pipeline(device, s.pipe_clump);
    destroy_pd_terrain_pipeline(device, s.pipe_terrain);
    destroy_compute_pipeline(device, s.pipe_brush);
    destroy_compute_pipeline(device, s.pipe_swe_step);

    vkDestroyDescriptorPool(device, s.desc_pool, nullptr);
    vkDestroySampler(device, s.sampler, nullptr);

    vmaDestroyBuffer(alloc, s.edge_flags_buf, s.edge_flags_alloc);

    destroy_array_image(device, alloc, s.moisture_pool);
    destroy_array_image(device, alloc, s.swe_output);
    destroy_array_image(device, alloc, s.swe_state_b);
    destroy_array_image(device, alloc, s.swe_state_a);
    destroy_array_image(device, alloc, s.terrain_pool);

    s.initialized = false;
}

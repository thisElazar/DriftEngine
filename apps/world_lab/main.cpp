// World Lab v0.0.1 — single-tile sandbox: terrain brush + SWE water + plants.
//
// Validates the loop: paint terrain -> water flows downhill -> moisture field
// derived from water depth -> Bestiary places plants by suitability.
// Replant runs on left-mouse-button release so cause and effect read clearly.

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

#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"
#include "environment.h"
#include "distribution.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

// ---------------------------------------------------------------------------
// Tile constants
// ---------------------------------------------------------------------------
constexpr uint32_t GRID_W   = 256;
constexpr uint32_t GRID_H   = 256;
constexpr float    DX       = 1.0f;        // 1 m per cell -> tile is 256m x 256m
constexpr float    TILE_HALF_X = GRID_W * DX * 0.5f;
constexpr float    TILE_HALF_Z = GRID_H * DX * 0.5f;

// ---------------------------------------------------------------------------
// PC structs (matching the existing engine shaders one-to-one)
// ---------------------------------------------------------------------------
struct SweInitPC {
    uint32_t grid_w;
    uint32_t grid_h;
    float    initial_water_level;
    float    _pad;
};

struct SweStepPC {
    float    time;
    float    dt;
    float    gravity;
    float    friction;
    float    dx;
    float    sea_level;
    float    damping;
    float    _pad0;
    uint32_t grid_w;
    uint32_t grid_h;
    float    pulse_x;
    float    pulse_y;
    float    pulse_radius;
    float    pulse_amount;
};

struct TerrainBrushPC {
    float    brush_x;
    float    brush_y;
    float    brush_radius;
    float    brush_amount;
    uint32_t grid_w;
    uint32_t grid_h;
    uint32_t _pad0;
    uint32_t _pad1;
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
    float     moisture_overlay; // 0 = normal terrain, 1 = moisture heatmap
    float     _pad0;
    float     _pad1;
    float     _pad2;
};

struct ClumpPC {
    glm::mat4 mvp;
    float     wind_dir[2];
    float     wind_speed;
    float     time;
};

namespace {

// ---------------------------------------------------------------------------
// SPIR-V loader
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

VkShaderModule make_shader(VkDevice device, const char* path)
{
    auto spv = load_spirv(path);
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(uint32_t);
    ci.pCode    = spv.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
    return mod;
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------
struct GpuBuffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

GpuBuffer create_host_buffer(VmaAllocator alloc, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size  = size;
    ci.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    GpuBuffer b{};
    VK_CHECK(vmaCreateBuffer(alloc, &ci, &ai, &b.buffer, &b.allocation, nullptr));
    return b;
}

GpuBuffer create_readback_buffer(VmaAllocator alloc, VkDeviceSize size)
{
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size  = size;
    ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
             | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    GpuBuffer b{};
    VK_CHECK(vmaCreateBuffer(alloc, &ci, &ai, &b.buffer, &b.allocation, nullptr));
    return b;
}

void destroy_buffer(VmaAllocator alloc, GpuBuffer& b)
{
    if (b.buffer) { vmaDestroyBuffer(alloc, b.buffer, b.allocation); b = {}; }
}

// ---------------------------------------------------------------------------
// One-shot command buffer helper
// ---------------------------------------------------------------------------
struct OneShot {
    VkDevice      device;
    VkQueue       queue;
    VkCommandPool pool;
    VkCommandBuffer cmd;
};

OneShot oneshot_begin(VkDevice device, VkQueue queue, uint32_t family)
{
    OneShot s{device, queue, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = family;
    VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &s.pool));

    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = s.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &cai, &s.cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(s.cmd, &bi));
    return s;
}

void oneshot_end(OneShot& s)
{
    VK_CHECK(vkEndCommandBuffer(s.cmd));
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(s.device, &fci, nullptr, &fence));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s.cmd;
    VK_CHECK(vkQueueSubmit(s.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(s.device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(s.device, fence, nullptr);
    vkDestroyCommandPool(s.device, s.pool, nullptr);
}

void image_barrier(VkCommandBuffer cmd, VkImage img,
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

// Upload float data into an existing R32_SFLOAT image (assumes GENERAL layout
// in/out, image was created with TRANSFER_DST_BIT — upload_heightmap does this).
void update_r32_image(VkDevice device, VmaAllocator alloc,
                     VkQueue queue, uint32_t family,
                     VkImage img, const std::vector<float>& data,
                     uint32_t w, uint32_t h)
{
    VkDeviceSize bytes = VkDeviceSize{w} * h * sizeof(float);
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
             | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo staging_info{};
    VK_CHECK(vmaCreateBuffer(alloc, &bci, &ai, &staging, &staging_alloc, &staging_info));
    std::memcpy(staging_info.pMappedData, data.data(), bytes);
    vmaFlushAllocation(alloc, staging_alloc, 0, VK_WHOLE_SIZE);

    OneShot s = oneshot_begin(device, queue, family);
    VkImageMemoryBarrier2 b1{};
    b1.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b1.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                     | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    b1.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b1.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    b1.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b1.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    b1.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b1.image         = img;
    b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d1{};
    d1.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d1.imageMemoryBarrierCount = 1; d1.pImageMemoryBarriers = &b1;
    vkCmdPipelineBarrier2(s.cmd, &d1);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(s.cmd, staging, img,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier2 b2 = b1;
    b2.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    b2.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b2.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                     | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    b2.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b2.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b2.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    VkDependencyInfo d2{};
    d2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d2.imageMemoryBarrierCount = 1; d2.pImageMemoryBarriers = &b2;
    vkCmdPipelineBarrier2(s.cmd, &d2);
    oneshot_end(s);
    vmaDestroyBuffer(alloc, staging, staging_alloc);
}

void compute_memory_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                     | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                     | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

// ---------------------------------------------------------------------------
// Compute pipelines
// ---------------------------------------------------------------------------
struct ComputePipeline {
    VkShaderModule   shader      = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl    = VK_NULL_HANDLE;
    VkPipelineLayout layout      = VK_NULL_HANDLE;
    VkPipeline       pipeline    = VK_NULL_HANDLE;
};

ComputePipeline make_compute_pipeline(VkDevice device, const char* spv,
                                      const std::vector<VkDescriptorType>& bindings,
                                      uint32_t push_size)
{
    ComputePipeline cp{};
    cp.shader = make_shader(device, spv);

    std::vector<VkDescriptorSetLayoutBinding> b(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        b[i].binding         = static_cast<uint32_t>(i);
        b[i].descriptorType  = bindings[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(b.size());
    dslci.pBindings    = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &cp.dsl));

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size       = push_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &cp.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &push;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &cp.layout));

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cp.shader;
    stage.pName  = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = cp.layout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &cp.pipeline));

    return cp;
}

void destroy_compute_pipeline(VkDevice device, ComputePipeline& cp)
{
    vkDestroyPipeline(device, cp.pipeline, nullptr);
    vkDestroyPipelineLayout(device, cp.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, cp.dsl, nullptr);
    vkDestroyShaderModule(device, cp.shader, nullptr);
    cp = {};
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
// Graphics pipeline — clump (plants)
// ---------------------------------------------------------------------------
struct ClumpPipeline {
    VkShaderModule   vs       = VK_NULL_HANDLE;
    VkShaderModule   fs       = VK_NULL_HANDLE;
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
};

ClumpPipeline create_clump_pipeline(VkDevice device)
{
    ClumpPipeline p{};
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

    VkVertexInputBindingDescription vb{};
    vb.binding   = 0;
    vb.stride    = sizeof(bestiary::VegetationVertex);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[4]{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, position)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, normal)};
    attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(bestiary::VegetationVertex, color)};
    attrs[3] = {3, 0, VK_FORMAT_R32_SFLOAT,        offsetof(bestiary::VegetationVertex, height_t)};

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

void destroy_clump_pipeline(VkDevice device, ClumpPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    p = {};
}

// ---------------------------------------------------------------------------
// Static terrain mesh: GRID_W x GRID_H quads, vertices are integer grid coords.
// ---------------------------------------------------------------------------
struct StaticGridMesh {
    GpuBuffer vbo{};
    GpuBuffer ibo{};
    uint32_t  index_count = 0;
};

StaticGridMesh build_grid_mesh(VmaAllocator alloc)
{
    const uint32_t W = GRID_W;
    const uint32_t H = GRID_H;
    const uint32_t verts_x = W + 1;
    const uint32_t verts_y = H + 1;

    std::vector<float> verts;
    verts.reserve(verts_x * verts_y * 2);
    for (uint32_t y = 0; y <= H; ++y) {
        for (uint32_t x = 0; x <= W; ++x) {
            verts.push_back(static_cast<float>(x));
            verts.push_back(static_cast<float>(y));
        }
    }

    std::vector<uint32_t> idx;
    idx.reserve(W * H * 6);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t i00 = y * verts_x + x;
            uint32_t i10 = i00 + 1;
            uint32_t i01 = i00 + verts_x;
            uint32_t i11 = i01 + 1;
            idx.push_back(i00); idx.push_back(i10); idx.push_back(i11);
            idx.push_back(i00); idx.push_back(i11); idx.push_back(i01);
        }
    }

    StaticGridMesh m{};
    m.index_count = static_cast<uint32_t>(idx.size());

    VkDeviceSize vbs = verts.size() * sizeof(float);
    VkDeviceSize ibs = idx.size()   * sizeof(uint32_t);
    m.vbo = create_host_buffer(alloc, vbs, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m.ibo = create_host_buffer(alloc, ibs, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    void* mapped = nullptr;
    VK_CHECK(vmaMapMemory(alloc, m.vbo.allocation, &mapped));
    std::memcpy(mapped, verts.data(), vbs);
    vmaUnmapMemory(alloc, m.vbo.allocation);

    VK_CHECK(vmaMapMemory(alloc, m.ibo.allocation, &mapped));
    std::memcpy(mapped, idx.data(), ibs);
    vmaUnmapMemory(alloc, m.ibo.allocation);

    return m;
}

// ---------------------------------------------------------------------------
// Plant mesh state (rebuilt on replant)
// ---------------------------------------------------------------------------
struct PlantMesh {
    GpuBuffer vbo{};
    GpuBuffer ibo{};
    uint32_t  index_count  = 0;
    uint32_t  vertex_count = 0;
};

void destroy_plant_mesh(VmaAllocator alloc, PlantMesh& m)
{
    destroy_buffer(alloc, m.vbo);
    destroy_buffer(alloc, m.ibo);
    m = {};
}

void upload_plant_mesh(VmaAllocator alloc, PlantMesh& dst, const bestiary::VegetationMesh& src)
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
// Heightmap CPU mirror — kept in sync with GPU brush dispatches.
// We replay the same falloff as terrain_brush.hlsl so we don't need readback.
// ---------------------------------------------------------------------------
void cpu_apply_brush(std::vector<float>& hm,
                     float bx, float by, float radius, float amount)
{
    int cx = static_cast<int>(bx);
    int cy = static_cast<int>(by);
    int half = static_cast<int>(std::ceil(radius * 3.0f));
    int x0 = std::max(0, cx - half);
    int x1 = std::min(static_cast<int>(GRID_W) - 1, cx + half);
    int y0 = std::max(0, cy - half);
    int y1 = std::min(static_cast<int>(GRID_H) - 1, cy + half);
    float r2 = radius * radius;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            float dx_ = static_cast<float>(x) - bx;
            float dy_ = static_cast<float>(y) - by;
            float d2 = dx_ * dx_ + dy_ * dy_;
            float falloff = std::exp(-d2 / r2);
            if (falloff < 0.001f) continue;
            hm[y * GRID_W + x] += amount * falloff;
        }
    }
}

float sample_hm_bilinear(const std::vector<float>& hm, float wx, float wz)
{
    float gx = (wx + TILE_HALF_X) / DX - 0.5f;
    float gy = (wz + TILE_HALF_Z) / DX - 0.5f;
    int x0 = std::clamp(static_cast<int>(std::floor(gx)), 0, static_cast<int>(GRID_W) - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(gy)), 0, static_cast<int>(GRID_H) - 1);
    int x1 = std::min(x0 + 1, static_cast<int>(GRID_W) - 1);
    int y1 = std::min(y0 + 1, static_cast<int>(GRID_H) - 1);
    float fx = gx - x0;
    float fy = gy - y0;
    float h00 = hm[y0 * GRID_W + x0];
    float h10 = hm[y0 * GRID_W + x1];
    float h01 = hm[y1 * GRID_W + x0];
    float h11 = hm[y1 * GRID_W + x1];
    return (1 - fx) * (1 - fy) * h00 + fx * (1 - fy) * h10
         + (1 - fx) * fy       * h01 + fx * fy       * h11;
}

// ---------------------------------------------------------------------------
// Half-float (RGBA16F) decoder — water_state stores .r as half float
// ---------------------------------------------------------------------------
float half_to_float(uint16_t h)
{
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h & 0x7C00u) >> 10;
    uint32_t m =  h & 0x03FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            // subnormal -> normalize
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            ++e; m &= 0x03FFu;
            f = s | ((e + (127 - 15)) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + (127 - 15)) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// ---------------------------------------------------------------------------
// Build moisture field from water depth grid (and a small capillary blur).
// ---------------------------------------------------------------------------
std::vector<float> build_moisture_grid(const std::vector<float>& water_depth,
                                       float capillary_depth,
                                       int   capillary_blur_radius)
{
    std::vector<float> raw(GRID_W * GRID_H, 0.0f);
    for (size_t i = 0; i < raw.size(); ++i) {
        // Where there's standing water -> saturate moisture quickly.
        float d = std::max(0.0f, water_depth[i]);
        raw[i] = std::clamp(d / capillary_depth, 0.0f, 1.0f);
    }

    if (capillary_blur_radius <= 0) return raw;

    // Box blur to spread moisture into adjacent dry cells (capillary effect).
    std::vector<float> tmp(GRID_W * GRID_H, 0.0f);
    int r = capillary_blur_radius;
    auto idx = [](int x, int y) {
        return y * static_cast<int>(GRID_W) + x;
    };
    for (int y = 0; y < (int)GRID_H; ++y) {
        for (int x = 0; x < (int)GRID_W; ++x) {
            float sum = 0.0f;
            int count = 0;
            for (int dy = -r; dy <= r; ++dy) {
                int yy = std::clamp(y + dy, 0, (int)GRID_H - 1);
                for (int dx = -r; dx <= r; ++dx) {
                    int xx = std::clamp(x + dx, 0, (int)GRID_W - 1);
                    sum += raw[idx(xx, yy)];
                    ++count;
                }
            }
            tmp[idx(x, y)] = sum / count;
        }
    }
    // Take the max of raw and blurred — water cells stay saturated, dry-but-near
    // cells gain moisture from neighbors.
    for (size_t i = 0; i < tmp.size(); ++i)
        tmp[i] = std::max(raw[i], tmp[i]);
    return tmp;
}

// ---------------------------------------------------------------------------
// Read water depth (state_a layer 0, channel R) back to CPU float array.
// state image is RGBA16F; we only care about .r.
// ---------------------------------------------------------------------------
std::vector<float> readback_water_depth(VkDevice device, VmaAllocator alloc,
                                        VkQueue queue, uint32_t family,
                                        VkImage state_img)
{
    constexpr VkDeviceSize bytes = VkDeviceSize{GRID_W} * GRID_H * 8; // RGBA16F = 8 bytes/pixel
    GpuBuffer staging = create_readback_buffer(alloc, bytes);

    OneShot s = oneshot_begin(device, queue, family);

    image_barrier(s.cmd, state_img,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_READ_BIT,
                  VK_IMAGE_LAYOUT_GENERAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent      = {GRID_W, GRID_H, 1};
    vkCmdCopyImageToBuffer(s.cmd, state_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging.buffer, 1, &copy);

    image_barrier(s.cmd, state_img,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                    | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_IMAGE_LAYOUT_GENERAL);

    oneshot_end(s);

    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(alloc, staging.allocation, &ai);
    auto* hp = static_cast<const uint16_t*>(ai.pMappedData);

    std::vector<float> depth(GRID_W * GRID_H, 0.0f);
    for (size_t i = 0; i < depth.size(); ++i) {
        depth[i] = half_to_float(hp[i * 4 + 0]); // .r channel
    }

    destroy_buffer(alloc, staging);
    return depth;
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
    renderer_init(renderer, 1280, 800, "World Lab");
    glfwSetScrollCallback(renderer.window, scroll_cb);
    ImGui_ImplGlfw_InstallCallbacks(renderer.window);

    VkDevice device   = renderer.device;
    VmaAllocator alloc = renderer.allocator;
    VkQueue gqueue    = renderer.graphics_queue;
    uint32_t gfamily  = renderer.gfx_family;

    // ---- Heightmap (initial: a gentle bowl) -------------------------------
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
                // Bowl: deep center, rising rim. ~20m total relief so the
                // shape reads at the default 220m camera distance.
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
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        sizeof(SweStepPC));

    ComputePipeline pipe_brush = make_compute_pipeline(device, "shaders/terrain_brush.spv",
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
        sizeof(TerrainBrushPC));

    // ---- Descriptor pool --------------------------------------------------
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,        16},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         8},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = 16;
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

    VkDescriptorSet ds_swe_init = alloc_set(pipe_swe_init.dsl);
    VkDescriptorSet ds_swe_step[2] = {
        alloc_set(pipe_swe_step.dsl),
        alloc_set(pipe_swe_step.dsl)
    };
    VkDescriptorSet ds_brush = alloc_set(pipe_brush.dsl);

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

    write_image(ds_swe_init, 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, hm_gpu.view,    VK_NULL_HANDLE);
    write_image(ds_swe_init, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, state_a.view,   VK_NULL_HANDLE);

    // step set 0: read A -> write B
    write_image(ds_swe_step[0], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, hm_gpu.view,  VK_NULL_HANDLE);
    write_image(ds_swe_step[0], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, state_a.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[0], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, state_b.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[0], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, water_out.view, VK_NULL_HANDLE);

    // step set 1: read B -> write A
    write_image(ds_swe_step[1], 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, hm_gpu.view,  VK_NULL_HANDLE);
    write_image(ds_swe_step[1], 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, state_b.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[1], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, state_a.view, VK_NULL_HANDLE);
    write_image(ds_swe_step[1], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, water_out.view, VK_NULL_HANDLE);

    write_image(ds_brush, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, hm_gpu.view, VK_NULL_HANDLE);

    // ---- Graphics pipelines ---------------------------------------------
    TerrainPipeline pipe_terrain = create_terrain_pipeline(device);
    ClumpPipeline   pipe_clump   = create_clump_pipeline(device);

    VkDescriptorSet ds_terrain = alloc_set(pipe_terrain.dsl);
    write_image(ds_terrain, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, hm_gpu.view,    sampler);
    write_image(ds_terrain, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, water_out.view, sampler);
    write_image(ds_terrain, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, moist_gpu.view, sampler);

    StaticGridMesh terrain_mesh = build_grid_mesh(alloc);

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
    PlantMesh plants{};

    // Plant + ecosystem params (sliders later)
    bestiary::ClumpParams      clump_params{};
    bestiary::ClumpExpression  clump_expr{};
    bestiary::BushParams       bush_params{};
    bestiary::BushExpression   bush_expr{};
    bestiary::TreeParams       tree_params{};
    bestiary::TreeExpression   tree_expr{};
    bestiary::EcosystemParams  eco_params{};
    eco_params.region_size   = static_cast<float>(GRID_W) * DX;
    eco_params.density_scale = 1.0f;
    eco_params.r_min         = 1.5f;
    eco_params.r_max         = 6.0f;

    // Drier-loving wider band so empty/dry areas still grow some grass
    eco_params.grass_suit = {0.0f, 0.05f, 0.95f, 1.0f,
                             0.0f, 0.0f, 1.0f, 1.0f, 4.0f};

    // Enable color variation so dry vs wet reads visually
    clump_expr.vary_color = true;
    clump_expr.dry_color[0] = 0.62f; clump_expr.dry_color[1] = 0.55f; clump_expr.dry_color[2] = 0.25f;
    clump_expr.wet_color[0] = 0.20f; clump_expr.wet_color[1] = 0.55f; clump_expr.wet_color[2] = 0.18f;
    clump_expr.blade_height.enabled = true;
    clump_expr.blade_height.low  = 0.20f;
    clump_expr.blade_height.high = 0.80f;
    clump_expr.blade_count.enabled = true;
    clump_expr.blade_count.low  = 4.0f;
    clump_expr.blade_count.high = 30.0f;

    // ---- Brush + sim controls --------------------------------------------
    int   ui_brush_radius_cells = 12;
    float ui_brush_amount       = 1.5f;
    bool  ui_swe_enabled        = true;
    int   ui_swe_substeps       = 4;
    float ui_swe_dt             = 0.05f;
    float ui_capillary_depth    = 0.05f;
    int   ui_capillary_blur     = 4;
    bool  ui_show_plants        = true;
    bool  ui_show_moisture      = false;
    int   replant_pending       = 0;     // counts down frames after stroke release

    bool brushing      = false;          // any modifier currently editing terrain
    bool brushed_this_stroke = false;
    bool prev_lmb      = false;
    bool prev_rain_key = false;

    int   swe_ping_pong = 0;
    int   wt_water_pulse_active = 0;

    // Initial plant placement (no water -> all dry)
    auto run_replant = [&](){
        // Freshest state slot: each step swaps; after N steps from slot A,
        // ping_pong & 1 == 0 means A is current.
        VkImage fresh = (swe_ping_pong & 1) ? state_b.image : state_a.image;
        std::vector<float> water_depth = readback_water_depth(
            device, alloc, gqueue, gfamily, fresh);

        std::vector<float> moisture = build_moisture_grid(
            water_depth, ui_capillary_depth, ui_capillary_blur);

        // Push the new moisture grid to GPU for the debug overlay.
        update_r32_image(device, alloc, gqueue, gfamily,
                         moist_gpu.image, moisture, GRID_W, GRID_H);

        bestiary::EnvironmentField env;
        env.sample = [moisture, &hm_cpu](float x, float z) -> bestiary::EnvironmentSample {
            float gx = (x + TILE_HALF_X) / DX;
            float gy = (z + TILE_HALF_Z) / DX;
            int ix = std::clamp((int)std::floor(gx), 0, (int)GRID_W - 1);
            int iy = std::clamp((int)std::floor(gy), 0, (int)GRID_H - 1);
            float m = moisture[iy * GRID_W + ix];
            // Temperature: subtly cooler at low elevations (proxy field), so
            // the suitability function has something other than moisture to lean on.
            float h = hm_cpu[iy * GRID_W + ix];
            float t = std::clamp(0.5f + 0.05f * h, 0.0f, 1.0f);
            return {m, t};
        };

        bestiary::VegetationMesh m = bestiary::generate_ecosystem(
            eco_params, env,
            clump_params, clump_expr,
            bush_params, bush_expr,
            tree_params, tree_expr,
            /*include_ground=*/false);

        // Displace plant vertices' y by terrain height at their (x,z).
        // Add a small bias so blade bases don't z-fight with the ground mesh.
        constexpr float plant_lift = 0.01f;
        for (auto& v : m.vertices) {
            float y = sample_hm_bilinear(hm_cpu, v.position[0], v.position[2]);
            v.position[1] += y + plant_lift;
        }

        vkDeviceWaitIdle(device);
        destroy_plant_mesh(alloc, plants);
        upload_plant_mesh(alloc, plants, m);
    };

    run_replant();

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

        // Cursor pick
        int win_w, win_h;
        glfwGetWindowSize(renderer.window, &win_w, &win_h);
        double mx, my;
        glfwGetCursorPos(renderer.window, &mx, &my);
        float aspect = float(win_w) / float(win_h);
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.5f, 1500.0f);
        proj[1][1] *= -1.0f;
        glm::mat4 view = orbit_view(camera);

        // Vulkan clip space (after proj[1][1] *= -1) has +Y down, matching GLFW
        // cursor coords, so no extra flip is needed.
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

        // ---- ImGui --------------------------------------------------------
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360, 600), ImGuiCond_FirstUseEver);
        ImGui::Begin("World Lab");
        ImGui::Text("LMB: raise terrain   Shift+LMB: lower");
        ImGui::Text("R (key): rain pulse   RMB: orbit   Scroll: zoom");
        ImGui::Separator();

        ImGui::SliderInt  ("brush radius (cells)", &ui_brush_radius_cells, 2, 60);
        ImGui::SliderFloat("brush amount",         &ui_brush_amount, 0.05f, 5.0f, "%.2f m");

        ImGui::Separator();
        ImGui::TextUnformatted("Water sim");
        ImGui::Checkbox  ("enable SWE",  &ui_swe_enabled);
        ImGui::SliderInt ("substeps",    &ui_swe_substeps, 1, 10);
        ImGui::SliderFloat("dt (s)",     &ui_swe_dt,       0.005f, 0.1f, "%.3f");

        ImGui::Separator();
        ImGui::TextUnformatted("Moisture field");
        ImGui::SliderFloat("capillary depth", &ui_capillary_depth, 0.005f, 1.0f, "%.3f");
        ImGui::SliderInt  ("capillary blur",  &ui_capillary_blur,  0, 12);

        ImGui::Separator();
        ImGui::Checkbox  ("show plants",   &ui_show_plants);
        ImGui::Checkbox  ("show moisture (debug)", &ui_show_moisture);
        ImGui::SliderFloat("density",    &eco_params.density_scale, 0.1f, 3.0f, "%.2f");
        ImGui::SliderFloat("r_min",      &eco_params.r_min, 0.5f, 4.0f, "%.2f m");
        ImGui::SliderFloat("r_max",      &eco_params.r_max, 1.0f, 10.0f, "%.2f m");
        if (ImGui::Button("Replant now")) replant_pending = 1;

        ImGui::Separator();
        ImGui::Text("plants: %u verts / %u tris",
                    plants.vertex_count, plants.index_count / 3);
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

        // Brush dispatch (terrain) + CPU mirror update
        bool brush_apply_now = lmb && have_pick;
        if (brush_apply_now) {
            float amount = (shift ? -1.0f : 1.0f) * ui_brush_amount;
            // GPU
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

            // CPU mirror (replay same falloff)
            cpu_apply_brush(hm_cpu, pick_gx, pick_gy, bpc.brush_radius, bpc.brush_amount);
            brushed_this_stroke = true;
        }

        // SWE step(s) — read A, write B; ping-pong; output texture is shared.
        if (ui_swe_enabled) {
            for (int sub = 0; sub < ui_swe_substeps; ++sub) {
                int cur = swe_ping_pong & 1;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_swe_step.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipe_swe_step.layout, 0, 1, &ds_swe_step[cur], 0, nullptr);
                SweStepPC pc{};
                pc.time     = accumulated_time;
                pc.dt       = ui_swe_dt;
                pc.gravity  = 9.81f;
                pc.friction = 0.05f;
                pc.dx       = DX;
                pc.sea_level = -10000.0f; // disable static sea floor
                pc.damping  = 0.0f;
                pc.grid_w   = GRID_W;
                pc.grid_h   = GRID_H;
                if (rain_key && have_pick) {
                    pc.pulse_x      = pick_gx;
                    pc.pulse_y      = pick_gy;
                    pc.pulse_radius = static_cast<float>(ui_brush_radius_cells);
                    pc.pulse_amount = 0.5f;
                    wt_water_pulse_active = 1;
                }
                vkCmdPushConstants(cmd, pipe_swe_step.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, (GRID_W + 7) / 8, (GRID_H + 7) / 8, 1);
                compute_memory_barrier(cmd);
                ++swe_ping_pong;
            }
        }

        // Detect LMB release for replant
        if (prev_lmb && !lmb && brushed_this_stroke) {
            replant_pending = 1; // schedule for next frame so the latest brush stroke is on GPU
            brushed_this_stroke = false;
        }

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

        // ---- Main pass: terrain + plants ---------------------------------
        glm::mat4 mvp = proj * view;

        VkRenderingAttachmentInfo color{};
        color.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView   = renderer.swapchain_views[image_index];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{0.45f, 0.55f, 0.70f, 1.0f}}; // sky

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
        tpc.moisture_overlay = ui_show_moisture ? 1.0f : 0.0f;
        vkCmdPushConstants(cmd, pipe_terrain.layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(tpc), &tpc);
        VkDeviceSize zero = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &terrain_mesh.vbo.buffer, &zero);
        vkCmdBindIndexBuffer(cmd, terrain_mesh.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, terrain_mesh.index_count, 1, 0, 0, 0);

        // Plants
        if (ui_show_plants && plants.index_count > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_clump.pipeline);
            ClumpPC cpc{};
            cpc.mvp = mvp;
            cpc.wind_dir[0] = 0.6f;
            cpc.wind_dir[1] = 0.4f;
            cpc.wind_speed  = 0.5f;
            cpc.time = accumulated_time;
            vkCmdPushConstants(cmd, pipe_clump.layout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(cpc), &cpc);
            vkCmdBindVertexBuffers(cmd, 0, 1, &plants.vbo.buffer, &zero);
            vkCmdBindIndexBuffer(cmd, plants.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, plants.index_count, 1, 0, 0, 0);
        }
        vkCmdEndRendering(cmd);

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

        prev_lmb = lmb;
        prev_rain_key = rain_key;
        (void)prev_rain_key; // silence unused
        (void)brushing;
        (void)wt_water_pulse_active;

        if (replant_pending > 0) {
            --replant_pending;
            if (replant_pending == 0) {
                run_replant();
            }
        }
    }

    vkDeviceWaitIdle(device);
    destroy_plant_mesh(alloc, plants);
    destroy_buffer(alloc, terrain_mesh.vbo);
    destroy_buffer(alloc, terrain_mesh.ibo);
    destroy_clump_pipeline(device, pipe_clump);
    destroy_terrain_pipeline(device, pipe_terrain);
    destroy_compute_pipeline(device, pipe_brush);
    destroy_compute_pipeline(device, pipe_swe_step);
    destroy_compute_pipeline(device, pipe_swe_init);

    vkDestroyDescriptorPool(device, desc_pool, nullptr);
    vkDestroySampler(device, sampler, nullptr);

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

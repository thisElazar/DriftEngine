// animals_lab.cpp — Animals Lab module implementation.

#include "animals_lab.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <filesystem>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void upload_to_buffer(VmaAllocator allocator, GpuBuffer& buf,
                             const void* data, VkDeviceSize size)
{
    void* mapped = nullptr;
    VK_CHECK(vmaMapMemory(allocator, buf.allocation, &mapped));
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(allocator, buf.allocation);
}

// ---------------------------------------------------------------------------
// Common pipeline state
// ---------------------------------------------------------------------------
static VkFormat g_color_fmt = VK_FORMAT_B8G8R8A8_UNORM;
static VkFormat g_depth_fmt = VK_FORMAT_D32_SFLOAT;

// ---------------------------------------------------------------------------
// Mesh pipeline (skinned animal mesh)
// ---------------------------------------------------------------------------
static MeshPipeline create_mesh_pipeline(VkDevice device)
{
    MeshPipeline p{};

    p.vs = make_shader(device, "shaders/mesh_vs.spv");
    p.fs = make_shader(device, "shaders/mesh_fs.spv");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.offset     = 0;
    push.size       = sizeof(glm::mat4);

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
    vb.stride    = sizeof(bestiary::SkinnedVertex);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[3]{};
    attrs[0].location = 0; attrs[0].binding = 0;
    attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset   = offsetof(bestiary::SkinnedVertex, position);
    attrs[1].location = 1; attrs[1].binding = 0;
    attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset   = offsetof(bestiary::SkinnedVertex, normal);
    attrs[2].location = 2; attrs[2].binding = 0;
    attrs[2].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[2].offset   = offsetof(bestiary::SkinnedVertex, color);

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
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_BACK_BIT;
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

    VkPipelineRenderingCreateInfo rendering_ci{};
    rendering_ci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_ci.colorAttachmentCount    = 1;
    rendering_ci.pColorAttachmentFormats = &g_color_fmt;
    rendering_ci.depthAttachmentFormat   = g_depth_fmt;

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

static void destroy_mesh_pipeline(VkDevice device, MeshPipeline& p)
{
    vkDestroyPipeline(device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyShaderModule(device, p.vs, nullptr);
    vkDestroyShaderModule(device, p.fs, nullptr);
}

// ---------------------------------------------------------------------------
// Skeleton pipeline (joint spheres + bone lines — overlay)
// ---------------------------------------------------------------------------
static SkeletonPipeline create_skeleton_pipeline(VkDevice device)
{
    SkeletonPipeline p{};

    p.sphere_vs = make_shader(device, "shaders/joint_vs.spv");
    p.sphere_fs = make_shader(device, "shaders/joint_fs.spv");
    p.line_vs   = make_shader(device, "shaders/bone_vs.spv");
    p.line_fs   = make_shader(device, "shaders/bone_fs.spv");

    VkDescriptorSetLayoutBinding ssbo_binding{};
    ssbo_binding.binding         = 0;
    ssbo_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssbo_binding.descriptorCount = 1;
    ssbo_binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 1;
    dsl_ci.pBindings    = &ssbo_binding;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &p.desc_layout));

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.offset     = 0;
    push.size       = sizeof(SkeletonPC);

    VkPipelineLayoutCreateInfo pl_ci{};
    pl_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount         = 1;
    pl_ci.pSetLayouts            = &p.desc_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges    = &push;
    VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.layout));

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

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

    VkPipelineRenderingCreateInfo rendering_ci{};
    rendering_ci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_ci.colorAttachmentCount    = 1;
    rendering_ci.pColorAttachmentFormats = &g_color_fmt;
    rendering_ci.depthAttachmentFormat   = g_depth_fmt;

    // Sphere pipeline
    {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = p.sphere_vs; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = p.sphere_fs; stages[1].pName = "main";

        VkVertexInputBindingDescription vb{}; vb.binding = 0; vb.stride = sizeof(SphereVertex); vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SphereVertex, position)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SphereVertex, normal)};

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
        vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

        VkGraphicsPipelineCreateInfo gp_ci{};
        gp_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; gp_ci.pNext = &rendering_ci;
        gp_ci.stageCount = 2; gp_ci.pStages = stages;
        gp_ci.pVertexInputState = &vi; gp_ci.pInputAssemblyState = &ia; gp_ci.pViewportState = &vp;
        gp_ci.pRasterizationState = &rs; gp_ci.pMultisampleState = &ms; gp_ci.pDepthStencilState = &ds;
        gp_ci.pColorBlendState = &cb; gp_ci.pDynamicState = &dyn; gp_ci.layout = p.layout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp_ci, nullptr, &p.sphere_pipe));
    }

    // Line pipeline
    {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = p.line_vs; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = p.line_fs; stages[1].pName = "main";

        VkVertexInputBindingDescription vb{}; vb.binding = 0; vb.stride = sizeof(uint32_t); vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32_UINT, 0};

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
        vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &attr;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

        VkGraphicsPipelineCreateInfo gp_ci{};
        gp_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; gp_ci.pNext = &rendering_ci;
        gp_ci.stageCount = 2; gp_ci.pStages = stages;
        gp_ci.pVertexInputState = &vi; gp_ci.pInputAssemblyState = &ia; gp_ci.pViewportState = &vp;
        gp_ci.pRasterizationState = &rs; gp_ci.pMultisampleState = &ms; gp_ci.pDepthStencilState = &ds;
        gp_ci.pColorBlendState = &cb; gp_ci.pDynamicState = &dyn; gp_ci.layout = p.layout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp_ci, nullptr, &p.line_pipe));
    }

    return p;
}

static void destroy_skeleton_pipeline(VkDevice device, SkeletonPipeline& p)
{
    vkDestroyPipeline(device, p.sphere_pipe, nullptr);
    vkDestroyPipeline(device, p.line_pipe, nullptr);
    vkDestroyPipelineLayout(device, p.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.desc_layout, nullptr);
    vkDestroyShaderModule(device, p.sphere_vs, nullptr);
    vkDestroyShaderModule(device, p.sphere_fs, nullptr);
    vkDestroyShaderModule(device, p.line_vs, nullptr);
    vkDestroyShaderModule(device, p.line_fs, nullptr);
}

// ---------------------------------------------------------------------------
// UV sphere mesh generation
// ---------------------------------------------------------------------------
static void generate_uv_sphere(std::vector<SphereVertex>& verts,
                                std::vector<uint32_t>& indices,
                                int slices, int stacks)
{
    verts.clear();
    indices.clear();

    verts.push_back({{0, 1, 0}, {0, 1, 0}});

    for (int i = 1; i < stacks; ++i) {
        float phi = 3.14159265f * static_cast<float>(i) / static_cast<float>(stacks);
        float sp = std::sin(phi), cp = std::cos(phi);
        for (int j = 0; j < slices; ++j) {
            float theta = 2.0f * 3.14159265f * static_cast<float>(j) / static_cast<float>(slices);
            float x = sp * std::cos(theta);
            float y = cp;
            float z = sp * std::sin(theta);
            verts.push_back({{x, y, z}, {x, y, z}});
        }
    }

    verts.push_back({{0, -1, 0}, {0, -1, 0}});

    for (int j = 0; j < slices; ++j) {
        indices.push_back(0);
        indices.push_back(1 + (j + 1) % slices);
        indices.push_back(1 + j);
    }

    for (int i = 0; i < stacks - 2; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = 1 + static_cast<uint32_t>(i * slices + j);
            uint32_t b = 1 + static_cast<uint32_t>(i * slices + (j + 1) % slices);
            uint32_t c = 1 + static_cast<uint32_t>((i + 1) * slices + j);
            uint32_t d = 1 + static_cast<uint32_t>((i + 1) * slices + (j + 1) % slices);
            indices.push_back(a); indices.push_back(b); indices.push_back(c);
            indices.push_back(b); indices.push_back(d); indices.push_back(c);
        }
    }

    uint32_t bottom = static_cast<uint32_t>(verts.size()) - 1;
    uint32_t base   = 1 + static_cast<uint32_t>((stacks - 2) * slices);
    for (int j = 0; j < slices; ++j) {
        indices.push_back(bottom);
        indices.push_back(base + static_cast<uint32_t>(j));
        indices.push_back(base + static_cast<uint32_t>((j + 1) % slices));
    }
}

// ---------------------------------------------------------------------------
// Ground plane (checkerboard)
// ---------------------------------------------------------------------------
static GroundPlane create_ground_plane(VmaAllocator allocator)
{
    std::vector<bestiary::SkinnedVertex> verts;
    std::vector<uint32_t> indices;

    const float tile = 0.5f;
    const int   nz   = 400;
    const int   nx   = 12;
    const float x0   = -static_cast<float>(nx) * tile * 0.5f;
    const float z0   = -5.0f;

    for (int iz = 0; iz < nz; ++iz) {
        for (int ix = 0; ix < nx; ++ix) {
            float zz = z0 + static_cast<float>(iz) * tile;
            float xx = x0 + static_cast<float>(ix) * tile;
            bool dark = ((ix + iz) % 2) == 0;
            float c = dark ? 0.13f : 0.19f;

            auto base = static_cast<uint32_t>(verts.size());
            bestiary::SkinnedVertex sv{};
            sv.normal[1] = 1.0f;
            sv.color[0] = c * 0.9f; sv.color[1] = c; sv.color[2] = c * 0.85f;

            auto push = [&](float px, float pz) {
                sv.position[0] = px; sv.position[1] = 0.0f; sv.position[2] = pz;
                verts.push_back(sv);
            };
            push(xx, zz);
            push(xx + tile, zz);
            push(xx + tile, zz + tile);
            push(xx, zz + tile);

            indices.push_back(base);     indices.push_back(base + 2); indices.push_back(base + 1);
            indices.push_back(base);     indices.push_back(base + 3); indices.push_back(base + 2);
        }
    }

    GroundPlane gp;
    gp.index_count = static_cast<uint32_t>(indices.size());
    gp.vb = create_host_buffer(allocator,
        verts.size() * sizeof(bestiary::SkinnedVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    upload_to_buffer(allocator, gp.vb,
        verts.data(), verts.size() * sizeof(bestiary::SkinnedVertex));
    gp.ib = create_host_buffer(allocator,
        indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    upload_to_buffer(allocator, gp.ib,
        indices.data(), indices.size() * sizeof(uint32_t));
    return gp;
}

// ---------------------------------------------------------------------------
// Compute world-space joint transforms
// ---------------------------------------------------------------------------
static std::vector<glm::mat4> compute_world_transforms(const bestiary::Skeleton& skel)
{
    std::vector<glm::mat4> world(skel.joints.size());
    for (size_t i = 0; i < skel.joints.size(); ++i) {
        world[i] = (skel.joints[i].parent == -1)
            ? skel.joints[i].local_bind
            : world[skel.joints[i].parent] * skel.joints[i].local_bind;
    }
    return world;
}

// ---------------------------------------------------------------------------
// Frame recording
// ---------------------------------------------------------------------------
static void record_frame(Renderer& r, FrameData& frame, uint32_t image_index, VkExtent2D extent,
                         const glm::mat4& mvp,
                         const MeshPipeline& mesh_pipe,
                         const GroundPlane& ground,
                         GpuBuffer& mesh_vb, GpuBuffer& mesh_ib, uint32_t mesh_index_count,
                         bool show_skeleton,
                         const SkeletonPipeline& skel_pipe,
                         GpuBuffer& sphere_vb, GpuBuffer& sphere_ib, uint32_t sphere_index_count,
                         GpuBuffer& bone_vb, uint32_t bone_vertex_count,
                         VkDescriptorSet desc_set, int joint_count, float joint_radius)
{
    VkCommandBuffer cmd = frame.cmd;

    // Transition swapchain + depth.
    {
        VkImageMemoryBarrier2 barriers[2]{};
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT; barriers[0].srcAccessMask = VK_ACCESS_2_NONE;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT; barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].image = r.swapchain_images[image_index]; barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT; barriers[1].srcAccessMask = VK_ACCESS_2_NONE;
        barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT; barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barriers[1].image = r.depth_buffer.image; barriers[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{}; dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 2; dep.pImageMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Main pass.
    {
        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = r.swapchain_views[image_index]; color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue.color = {{0.07f, 0.08f, 0.10f, 1.0f}};

        VkRenderingAttachmentInfo depth{};
        depth.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth.imageView = r.depth_buffer.view; depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil = {1.0f, 0};

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO; ri.renderArea = {{0, 0}, extent};
        ri.layerCount = 1; ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color; ri.pDepthAttachment = &depth;
        vkCmdBeginRendering(cmd, &ri);

        VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{{0, 0}, extent};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Draw ground plane.
        if (ground.index_count > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipe.pipeline);
            vkCmdPushConstants(cmd, mesh_pipe.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &ground.vb.buffer, &off);
            vkCmdBindIndexBuffer(cmd, ground.ib.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, ground.index_count, 1, 0, 0, 0);
        }

        // Draw animal mesh.
        if (mesh_index_count > 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipe.pipeline);
            vkCmdPushConstants(cmd, mesh_pipe.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh_vb.buffer, &offset);
            vkCmdBindIndexBuffer(cmd, mesh_ib.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh_index_count, 1, 0, 0, 0);
        }

        // Draw skeleton overlay.
        if (show_skeleton && joint_count > 0) {
            SkeletonPC spc{};
            spc.mvp = mvp;
            spc.joint_radius = joint_radius;

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    skel_pipe.layout, 0, 1, &desc_set, 0, nullptr);
            vkCmdPushConstants(cmd, skel_pipe.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(spc), &spc);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skel_pipe.sphere_pipe);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &sphere_vb.buffer, &offset);
            vkCmdBindIndexBuffer(cmd, sphere_ib.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, sphere_index_count, static_cast<uint32_t>(joint_count), 0, 0, 0);

            if (bone_vertex_count > 0) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skel_pipe.line_pipe);
                offset = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &bone_vb.buffer, &offset);
                vkCmdDraw(cmd, bone_vertex_count, 1, 0, 0);
            }
        }

        vkCmdEndRendering(cmd);
    }

    // ImGui pass.
    {
        VkRenderingAttachmentInfo color{};
        color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        color.imageView = r.swapchain_views[image_index]; color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO; ri.renderArea = {{0, 0}, extent};
        ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &color;
        vkCmdBeginRendering(cmd, &ri);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
        vkCmdEndRendering(cmd);
    }

    // Present transition.
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT; barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT; barrier.dstAccessMask = VK_ACCESS_2_NONE;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.image = r.swapchain_images[image_index]; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{}; dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

// ===========================================================================
// Module API
// ===========================================================================

void animals_lab_init(AnimalsLabState& s, Renderer& r)
{
    s.mesh_pipe = create_mesh_pipeline(r.device);
    s.skel_pipe = create_skeleton_pipeline(r.device);

    // Camera defaults for animals
    s.camera.yaw      = 0.785f;
    s.camera.pitch    = 0.06f;
    s.camera.distance = 3.5f;
    s.camera.target   = {0.0f, 1.0f, 0.0f};

    // Sphere mesh for skeleton overlay (static)
    std::vector<SphereVertex> sphere_verts;
    std::vector<uint32_t>     sphere_indices;
    generate_uv_sphere(sphere_verts, sphere_indices, 8, 6);

    s.sphere_vb = create_host_buffer(r.allocator,
        sphere_verts.size() * sizeof(SphereVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    upload_to_buffer(r.allocator, s.sphere_vb,
        sphere_verts.data(), sphere_verts.size() * sizeof(SphereVertex));

    s.sphere_ib = create_host_buffer(r.allocator,
        sphere_indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    upload_to_buffer(r.allocator, s.sphere_ib,
        sphere_indices.data(), sphere_indices.size() * sizeof(uint32_t));
    s.sphere_index_count = static_cast<uint32_t>(sphere_indices.size());

    // Joint transform SSBO
    constexpr int MAX_JOINTS = 26;
    s.joint_ssbo = create_host_buffer(r.allocator,
        MAX_JOINTS * sizeof(glm::mat4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    {
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.maxSets = 1; pool_ci.poolSizeCount = 1; pool_ci.pPoolSizes = &pool_size;
        VK_CHECK(vkCreateDescriptorPool(r.device, &pool_ci, nullptr, &s.desc_pool));
    }

    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = s.desc_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &s.skel_pipe.desc_layout;
        VK_CHECK(vkAllocateDescriptorSets(r.device, &ai, &s.desc_set));

        VkDescriptorBufferInfo bi{s.joint_ssbo.buffer, 0, MAX_JOINTS * sizeof(glm::mat4)};
        VkWriteDescriptorSet wr{};
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = s.desc_set; wr.dstBinding = 0; wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr.pBufferInfo = &bi;
        vkUpdateDescriptorSets(r.device, 1, &wr, 0, nullptr);
    }

    // Initial creature
    s.animal_mesh = bestiary::generate_herbivore_mesh(s.herb_params);
    s.gaits[0] = bestiary::make_herbivore_walk(s.herb_params);
    s.gaits[1] = bestiary::make_herbivore_trot(s.herb_params);
    s.gaits[2] = bestiary::make_herbivore_run(s.herb_params);
    s.gaits[3] = bestiary::make_herbivore_idle(s.herb_params);
    s.gaits[4] = bestiary::make_herbivore_graze(s.herb_params, s.feed_height);
    s.joint_count = static_cast<int>(s.animal_mesh.skeleton.joints.size());

    auto world_transforms = compute_world_transforms(s.animal_mesh.skeleton);
    upload_to_buffer(r.allocator, s.joint_ssbo,
        world_transforms.data(), world_transforms.size() * sizeof(glm::mat4));

    std::vector<uint32_t> bone_indices;
    for (size_t i = 0; i < s.animal_mesh.skeleton.joints.size(); ++i) {
        if (s.animal_mesh.skeleton.joints[i].parent >= 0) {
            bone_indices.push_back(static_cast<uint32_t>(s.animal_mesh.skeleton.joints[i].parent));
            bone_indices.push_back(static_cast<uint32_t>(i));
        }
    }
    s.bone_vertex_count = static_cast<uint32_t>(bone_indices.size());

    s.bone_vb = create_host_buffer(r.allocator,
        bone_indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    upload_to_buffer(r.allocator, s.bone_vb,
        bone_indices.data(), bone_indices.size() * sizeof(uint32_t));

    // Ground plane
    s.ground = create_ground_plane(r.allocator);

    // CPU-skinned mesh buffers
    s.joint_palette.resize(s.joint_count);
    for (int i = 0; i < s.joint_count; ++i)
        s.joint_palette[i] = world_transforms[i] * s.animal_mesh.skeleton.inverse_bind[i];

    bestiary::cpu_skin(s.animal_mesh.vertices, s.joint_palette, s.skinned_verts);

    if (!s.skinned_verts.empty()) {
        s.mesh_vb = create_host_buffer(r.allocator,
            s.skinned_verts.size() * sizeof(bestiary::SkinnedVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        upload_to_buffer(r.allocator, s.mesh_vb,
            s.skinned_verts.data(), s.skinned_verts.size() * sizeof(bestiary::SkinnedVertex));

        s.mesh_ib = create_host_buffer(r.allocator,
            s.animal_mesh.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        upload_to_buffer(r.allocator, s.mesh_ib,
            s.animal_mesh.indices.data(), s.animal_mesh.indices.size() * sizeof(uint32_t));
        s.mesh_index_count = static_cast<uint32_t>(s.animal_mesh.indices.size());
    }

    // Dirty-check: force initial regen
    std::memset(&s.last_herb_params,   0xFF, sizeof(s.last_herb_params));
    std::memset(&s.last_pred_params,   0xFF, sizeof(s.last_pred_params));
    std::memset(&s.last_rabbit_params, 0xFF, sizeof(s.last_rabbit_params));
    std::memset(&s.last_bird_params,   0xFF, sizeof(s.last_bird_params));
    std::memset(&s.last_raptor_params, 0xFF, sizeof(s.last_raptor_params));
    std::memset(&s.last_snake_params,  0xFF, sizeof(s.last_snake_params));

    // Raptor defaults — powerful build, long broad wings, hooked beak
    s.raptor_params.body_length      = 0.38f;
    s.raptor_params.body_girth       = 0.20f;
    s.raptor_params.neck_length      = 0.10f;
    s.raptor_params.head_size        = 0.055f;
    s.raptor_params.beak_length      = 0.045f;
    s.raptor_params.wing_length      = 0.55f;
    s.raptor_params.wing_width       = 0.12f;
    s.raptor_params.wing_taper       = 0.20f;
    s.raptor_params.leg_length_upper = 0.12f;
    s.raptor_params.leg_length_lower = 0.08f;
    s.raptor_params.leg_thickness    = 0.012f;
    s.raptor_params.foot_size        = 0.035f;
    s.raptor_params.tail_length      = 0.14f;
    s.raptor_params.coat_color[0] = 0.30f; s.raptor_params.coat_color[1] = 0.22f; s.raptor_params.coat_color[2] = 0.15f;
    s.raptor_params.belly_color[0] = 0.65f; s.raptor_params.belly_color[1] = 0.55f; s.raptor_params.belly_color[2] = 0.45f;
    s.raptor_params.walk_period_seconds = 0.6f;
    s.raptor_params.flap_period    = 0.6f;
    s.raptor_params.flap_amplitude = 35.0f;
    s.raptor_params.flap_sweep     = 0.6f;
    s.raptor_params.fly_height     = 1.2f;
    s.raptor_params.hop_height     = 0.05f;

    // File I/O
    s.fio.refresh_files();

    s.initialized = true;
}

bool animals_lab_tick(AnimalsLabState& s, Renderer& r, float dt)
{
    update_orbit(s.camera, r.window, 10.0f);

    // --- Dirty-check params -> regen mesh/skeleton/gaits ---
    bool params_changed = false;
    if (s.creature_type != s.last_creature_type) {
        params_changed = true;
    } else if (s.creature_type == CREATURE_HERBIVORE) {
        params_changed = std::memcmp(&s.herb_params, &s.last_herb_params, sizeof(s.herb_params)) != 0;
    } else if (s.creature_type == CREATURE_PREDATOR) {
        params_changed = std::memcmp(&s.pred_params, &s.last_pred_params, sizeof(s.pred_params)) != 0;
    } else if (s.creature_type == CREATURE_RABBIT) {
        params_changed = std::memcmp(&s.rabbit_params, &s.last_rabbit_params, sizeof(s.rabbit_params)) != 0;
    } else if (s.creature_type == CREATURE_BIRD) {
        params_changed = std::memcmp(&s.bird_params, &s.last_bird_params, sizeof(s.bird_params)) != 0;
    } else if (s.creature_type == CREATURE_RAPTOR) {
        params_changed = std::memcmp(&s.raptor_params, &s.last_raptor_params, sizeof(s.raptor_params)) != 0;
    } else {
        params_changed = std::memcmp(&s.snake_params, &s.last_snake_params, sizeof(s.snake_params)) != 0;
    }

    if (params_changed) {
        vkDeviceWaitIdle(r.device);

        if (s.creature_type == CREATURE_HERBIVORE) {
            s.animal_mesh = bestiary::generate_herbivore_mesh(s.herb_params);
            s.gaits[0] = bestiary::make_herbivore_walk(s.herb_params);
            s.gaits[1] = bestiary::make_herbivore_trot(s.herb_params);
            s.gaits[2] = bestiary::make_herbivore_run(s.herb_params);
            s.gaits[3] = bestiary::make_herbivore_idle(s.herb_params);
            s.gaits[4] = bestiary::make_herbivore_graze(s.herb_params, s.feed_height);
            s.last_feed_h = s.feed_height;
            s.last_herb_params = s.herb_params;
        } else if (s.creature_type == CREATURE_PREDATOR) {
            s.animal_mesh = bestiary::generate_predator_mesh(s.pred_params);
            s.gaits[0] = bestiary::make_predator_walk(s.pred_params);
            s.gaits[1] = bestiary::make_predator_trot(s.pred_params);
            s.gaits[2] = bestiary::make_predator_run(s.pred_params);
            s.gaits[3] = bestiary::make_predator_idle(s.pred_params);
            s.gaits[4] = bestiary::make_predator_stalk(s.pred_params);
            s.last_pred_params = s.pred_params;
        } else if (s.creature_type == CREATURE_RABBIT) {
            s.animal_mesh = bestiary::generate_rabbit_mesh(s.rabbit_params);
            s.gaits[0] = bestiary::make_rabbit_hop(s.rabbit_params);
            s.gaits[1] = bestiary::make_rabbit_hop(s.rabbit_params);
            s.gaits[2] = bestiary::make_rabbit_run(s.rabbit_params);
            s.gaits[3] = bestiary::make_rabbit_idle(s.rabbit_params);
            s.gaits[4] = bestiary::make_rabbit_graze(s.rabbit_params);
            s.last_rabbit_params = s.rabbit_params;
        } else if (s.creature_type == CREATURE_BIRD) {
            s.animal_mesh = bestiary::generate_bird_mesh(s.bird_params);
            s.gaits[0] = bestiary::make_bird_walk(s.bird_params);
            s.gaits[1] = bestiary::make_bird_hop(s.bird_params);
            s.gaits[2] = bestiary::make_bird_fly(s.bird_params);
            s.gaits[3] = bestiary::make_bird_idle(s.bird_params);
            s.gaits[4] = bestiary::make_bird_peck(s.bird_params);
            s.gaits[5] = bestiary::make_bird_takeoff(s.bird_params);
            s.gaits[6] = bestiary::make_bird_land(s.bird_params);
            s.gaits[7] = bestiary::make_bird_perch(s.bird_params);
            s.last_bird_params = s.bird_params;
        } else if (s.creature_type == CREATURE_RAPTOR) {
            s.animal_mesh = bestiary::generate_bird_mesh(s.raptor_params);
            s.gaits[0] = bestiary::make_bird_walk(s.raptor_params);
            s.gaits[1] = bestiary::make_bird_hop(s.raptor_params);
            s.gaits[2] = bestiary::make_bird_soar(s.raptor_params);
            s.gaits[3] = bestiary::make_bird_idle(s.raptor_params);
            s.gaits[4] = bestiary::make_bird_dive(s.raptor_params);
            s.gaits[5] = bestiary::make_bird_takeoff(s.raptor_params);
            s.gaits[6] = bestiary::make_bird_land(s.raptor_params);
            s.gaits[7] = bestiary::make_bird_perch(s.raptor_params);
            s.last_raptor_params = s.raptor_params;
        } else {
            s.animal_mesh = bestiary::generate_snake_mesh(s.snake_params);
            s.gaits[0] = bestiary::make_snake_slither(s.snake_params);
            s.gaits[1] = bestiary::make_snake_slither(s.snake_params);
            s.gaits[2] = bestiary::make_snake_fast(s.snake_params);
            s.gaits[3] = bestiary::make_snake_idle(s.snake_params);
            s.gaits[4] = bestiary::make_snake_strike(s.snake_params);
            s.last_snake_params = s.snake_params;
        }

        s.joint_count = static_cast<int>(s.animal_mesh.skeleton.joints.size());
        auto world_transforms = compute_world_transforms(s.animal_mesh.skeleton);
        upload_to_buffer(r.allocator, s.joint_ssbo,
            world_transforms.data(), world_transforms.size() * sizeof(glm::mat4));

        std::vector<uint32_t> bone_indices;
        for (size_t i = 0; i < s.animal_mesh.skeleton.joints.size(); ++i) {
            if (s.animal_mesh.skeleton.joints[i].parent >= 0) {
                bone_indices.push_back(static_cast<uint32_t>(s.animal_mesh.skeleton.joints[i].parent));
                bone_indices.push_back(static_cast<uint32_t>(i));
            }
        }
        s.bone_vertex_count = static_cast<uint32_t>(bone_indices.size());
        destroy_buffer(r.allocator, s.bone_vb);
        s.bone_vb = create_host_buffer(r.allocator,
            bone_indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        upload_to_buffer(r.allocator, s.bone_vb,
            bone_indices.data(), bone_indices.size() * sizeof(uint32_t));

        s.joint_palette.resize(s.joint_count);
        for (int i = 0; i < s.joint_count; ++i)
            s.joint_palette[i] = world_transforms[i] * s.animal_mesh.skeleton.inverse_bind[i];

        bestiary::cpu_skin(s.animal_mesh.vertices, s.joint_palette, s.skinned_verts);

        destroy_buffer(r.allocator, s.mesh_vb);
        destroy_buffer(r.allocator, s.mesh_ib);
        s.mesh_index_count = 0;

        if (!s.skinned_verts.empty()) {
            s.mesh_vb = create_host_buffer(r.allocator,
                s.skinned_verts.size() * sizeof(bestiary::SkinnedVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            upload_to_buffer(r.allocator, s.mesh_vb,
                s.skinned_verts.data(), s.skinned_verts.size() * sizeof(bestiary::SkinnedVertex));

            s.mesh_ib = create_host_buffer(r.allocator,
                s.animal_mesh.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            upload_to_buffer(r.allocator, s.mesh_ib,
                s.animal_mesh.indices.data(), s.animal_mesh.indices.size() * sizeof(uint32_t));
            s.mesh_index_count = static_cast<uint32_t>(s.animal_mesh.indices.size());
        }

        if (s.creature_type != s.last_creature_type) {
            s.anim_phase = 0.0f;
            s.root_offset = 0.0f;
            s.anim_mode = 3;
        }
        s.last_creature_type = s.creature_type;
        s.last_phase = -1.0f;
    }

    if (s.creature_type == CREATURE_HERBIVORE && s.feed_height != s.last_feed_h) {
        s.gaits[4] = bestiary::make_herbivore_graze(s.herb_params, s.feed_height);
        s.last_feed_h = s.feed_height;
        if (s.anim_mode == 5) s.last_phase = -1.0f;
    }

    bool is_bird_type = (s.creature_type == CREATURE_BIRD || s.creature_type == CREATURE_RAPTOR);
    bool is_snake = (s.creature_type == CREATURE_SNAKE);
    auto& bp = (s.creature_type == CREATURE_RAPTOR) ? s.raptor_params : s.bird_params;

    // Resolve blend pair from mode
    int gait_a = 0, gait_b = 0;
    float blend_t = 0.0f;
    bool locomotion = true;

    switch (s.anim_mode) {
    case 0: gait_a = gait_b = 0; break;                       // walk
    case 1: gait_a = gait_b = 1; break;                       // trot
    case 2: gait_a = gait_b = 2; break;                       // run
    case 3:                                                     // move (blended)
        if (s.gait_blend <= 0.5f) { gait_a = 0; gait_b = 1; blend_t = s.gait_blend * 2.0f; }
        else { gait_a = 1; gait_b = 2; blend_t = (s.gait_blend - 0.5f) * 2.0f; }
        break;
    case 4: gait_a = gait_b = 3; locomotion = false; break;   // idle
    case 5: gait_a = gait_b = 4;
        locomotion = (s.creature_type == CREATURE_PREDATOR);
        break;
    case 6: gait_a = gait_b = 5; break;                       // takeoff
    case 7: gait_a = gait_b = 6; break;                       // landing
    case 8: gait_a = gait_b = 7; locomotion = false; break;   // perch
    }

    float blended_period  = glm::mix(s.gaits[gait_a].period_seconds,  s.gaits[gait_b].period_seconds,  blend_t);
    float blended_swing   = glm::mix(s.gaits[gait_a].hip_swing_deg,   s.gaits[gait_b].hip_swing_deg,   blend_t);
    float blended_stance  = glm::mix(s.gaits[gait_a].stance_fraction, s.gaits[gait_b].stance_fraction, blend_t);

    // Animation update
    if (s.anim_playing) {
        float delta_phase = dt / blended_period * s.anim_speed;
        s.anim_phase += delta_phase;
        s.anim_phase = std::fmod(s.anim_phase, 1.0f);
        if (s.anim_phase < 0.0f) s.anim_phase += 1.0f;

        if (s.root_motion && locomotion && blended_swing > 0.0f) {
            float leg;
            if (s.creature_type == CREATURE_HERBIVORE)
                leg = s.herb_params.leg_length_back + s.herb_params.hoof_size;
            else if (s.creature_type == CREATURE_PREDATOR)
                leg = s.pred_params.leg_length_back + s.pred_params.paw_size;
            else if (s.creature_type == CREATURE_RABBIT)
                leg = s.rabbit_params.leg_length_back + s.rabbit_params.paw_size;
            else if (is_bird_type)
                leg = bp.leg_length_upper + bp.leg_length_lower + bp.foot_size;
            else
                leg = s.snake_params.body_length * 0.5f;
            float stride = 2.0f * leg
                         * std::sin(glm::radians(blended_swing))
                         / blended_stance;
            s.root_offset += delta_phase * stride;
        }
    }

    if (s.anim_phase != s.last_phase && s.mesh_vb.buffer != VK_NULL_HANDLE) {
        // Sample and blend
        auto rots_a = bestiary::sample_walk(s.gaits[gait_a], s.anim_phase, s.joint_count);
        auto rots_b = bestiary::sample_walk(s.gaits[gait_b], s.anim_phase, s.joint_count);
        std::vector<glm::quat> rotations(static_cast<size_t>(s.joint_count));
        for (int j = 0; j < s.joint_count; ++j)
            rotations[j] = glm::slerp(rots_a[j], rots_b[j], blend_t);

        float pi = 3.14159265f;
        float bob;
        if (locomotion && s.creature_type == CREATURE_RABBIT) {
            float hop_h = s.rabbit_params.hop_height;
            float arc = std::sin(s.anim_phase * pi);
            bob = hop_h * arc * arc;
        } else if (locomotion && is_bird_type && s.anim_mode == 1) {
            float hop_h = bp.hop_height;
            float arc = std::sin(s.anim_phase * pi);
            bob = hop_h * arc * arc;
        } else if (locomotion && is_bird_type &&
                   (s.anim_mode == 2 || (s.anim_mode == 3 && s.gait_blend > 0.5f))) {
            float fly_factor = (s.anim_mode == 2) ? 1.0f : (s.gait_blend - 0.5f) * 2.0f;
            float fly_h = bp.fly_height * fly_factor;
            float beat = 0.02f * std::cos(s.anim_phase * 2.0f * pi);
            bob = fly_h + beat;
        } else if (is_bird_type && s.anim_mode == 6) {
            bob = bp.fly_height * s.anim_phase;
        } else if (is_bird_type && s.anim_mode == 7) {
            bob = bp.fly_height * (1.0f - s.anim_phase);
        } else if (is_bird_type && s.anim_mode == 8) {
            bob = 0.3f + 0.003f * std::cos(s.anim_phase * 2.0f * pi);
        } else if (is_snake) {
            bob = 0.0f;
        } else if (locomotion && is_bird_type) {
            float bob_amp = 0.012f;
            bob = bob_amp * std::cos(s.anim_phase * 2.0f * pi);
        } else if (locomotion) {
            float bob_amp = glm::mix(0.025f, 0.04f,
                (s.anim_mode == 3) ? s.gait_blend : (s.anim_mode * 0.5f));
            float raw = std::cos(s.anim_phase * 4.0f * pi);
            bob = (raw > 0.0f ? bob_amp * 0.8f : bob_amp * 1.2f) * raw;
        } else if (s.anim_mode == 5) {
            bob = 0.003f * std::cos(s.anim_phase * 2.0f * pi) - s.gaits[4].pelvis_drop;
        } else {
            bob = 0.003f * std::cos(s.anim_phase * 2.0f * pi);
        }

        std::vector<glm::mat4> anim_world(s.joint_count);
        for (int j = 0; j < s.joint_count; ++j) {
            glm::mat4 local_anim;
            if (j == 0) {
                glm::vec3 bind_pos(s.animal_mesh.skeleton.joints[0].local_bind[3]);
                float rz = s.root_motion ? s.root_offset : 0.0f;
                local_anim = glm::translate(glm::mat4(1.0f),
                                            bind_pos + glm::vec3(0.0f, bob, rz))
                           * glm::mat4_cast(rotations[0]);
            } else {
                local_anim = s.animal_mesh.skeleton.joints[j].local_bind
                           * glm::mat4_cast(rotations[j]);
            }
            int p = s.animal_mesh.skeleton.joints[j].parent;
            anim_world[j] = (p == -1) ? local_anim : anim_world[p] * local_anim;
        }

        s.joint_palette.resize(s.joint_count);
        for (int j = 0; j < s.joint_count; ++j)
            s.joint_palette[j] = anim_world[j] * s.animal_mesh.skeleton.inverse_bind[j];

        bestiary::cpu_skin(s.animal_mesh.vertices, s.joint_palette, s.skinned_verts);
        upload_to_buffer(r.allocator, s.mesh_vb,
            s.skinned_verts.data(), s.skinned_verts.size() * sizeof(bestiary::SkinnedVertex));

        upload_to_buffer(r.allocator, s.joint_ssbo,
            anim_world.data(), anim_world.size() * sizeof(glm::mat4));

        s.last_phase = s.anim_phase;
    }

    // Camera follow
    {
        float rz = s.root_motion ? s.root_offset : 0.0f;
        float leg_h;
        if (s.creature_type == CREATURE_HERBIVORE)
            leg_h = s.herb_params.leg_length_back;
        else if (s.creature_type == CREATURE_PREDATOR)
            leg_h = s.pred_params.leg_length_back;
        else if (s.creature_type == CREATURE_RABBIT)
            leg_h = s.rabbit_params.leg_length_back;
        else if (is_bird_type)
            leg_h = bp.leg_length_upper + bp.leg_length_lower;
        else
            leg_h = s.snake_params.body_thickness;
        float cam_y = leg_h * 0.8f;
        if (is_bird_type && s.anim_playing) {
            if (s.anim_mode == 2)
                cam_y += bp.fly_height;
            else if (s.anim_mode == 6)
                cam_y += bp.fly_height * s.anim_phase;
            else if (s.anim_mode == 7)
                cam_y += bp.fly_height * (1.0f - s.anim_phase);
            else if (s.anim_mode == 8)
                cam_y += 0.3f;
        }
        s.camera.target = glm::vec3(0.0f, cam_y, rz);
    }

    // --- ImGui ---
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340, 640), ImGuiCond_FirstUseEver);
    ImGui::Begin("Animals Lab v0.0.5");

    if (s.embedded) {
        if (ImGui::Button("<- Back")) {
            ImGui::End();
            ImGui::Render();
            return false;
        }
        ImGui::Separator();
    }

    {
        const char* creatures[] = {"Deer", "Wolf", "Rabbit", "Bird", "Raptor", "Snake"};
        ImGui::Combo("creature", &s.creature_type, creatures, 6);
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Morphology");
    ImGui::Separator();

    if (s.creature_type == CREATURE_HERBIVORE) {
        ImGui::SliderFloat("torso_length",     &s.herb_params.torso_length,     0.5f,  2.5f, "%.2f m");
        ImGui::SliderFloat("torso_girth",      &s.herb_params.torso_girth,      0.2f,  0.8f, "%.2f m");
        ImGui::SliderFloat("neck_length",      &s.herb_params.neck_length,      0.1f,  1.0f, "%.2f m");
        ImGui::SliderFloat("head_length",      &s.herb_params.head_length,      0.1f,  0.6f, "%.2f m");
        ImGui::SliderFloat("leg_length_front", &s.herb_params.leg_length_front, 0.3f,  1.5f, "%.2f m");
        ImGui::SliderFloat("leg_length_back",  &s.herb_params.leg_length_back,  0.3f,  1.5f, "%.2f m");
        ImGui::SliderFloat("leg_thickness",    &s.herb_params.leg_thickness,    0.02f, 0.15f, "%.3f m");
        ImGui::SliderFloat("hoof_size",        &s.herb_params.hoof_size,        0.02f, 0.12f, "%.3f m");
        ImGui::Spacing();
        ImGui::ColorEdit3("coat_color", s.herb_params.coat_color);
        ImGui::ColorEdit3("belly_color", s.herb_params.belly_color);
    } else if (s.creature_type == CREATURE_PREDATOR) {
        ImGui::SliderFloat("torso_length",     &s.pred_params.torso_length,     0.4f,  1.8f, "%.2f m");
        ImGui::SliderFloat("chest_girth",      &s.pred_params.chest_girth,      0.2f,  0.7f, "%.2f m");
        ImGui::SliderFloat("waist_girth",      &s.pred_params.waist_girth,      0.15f, 0.5f, "%.2f m");
        ImGui::SliderFloat("neck_length",      &s.pred_params.neck_length,      0.1f,  0.6f, "%.2f m");
        ImGui::SliderFloat("snout_length",     &s.pred_params.snout_length,     0.1f,  0.5f, "%.2f m");
        ImGui::SliderFloat("head_width",       &s.pred_params.head_width,       0.08f, 0.3f, "%.2f m");
        ImGui::SliderFloat("leg_length_front", &s.pred_params.leg_length_front, 0.3f,  1.2f, "%.2f m");
        ImGui::SliderFloat("leg_length_back",  &s.pred_params.leg_length_back,  0.3f,  1.3f, "%.2f m");
        ImGui::SliderFloat("leg_thickness",    &s.pred_params.leg_thickness,    0.02f, 0.10f, "%.3f m");
        ImGui::SliderFloat("paw_size",         &s.pred_params.paw_size,         0.02f, 0.08f, "%.3f m");
        ImGui::SliderFloat("tail_length",      &s.pred_params.tail_length,      0.1f,  0.8f, "%.2f m");
        ImGui::Spacing();
        ImGui::ColorEdit3("coat_color", s.pred_params.coat_color);
        ImGui::ColorEdit3("belly_color", s.pred_params.belly_color);
    } else if (s.creature_type == CREATURE_RABBIT) {
        ImGui::SliderFloat("body_length",      &s.rabbit_params.body_length,      0.15f, 0.5f,  "%.2f m");
        ImGui::SliderFloat("body_girth",       &s.rabbit_params.body_girth,       0.08f, 0.25f, "%.2f m");
        ImGui::SliderFloat("head_width",       &s.rabbit_params.head_width,       0.04f, 0.12f, "%.2f m");
        ImGui::SliderFloat("ear_length",       &s.rabbit_params.ear_length,       0.04f, 0.25f, "%.2f m");
        ImGui::SliderFloat("leg_length_front", &s.rabbit_params.leg_length_front, 0.08f, 0.3f,  "%.2f m");
        ImGui::SliderFloat("leg_length_back",  &s.rabbit_params.leg_length_back,  0.10f, 0.4f,  "%.2f m");
        ImGui::SliderFloat("leg_thickness",    &s.rabbit_params.leg_thickness,    0.008f, 0.04f, "%.3f m");
        ImGui::Spacing();
        ImGui::ColorEdit3("coat_color", s.rabbit_params.coat_color);
        ImGui::ColorEdit3("belly_color", s.rabbit_params.belly_color);
    } else if (is_bird_type) {
        ImGui::SliderFloat("body_length",      &bp.body_length,      0.10f, 0.5f,  "%.2f m");
        ImGui::SliderFloat("body_girth",       &bp.body_girth,       0.06f, 0.30f, "%.2f m");
        ImGui::SliderFloat("neck_length",      &bp.neck_length,      0.04f, 0.25f, "%.2f m");
        ImGui::SliderFloat("head_size",        &bp.head_size,        0.02f, 0.10f, "%.2f m");
        ImGui::SliderFloat("beak_length",      &bp.beak_length,      0.01f, 0.08f, "%.2f m");
        ImGui::SliderFloat("wing_length",      &bp.wing_length,      0.08f, 0.60f, "%.2f m");
        ImGui::SliderFloat("wing_width",       &bp.wing_width,       0.02f, 0.15f, "%.3f m");
        ImGui::SliderFloat("wing_taper",       &bp.wing_taper,       0.1f,  0.8f,  "%.2f");
        ImGui::SliderFloat("leg_length_upper", &bp.leg_length_upper, 0.04f, 0.25f, "%.2f m");
        ImGui::SliderFloat("leg_length_lower", &bp.leg_length_lower, 0.02f, 0.15f, "%.2f m");
        ImGui::SliderFloat("leg_thickness",    &bp.leg_thickness,    0.004f, 0.02f, "%.3f m");
        ImGui::SliderFloat("foot_size",        &bp.foot_size,        0.01f, 0.06f, "%.3f m");
        ImGui::SliderFloat("tail_length",      &bp.tail_length,      0.02f, 0.15f, "%.2f m");
        ImGui::Spacing();
        ImGui::ColorEdit3("coat_color", bp.coat_color);
        ImGui::ColorEdit3("belly_color", bp.belly_color);
    } else {
        ImGui::SliderFloat("body_length",    &s.snake_params.body_length,    0.3f, 2.0f,  "%.2f m");
        ImGui::SliderFloat("body_thickness", &s.snake_params.body_thickness, 0.01f, 0.08f, "%.3f m");
        ImGui::SliderFloat("head_width",     &s.snake_params.head_width,     0.015f, 0.06f, "%.3f m");
        ImGui::SliderFloat("head_length",    &s.snake_params.head_length,    0.02f, 0.08f, "%.2f m");
        ImGui::SliderFloat("taper_tail",     &s.snake_params.taper_tail,     0.05f, 0.5f,  "%.2f");
        ImGui::Spacing();
        ImGui::ColorEdit3("coat_color", s.snake_params.coat_color);
        ImGui::ColorEdit3("belly_color", s.snake_params.belly_color);
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Animation");
    ImGui::Separator();

    if (s.creature_type == CREATURE_HERBIVORE) {
        const char* modes[] = {"Walk", "Trot", "Run", "Move", "Idle", "Graze"};
        ImGui::Combo("mode", &s.anim_mode, modes, 6);
    } else if (s.creature_type == CREATURE_PREDATOR) {
        const char* modes[] = {"Walk", "Trot", "Run", "Move", "Idle", "Stalk"};
        ImGui::Combo("mode", &s.anim_mode, modes, 6);
    } else if (s.creature_type == CREATURE_RABBIT) {
        const char* modes[] = {"Hop", "Hop", "Run", "Hop+Run", "Idle", "Graze"};
        ImGui::Combo("mode", &s.anim_mode, modes, 6);
    } else if (s.creature_type == CREATURE_BIRD) {
        const char* modes[] = {"Walk", "Hop", "Fly", "Move", "Idle", "Peck", "Takeoff", "Landing", "Perch"};
        ImGui::Combo("mode", &s.anim_mode, modes, 9);
    } else if (s.creature_type == CREATURE_RAPTOR) {
        const char* modes[] = {"Walk", "Hop", "Soar", "Move", "Idle", "Dive", "Takeoff", "Landing", "Perch"};
        ImGui::Combo("mode", &s.anim_mode, modes, 9);
    } else {
        const char* modes[] = {"Slither", "Slither", "Fast", "Move", "Idle", "Strike"};
        ImGui::Combo("mode", &s.anim_mode, modes, 6);
    }

    if (s.anim_mode == 3)
        ImGui::SliderFloat("gait", &s.gait_blend, 0.0f, 1.0f, "%.2f");

    if (s.anim_mode == 5 && s.creature_type == CREATURE_HERBIVORE) {
        float shoulder_h = s.herb_params.leg_length_front;
        float neck_reach = s.herb_params.neck_length + s.herb_params.head_length;
        float low_shoulder = shoulder_h * 0.80f;
        float min_reach = std::max(0.0f, low_shoulder - neck_reach * 0.94f);
        float max_reach = shoulder_h + neck_reach;

        ImGui::SliderFloat("feed_height", &s.feed_height, 0.0f, max_reach, "%.2f m");
        if (s.feed_height < min_reach)
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f), "can't reach! (min %.2f m)", min_reach);
        ImGui::Text("reach: %.2f - %.2f m", min_reach, max_reach);
        ImGui::Text("niche: %s",
            min_reach < 0.05f ? "ground grazer" :
            min_reach < 0.3f  ? "low browser" :
                                "canopy browser");
    }

    if (s.creature_type == CREATURE_HERBIVORE) {
        ImGui::SliderFloat("base_period",      &s.herb_params.walk_period_seconds, 0.3f, 3.0f, "%.2f s");
        ImGui::SliderFloat("foot_lift_height",  &s.herb_params.foot_lift_height,    0.02f, 0.2f, "%.3f m");
    } else if (s.creature_type == CREATURE_PREDATOR) {
        ImGui::SliderFloat("base_period",      &s.pred_params.walk_period_seconds, 0.3f, 3.0f, "%.2f s");
        ImGui::SliderFloat("foot_lift_height",  &s.pred_params.foot_lift_height,    0.02f, 0.2f, "%.3f m");
    } else if (s.creature_type == CREATURE_RABBIT) {
        ImGui::SliderFloat("hop_period",        &s.rabbit_params.hop_period_seconds, 0.2f, 1.5f, "%.2f s");
        ImGui::SliderFloat("hop_height",        &s.rabbit_params.hop_height,         0.02f, 0.15f, "%.3f m");
    } else if (is_bird_type) {
        ImGui::SliderFloat("walk_period",       &bp.walk_period_seconds, 0.2f, 1.5f, "%.2f s");
        ImGui::SliderFloat("foot_lift_height",  &bp.foot_lift_height,    0.01f, 0.10f, "%.3f m");
        ImGui::SliderFloat("hop_height",        &bp.hop_height,          0.01f, 0.10f, "%.3f m");
        ImGui::Spacing();
        ImGui::SliderFloat("flap_period",       &bp.flap_period,         0.1f,  2.0f,  "%.2f s");
        ImGui::SliderFloat("flap_amplitude",    &bp.flap_amplitude,      10.0f, 70.0f, "%.0f deg");
        ImGui::SliderFloat("flap_sweep",        &bp.flap_sweep,          0.0f,  1.0f,  "%.2f");
        ImGui::SliderFloat("fly_height",        &bp.fly_height,          0.1f,  3.0f,  "%.2f m");
    } else {
        ImGui::SliderFloat("slither_period",    &s.snake_params.slither_period,    0.5f, 3.0f,  "%.2f s");
        ImGui::SliderFloat("slither_amplitude", &s.snake_params.slither_amplitude, 5.0f, 30.0f, "%.1f deg");
        ImGui::SliderFloat("slither_waves",     &s.snake_params.slither_waves,     1.0f, 4.0f,  "%.1f");
    }

    if (ImGui::Button(s.anim_playing ? "Pause" : "Play")) s.anim_playing = !s.anim_playing;
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        s.anim_phase = 0.0f; s.anim_playing = false;
        s.root_offset = 0.0f;
    }
    ImGui::SliderFloat("phase", &s.anim_phase, 0.0f, 0.999f, "%.3f");
    ImGui::SliderFloat("time_scale", &s.anim_speed, 0.0f, 3.0f, "%.1fx");

    if (locomotion) {
        ImGui::Checkbox("root motion", &s.root_motion);
        float leg;
        if (s.creature_type == CREATURE_HERBIVORE)
            leg = s.herb_params.leg_length_back + s.herb_params.hoof_size;
        else if (s.creature_type == CREATURE_PREDATOR)
            leg = s.pred_params.leg_length_back + s.pred_params.paw_size;
        else if (s.creature_type == CREATURE_RABBIT)
            leg = s.rabbit_params.leg_length_back + s.rabbit_params.paw_size;
        else if (is_bird_type)
            leg = bp.leg_length_upper + bp.leg_length_lower + bp.foot_size;
        else
            leg = s.snake_params.body_length * 0.5f;
        float stride = 2.0f * leg * std::sin(glm::radians(blended_swing)) / blended_stance;
        float spd = stride / blended_period;
        ImGui::Text("period: %.2fs  stride: %.3fm  speed: %.2f m/s",
                    blended_period, stride, spd);
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Display");
    ImGui::Separator();
    ImGui::Checkbox("show skeleton", &s.show_skeleton);

    ImGui::Spacing();
    ImGui::TextUnformatted("Save / Load");
    ImGui::Separator();
    ImGui::InputText("name", s.fio.name_buf, sizeof(s.fio.name_buf));

    const auto& sdir = species_dir();
    if (ImGui::Button("Save")) {
        if (!std::filesystem::exists(sdir))
            std::filesystem::create_directories(sdir);
        auto path = sdir / (std::string(s.fio.name_buf) + ".toml");
        auto arch = static_cast<bestiary::Archetype>(s.creature_type);
        bool ok = bestiary::save_animal(path, arch,
            &s.herb_params, &s.pred_params, &s.rabbit_params, &s.bird_params,
            &s.snake_params, s.fio.name_buf);
        s.fio.set_status(ok ? "Saved." : "Save failed.");
        if (ok) s.fio.refresh_files();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load") && s.fio.selected_file >= 0) {
        auto path = sdir / (s.fio.file_list[static_cast<size_t>(s.fio.selected_file)] + ".toml");
        bestiary::Archetype loaded_arch{};
        std::string loaded_name;
        bool ok = bestiary::load_animal(path, loaded_arch, s.herb_params, s.pred_params,
                                        s.rabbit_params, s.bird_params, s.snake_params,
                                        loaded_name);
        if (ok) {
            s.creature_type = static_cast<int>(loaded_arch);
            std::snprintf(s.fio.name_buf, sizeof(s.fio.name_buf), "%s", loaded_name.c_str());
            s.fio.set_status("Loaded.");
        } else {
            s.fio.set_status("Load failed.");
        }
    }

    if (!s.fio.file_list.empty()) {
        std::vector<const char*> items;
        for (auto& str : s.fio.file_list) items.push_back(str.c_str());
        ImGui::ListBox("##files", &s.fio.selected_file,
                       items.data(), static_cast<int>(items.size()), 4);
    }

    s.fio.status_timer -= dt;
    if (s.fio.status_timer > 0.0f)
        ImGui::TextUnformatted(s.fio.status_buf);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("[INFO] joints: %d  verts: %u  tris: %u",
                s.joint_count,
                static_cast<uint32_t>(s.skinned_verts.size()),
                s.mesh_index_count / 3);
    ImGui::Text("%.1f ms/frame", dt * 1000.0f);
    ImGui::TextUnformatted("RMB drag to orbit, scroll to zoom.");

    ImGui::End();
    ImGui::Render();

    return true;
}

void animals_lab_render(AnimalsLabState& s, Renderer& r,
                        FrameData& frame, uint32_t image_index, VkExtent2D extent)
{
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 view = orbit_view(s.camera);
    glm::mat4 mvp  = proj * view;

    float joint_radius =
        (s.creature_type == CREATURE_HERBIVORE) ? s.herb_params.leg_thickness
        : (s.creature_type == CREATURE_PREDATOR) ? s.pred_params.leg_thickness
        : (s.creature_type == CREATURE_RABBIT) ? s.rabbit_params.leg_thickness
        : (s.creature_type == CREATURE_RAPTOR) ? s.raptor_params.leg_thickness
        : (s.creature_type == CREATURE_SNAKE) ? s.snake_params.body_thickness
        : s.bird_params.leg_thickness;

    record_frame(r, frame, image_index, extent,
                 mvp,
                 s.mesh_pipe, s.ground,
                 s.mesh_vb, s.mesh_ib, s.mesh_index_count,
                 s.show_skeleton,
                 s.skel_pipe, s.sphere_vb, s.sphere_ib, s.sphere_index_count,
                 s.bone_vb, s.bone_vertex_count,
                 s.desc_set, s.joint_count, joint_radius);
}

void animals_lab_shutdown(AnimalsLabState& s, Renderer& r)
{
    vkDeviceWaitIdle(r.device);

    destroy_buffer(r.allocator, s.mesh_ib);
    destroy_buffer(r.allocator, s.mesh_vb);
    destroy_buffer(r.allocator, s.ground.ib);
    destroy_buffer(r.allocator, s.ground.vb);
    destroy_buffer(r.allocator, s.bone_vb);
    destroy_buffer(r.allocator, s.joint_ssbo);
    destroy_buffer(r.allocator, s.sphere_ib);
    destroy_buffer(r.allocator, s.sphere_vb);
    vkDestroyDescriptorPool(r.device, s.desc_pool, nullptr);
    destroy_skeleton_pipeline(r.device, s.skel_pipe);
    destroy_mesh_pipeline(r.device, s.mesh_pipe);

    s.initialized = false;
}

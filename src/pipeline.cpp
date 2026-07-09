#include "pipeline.h"
#include "vk_util.h"
#include <cstdio>

// Standard single-set compute pipeline quartet (shader module, descriptor set
// layout with binding i = bindings[i], pipeline layout with one push-constant
// range, pipeline), unpacked into the Pipelines struct's loose fields.
static void create_compute(VkDevice device, const char* spv,
                           const std::vector<VkDescriptorType>& bindings,
                           uint32_t push_size,
                           VkShaderModule& shader, VkDescriptorSetLayout& dsl,
                           VkPipelineLayout& layout, VkPipeline& pipeline)
{
    ComputePipeline cp = make_compute_pipeline(device, spv, bindings, push_size);
    shader   = cp.shader;
    dsl      = cp.dsl;
    layout   = cp.layout;
    pipeline = cp.pipeline;
}

void pipelines_create(Pipelines& p, VkDevice device, VkFormat color_format,
                      VkFormat present_format)
{
    p.color_format = color_format;
    p.present_format = (present_format == VK_FORMAT_UNDEFINED) ? color_format
                                                               : present_format;

    // ---- Compute pipelines ---------------------------------------------------
    create_compute(device, "shaders/swe_init.spv",
                   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                   sizeof(SweInitPC),
                   p.swe_init_shader, p.swe_init_desc_layout,
                   p.swe_init_pipeline_layout, p.swe_init_pipeline);

    create_compute(device, "shaders/swe_step.spv",
                   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLER},
                   sizeof(SweStepPC),
                   p.swe_step_shader, p.swe_step_desc_layout,
                   p.swe_step_pipeline_layout, p.swe_step_pipeline);

    create_compute(device, "shaders/terrain_brush.spv",
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                   sizeof(TerrainBrushPC),
                   p.terrain_brush_shader, p.tb_desc_layout,
                   p.tb_pipeline_layout, p.terrain_brush_pipeline);

    create_compute(device, "shaders/erosion.spv",
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLER},
                   sizeof(ErosionPC),
                   p.erosion_shader, p.ero_desc_layout,
                   p.ero_pipeline_layout, p.erosion_pipeline);

    create_compute(device, "shaders/atmosphere3d_cs.spv",
                   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_SAMPLER},
                   sizeof(Atmo3DPC),
                   p.atmo_shader, p.atmo_desc_layout,
                   p.atmo_pipeline_layout, p.atmo_pipeline);

    create_compute(device, "shaders/planet_gen_cs.spv",
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(PlanetGenPC),
                   p.terrain_gen_shader, p.terrain_gen_desc_layout,
                   p.terrain_gen_pipeline_layout, p.terrain_gen_pipeline);

    create_compute(device, "shaders/planet_swe_init_cs.spv",
                   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,   // hydrology field (water surface)
                    VK_DESCRIPTOR_TYPE_SAMPLER},
                   sizeof(PlanetSweInitPC),
                   p.planet_swe_init_shader, p.planet_swe_init_desc_layout,
                   p.planet_swe_init_pipeline_layout, p.planet_swe_init_pipeline);

    create_compute(device, "shaders/planet_swe_step_cs.spv",
                   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                   sizeof(PlanetSweStepPC),
                   p.planet_swe_step_shader, p.planet_swe_step_desc_layout,
                   p.planet_swe_step_pipeline_layout, p.planet_swe_step_pipeline);

    create_compute(device, "shaders/planet_swe_h_adjust_cs.spv",
                   {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                   sizeof(PlanetSweHAdjustPC),
                   p.planet_swe_h_adjust_shader, p.planet_swe_h_adjust_desc_layout,
                   p.planet_swe_h_adjust_pipeline_layout, p.planet_swe_h_adjust_pipeline);

    create_compute(device, "shaders/sand_sim_cs.spv",
                   {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    VK_DESCRIPTOR_TYPE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_SAMPLER,
                    VK_DESCRIPTOR_TYPE_SAMPLER},
                   sizeof(SandSimPC),
                   p.sand_sim_shader, p.sand_sim_desc_layout,
                   p.sand_sim_pipeline_layout, p.sand_sim_pipeline);

    // ---- Graphics pipeline (terrain) ----------------------------------------
    p.terrain_vs = make_shader(device, "shaders/terrain_vs.spv");
    p.terrain_fs = make_shader(device, "shaders/terrain_fs.spv");

    // Graphics descriptor set layout: UBO + 7 textures (SAMPLED_IMAGE) + 1 shared sampler
    VkDescriptorSetLayoutBinding gfx_bindings[10]{};
    gfx_bindings[0].binding = 0;
    gfx_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    gfx_bindings[0].descriptorCount = 1;
    gfx_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[1].binding = 1;
    gfx_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    gfx_bindings[1].descriptorCount = 1;
    gfx_bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[2].binding = 2;
    gfx_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    gfx_bindings[2].descriptorCount = 1;
    gfx_bindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[3].binding = 3;
    gfx_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    gfx_bindings[3].descriptorCount = 1;
    gfx_bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[4].binding = 4;
    gfx_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    gfx_bindings[4].descriptorCount = 1;
    gfx_bindings[4].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[5].binding = 5;
    gfx_bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    gfx_bindings[5].descriptorCount = 1;
    gfx_bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[6].binding = 6;
    gfx_bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    gfx_bindings[6].descriptorCount = 1;
    gfx_bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[7].binding = 7;
    gfx_bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    gfx_bindings[7].descriptorCount = 1;
    gfx_bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[8].binding = 8;
    gfx_bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    gfx_bindings[8].descriptorCount = 1;
    gfx_bindings[8].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[9].binding = 9;
    gfx_bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;   // climate (sst/current)
    gfx_bindings[9].descriptorCount = 1;
    gfx_bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo gfx_dsl_ci{};
    gfx_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    gfx_dsl_ci.bindingCount = 10;
    gfx_dsl_ci.pBindings = gfx_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &gfx_dsl_ci, nullptr, &p.gfx_desc_layout));

    VkPushConstantRange gfx_push_range{};
    gfx_push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    gfx_push_range.offset = 0;
    gfx_push_range.size = sizeof(GfxPC);

    VkPipelineLayoutCreateInfo gfx_pl_ci{};
    gfx_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_pl_ci.setLayoutCount = 1;
    gfx_pl_ci.pSetLayouts = &p.gfx_desc_layout;
    gfx_pl_ci.pushConstantRangeCount = 1;
    gfx_pl_ci.pPushConstantRanges = &gfx_push_range;

    VK_CHECK(vkCreatePipelineLayout(device, &gfx_pl_ci, nullptr, &p.gfx_pipeline_layout));

    VkPushConstantRange rm_push_range{};
    rm_push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    rm_push_range.offset = 0;
    rm_push_range.size = sizeof(RaymarchPC);

    VkPipelineLayoutCreateInfo rm_pl_ci{};
    rm_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    rm_pl_ci.setLayoutCount = 1;
    rm_pl_ci.pSetLayouts = &p.gfx_desc_layout;
    rm_pl_ci.pushConstantRangeCount = 1;
    rm_pl_ci.pPushConstantRanges = &rm_push_range;

    VK_CHECK(vkCreatePipelineLayout(device, &rm_pl_ci, nullptr, &p.raymarch_pipeline_layout));

    VkPipelineShaderStageCreateInfo gfx_stages[2]{};
    gfx_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    gfx_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    gfx_stages[0].module = p.terrain_vs;
    gfx_stages[0].pName = "main";

    gfx_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    gfx_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    gfx_stages[1].module = p.terrain_fs;
    gfx_stages[1].pName = "main";

    VkVertexInputBindingDescription vertex_binding{};
    vertex_binding.binding = 0;
    vertex_binding.stride = sizeof(float) * 2;
    vertex_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vertex_attr{};
    vertex_attr.binding = 0;
    vertex_attr.location = 0;
    vertex_attr.format = VK_FORMAT_R32G32_SFLOAT;
    vertex_attr.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &vertex_binding;
    vertex_input.vertexAttributeDescriptionCount = 1;
    vertex_input.pVertexAttributeDescriptions = &vertex_attr;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                    | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &blend_attachment;

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    VkPipelineRenderingCreateInfo rendering_ci{};
    rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_ci.colorAttachmentCount = 1;
    rendering_ci.pColorAttachmentFormats = &p.color_format;
    rendering_ci.depthAttachmentFormat = depth_format;

    VkGraphicsPipelineCreateInfo gfx_pipe_ci{};
    gfx_pipe_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_pipe_ci.pNext = &rendering_ci;
    gfx_pipe_ci.stageCount = 2;
    gfx_pipe_ci.pStages = gfx_stages;
    gfx_pipe_ci.pVertexInputState = &vertex_input;
    gfx_pipe_ci.pInputAssemblyState = &input_assembly;
    gfx_pipe_ci.pViewportState = &viewport_state;
    gfx_pipe_ci.pRasterizationState = &rasterizer;
    gfx_pipe_ci.pMultisampleState = &multisample;
    gfx_pipe_ci.pDepthStencilState = &depth_stencil;
    gfx_pipe_ci.pColorBlendState = &color_blend;
    gfx_pipe_ci.pDynamicState = &dynamic_state;
    gfx_pipe_ci.layout = p.gfx_pipeline_layout;

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gfx_pipe_ci, nullptr, &p.gfx_pipeline));

    // ---- Planet terrain graphics pipeline layout + pipeline -------------------
    VkPushConstantRange clip_gfx_push{};
    clip_gfx_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    clip_gfx_push.offset = 0;
    clip_gfx_push.size = sizeof(PlanetTilePC);

    VkPipelineLayoutCreateInfo clip_gfx_pl_ci{};
    clip_gfx_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    clip_gfx_pl_ci.setLayoutCount = 1;
    clip_gfx_pl_ci.pSetLayouts = &p.gfx_desc_layout;
    clip_gfx_pl_ci.pushConstantRangeCount = 1;
    clip_gfx_pl_ci.pPushConstantRanges = &clip_gfx_push;

    VK_CHECK(vkCreatePipelineLayout(device, &clip_gfx_pl_ci, nullptr, &p.clipmap_gfx_pipeline_layout));

    VkGraphicsPipelineCreateInfo clip_gfx_pipe_ci = gfx_pipe_ci;
    clip_gfx_pipe_ci.layout = p.clipmap_gfx_pipeline_layout;

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &clip_gfx_pipe_ci, nullptr, &p.clipmap_terrain_pipeline));

    // ---- Water pipeline (transparent, same layout) --------------------------
    p.water_vs = make_shader(device, "shaders/water_vs.spv");
    p.water_fs = make_shader(device, "shaders/water_fs.spv");

    VkPipelineShaderStageCreateInfo water_stages[2]{};
    water_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    water_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    water_stages[0].module = p.water_vs;
    water_stages[0].pName = "main";

    water_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    water_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    water_stages[1].module = p.water_fs;
    water_stages[1].pName = "main";

    VkPipelineDepthStencilStateCreateInfo water_depth_stencil{};
    water_depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    water_depth_stencil.depthTestEnable = VK_TRUE;
    water_depth_stencil.depthWriteEnable = VK_FALSE;
    water_depth_stencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    VkPipelineColorBlendAttachmentState water_blend_attachment{};
    water_blend_attachment.blendEnable = VK_TRUE;
    water_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    water_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    water_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    water_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    water_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    water_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    water_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo water_color_blend{};
    water_color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    water_color_blend.attachmentCount = 1;
    water_color_blend.pAttachments = &water_blend_attachment;

    VkGraphicsPipelineCreateInfo water_pipe_ci = gfx_pipe_ci;
    water_pipe_ci.pStages = water_stages;
    water_pipe_ci.pDepthStencilState = &water_depth_stencil;
    water_pipe_ci.pColorBlendState = &water_color_blend;

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &water_pipe_ci, nullptr, &p.water_pipeline));

    // ---- River overlay pipeline (animated drainage, alpha blend, no depth write)
    {
        p.river_vs = make_shader(device, "shaders/river_overlay_vs.spv");
        p.river_fs = make_shader(device, "shaders/river_overlay_fs.spv");

        // Own minimal descriptor layout: camera UBO + heightmap pool + hydrology + sampler.
        VkDescriptorSetLayoutBinding rb[4]{};
        rb[0].binding = 0;
        rb[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        rb[0].descriptorCount = 1;
        rb[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; // FS reads sun for water lighting
        rb[1].binding = 1;
        rb[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;   // terrain heightmap pool
        rb[1].descriptorCount = 1;
        rb[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        rb[2].binding = 2;
        rb[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;   // hydrology field
        rb[2].descriptorCount = 1;
        rb[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        rb[3].binding = 3;
        rb[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        rb[3].descriptorCount = 1;
        rb[3].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo rdsl_ci{};
        rdsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        rdsl_ci.bindingCount = 4;
        rdsl_ci.pBindings = rb;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &rdsl_ci, nullptr, &p.river_desc_layout));

        VkPushConstantRange river_push{};
        river_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        river_push.offset = 0;
        river_push.size = sizeof(RiverOverlayPC);

        VkPipelineLayoutCreateInfo river_pl_ci{};
        river_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        river_pl_ci.setLayoutCount = 1;
        river_pl_ci.pSetLayouts = &p.river_desc_layout;
        river_pl_ci.pushConstantRangeCount = 1;
        river_pl_ci.pPushConstantRanges = &river_push;
        VK_CHECK(vkCreatePipelineLayout(device, &river_pl_ci, nullptr, &p.river_pipeline_layout));

        VkPipelineShaderStageCreateInfo river_stages[2]{};
        river_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        river_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        river_stages[0].module = p.river_vs;
        river_stages[0].pName = "main";
        river_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        river_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        river_stages[1].module = p.river_fs;
        river_stages[1].pName = "main";

        // Same tile grid mesh, alpha blend, depth test on / write off (like water).
        VkGraphicsPipelineCreateInfo river_pipe_ci = gfx_pipe_ci;
        river_pipe_ci.pStages = river_stages;
        river_pipe_ci.pDepthStencilState = &water_depth_stencil;
        river_pipe_ci.pColorBlendState = &water_color_blend;
        river_pipe_ci.layout = p.river_pipeline_layout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &river_pipe_ci, nullptr, &p.river_pipeline));
    }

    // ---- Cloud raymarch pipeline (fullscreen, alpha blend, no depth) ---------
    p.raymarch_vs = make_shader(device, "shaders/cloud_raymarch_vs.spv");
    p.raymarch_fs = make_shader(device, "shaders/cloud_raymarch_fs.spv");

    VkPipelineShaderStageCreateInfo rm_stages[2]{};
    rm_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rm_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    rm_stages[0].module = p.raymarch_vs;
    rm_stages[0].pName = "main";
    rm_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rm_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    rm_stages[1].module = p.raymarch_fs;
    rm_stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo rm_vi{};
    rm_vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineDepthStencilStateCreateInfo rm_ds{};
    rm_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    rm_ds.depthTestEnable = VK_FALSE;
    rm_ds.depthWriteEnable = VK_FALSE;

    VkGraphicsPipelineCreateInfo rm_pipe_ci = gfx_pipe_ci;
    rm_pipe_ci.pStages = rm_stages;
    rm_pipe_ci.pVertexInputState = &rm_vi;
    rm_pipe_ci.pDepthStencilState = &rm_ds;
    rm_pipe_ci.pColorBlendState = &water_color_blend;
    rm_pipe_ci.layout = p.raymarch_pipeline_layout;

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &rm_pipe_ci, nullptr, &p.raymarch_pipeline));

    // ---- Sand render descriptor set layout + pipeline layout --------------------
    VkDescriptorSetLayoutBinding sand_render_bindings[2]{};
    sand_render_bindings[0].binding = 0;
    sand_render_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sand_render_bindings[0].descriptorCount = 1;
    sand_render_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    sand_render_bindings[1].binding = 1;
    sand_render_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sand_render_bindings[1].descriptorCount = 1;
    sand_render_bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo sand_render_dsl_ci{};
    sand_render_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sand_render_dsl_ci.bindingCount = 2;
    sand_render_dsl_ci.pBindings = sand_render_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &sand_render_dsl_ci, nullptr, &p.sand_render_desc_layout));

    VkPushConstantRange sand_render_push{};
    sand_render_push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    sand_render_push.offset = 0;
    sand_render_push.size = sizeof(SandRenderPC);

    VkPipelineLayoutCreateInfo sand_render_pl_ci{};
    sand_render_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    sand_render_pl_ci.setLayoutCount = 1;
    sand_render_pl_ci.pSetLayouts = &p.sand_render_desc_layout;
    sand_render_pl_ci.pushConstantRangeCount = 1;
    sand_render_pl_ci.pPushConstantRanges = &sand_render_push;

    VK_CHECK(vkCreatePipelineLayout(device, &sand_render_pl_ci, nullptr, &p.sand_render_pipeline_layout));

    // ---- Sand render shaders + pipeline -----------------------------------------
    p.sand_render_vs = make_shader(device, "shaders/sand_render_vs.spv");
    p.sand_render_fs = make_shader(device, "shaders/sand_render_fs.spv");

    {
        VkPipelineShaderStageCreateInfo sand_stages[2]{};
        sand_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sand_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        sand_stages[0].module = p.sand_render_vs;
        sand_stages[0].pName = "main";
        sand_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sand_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        sand_stages[1].module = p.sand_render_fs;
        sand_stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo sand_vi{};
        sand_vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo sand_ia{};
        sand_ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        sand_ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

        VkPipelineDepthStencilStateCreateInfo sand_ds{};
        sand_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        sand_ds.depthTestEnable = VK_TRUE;
        sand_ds.depthWriteEnable = VK_FALSE;
        sand_ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

        VkPipelineColorBlendAttachmentState sand_ba{};
        sand_ba.blendEnable = VK_TRUE;
        sand_ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        sand_ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        sand_ba.colorBlendOp = VK_BLEND_OP_ADD;
        sand_ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        sand_ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        sand_ba.alphaBlendOp = VK_BLEND_OP_ADD;
        sand_ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                               | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo sand_cb{};
        sand_cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        sand_cb.attachmentCount = 1;
        sand_cb.pAttachments = &sand_ba;

        VkGraphicsPipelineCreateInfo sand_pipe_ci = gfx_pipe_ci;
        sand_pipe_ci.pStages = sand_stages;
        sand_pipe_ci.pVertexInputState = &sand_vi;
        sand_pipe_ci.pInputAssemblyState = &sand_ia;
        sand_pipe_ci.pDepthStencilState = &sand_ds;
        sand_pipe_ci.pColorBlendState = &sand_cb;
        sand_pipe_ci.layout = p.sand_render_pipeline_layout;

        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &sand_pipe_ci, nullptr, &p.sand_render_pipeline));
    }

    // ---- Sky pipeline (fullscreen atmosphere raymarch) ------------------------
    // Depth EQUAL against the reverse-Z clear value (0): the fullscreen triangle
    // sits at z = 0, so only pixels no geometry touched get sky. Opaque.
    {
        p.fullscreen_vs = make_shader(device, "shaders/fullscreen_vs.spv");
        p.sky_fs = make_shader(device, "shaders/sky_fs.spv");

        VkDescriptorSetLayoutBinding sb{};
        sb.binding = 0;
        sb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;   // camera UBO
        sb.descriptorCount = 1;
        sb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo sdsl_ci{};
        sdsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        sdsl_ci.bindingCount = 1;
        sdsl_ci.pBindings = &sb;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &sdsl_ci, nullptr, &p.sky_desc_layout));

        VkPushConstantRange sky_push{};
        sky_push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        sky_push.offset = 0;
        sky_push.size = sizeof(SkyPC);

        VkPipelineLayoutCreateInfo sky_pl_ci{};
        sky_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        sky_pl_ci.setLayoutCount = 1;
        sky_pl_ci.pSetLayouts = &p.sky_desc_layout;
        sky_pl_ci.pushConstantRangeCount = 1;
        sky_pl_ci.pPushConstantRanges = &sky_push;
        VK_CHECK(vkCreatePipelineLayout(device, &sky_pl_ci, nullptr, &p.sky_pipeline_layout));

        VkPipelineShaderStageCreateInfo sky_stages[2]{};
        sky_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sky_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        sky_stages[0].module = p.fullscreen_vs;
        sky_stages[0].pName = "main";
        sky_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sky_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        sky_stages[1].module = p.sky_fs;
        sky_stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo sky_vi{};
        sky_vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineDepthStencilStateCreateInfo sky_ds{};
        sky_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        sky_ds.depthTestEnable = VK_TRUE;
        sky_ds.depthWriteEnable = VK_FALSE;
        sky_ds.depthCompareOp = VK_COMPARE_OP_EQUAL;

        VkGraphicsPipelineCreateInfo sky_pipe_ci = gfx_pipe_ci;
        sky_pipe_ci.pStages = sky_stages;
        sky_pipe_ci.pVertexInputState = &sky_vi;
        sky_pipe_ci.pDepthStencilState = &sky_ds;
        sky_pipe_ci.pColorBlendState = &color_blend;   // opaque
        sky_pipe_ci.layout = p.sky_pipeline_layout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &sky_pipe_ci, nullptr, &p.sky_pipeline));
    }

    // ---- Tonemap pipeline (HDR scene → present format) ------------------------
    {
        p.tonemap_fs = make_shader(device, "shaders/tonemap_fs.spv");

        VkDescriptorSetLayoutBinding tb2[2]{};
        tb2[0].binding = 0;
        tb2[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;   // HDR scene
        tb2[0].descriptorCount = 1;
        tb2[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        tb2[1].binding = 1;
        tb2[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        tb2[1].descriptorCount = 1;
        tb2[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo tdsl_ci{};
        tdsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        tdsl_ci.bindingCount = 2;
        tdsl_ci.pBindings = tb2;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &tdsl_ci, nullptr, &p.tonemap_desc_layout));

        VkPushConstantRange tm_push{};
        tm_push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        tm_push.offset = 0;
        tm_push.size = sizeof(TonemapPC);

        VkPipelineLayoutCreateInfo tm_pl_ci{};
        tm_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        tm_pl_ci.setLayoutCount = 1;
        tm_pl_ci.pSetLayouts = &p.tonemap_desc_layout;
        tm_pl_ci.pushConstantRangeCount = 1;
        tm_pl_ci.pPushConstantRanges = &tm_push;
        VK_CHECK(vkCreatePipelineLayout(device, &tm_pl_ci, nullptr, &p.tonemap_pipeline_layout));

        VkPipelineShaderStageCreateInfo tm_stages[2]{};
        tm_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        tm_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        tm_stages[0].module = p.fullscreen_vs;
        tm_stages[0].pName = "main";
        tm_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        tm_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        tm_stages[1].module = p.tonemap_fs;
        tm_stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo tm_vi{};
        tm_vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineDepthStencilStateCreateInfo tm_ds{};
        tm_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

        // Own rendering info: writes the PRESENT format, no depth attachment.
        VkPipelineRenderingCreateInfo tm_rendering_ci{};
        tm_rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        tm_rendering_ci.colorAttachmentCount = 1;
        tm_rendering_ci.pColorAttachmentFormats = &p.present_format;
        tm_rendering_ci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

        VkGraphicsPipelineCreateInfo tm_pipe_ci = gfx_pipe_ci;
        tm_pipe_ci.pNext = &tm_rendering_ci;
        tm_pipe_ci.pStages = tm_stages;
        tm_pipe_ci.pVertexInputState = &tm_vi;
        tm_pipe_ci.pDepthStencilState = &tm_ds;
        tm_pipe_ci.pColorBlendState = &color_blend;
        tm_pipe_ci.layout = p.tonemap_pipeline_layout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &tm_pipe_ci, nullptr, &p.tonemap_pipeline));
    }
}

void pipelines_reload(Pipelines& p, VkDevice device)
{
    std::fprintf(stderr, "[shader reload] Compiling shaders...\n");

    // Shader entries: {hlsl_relative, spv_relative, stage}
    struct ShaderEntry {
        const char* hlsl;
        const char* spv;
        const char* stage;
    };

    const ShaderEntry entries[] = {
        {"../shaders/swe_init.hlsl",       "shaders/swe_init.spv",       "compute"},
        {"../shaders/swe_step.hlsl",       "shaders/swe_step.spv",       "compute"},
        {"../shaders/terrain.vs.hlsl",     "shaders/terrain_vs.spv",     "vertex"},
        {"../shaders/terrain.fs.hlsl",     "shaders/terrain_fs.spv",     "fragment"},
        {"../shaders/water.vs.hlsl",       "shaders/water_vs.spv",       "vertex"},
        {"../shaders/water.fs.hlsl",       "shaders/water_fs.spv",       "fragment"},
        {"../shaders/river_overlay.vs.hlsl", "shaders/river_overlay_vs.spv", "vertex"},
        {"../shaders/river_overlay.fs.hlsl", "shaders/river_overlay_fs.spv", "fragment"},
        {"../shaders/terrain_brush.hlsl",  "shaders/terrain_brush.spv",  "compute"},
        {"../shaders/erosion.hlsl",        "shaders/erosion.spv",        "compute"},
        {"../shaders/atmosphere3d.cs.hlsl", "shaders/atmosphere3d_cs.spv", "compute"},
        {"../shaders/cloud_raymarch.vs.hlsl", "shaders/cloud_raymarch_vs.spv", "vertex"},
        {"../shaders/cloud_raymarch.fs.hlsl", "shaders/cloud_raymarch_fs.spv", "fragment"},
        {"../shaders/sand_sim.cs.hlsl",      "shaders/sand_sim_cs.spv",      "compute"},
        {"../shaders/sand_render.vs.hlsl",   "shaders/sand_render_vs.spv",   "vertex"},
        {"../shaders/sand_render.fs.hlsl",   "shaders/sand_render_fs.spv",   "fragment"},
        {"../shaders/planet_gen.cs.hlsl",    "shaders/planet_gen_cs.spv",    "compute"},
        {"../shaders/planet_swe_init.cs.hlsl", "shaders/planet_swe_init_cs.spv", "compute"},
        {"../shaders/planet_swe_step.cs.hlsl", "shaders/planet_swe_step_cs.spv", "compute"},
        {"../shaders/planet_swe_h_adjust.cs.hlsl", "shaders/planet_swe_h_adjust_cs.spv", "compute"},
        {"../shaders/fullscreen.vs.hlsl",  "shaders/fullscreen_vs.spv",  "vertex"},
        {"../shaders/sky.fs.hlsl",         "shaders/sky_fs.spv",         "fragment"},
        {"../shaders/tonemap.fs.hlsl",     "shaders/tonemap_fs.spv",     "fragment"},
    };
    constexpr int NUM_SHADERS = static_cast<int>(sizeof(entries) / sizeof(entries[0]));

    // Compile all shaders first; abort on any failure
    for (int i = 0; i < NUM_SHADERS; ++i) {
        char cmd[512];
        std::snprintf(cmd, sizeof(cmd),
            "glslc -x hlsl -fshader-stage=%s -fentry-point=main "
            "--target-env=vulkan1.3 %s -o %s 2>&1",
            entries[i].stage, entries[i].hlsl, entries[i].spv);

        FILE* proc = popen(cmd, "r");
        if (!proc) {
            std::fprintf(stderr, "[shader reload] Failed to run glslc for %s\n", entries[i].hlsl);
            return;
        }

        char line[1024];
        while (std::fgets(line, sizeof(line), proc)) {
            std::fprintf(stderr, "  %s", line);
        }

        int status = pclose(proc);
        if (status != 0) {
            std::fprintf(stderr, "[shader reload] FAILED to compile %s (exit %d). Pipelines unchanged.\n",
                         entries[i].hlsl, status);
            return;
        }
    }

    std::fprintf(stderr, "[shader reload] All %d shaders compiled. Recreating pipelines...\n", NUM_SHADERS);

    // Tear everything down and rebuild from the fresh SPIR-V. Recreating the
    // descriptor set layouts is safe: the new layouts are identical, so
    // descriptor sets allocated against the old ones stay binding-compatible.
    vkDeviceWaitIdle(device);
    pipelines_destroy(p, device);
    pipelines_create(p, device, p.color_format, p.present_format);
}

void pipelines_destroy(Pipelines& p, VkDevice device)
{
    // Destroy pipelines first
    vkDestroyPipeline(device, p.clipmap_terrain_pipeline, nullptr);
    vkDestroyPipeline(device, p.terrain_gen_pipeline, nullptr);
    vkDestroyPipeline(device, p.sand_render_pipeline, nullptr);
    vkDestroyPipeline(device, p.sand_sim_pipeline, nullptr);
    vkDestroyPipeline(device, p.water_pipeline, nullptr);
    vkDestroyPipeline(device, p.raymarch_pipeline, nullptr);
    vkDestroyPipeline(device, p.atmo_pipeline, nullptr);
    vkDestroyPipeline(device, p.erosion_pipeline, nullptr);
    vkDestroyPipeline(device, p.terrain_brush_pipeline, nullptr);
    vkDestroyPipeline(device, p.gfx_pipeline, nullptr);
    vkDestroyPipeline(device, p.swe_step_pipeline, nullptr);
    vkDestroyPipeline(device, p.swe_init_pipeline, nullptr);
    vkDestroyPipeline(device, p.planet_swe_init_pipeline, nullptr);
    vkDestroyPipeline(device, p.planet_swe_step_pipeline, nullptr);
    vkDestroyPipeline(device, p.planet_swe_h_adjust_pipeline, nullptr);
    vkDestroyPipeline(device, p.river_pipeline, nullptr);
    vkDestroyPipeline(device, p.sky_pipeline, nullptr);
    vkDestroyPipeline(device, p.tonemap_pipeline, nullptr);

    // Destroy pipeline layouts
    vkDestroyPipelineLayout(device, p.river_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.sky_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.tonemap_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.clipmap_gfx_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.terrain_gen_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.sand_render_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.sand_sim_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.raymarch_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.atmo_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.ero_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.tb_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.gfx_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.swe_step_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.swe_init_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.planet_swe_init_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.planet_swe_step_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.planet_swe_h_adjust_pipeline_layout, nullptr);

    // Destroy descriptor set layouts
    vkDestroyDescriptorSetLayout(device, p.river_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.sky_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.tonemap_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.sand_render_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.sand_sim_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.terrain_gen_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.atmo_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.ero_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.tb_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.gfx_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.swe_step_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.swe_init_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.planet_swe_init_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.planet_swe_step_desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, p.planet_swe_h_adjust_desc_layout, nullptr);

    // Destroy shader modules
    vkDestroyShaderModule(device, p.river_vs, nullptr);
    vkDestroyShaderModule(device, p.river_fs, nullptr);
    vkDestroyShaderModule(device, p.fullscreen_vs, nullptr);
    vkDestroyShaderModule(device, p.sky_fs, nullptr);
    vkDestroyShaderModule(device, p.tonemap_fs, nullptr);
    vkDestroyShaderModule(device, p.terrain_gen_shader, nullptr);
    vkDestroyShaderModule(device, p.sand_render_fs, nullptr);
    vkDestroyShaderModule(device, p.sand_render_vs, nullptr);
    vkDestroyShaderModule(device, p.sand_sim_shader, nullptr);
    vkDestroyShaderModule(device, p.raymarch_fs, nullptr);
    vkDestroyShaderModule(device, p.raymarch_vs, nullptr);
    vkDestroyShaderModule(device, p.atmo_shader, nullptr);
    vkDestroyShaderModule(device, p.erosion_shader, nullptr);
    vkDestroyShaderModule(device, p.terrain_brush_shader, nullptr);
    vkDestroyShaderModule(device, p.water_fs, nullptr);
    vkDestroyShaderModule(device, p.water_vs, nullptr);
    vkDestroyShaderModule(device, p.terrain_fs, nullptr);
    vkDestroyShaderModule(device, p.terrain_vs, nullptr);
    vkDestroyShaderModule(device, p.swe_step_shader, nullptr);
    vkDestroyShaderModule(device, p.swe_init_shader, nullptr);
    vkDestroyShaderModule(device, p.planet_swe_init_shader, nullptr);
    vkDestroyShaderModule(device, p.planet_swe_step_shader, nullptr);
    vkDestroyShaderModule(device, p.planet_swe_h_adjust_shader, nullptr);
}

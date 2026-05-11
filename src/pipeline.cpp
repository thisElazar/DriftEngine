#include "pipeline.h"
#include "vk_util.h"
#include <cstdio>
#include <cstring>

void pipelines_create(Pipelines& p, VkDevice device)
{
    // ---- SWE init pipeline --------------------------------------------------
    auto swe_init_spirv = load_spirv("shaders/swe_init.spv");

    VkShaderModuleCreateInfo swe_init_sm_ci{};
    swe_init_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    swe_init_sm_ci.codeSize = swe_init_spirv.size() * sizeof(uint32_t);
    swe_init_sm_ci.pCode = swe_init_spirv.data();

    VK_CHECK(vkCreateShaderModule(device, &swe_init_sm_ci, nullptr, &p.swe_init_shader));

    VkDescriptorSetLayoutBinding swe_init_bindings[2]{};
    swe_init_bindings[0].binding = 0;
    swe_init_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    swe_init_bindings[0].descriptorCount = 1;
    swe_init_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    swe_init_bindings[1].binding = 1;
    swe_init_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swe_init_bindings[1].descriptorCount = 1;
    swe_init_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo swe_init_dsl_ci{};
    swe_init_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    swe_init_dsl_ci.bindingCount = 2;
    swe_init_dsl_ci.pBindings = swe_init_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &swe_init_dsl_ci, nullptr, &p.swe_init_desc_layout));

    VkPushConstantRange swe_init_push{};
    swe_init_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    swe_init_push.offset = 0;
    swe_init_push.size = sizeof(SweInitPC);

    VkPipelineLayoutCreateInfo swe_init_pl_ci{};
    swe_init_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    swe_init_pl_ci.setLayoutCount = 1;
    swe_init_pl_ci.pSetLayouts = &p.swe_init_desc_layout;
    swe_init_pl_ci.pushConstantRangeCount = 1;
    swe_init_pl_ci.pPushConstantRanges = &swe_init_push;

    VK_CHECK(vkCreatePipelineLayout(device, &swe_init_pl_ci, nullptr, &p.swe_init_pipeline_layout));

    VkPipelineShaderStageCreateInfo swe_init_stage{};
    swe_init_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    swe_init_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    swe_init_stage.module = p.swe_init_shader;
    swe_init_stage.pName = "main";

    VkComputePipelineCreateInfo swe_init_cp_ci{};
    swe_init_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    swe_init_cp_ci.stage = swe_init_stage;
    swe_init_cp_ci.layout = p.swe_init_pipeline_layout;

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &swe_init_cp_ci, nullptr, &p.swe_init_pipeline));

    // ---- SWE step pipeline --------------------------------------------------
    auto swe_step_spirv = load_spirv("shaders/swe_step.spv");

    VkShaderModuleCreateInfo swe_step_sm_ci{};
    swe_step_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    swe_step_sm_ci.codeSize = swe_step_spirv.size() * sizeof(uint32_t);
    swe_step_sm_ci.pCode = swe_step_spirv.data();

    VK_CHECK(vkCreateShaderModule(device, &swe_step_sm_ci, nullptr, &p.swe_step_shader));

    VkDescriptorSetLayoutBinding swe_step_bindings[4]{};
    swe_step_bindings[0].binding = 0;
    swe_step_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    swe_step_bindings[0].descriptorCount = 1;
    swe_step_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    swe_step_bindings[1].binding = 1;
    swe_step_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    swe_step_bindings[1].descriptorCount = 1;
    swe_step_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    swe_step_bindings[2].binding = 2;
    swe_step_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swe_step_bindings[2].descriptorCount = 1;
    swe_step_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    swe_step_bindings[3].binding = 3;
    swe_step_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swe_step_bindings[3].descriptorCount = 1;
    swe_step_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo swe_step_dsl_ci{};
    swe_step_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    swe_step_dsl_ci.bindingCount = 4;
    swe_step_dsl_ci.pBindings = swe_step_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &swe_step_dsl_ci, nullptr, &p.swe_step_desc_layout));

    VkPushConstantRange swe_step_push{};
    swe_step_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    swe_step_push.offset = 0;
    swe_step_push.size = sizeof(SweStepPC);

    VkPipelineLayoutCreateInfo swe_step_pl_ci{};
    swe_step_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    swe_step_pl_ci.setLayoutCount = 1;
    swe_step_pl_ci.pSetLayouts = &p.swe_step_desc_layout;
    swe_step_pl_ci.pushConstantRangeCount = 1;
    swe_step_pl_ci.pPushConstantRanges = &swe_step_push;

    VK_CHECK(vkCreatePipelineLayout(device, &swe_step_pl_ci, nullptr, &p.swe_step_pipeline_layout));

    VkPipelineShaderStageCreateInfo swe_step_stage{};
    swe_step_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    swe_step_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    swe_step_stage.module = p.swe_step_shader;
    swe_step_stage.pName = "main";

    VkComputePipelineCreateInfo swe_step_cp_ci{};
    swe_step_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    swe_step_cp_ci.stage = swe_step_stage;
    swe_step_cp_ci.layout = p.swe_step_pipeline_layout;

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &swe_step_cp_ci, nullptr, &p.swe_step_pipeline));

    // ---- Graphics pipeline (terrain) ----------------------------------------
    auto terrain_vs_spirv = load_spirv("shaders/terrain_vs.spv");
    auto terrain_fs_spirv = load_spirv("shaders/terrain_fs.spv");

    VkShaderModuleCreateInfo vs_sm_ci{};
    vs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vs_sm_ci.codeSize = terrain_vs_spirv.size() * sizeof(uint32_t);
    vs_sm_ci.pCode = terrain_vs_spirv.data();

    VK_CHECK(vkCreateShaderModule(device, &vs_sm_ci, nullptr, &p.terrain_vs));

    VkShaderModuleCreateInfo fs_sm_ci{};
    fs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fs_sm_ci.codeSize = terrain_fs_spirv.size() * sizeof(uint32_t);
    fs_sm_ci.pCode = terrain_fs_spirv.data();

    VK_CHECK(vkCreateShaderModule(device, &fs_sm_ci, nullptr, &p.terrain_fs));

    // Graphics descriptor set layout: UBO + heightmap + swe_output + sediment + atmo_shadow + cloud_vol_3d + wind_vol_3d + sand_deposit
    VkDescriptorSetLayoutBinding gfx_bindings[8]{};
    gfx_bindings[0].binding = 0;
    gfx_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    gfx_bindings[0].descriptorCount = 1;
    gfx_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[1].binding = 1;
    gfx_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[1].descriptorCount = 1;
    gfx_bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[2].binding = 2;
    gfx_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[2].descriptorCount = 1;
    gfx_bindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[3].binding = 3;
    gfx_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[3].descriptorCount = 1;
    gfx_bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[4].binding = 4;
    gfx_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[4].descriptorCount = 1;
    gfx_bindings[4].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[5].binding = 5;
    gfx_bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[5].descriptorCount = 1;
    gfx_bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[6].binding = 6;
    gfx_bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[6].descriptorCount = 1;
    gfx_bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    gfx_bindings[7].binding = 7;
    gfx_bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    gfx_bindings[7].descriptorCount = 1;
    gfx_bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo gfx_dsl_ci{};
    gfx_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    gfx_dsl_ci.bindingCount = 8;
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

    VkFormat color_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    VkPipelineRenderingCreateInfo rendering_ci{};
    rendering_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_ci.colorAttachmentCount = 1;
    rendering_ci.pColorAttachmentFormats = &color_format;
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
    auto water_vs_spirv = load_spirv("shaders/water_vs.spv");
    auto water_fs_spirv = load_spirv("shaders/water_fs.spv");

    VkShaderModuleCreateInfo water_vs_sm_ci{};
    water_vs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    water_vs_sm_ci.codeSize = water_vs_spirv.size() * sizeof(uint32_t);
    water_vs_sm_ci.pCode = water_vs_spirv.data();

    VK_CHECK(vkCreateShaderModule(device, &water_vs_sm_ci, nullptr, &p.water_vs));

    VkShaderModuleCreateInfo water_fs_sm_ci{};
    water_fs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    water_fs_sm_ci.codeSize = water_fs_spirv.size() * sizeof(uint32_t);
    water_fs_sm_ci.pCode = water_fs_spirv.data();

    VK_CHECK(vkCreateShaderModule(device, &water_fs_sm_ci, nullptr, &p.water_fs));

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

    // ---- Cloud raymarch pipeline (fullscreen, alpha blend, no depth) ---------
    auto rm_vs_spirv = load_spirv("shaders/cloud_raymarch_vs.spv");
    auto rm_fs_spirv = load_spirv("shaders/cloud_raymarch_fs.spv");

    VkShaderModuleCreateInfo rm_vs_sm_ci{};
    rm_vs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    rm_vs_sm_ci.codeSize = rm_vs_spirv.size() * sizeof(uint32_t);
    rm_vs_sm_ci.pCode = rm_vs_spirv.data();
    VK_CHECK(vkCreateShaderModule(device, &rm_vs_sm_ci, nullptr, &p.raymarch_vs));

    VkShaderModuleCreateInfo rm_fs_sm_ci{};
    rm_fs_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    rm_fs_sm_ci.codeSize = rm_fs_spirv.size() * sizeof(uint32_t);
    rm_fs_sm_ci.pCode = rm_fs_spirv.data();
    VK_CHECK(vkCreateShaderModule(device, &rm_fs_sm_ci, nullptr, &p.raymarch_fs));

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

    // ---- Terrain brush compute pipeline -------------------------------------
    auto terrain_brush_spirv = load_spirv("shaders/terrain_brush.spv");

    VkShaderModuleCreateInfo tb_sm_ci{};
    tb_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    tb_sm_ci.codeSize = terrain_brush_spirv.size() * sizeof(uint32_t);
    tb_sm_ci.pCode = terrain_brush_spirv.data();

    VK_CHECK(vkCreateShaderModule(device, &tb_sm_ci, nullptr, &p.terrain_brush_shader));

    VkDescriptorSetLayoutBinding tb_binding{};
    tb_binding.binding = 0;
    tb_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    tb_binding.descriptorCount = 1;
    tb_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo tb_dsl_ci{};
    tb_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    tb_dsl_ci.bindingCount = 1;
    tb_dsl_ci.pBindings = &tb_binding;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &tb_dsl_ci, nullptr, &p.tb_desc_layout));

    VkPushConstantRange tb_push{};
    tb_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tb_push.offset = 0;
    tb_push.size = sizeof(TerrainBrushPC);

    VkPipelineLayoutCreateInfo tb_pl_ci{};
    tb_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tb_pl_ci.setLayoutCount = 1;
    tb_pl_ci.pSetLayouts = &p.tb_desc_layout;
    tb_pl_ci.pushConstantRangeCount = 1;
    tb_pl_ci.pPushConstantRanges = &tb_push;

    VK_CHECK(vkCreatePipelineLayout(device, &tb_pl_ci, nullptr, &p.tb_pipeline_layout));

    VkPipelineShaderStageCreateInfo tb_stage{};
    tb_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    tb_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    tb_stage.module = p.terrain_brush_shader;
    tb_stage.pName = "main";

    VkComputePipelineCreateInfo tb_cp_ci{};
    tb_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    tb_cp_ci.stage = tb_stage;
    tb_cp_ci.layout = p.tb_pipeline_layout;

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &tb_cp_ci, nullptr, &p.terrain_brush_pipeline));

    // ---- Erosion compute pipeline -------------------------------------------
    auto erosion_spirv = load_spirv("shaders/erosion.spv");

    VkShaderModuleCreateInfo ero_sm_ci{};
    ero_sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ero_sm_ci.codeSize = erosion_spirv.size() * sizeof(uint32_t);
    ero_sm_ci.pCode = erosion_spirv.data();

    VK_CHECK(vkCreateShaderModule(device, &ero_sm_ci, nullptr, &p.erosion_shader));

    VkDescriptorSetLayoutBinding ero_bindings[4]{};
    ero_bindings[0].binding = 0;
    ero_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ero_bindings[0].descriptorCount = 1;
    ero_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    ero_bindings[1].binding = 1;
    ero_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    ero_bindings[1].descriptorCount = 1;
    ero_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    ero_bindings[2].binding = 2;
    ero_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    ero_bindings[2].descriptorCount = 1;
    ero_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    ero_bindings[3].binding = 3;
    ero_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ero_bindings[3].descriptorCount = 1;
    ero_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo ero_dsl_ci{};
    ero_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ero_dsl_ci.bindingCount = 4;
    ero_dsl_ci.pBindings = ero_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &ero_dsl_ci, nullptr, &p.ero_desc_layout));

    VkPushConstantRange ero_push{};
    ero_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ero_push.offset = 0;
    ero_push.size = sizeof(ErosionPC);

    VkPipelineLayoutCreateInfo ero_pl_ci{};
    ero_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ero_pl_ci.setLayoutCount = 1;
    ero_pl_ci.pSetLayouts = &p.ero_desc_layout;
    ero_pl_ci.pushConstantRangeCount = 1;
    ero_pl_ci.pPushConstantRanges = &ero_push;

    VK_CHECK(vkCreatePipelineLayout(device, &ero_pl_ci, nullptr, &p.ero_pipeline_layout));

    VkPipelineShaderStageCreateInfo ero_stage{};
    ero_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ero_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ero_stage.module = p.erosion_shader;
    ero_stage.pName = "main";

    VkComputePipelineCreateInfo ero_cp_ci{};
    ero_cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ero_cp_ci.stage = ero_stage;
    ero_cp_ci.layout = p.ero_pipeline_layout;

    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ero_cp_ci, nullptr, &p.erosion_pipeline));

    // ---- Atmosphere descriptor set layout --------------------------------------
    VkDescriptorSetLayoutBinding atmo_bindings[6]{};
    atmo_bindings[0].binding = 0;
    atmo_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    atmo_bindings[0].descriptorCount = 1;
    atmo_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[1].binding = 1;
    atmo_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    atmo_bindings[1].descriptorCount = 1;
    atmo_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[2].binding = 2;
    atmo_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    atmo_bindings[2].descriptorCount = 1;
    atmo_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[3].binding = 3;
    atmo_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    atmo_bindings[3].descriptorCount = 1;
    atmo_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[4].binding = 4;
    atmo_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    atmo_bindings[4].descriptorCount = 1;
    atmo_bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    atmo_bindings[5].binding = 5;
    atmo_bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    atmo_bindings[5].descriptorCount = 1;
    atmo_bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo atmo_dsl_ci{};
    atmo_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    atmo_dsl_ci.bindingCount = 6;
    atmo_dsl_ci.pBindings = atmo_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &atmo_dsl_ci, nullptr, &p.atmo_desc_layout));

    // ---- Terrain gen compute descriptor set layout + pipeline layout -----------
    VkDescriptorSetLayoutBinding tgen_bindings[2]{};
    tgen_bindings[0].binding = 0;
    tgen_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    tgen_bindings[0].descriptorCount = 1;
    tgen_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tgen_bindings[1].binding = 1;
    tgen_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    tgen_bindings[1].descriptorCount = 1;
    tgen_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo tgen_dsl_ci{};
    tgen_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    tgen_dsl_ci.bindingCount = 2;
    tgen_dsl_ci.pBindings = tgen_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &tgen_dsl_ci, nullptr, &p.terrain_gen_desc_layout));

    VkPushConstantRange tgen_push{};
    tgen_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tgen_push.offset = 0;
    tgen_push.size = sizeof(PlanetGenPC);

    VkPipelineLayoutCreateInfo tgen_pl_ci{};
    tgen_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tgen_pl_ci.setLayoutCount = 1;
    tgen_pl_ci.pSetLayouts = &p.terrain_gen_desc_layout;
    tgen_pl_ci.pushConstantRangeCount = 1;
    tgen_pl_ci.pPushConstantRanges = &tgen_push;

    VK_CHECK(vkCreatePipelineLayout(device, &tgen_pl_ci, nullptr, &p.terrain_gen_pipeline_layout));

    {
        auto spv = load_spirv("shaders/planet_gen_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &p.terrain_gen_shader));
    }

    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.terrain_gen_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.terrain_gen_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.terrain_gen_pipeline));
    }

    // ---- Planet SWE init pipeline -----------------------------------------------
    {
        VkDescriptorSetLayoutBinding bindings[5]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        for (int i = 1; i <= 3; ++i) {
            bindings[i].binding = (uint32_t)i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dsl_ci{};
        dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_ci.bindingCount = 5;
        dsl_ci.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &p.planet_swe_init_desc_layout));

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(PlanetSweInitPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.planet_swe_init_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.planet_swe_init_pipeline_layout));

        auto spv = load_spirv("shaders/planet_swe_init_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &p.planet_swe_init_shader));

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.planet_swe_init_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.planet_swe_init_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.planet_swe_init_pipeline));
    }

    // ---- Planet SWE step pipeline -----------------------------------------------
    {
        VkDescriptorSetLayoutBinding bindings[5]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dsl_ci{};
        dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_ci.bindingCount = 5;
        dsl_ci.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &p.planet_swe_step_desc_layout));

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(PlanetSweStepPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.planet_swe_step_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.planet_swe_step_pipeline_layout));

        auto spv = load_spirv("shaders/planet_swe_step_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &p.planet_swe_step_shader));

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.planet_swe_step_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.planet_swe_step_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.planet_swe_step_pipeline));
    }

    // ---- Planet SWE h-adjust pipeline ------------------------------------------
    {
        VkDescriptorSetLayoutBinding bindings[2]{};
        for (int i = 0; i < 2; ++i) {
            bindings[i].binding = (uint32_t)i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dsl_ci{};
        dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_ci.bindingCount = 2;
        dsl_ci.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dsl_ci, nullptr, &p.planet_swe_h_adjust_desc_layout));

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(PlanetSweHAdjustPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.planet_swe_h_adjust_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.planet_swe_h_adjust_pipeline_layout));

        auto spv = load_spirv("shaders/planet_swe_h_adjust_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &p.planet_swe_h_adjust_shader));

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.planet_swe_h_adjust_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.planet_swe_h_adjust_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.planet_swe_h_adjust_pipeline));
    }

    // ---- Atmosphere pipeline layout + pipeline ---------------------------------
    VkPushConstantRange atmo_push{};
    atmo_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    atmo_push.offset = 0;
    atmo_push.size = sizeof(Atmo3DPC);

    VkPipelineLayoutCreateInfo atmo_pl_ci{};
    atmo_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    atmo_pl_ci.setLayoutCount = 1;
    atmo_pl_ci.pSetLayouts = &p.atmo_desc_layout;
    atmo_pl_ci.pushConstantRangeCount = 1;
    atmo_pl_ci.pPushConstantRanges = &atmo_push;

    VK_CHECK(vkCreatePipelineLayout(device, &atmo_pl_ci, nullptr, &p.atmo_pipeline_layout));

    {
        auto spv = load_spirv("shaders/atmosphere3d_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &p.atmo_shader));
    }

    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.atmo_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.atmo_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.atmo_pipeline));
    }

    // ---- Sand compute descriptor set layout ------------------------------------
    VkDescriptorSetLayoutBinding sand_sim_bindings[4]{};
    sand_sim_bindings[0].binding = 0;
    sand_sim_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sand_sim_bindings[0].descriptorCount = 1;
    sand_sim_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    sand_sim_bindings[1].binding = 1;
    sand_sim_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sand_sim_bindings[1].descriptorCount = 1;
    sand_sim_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    sand_sim_bindings[2].binding = 2;
    sand_sim_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sand_sim_bindings[2].descriptorCount = 1;
    sand_sim_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    sand_sim_bindings[3].binding = 3;
    sand_sim_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sand_sim_bindings[3].descriptorCount = 1;
    sand_sim_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo sand_sim_dsl_ci{};
    sand_sim_dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sand_sim_dsl_ci.bindingCount = 4;
    sand_sim_dsl_ci.pBindings = sand_sim_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(device, &sand_sim_dsl_ci, nullptr, &p.sand_sim_desc_layout));

    // ---- Sand compute pipeline layout + pipeline --------------------------------
    VkPushConstantRange sand_sim_push{};
    sand_sim_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    sand_sim_push.offset = 0;
    sand_sim_push.size = sizeof(SandSimPC);

    VkPipelineLayoutCreateInfo sand_sim_pl_ci{};
    sand_sim_pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    sand_sim_pl_ci.setLayoutCount = 1;
    sand_sim_pl_ci.pSetLayouts = &p.sand_sim_desc_layout;
    sand_sim_pl_ci.pushConstantRangeCount = 1;
    sand_sim_pl_ci.pPushConstantRanges = &sand_sim_push;

    VK_CHECK(vkCreatePipelineLayout(device, &sand_sim_pl_ci, nullptr, &p.sand_sim_pipeline_layout));

    {
        auto spv = load_spirv("shaders/sand_sim_cs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &p.sand_sim_shader));
    }

    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.sand_sim_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.sand_sim_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.sand_sim_pipeline));
    }

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
    {
        auto spv = load_spirv("shaders/sand_render_vs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &p.sand_render_vs));
    }
    {
        auto spv = load_spirv("shaders/sand_render_fs.spv");
        VkShaderModuleCreateInfo sm_ci{};
        sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm_ci.codeSize = spv.size() * sizeof(uint32_t);
        sm_ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &sm_ci, nullptr, &p.sand_render_fs));
    }

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

    ShaderEntry entries[] = {
        {"../shaders/swe_init.hlsl",       "shaders/swe_init.spv",       "compute"},
        {"../shaders/swe_step.hlsl",       "shaders/swe_step.spv",       "compute"},
        {"../shaders/terrain.vs.hlsl",     "shaders/terrain_vs.spv",     "vertex"},
        {"../shaders/terrain.fs.hlsl",     "shaders/terrain_fs.spv",     "fragment"},
        {"../shaders/water.vs.hlsl",       "shaders/water_vs.spv",       "vertex"},
        {"../shaders/water.fs.hlsl",       "shaders/water_fs.spv",       "fragment"},
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
    };
    constexpr int NUM_SHADERS = 18;

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

    // Wait for GPU idle before touching anything
    vkDeviceWaitIdle(device);

    // Destroy old shader modules
    vkDestroyShaderModule(device, p.swe_init_shader, nullptr);
    vkDestroyShaderModule(device, p.swe_step_shader, nullptr);
    vkDestroyShaderModule(device, p.terrain_vs, nullptr);
    vkDestroyShaderModule(device, p.terrain_fs, nullptr);
    vkDestroyShaderModule(device, p.water_vs, nullptr);
    vkDestroyShaderModule(device, p.water_fs, nullptr);
    vkDestroyShaderModule(device, p.terrain_brush_shader, nullptr);
    vkDestroyShaderModule(device, p.erosion_shader, nullptr);
    vkDestroyShaderModule(device, p.atmo_shader, nullptr);
    vkDestroyShaderModule(device, p.raymarch_vs, nullptr);
    vkDestroyShaderModule(device, p.raymarch_fs, nullptr);
    vkDestroyShaderModule(device, p.sand_sim_shader, nullptr);
    vkDestroyShaderModule(device, p.sand_render_vs, nullptr);
    vkDestroyShaderModule(device, p.sand_render_fs, nullptr);
    vkDestroyShaderModule(device, p.terrain_gen_shader, nullptr);
    vkDestroyShaderModule(device, p.planet_swe_init_shader, nullptr);
    vkDestroyShaderModule(device, p.planet_swe_step_shader, nullptr);
    vkDestroyShaderModule(device, p.planet_swe_h_adjust_shader, nullptr);

    // Destroy old pipelines
    vkDestroyPipeline(device, p.swe_init_pipeline, nullptr);
    vkDestroyPipeline(device, p.swe_step_pipeline, nullptr);
    vkDestroyPipeline(device, p.gfx_pipeline, nullptr);
    vkDestroyPipeline(device, p.water_pipeline, nullptr);
    vkDestroyPipeline(device, p.raymarch_pipeline, nullptr);
    vkDestroyPipeline(device, p.terrain_brush_pipeline, nullptr);
    vkDestroyPipeline(device, p.erosion_pipeline, nullptr);
    vkDestroyPipeline(device, p.atmo_pipeline, nullptr);
    vkDestroyPipeline(device, p.sand_sim_pipeline, nullptr);
    vkDestroyPipeline(device, p.sand_render_pipeline, nullptr);
    vkDestroyPipeline(device, p.terrain_gen_pipeline, nullptr);
    vkDestroyPipeline(device, p.clipmap_terrain_pipeline, nullptr);
    vkDestroyPipeline(device, p.planet_swe_init_pipeline, nullptr);
    vkDestroyPipeline(device, p.planet_swe_step_pipeline, nullptr);
    vkDestroyPipeline(device, p.planet_swe_h_adjust_pipeline, nullptr);

    // Destroy old pipeline layouts
    vkDestroyPipelineLayout(device, p.swe_init_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.swe_step_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.gfx_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.raymarch_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.tb_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.ero_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.atmo_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.sand_sim_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.sand_render_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.terrain_gen_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.clipmap_gfx_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.planet_swe_init_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.planet_swe_step_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(device, p.planet_swe_h_adjust_pipeline_layout, nullptr);

    // Reload SPIR-V and recreate shader modules
    {
        auto spv = load_spirv("shaders/swe_init.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.swe_init_shader));
    }
    {
        auto spv = load_spirv("shaders/swe_step.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.swe_step_shader));
    }
    {
        auto spv = load_spirv("shaders/terrain_vs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.terrain_vs));
    }
    {
        auto spv = load_spirv("shaders/terrain_fs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.terrain_fs));
    }
    {
        auto spv = load_spirv("shaders/water_vs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.water_vs));
    }
    {
        auto spv = load_spirv("shaders/water_fs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.water_fs));
    }
    {
        auto spv = load_spirv("shaders/terrain_brush.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.terrain_brush_shader));
    }
    {
        auto spv = load_spirv("shaders/erosion.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.erosion_shader));
    }
    {
        auto spv = load_spirv("shaders/atmosphere3d_cs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.atmo_shader));
    }
    {
        auto spv = load_spirv("shaders/cloud_raymarch_vs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.raymarch_vs));
    }
    {
        auto spv = load_spirv("shaders/cloud_raymarch_fs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.raymarch_fs));
    }
    {
        auto spv = load_spirv("shaders/sand_sim_cs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.sand_sim_shader));
    }
    {
        auto spv = load_spirv("shaders/sand_render_vs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.sand_render_vs));
    }
    {
        auto spv = load_spirv("shaders/sand_render_fs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.sand_render_fs));
    }
    {
        auto spv = load_spirv("shaders/planet_gen_cs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.terrain_gen_shader));
    }
    {
        auto spv = load_spirv("shaders/planet_swe_init_cs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.planet_swe_init_shader));
    }
    {
        auto spv = load_spirv("shaders/planet_swe_step_cs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.planet_swe_step_shader));
    }
    {
        auto spv = load_spirv("shaders/planet_swe_h_adjust_cs.spv");
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size() * sizeof(uint32_t);
        ci.pCode = spv.data();
        VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &p.planet_swe_h_adjust_shader));
    }

    // Recreate pipeline layouts
    // (descriptor set layouts are unchanged, only pipeline layouts are rebuilt)
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(SweInitPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.swe_init_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.swe_init_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(SweStepPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.swe_step_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.swe_step_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(GfxPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.gfx_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.gfx_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(RaymarchPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.gfx_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.raymarch_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(TerrainBrushPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.tb_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.tb_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(ErosionPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.ero_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.ero_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(Atmo3DPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.atmo_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.atmo_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(SandSimPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.sand_sim_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.sand_sim_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(SandRenderPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.sand_render_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.sand_render_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(PlanetGenPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.terrain_gen_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.terrain_gen_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        push.offset = 0;
        push.size = sizeof(PlanetTilePC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.gfx_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.clipmap_gfx_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(PlanetSweInitPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.planet_swe_init_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.planet_swe_init_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(PlanetSweStepPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.planet_swe_step_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.planet_swe_step_pipeline_layout));
    }
    {
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.offset = 0;
        push.size = sizeof(PlanetSweHAdjustPC);

        VkPipelineLayoutCreateInfo pl_ci{};
        pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl_ci.setLayoutCount = 1;
        pl_ci.pSetLayouts = &p.planet_swe_h_adjust_desc_layout;
        pl_ci.pushConstantRangeCount = 1;
        pl_ci.pPushConstantRanges = &push;
        VK_CHECK(vkCreatePipelineLayout(device, &pl_ci, nullptr, &p.planet_swe_h_adjust_pipeline_layout));
    }

    // Recreate compute pipelines
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.swe_init_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.swe_init_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.swe_init_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.swe_step_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.swe_step_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.swe_step_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.terrain_brush_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.tb_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.terrain_brush_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.erosion_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.ero_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.erosion_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.atmo_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.atmo_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.atmo_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.sand_sim_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.sand_sim_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.sand_sim_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.terrain_gen_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.terrain_gen_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.terrain_gen_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.planet_swe_init_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.planet_swe_init_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.planet_swe_init_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.planet_swe_step_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.planet_swe_step_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.planet_swe_step_pipeline));
    }
    {
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = p.planet_swe_h_adjust_shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cp_ci{};
        cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_ci.stage = stage;
        cp_ci.layout = p.planet_swe_h_adjust_pipeline_layout;
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &p.planet_swe_h_adjust_pipeline));
    }

    // Recreate graphics pipelines (terrain + water)
    {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = p.terrain_vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = p.terrain_fs;
        stages[1].pName = "main";

        VkVertexInputBindingDescription vb{};
        vb.binding = 0;
        vb.stride = sizeof(float) * 2;
        vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription va{};
        va.binding = 0;
        va.location = 0;
        va.format = VK_FORMAT_R32G32_SFLOAT;
        va.offset = 0;

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &vb;
        vi.vertexAttributeDescriptionCount = 1;
        vi.pVertexAttributeDescriptions = &va;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                          | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &ba;

        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{};
        dy.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dy.dynamicStateCount = 2;
        dy.pDynamicStates = dyn;

        VkFormat cf = VK_FORMAT_B8G8R8A8_UNORM;
        VkFormat df = VK_FORMAT_D32_SFLOAT;

        VkPipelineRenderingCreateInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachmentFormats = &cf;
        ri.depthAttachmentFormat = df;

        VkGraphicsPipelineCreateInfo gp_ci{};
        gp_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_ci.pNext = &ri;
        gp_ci.stageCount = 2;
        gp_ci.pStages = stages;
        gp_ci.pVertexInputState = &vi;
        gp_ci.pInputAssemblyState = &ia;
        gp_ci.pViewportState = &vp;
        gp_ci.pRasterizationState = &rs;
        gp_ci.pMultisampleState = &ms;
        gp_ci.pDepthStencilState = &ds;
        gp_ci.pColorBlendState = &cb;
        gp_ci.pDynamicState = &dy;
        gp_ci.layout = p.gfx_pipeline_layout;

        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp_ci, nullptr, &p.gfx_pipeline));

        // Clipmap terrain pipeline (same state, different layout)
        VkGraphicsPipelineCreateInfo clip_ci = gp_ci;
        clip_ci.layout = p.clipmap_gfx_pipeline_layout;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &clip_ci, nullptr, &p.clipmap_terrain_pipeline));

        // Water pipeline (alpha blend, no depth write)
        VkPipelineShaderStageCreateInfo w_stages[2]{};
        w_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        w_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        w_stages[0].module = p.water_vs;
        w_stages[0].pName = "main";
        w_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        w_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        w_stages[1].module = p.water_fs;
        w_stages[1].pName = "main";

        VkPipelineDepthStencilStateCreateInfo w_ds{};
        w_ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        w_ds.depthTestEnable = VK_TRUE;
        w_ds.depthWriteEnable = VK_FALSE;
        w_ds.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

        VkPipelineColorBlendAttachmentState w_ba{};
        w_ba.blendEnable = VK_TRUE;
        w_ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        w_ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        w_ba.colorBlendOp = VK_BLEND_OP_ADD;
        w_ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        w_ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        w_ba.alphaBlendOp = VK_BLEND_OP_ADD;
        w_ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo w_cb{};
        w_cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        w_cb.attachmentCount = 1;
        w_cb.pAttachments = &w_ba;

        VkGraphicsPipelineCreateInfo w_ci = gp_ci;
        w_ci.pStages = w_stages;
        w_ci.pDepthStencilState = &w_ds;
        w_ci.pColorBlendState = &w_cb;

        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &w_ci, nullptr, &p.water_pipeline));

        // Raymarch pipeline (fullscreen, no depth, alpha blend)
        VkPipelineShaderStageCreateInfo rm_stg[2]{};
        rm_stg[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        rm_stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        rm_stg[0].module = p.raymarch_vs;
        rm_stg[0].pName = "main";
        rm_stg[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        rm_stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        rm_stg[1].module = p.raymarch_fs;
        rm_stg[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo rm_vi_s{};
        rm_vi_s.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineDepthStencilStateCreateInfo rm_ds_s{};
        rm_ds_s.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        rm_ds_s.depthTestEnable = VK_FALSE;
        rm_ds_s.depthWriteEnable = VK_FALSE;

        VkGraphicsPipelineCreateInfo rm_ci = gp_ci;
        rm_ci.pStages = rm_stg;
        rm_ci.pVertexInputState = &rm_vi_s;
        rm_ci.pDepthStencilState = &rm_ds_s;
        rm_ci.pColorBlendState = &w_cb;
        rm_ci.layout = p.raymarch_pipeline_layout;

        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &rm_ci, nullptr, &p.raymarch_pipeline));

        // Sand render pipeline (line list, no vertex input, depth test no write, alpha blend)
        VkPipelineShaderStageCreateInfo sand_stg[2]{};
        sand_stg[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sand_stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        sand_stg[0].module = p.sand_render_vs;
        sand_stg[0].pName = "main";
        sand_stg[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sand_stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        sand_stg[1].module = p.sand_render_fs;
        sand_stg[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo sand_vi_s{};
        sand_vi_s.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo sand_ia_s{};
        sand_ia_s.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        sand_ia_s.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

        VkPipelineDepthStencilStateCreateInfo sand_ds_s{};
        sand_ds_s.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        sand_ds_s.depthTestEnable = VK_TRUE;
        sand_ds_s.depthWriteEnable = VK_FALSE;
        sand_ds_s.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

        VkPipelineColorBlendAttachmentState sand_ba_s{};
        sand_ba_s.blendEnable = VK_TRUE;
        sand_ba_s.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        sand_ba_s.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        sand_ba_s.colorBlendOp = VK_BLEND_OP_ADD;
        sand_ba_s.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        sand_ba_s.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        sand_ba_s.alphaBlendOp = VK_BLEND_OP_ADD;
        sand_ba_s.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                 | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo sand_cb_s{};
        sand_cb_s.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        sand_cb_s.attachmentCount = 1;
        sand_cb_s.pAttachments = &sand_ba_s;

        VkGraphicsPipelineCreateInfo sand_ci = gp_ci;
        sand_ci.pStages = sand_stg;
        sand_ci.pVertexInputState = &sand_vi_s;
        sand_ci.pInputAssemblyState = &sand_ia_s;
        sand_ci.pDepthStencilState = &sand_ds_s;
        sand_ci.pColorBlendState = &sand_cb_s;
        sand_ci.layout = p.sand_render_pipeline_layout;

        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &sand_ci, nullptr, &p.sand_render_pipeline));
    }

    std::fprintf(stderr, "[shader reload] Done — %d shaders reloaded, all pipelines recreated.\n", NUM_SHADERS);
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

    // Destroy pipeline layouts
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

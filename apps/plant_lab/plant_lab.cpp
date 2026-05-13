// plant_lab.cpp — Plant Lab module implementation.

#include "plant_lab.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "pipeline.h"       // ClumpPC
#include "species_file.h"

#include <cstring>
#include <cstdio>
#include <cmath>

// ---------------------------------------------------------------------------
// Pipeline creation (uses ClumpPC from pipeline.h)
// ---------------------------------------------------------------------------
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
    vb.stride    = sizeof(bestiary::VegetationVertex);
    vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[4]{};
    attrs[0].location = 0;  attrs[0].binding = 0;
    attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;  attrs[0].offset = offsetof(bestiary::VegetationVertex, position);
    attrs[1].location = 1;  attrs[1].binding = 0;
    attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;  attrs[1].offset = offsetof(bestiary::VegetationVertex, normal);
    attrs[2].location = 2;  attrs[2].binding = 0;
    attrs[2].format   = VK_FORMAT_R32G32B32_SFLOAT;  attrs[2].offset = offsetof(bestiary::VegetationVertex, color);
    attrs[3].location = 3;  attrs[3].binding = 0;
    attrs[3].format   = VK_FORMAT_R32_SFLOAT;         attrs[3].offset = offsetof(bestiary::VegetationVertex, height_t);

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
// draw_lab_panel (static — only called from plant_lab_tick)
// ---------------------------------------------------------------------------
static bool draw_lab_panel(PlantLabState& s, float dt)
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 780), ImGuiCond_FirstUseEver);
    ImGui::Begin("Plant Lab v0.0.5");

    const char* kinds[] = {"Grass Clump", "Bush", "Tree", "L-Plant", "Reed", "Wildflower"};
    ImGui::Combo("species", &s.species_kind, kinds, 6);

    const char* modes[] = {"Single", "Field", "Ecosystem"};
    ImGui::Combo("mode", &s.view_mode, modes, 3);

    ImGui::Spacing();
    ImGui::TextUnformatted("Morphology (base)");
    ImGui::Separator();

    if (s.species_kind == 0) {
        ImGui::SliderInt  ("blade_count",  &s.clump_params.blade_count,  1,     60);
        ImGui::SliderFloat("blade_height", &s.clump_params.blade_height, 0.05f, 2.0f,  "%.3f m");
        ImGui::SliderFloat("blade_width",  &s.clump_params.blade_width,  0.005f,0.05f, "%.4f m");
        ImGui::SliderFloat("splay_angle",  &s.clump_params.splay_angle,  0.0f,  80.0f, "%.1f deg");
        ImGui::SliderFloat("clump_radius", &s.clump_params.clump_radius, 0.0f,  0.5f,  "%.3f m");

        ImGui::Spacing();
        ImGui::TextUnformatted("Appearance");
        ImGui::Separator();
        ImGui::ColorEdit3("base_color", s.clump_params.base_color);
    } else if (s.species_kind == 1) {
        ImGui::SliderInt  ("leaf_count",  &s.bush_params.leaf_count,  10,    80);
        ImGui::SliderFloat("leaf_length", &s.bush_params.leaf_length, 0.05f, 0.3f,  "%.3f m");
        ImGui::SliderFloat("leaf_width",  &s.bush_params.leaf_width,  0.02f, 0.15f, "%.3f m");
        ImGui::SliderFloat("bush_height", &s.bush_params.bush_height, 0.1f,  1.5f,  "%.2f m");
        ImGui::SliderFloat("bush_radius", &s.bush_params.bush_radius, 0.05f, 0.5f,  "%.3f m");
        ImGui::SliderFloat("stem_height", &s.bush_params.stem_height, 0.0f,  0.3f,  "%.3f m");

        if (ImGui::TreeNode("Branching")) {
            ImGui::SliderInt  ("n_stems",         &s.bush_params.n_stems,         1,     8);
            ImGui::SliderInt  ("attractor_count", &s.bush_params.attractor_count, 50,    2000);
            ImGui::SliderFloat("kill_ratio",      &s.bush_params.kill_ratio,      0.5f,  5.0f,  "%.2f");
            ImGui::SliderFloat("influence_ratio", &s.bush_params.influence_ratio, 2.0f,  30.0f, "%.1f");
            ImGui::SliderFloat("tropism",         &s.bush_params.tropism,        -1.0f,  1.0f,  "%.2f");
            ImGui::SliderFloat("surface_bias",    &s.bush_params.surface_bias,    0.0f,  1.0f,  "%.2f");
            const char* shapes[] = {"Ellipsoid", "Hemisphere", "Cylinder", "Cone"};
            ImGui::Combo("envelope_shape", &s.bush_params.envelope_shape, shapes, 4);
            ImGui::SliderFloat("branch_taper",     &s.bush_params.branch_taper,     1.5f,   3.5f,  "%.2f");
            ImGui::SliderFloat("branch_width_min", &s.bush_params.branch_width_min, 0.001f, 0.02f, "%.3f m");
            ImGui::SliderFloat("branch_width_max", &s.bush_params.branch_width_max, 0.01f,  0.1f,  "%.3f m");
            ImGui::SliderFloat("branch_gravity",   &s.bush_params.branch_gravity,   0.0f,   1.0f,  "%.2f");
            ImGui::SliderFloat("branch_wobble",    &s.bush_params.branch_wobble,    0.0f,   1.0f,  "%.2f");
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Leaf Placement")) {
            ImGui::SliderFloat("tip_leaf_bias",   &s.bush_params.tip_leaf_bias,   0.0f,  1.0f,  "%.2f");
            ImGui::SliderFloat("leaf_droop",      &s.bush_params.leaf_droop,      0.0f,  1.0f,  "%.2f");
            ImGui::TreePop();
        }

        ImGui::Spacing();
        ImGui::TextUnformatted("Appearance");
        ImGui::Separator();
        ImGui::ColorEdit3("base_color", s.bush_params.base_color);
    } else if (s.species_kind == 2) {
        ImGui::SliderFloat("tree_height",  &s.tree_params.tree_height,  1.0f,  15.0f, "%.1f m");
        ImGui::SliderFloat("trunk_height", &s.tree_params.trunk_height, 0.5f,  10.0f, "%.1f m");
        ImGui::SliderFloat("crown_radius", &s.tree_params.crown_radius, 0.5f,  5.0f,  "%.2f m");
        ImGui::SliderFloat("crown_height", &s.tree_params.crown_height, 0.5f,  8.0f,  "%.1f m");
        ImGui::SliderFloat("trunk_width",  &s.tree_params.trunk_width,  0.02f, 0.3f,  "%.3f m");

        ImGui::SliderInt  ("leaf_count",   &s.tree_params.leaf_count,   20,    2000);
        ImGui::SliderFloat("leaf_length",  &s.tree_params.leaf_length,  0.03f, 0.3f,  "%.3f m");
        ImGui::SliderFloat("leaf_width",   &s.tree_params.leaf_width,   0.02f, 0.15f, "%.3f m");

        if (ImGui::TreeNode("Branching")) {
            ImGui::SliderInt  ("attractor_count",  &s.tree_params.attractor_count,  100,    3000);
            ImGui::SliderFloat("kill_ratio",       &s.tree_params.kill_ratio,       0.5f,   5.0f,  "%.2f");
            ImGui::SliderFloat("influence_ratio",  &s.tree_params.influence_ratio,  2.0f,   30.0f, "%.1f");
            ImGui::SliderFloat("tropism",          &s.tree_params.tropism,         -1.0f,   1.0f,  "%.2f");
            ImGui::SliderFloat("surface_bias",     &s.tree_params.surface_bias,     0.0f,   1.0f,  "%.2f");
            const char* shapes[] = {"Ellipsoid", "Hemisphere", "Cylinder", "Cone"};
            ImGui::Combo("crown_shape", &s.tree_params.crown_shape, shapes, 4);
            ImGui::SliderFloat("branch_taper",     &s.tree_params.branch_taper,     1.5f,   3.5f,  "%.2f");
            ImGui::SliderFloat("branch_width_min", &s.tree_params.branch_width_min, 0.001f, 0.03f, "%.3f m");
            ImGui::SliderFloat("branch_width_max", &s.tree_params.branch_width_max, 0.02f,  0.3f,  "%.3f m");
            ImGui::SliderFloat("branch_gravity",   &s.tree_params.branch_gravity,   0.0f,   1.0f,  "%.2f");
            ImGui::SliderFloat("branch_wobble",    &s.tree_params.branch_wobble,    0.0f,   1.0f,  "%.2f");
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Leaf Placement")) {
            ImGui::SliderFloat("tip_leaf_bias",   &s.tree_params.tip_leaf_bias,   0.0f,  1.0f,  "%.2f");
            ImGui::SliderFloat("leaf_droop",      &s.tree_params.leaf_droop,      0.0f,  1.0f,  "%.2f");
            ImGui::TreePop();
        }

        ImGui::Spacing();
        ImGui::TextUnformatted("Appearance");
        ImGui::Separator();
        ImGui::ColorEdit3("base_color",  s.tree_params.base_color);
        ImGui::ColorEdit3("trunk_color", s.tree_params.trunk_color);
    }

    if (s.species_kind == 3 && s.view_mode != 2) {
        const char* archetypes[] = {"Monopodial", "Sympodial", "Dichotomous", "MultistemBush", "Whorled"};
        int arch = static_cast<int>(s.lplant_params.archetype);
        if (ImGui::Combo("archetype", &arch, archetypes, 5))
            s.lplant_params.archetype = static_cast<bestiary::GrowthArchetype>(arch);

        ImGui::SliderFloat("total_height",    &s.lplant_params.total_height,    1.0f,  15.0f, "%.1f m");
        ImGui::SliderFloat("trunk_height",    &s.lplant_params.trunk_height,    0.0f,  10.0f, "%.1f m");
        ImGui::SliderFloat("crown_radius",    &s.lplant_params.crown_radius,    0.5f,  5.0f,  "%.2f m");
        ImGui::SliderFloat("crown_height",    &s.lplant_params.crown_height,    0.5f,  8.0f,  "%.1f m");
        ImGui::SliderFloat("trunk_width",     &s.lplant_params.trunk_width,     0.02f, 0.3f,  "%.3f m");

        if (ImGui::TreeNode("L-System")) {
            ImGui::SliderInt  ("growth_steps",     &s.lplant_params.growth_steps,     4,      25);
            ImGui::SliderFloat("branch_angle",     &s.lplant_params.branch_angle,     0.1f,   1.2f,  "%.2f rad");
            ImGui::SliderFloat("branch_angle_var", &s.lplant_params.branch_angle_var, 0.0f,   0.5f,  "%.2f");
            ImGui::SliderFloat("phyllotaxis_angle",&s.lplant_params.phyllotaxis_angle,90.0f,  180.0f,"%.1f deg");
            ImGui::SliderFloat("internode_length", &s.lplant_params.internode_length, 0.05f,  1.0f,  "%.3f m");
            ImGui::SliderFloat("length_decay",     &s.lplant_params.length_decay,     0.5f,   1.0f,  "%.2f");
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Resources (Borchert-Honda)")) {
            ImGui::SliderFloat("lambda",         &s.lplant_params.lambda,         0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("resource_alpha", &s.lplant_params.resource_alpha, 0.5f, 4.0f, "%.1f");
            ImGui::SliderFloat("v_threshold",    &s.lplant_params.v_threshold,    0.0f, 0.3f, "%.3f");
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Space Colonization")) {
            ImGui::SliderInt  ("attractor_count", &s.lplant_params.attractor_count, 100,   3000);
            ImGui::SliderFloat("kill_ratio",      &s.lplant_params.kill_ratio,      0.5f,  5.0f, "%.2f");
            ImGui::SliderFloat("influence_ratio", &s.lplant_params.influence_ratio, 2.0f,  30.0f,"%.1f");
            ImGui::SliderFloat("tropism",         &s.lplant_params.tropism,        -1.0f,  1.0f, "%.2f");
            ImGui::SliderFloat("surface_bias",    &s.lplant_params.surface_bias,    0.0f,  1.0f, "%.2f");
            const char* shapes[] = {"Ellipsoid", "Hemisphere", "Cylinder", "Cone"};
            ImGui::Combo("envelope_shape", &s.lplant_params.envelope_shape, shapes, 4);
            ImGui::SliderFloat("branch_taper",    &s.lplant_params.branch_taper,    1.5f,  3.5f, "%.2f");
            ImGui::SliderFloat("branch_gravity",  &s.lplant_params.branch_gravity,  0.0f,  1.0f, "%.2f");
            ImGui::SliderFloat("branch_wobble",   &s.lplant_params.branch_wobble,   0.0f,  1.0f, "%.2f");
            ImGui::TreePop();
        }

        ImGui::SliderInt  ("leaf_count",   &s.lplant_params.leaf_count,   20,    2000);
        ImGui::SliderFloat("tip_leaf_bias",&s.lplant_params.tip_leaf_bias,0.0f,  1.0f, "%.2f");
        ImGui::SliderFloat("leaf_droop",   &s.lplant_params.leaf_droop,   0.0f,  1.0f, "%.2f");

        ImGui::Spacing();
        ImGui::TextUnformatted("Appearance");
        ImGui::Separator();
        ImGui::ColorEdit3("base_color",  s.lplant_params.base_color);
        ImGui::ColorEdit3("trunk_color", s.lplant_params.trunk_color);
    }

    if (s.species_kind == 4) {
        ImGui::SliderInt  ("blade_count",  &s.reed_params.blade_count,  1,     30);
        ImGui::SliderFloat("blade_height", &s.reed_params.blade_height, 0.1f,  2.0f,  "%.3f m");
        ImGui::SliderFloat("blade_width",  &s.reed_params.blade_width,  0.003f,0.02f, "%.4f m");
        ImGui::SliderFloat("splay_angle",  &s.reed_params.splay_angle,  0.0f,  30.0f, "%.1f deg");
        ImGui::SliderFloat("clump_radius", &s.reed_params.clump_radius, 0.0f,  0.3f,  "%.3f m");
        ImGui::Spacing();
        ImGui::TextUnformatted("Appearance");
        ImGui::Separator();
        ImGui::ColorEdit3("base_color", s.reed_params.base_color);
    }

    if (s.species_kind == 5) {
        ImGui::SliderInt  ("flower_count",  &s.wildflower_params.flower_count,  1,     20);
        ImGui::SliderFloat("stem_height",   &s.wildflower_params.stem_height,   0.03f, 0.5f,  "%.3f m");
        ImGui::SliderFloat("stem_width",    &s.wildflower_params.stem_width,    0.002f,0.01f, "%.4f m");
        ImGui::SliderFloat("petal_radius",  &s.wildflower_params.petal_radius,  0.005f,0.08f, "%.3f m");
        ImGui::SliderInt  ("petal_count",   &s.wildflower_params.petal_count,   3,     8);
        ImGui::SliderFloat("clump_radius",  &s.wildflower_params.clump_radius,  0.0f,  0.2f,  "%.3f m");
        ImGui::Spacing();
        ImGui::TextUnformatted("Appearance");
        ImGui::Separator();
        ImGui::ColorEdit3("petal_color", s.wildflower_params.petal_color);
        ImGui::ColorEdit3("stem_color",  s.wildflower_params.stem_color);
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

    if (s.view_mode == 1) {
        ImGui::Spacing();
        ImGui::TextUnformatted("Phenotype Expression (moisture)");
        ImGui::Separator();
        ImGui::TextDisabled("dry (left) -> wet (right)");

        if (s.species_kind == 0) {
            range_ui("blade_count",  s.clump_expr.blade_count,  1.0f,  60.0f, "%.0f");
            range_ui("blade_height", s.clump_expr.blade_height, 0.05f, 2.0f,  "%.3f m");
            range_ui("blade_width",  s.clump_expr.blade_width,  0.005f,0.05f, "%.4f m");
            range_ui("splay_angle",  s.clump_expr.splay_angle,  0.0f,  80.0f, "%.1f deg");
            range_ui("clump_radius", s.clump_expr.clump_radius, 0.0f,  0.5f,  "%.3f m");

            ImGui::Spacing();
            ImGui::Checkbox("vary_color", &s.clump_expr.vary_color);
            if (s.clump_expr.vary_color) {
                ImGui::Indent();
                ImGui::ColorEdit3("dry_color", s.clump_expr.dry_color);
                ImGui::ColorEdit3("wet_color", s.clump_expr.wet_color);
                ImGui::Unindent();
            }
        } else if (s.species_kind == 1) {
            range_ui("leaf_count",      s.bush_expr.leaf_count,      10.0f, 80.0f,   "%.0f");
            range_ui("leaf_length",    s.bush_expr.leaf_length,    0.05f, 0.3f,    "%.3f m");
            range_ui("leaf_width",     s.bush_expr.leaf_width,     0.02f, 0.15f,   "%.3f m");
            range_ui("bush_height",    s.bush_expr.bush_height,    0.1f,  1.5f,    "%.2f m");
            range_ui("bush_radius",    s.bush_expr.bush_radius,    0.05f, 0.5f,    "%.3f m");
            range_ui("stem_height",    s.bush_expr.stem_height,    0.0f,  0.3f,    "%.3f m");
            range_ui("n_stems",        s.bush_expr.n_stems,        1.0f,  8.0f,    "%.0f");
            range_ui("attractor_count",s.bush_expr.attractor_count,50.0f, 2000.0f, "%.0f");
            range_ui("kill_ratio",     s.bush_expr.kill_ratio,     0.5f,  5.0f,    "%.2f");
            range_ui("influence_ratio",s.bush_expr.influence_ratio,2.0f,  30.0f,   "%.1f");
            range_ui("tropism",        s.bush_expr.tropism,       -1.0f,  1.0f,    "%.2f");
            range_ui("surface_bias",    s.bush_expr.surface_bias,    0.0f,   1.0f,    "%.2f");
            range_ui("branch_width_min",s.bush_expr.branch_width_min,0.001f,0.02f, "%.3f m");
            range_ui("branch_width_max",s.bush_expr.branch_width_max,0.01f, 0.1f,  "%.3f m");
            range_ui("branch_gravity", s.bush_expr.branch_gravity,  0.0f,   1.0f,   "%.2f");
            range_ui("branch_wobble",  s.bush_expr.branch_wobble,   0.0f,   1.0f,   "%.2f");
            range_ui("tip_leaf_bias",  s.bush_expr.tip_leaf_bias,   0.0f,   1.0f,   "%.2f");
            range_ui("leaf_droop",     s.bush_expr.leaf_droop,      0.0f,   1.0f,   "%.2f");

            ImGui::Spacing();
            ImGui::Checkbox("vary_color", &s.bush_expr.vary_color);
            if (s.bush_expr.vary_color) {
                ImGui::Indent();
                ImGui::ColorEdit3("dry_color", s.bush_expr.dry_color);
                ImGui::ColorEdit3("wet_color", s.bush_expr.wet_color);
                ImGui::Unindent();
            }
        } else if (s.species_kind == 2) {
            range_ui("tree_height",     s.tree_expr.tree_height,     1.0f,  15.0f,   "%.1f m");
            range_ui("trunk_height",    s.tree_expr.trunk_height,    0.5f,  10.0f,   "%.1f m");
            range_ui("crown_radius",    s.tree_expr.crown_radius,    0.5f,  5.0f,    "%.2f m");
            range_ui("crown_height",    s.tree_expr.crown_height,    0.5f,  8.0f,    "%.1f m");
            range_ui("trunk_width",     s.tree_expr.trunk_width,     0.02f, 0.3f,    "%.3f m");
            range_ui("leaf_count",      s.tree_expr.leaf_count,      20.0f, 400.0f,  "%.0f");
            range_ui("leaf_length",     s.tree_expr.leaf_length,     0.03f, 0.3f,    "%.3f m");
            range_ui("leaf_width",      s.tree_expr.leaf_width,      0.02f, 0.15f,   "%.3f m");
            range_ui("attractor_count", s.tree_expr.attractor_count, 100.0f,3000.0f, "%.0f");
            range_ui("kill_ratio",      s.tree_expr.kill_ratio,      0.5f,  5.0f,    "%.2f");
            range_ui("influence_ratio", s.tree_expr.influence_ratio, 2.0f,  30.0f,   "%.1f");
            range_ui("tropism",         s.tree_expr.tropism,        -1.0f,  1.0f,    "%.2f");
            range_ui("surface_bias",    s.tree_expr.surface_bias,    0.0f,   1.0f,   "%.2f");
            range_ui("branch_width_min",s.tree_expr.branch_width_min, 0.001f,0.03f,  "%.3f m");
            range_ui("branch_width_max",s.tree_expr.branch_width_max, 0.02f, 0.3f,   "%.3f m");
            range_ui("branch_gravity",  s.tree_expr.branch_gravity,   0.0f,  1.0f,   "%.2f");
            range_ui("branch_wobble",   s.tree_expr.branch_wobble,    0.0f,  1.0f,   "%.2f");
            range_ui("tip_leaf_bias",   s.tree_expr.tip_leaf_bias,    0.0f,  1.0f,   "%.2f");
            range_ui("leaf_droop",      s.tree_expr.leaf_droop,       0.0f,  1.0f,   "%.2f");

            ImGui::Spacing();
            ImGui::Checkbox("vary_color", &s.tree_expr.vary_color);
            if (s.tree_expr.vary_color) {
                ImGui::Indent();
                ImGui::ColorEdit3("dry_color", s.tree_expr.dry_color);
                ImGui::ColorEdit3("wet_color", s.tree_expr.wet_color);
                ImGui::Unindent();
            }
        } else if (s.species_kind == 4) {
            range_ui("blade_count",  s.reed_expr.blade_count,  1.0f,  30.0f, "%.0f");
            range_ui("blade_height", s.reed_expr.blade_height, 0.1f,  2.0f,  "%.3f m");
            range_ui("blade_width",  s.reed_expr.blade_width,  0.003f,0.02f, "%.4f m");
            range_ui("splay_angle",  s.reed_expr.splay_angle,  0.0f,  30.0f, "%.1f deg");
            range_ui("clump_radius", s.reed_expr.clump_radius, 0.0f,  0.3f,  "%.3f m");
        } else if (s.species_kind == 5) {
            range_ui("flower_count",  s.wildflower_expr.flower_count,  1.0f, 20.0f, "%.0f");
            range_ui("stem_height",   s.wildflower_expr.stem_height,   0.03f,0.5f,  "%.3f m");
            range_ui("petal_radius",  s.wildflower_expr.petal_radius,  0.005f,0.08f,"%.3f m");
            range_ui("clump_radius",  s.wildflower_expr.clump_radius,  0.0f, 0.2f,  "%.3f m");

            ImGui::Spacing();
            ImGui::Checkbox("vary_color", &s.wildflower_expr.vary_color);
            if (s.wildflower_expr.vary_color) {
                ImGui::Indent();
                ImGui::ColorEdit3("dry_color", s.wildflower_expr.dry_color);
                ImGui::ColorEdit3("wet_color", s.wildflower_expr.wet_color);
                ImGui::Unindent();
            }
        }

        if (s.view_mode == 1) {
            ImGui::Spacing();
            ImGui::TextUnformatted("Field Settings");
            ImGui::Separator();
            ImGui::SliderInt("grid_n",    &s.field_params.grid_n,  2,    16);
            float max_spacing = (s.species_kind == 2) ? 8.0f : 1.0f;
            ImGui::SliderFloat("spacing", &s.field_params.spacing, 0.1f, max_spacing, "%.2f m");
        }
    }

    if (s.view_mode == 2) {
        ImGui::Spacing();
        ImGui::TextUnformatted("Environment");
        ImGui::Separator();
        ImGui::SliderFloat("region_size",   &s.eco_params.region_size,   5.0f, 40.0f, "%.0f m");
        ImGui::SliderFloat("density_scale",      &s.eco_params.density_scale,      0.1f, 3.0f,  "%.2f");
        ImGui::SliderFloat("phenotype_variance", &s.eco_params.phenotype_variance, 0.0f, 1.0f,  "%.2f");
        ImGui::SliderFloat("r_min",              &s.eco_params.r_min,              0.1f, 2.0f,  "%.2f m");
        ImGui::SliderFloat("r_max",         &s.eco_params.r_max,         1.0f, 8.0f,  "%.1f m");
        int seed_int = static_cast<int>(s.eco_params.seed);
        if (ImGui::SliderInt("seed", &seed_int, 0, 999))
            s.eco_params.seed = static_cast<uint32_t>(seed_int);
        ImGui::SliderFloat("moisture_freq", &s.noise_params.moisture_freq, 0.5f, 5.0f, "%.2f");
        ImGui::SliderFloat("temp_freq",     &s.noise_params.temp_freq,     0.5f, 5.0f, "%.2f");

        auto suit_ui = [](const char* label, bool& enabled, bestiary::SpeciesSuitability& suit) {
            ImGui::PushID(label);
            ImGui::Checkbox(label, &enabled);
            if (enabled && ImGui::TreeNode("##suit")) {
                ImGui::SliderFloat("moisture min/max", &suit.moisture_min, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("moisture opt lo",  &suit.moisture_opt_lo, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("moisture opt hi",  &suit.moisture_opt_hi, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("moisture max",     &suit.moisture_max, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("temp min",         &suit.temp_min, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("temp opt lo",      &suit.temp_opt_lo, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("temp opt hi",      &suit.temp_opt_hi, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("temp max",         &suit.temp_max, 0.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("base_density",     &suit.base_density, 0.01f, 10.0f, "%.2f");
                ImGui::TreePop();
            }
            ImGui::PopID();
        };

        ImGui::Spacing();
        ImGui::TextUnformatted("Species");
        ImGui::Separator();
        suit_ui("Grass",      s.eco_params.grass_enabled,      s.eco_params.grass_suit);
        suit_ui("Bush",       s.eco_params.bush_enabled,       s.eco_params.bush_suit);
        suit_ui("Tree",       s.eco_params.tree_enabled,       s.eco_params.tree_suit);
        suit_ui("Reed",       s.eco_params.reed_enabled,       s.eco_params.reed_suit);
        suit_ui("Wildflower", s.eco_params.wildflower_enabled, s.eco_params.wildflower_suit);
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Wind");
    ImGui::Separator();
    ImGui::SliderFloat("direction", &s.wind.angle, 0.0f, 360.0f, "%.0f deg");
    ImGui::SliderFloat("speed",     &s.wind.speed, 0.0f, 2.0f,   "%.2f");

    ImGui::Spacing();
    ImGui::TextUnformatted("File");
    ImGui::Separator();

    ImGui::InputText("name", s.fio.name_buf, sizeof(s.fio.name_buf));

    if (ImGui::Button("Save")) {
        std::filesystem::create_directories(species_dir());
        auto path = species_dir() / (std::string(s.fio.name_buf) + ".toml");
        bool ok = false;
        if (s.species_kind == 0)
            ok = bestiary::save_clump(path, s.clump_params, s.fio.name_buf, &s.clump_expr);
        else if (s.species_kind == 1)
            ok = bestiary::save_bush(path, s.bush_params, s.fio.name_buf, &s.bush_expr);
        else if (s.species_kind == 2)
            ok = bestiary::save_tree(path, s.tree_params, s.fio.name_buf, &s.tree_expr);
        else
            ok = bestiary::save_lplant(path, s.lplant_params, s.fio.name_buf, &s.lplant_expr);
        s.fio.set_status(ok ? "Saved." : "Save failed.");
        if (ok) s.fio.refresh_files();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load") && s.fio.selected_file >= 0) {
        auto path = species_dir() / (s.fio.file_list[static_cast<size_t>(s.fio.selected_file)] + ".toml");
        std::string kind = bestiary::detect_species_kind(path);
        std::string loaded_name;
        bool ok = false;
        if (kind == "grass_clump") {
            ok = bestiary::load_clump(path, s.clump_params, loaded_name, &s.clump_expr);
            if (ok) s.species_kind = 0;
        } else if (kind == "bush") {
            ok = bestiary::load_bush(path, s.bush_params, loaded_name, &s.bush_expr);
            if (ok) s.species_kind = 1;
        } else if (kind == "tree") {
            ok = bestiary::load_tree(path, s.tree_params, loaded_name, &s.tree_expr);
            if (ok) s.species_kind = 2;
        } else if (kind == "lplant") {
            ok = bestiary::load_lplant(path, s.lplant_params, loaded_name, &s.lplant_expr);
            if (ok) s.species_kind = 3;
        }
        if (ok) {
            std::snprintf(s.fio.name_buf, sizeof(s.fio.name_buf), "%s", loaded_name.c_str());
            s.fio.set_status("Loaded.");
        } else {
            s.fio.set_status("Load failed.");
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) {
        s.fio.refresh_files();
    }

    if (s.fio.status_timer > 0.0f) {
        s.fio.status_timer -= dt;
        ImGui::TextUnformatted(s.fio.status_buf);
    }

    if (!s.fio.file_list.empty()) {
        ImGui::BeginChild("file_list", ImVec2(0, 120), ImGuiChildFlags_Borders);
        for (int i = 0; i < static_cast<int>(s.fio.file_list.size()); ++i) {
            bool selected = (i == s.fio.selected_file);
            if (ImGui::Selectable(s.fio.file_list[static_cast<size_t>(i)].c_str(), selected))
                s.fio.selected_file = i;
        }
        ImGui::EndChild();
    }

    ImGui::Spacing();
    ImGui::Separator();
    if (s.view_mode == 0) {
        ImGui::Text("[INFO] mesh: %u verts, %u tris",
                    s.mesh.vertex_count, s.mesh.index_count / 3);
    } else {
        ImGui::Text("[INFO] field: %dx%d, mesh: %u verts, %u tris",
                    s.field_params.grid_n, s.field_params.grid_n,
                    s.mesh.vertex_count, s.mesh.index_count / 3);
    }
    ImGui::TextUnformatted("RMB drag to orbit, scroll to zoom.");

    bool back_pressed = false;
    if (s.embedded) {
        ImGui::Spacing();
        ImGui::Separator();
        if (ImGui::Button("Back", ImVec2(-1, 0)))
            back_pressed = true;
    }

    ImGui::End();
    ImGui::Render();
    return back_pressed;
}

// ---------------------------------------------------------------------------
// Module API
// ---------------------------------------------------------------------------

void plant_lab_init(PlantLabState& s, Renderer& r)
{
    s.cpipe = create_clump_pipeline(r.device);

    // Reed defaults — tall, thin, upright
    s.reed_params.blade_count  = 8;
    s.reed_params.blade_height = 0.80f;
    s.reed_params.blade_width  = 0.008f;
    s.reed_params.splay_angle  = 8.0f;
    s.reed_params.clump_radius = 0.04f;
    s.reed_params.base_color[0] = 0.40f;
    s.reed_params.base_color[1] = 0.48f;
    s.reed_params.base_color[2] = 0.22f;

    // Force first-frame regen by poisoning the snapshot.
    std::memset(&s.last_snapshot, 0xFF, sizeof(s.last_snapshot));

    s.fio.kind_filter = {"grass_clump", "bush", "tree", "lplant", "reed", "wildflower"};
    s.fio.refresh_files();
    s.initialized = true;
}

bool plant_lab_tick(PlantLabState& s, Renderer& r, float dt)
{
    if (!s.pending_file.empty()) {
        auto path = std::filesystem::path(s.pending_file);
        s.pending_file.clear();
        std::string kind = bestiary::detect_species_kind(path);
        std::string loaded_name;
        bool ok = false;
        if (kind == "grass_clump") {
            ok = bestiary::load_clump(path, s.clump_params, loaded_name, &s.clump_expr);
            if (ok) s.species_kind = 0;
        } else if (kind == "bush") {
            ok = bestiary::load_bush(path, s.bush_params, loaded_name, &s.bush_expr);
            if (ok) s.species_kind = 1;
        } else if (kind == "tree") {
            ok = bestiary::load_tree(path, s.tree_params, loaded_name, &s.tree_expr);
            if (ok) s.species_kind = 2;
        } else if (kind == "lplant") {
            ok = bestiary::load_lplant(path, s.lplant_params, loaded_name, &s.lplant_expr);
            if (ok) s.species_kind = 3;
        } else if (kind == "reed") {
            ok = bestiary::load_clump(path, s.reed_params, loaded_name, &s.reed_expr);
            if (ok) s.species_kind = 4;
        } else if (kind == "wildflower") {
            ok = bestiary::load_wildflower(path, s.wildflower_params, loaded_name, &s.wildflower_expr);
            if (ok) s.species_kind = 5;
        }
        if (ok)
            std::snprintf(s.fio.name_buf, sizeof(s.fio.name_buf), "%s", loaded_name.c_str());
    }

    // Camera.
    float max_zoom = (s.view_mode == 2) ? 60.0f
                   : (s.view_mode == 1) ? 40.0f
                   : (s.species_kind >= 2 ? 25.0f : 5.0f);
    update_orbit(s.camera, r.window, max_zoom);

    // Auto-adjust camera when switching view modes.
    if (s.view_mode != s.last_view_mode) {
        if (s.view_mode == 2) {
            s.camera.distance = s.eco_params.region_size * 0.8f;
            s.camera.target   = {0.0f, 0.5f, 0.0f};
            s.camera.pitch    = 0.7f;
        } else if (s.view_mode == 1) {
            float field_extent = static_cast<float>(s.field_params.grid_n - 1) * s.field_params.spacing;
            s.camera.distance = field_extent * 1.2f;
            s.camera.target   = {0.0f, 0.1f, 0.0f};
            s.camera.pitch    = 0.6f;
        } else {
            s.camera.distance = 1.5f;
            s.camera.target   = {0.0f, 0.2f, 0.0f};
            s.camera.pitch    = 0.3f;
        }
        s.last_view_mode = s.view_mode;
    }

    // Dirty-check snapshot -> regen mesh.
    PlantLabSnapshot current{};
    std::memset(&current, 0, sizeof(current));
    current.species_kind      = s.species_kind;
    current.view_mode         = s.view_mode;
    current.clump_params      = s.clump_params;
    current.clump_expr        = s.clump_expr;
    current.bush_params       = s.bush_params;
    current.bush_expr         = s.bush_expr;
    current.tree_params       = s.tree_params;
    current.tree_expr         = s.tree_expr;
    current.lplant_params     = s.lplant_params;
    current.lplant_expr       = s.lplant_expr;
    current.reed_params       = s.reed_params;
    current.reed_expr         = s.reed_expr;
    current.wildflower_params = s.wildflower_params;
    current.wildflower_expr   = s.wildflower_expr;
    current.field             = s.field_params;
    current.eco_params        = s.eco_params;
    current.noise_params      = s.noise_params;

    if (std::memcmp(&current, &s.last_snapshot, sizeof(PlantLabSnapshot)) != 0) {
        vkDeviceWaitIdle(r.device);
        destroy_mesh(r.allocator, s.mesh);

        if (s.view_mode == 2) {
            auto env = bestiary::make_noise_field(s.noise_params);
            bestiary::PlantMeshParams pmp;
            pmp.cp = s.clump_params;      pmp.ce = s.clump_expr;
            pmp.bp = s.bush_params;       pmp.be = s.bush_expr;
            pmp.tp = s.tree_params;       pmp.te = s.tree_expr;
            pmp.reed_cp = s.reed_params;  pmp.reed_ce = s.reed_expr;
            pmp.wfp = s.wildflower_params; pmp.wfe = s.wildflower_expr;
            pmp.phenotype_variance = s.eco_params.phenotype_variance;
            auto m = bestiary::generate_ecosystem(s.eco_params, env, pmp);
            upload_mesh(r.allocator, s.mesh, m);
        } else if (s.species_kind == 0) {
            if (s.view_mode == 0) {
                auto m = bestiary::generate_clump(s.clump_params);
                upload_mesh(r.allocator, s.mesh, m);
            } else {
                auto m = bestiary::generate_field(s.clump_params, s.clump_expr, s.field_params);
                upload_mesh(r.allocator, s.mesh, m);
            }
        } else if (s.species_kind == 1) {
            if (s.view_mode == 0) {
                auto m = bestiary::generate_bush(s.bush_params);
                upload_mesh(r.allocator, s.mesh, m);
            } else {
                auto m = bestiary::generate_bush_field(s.bush_params, s.bush_expr, s.field_params);
                upload_mesh(r.allocator, s.mesh, m);
            }
        } else if (s.species_kind == 2) {
            if (s.view_mode == 0) {
                auto m = bestiary::generate_tree(s.tree_params);
                upload_mesh(r.allocator, s.mesh, m);
            } else {
                auto m = bestiary::generate_tree_field(s.tree_params, s.tree_expr, s.field_params);
                upload_mesh(r.allocator, s.mesh, m);
            }
        } else if (s.species_kind == 3) {
            if (s.view_mode == 0) {
                auto m = bestiary::generate_lplant(s.lplant_params);
                upload_mesh(r.allocator, s.mesh, m);
            } else {
                auto m = bestiary::generate_lplant_field(s.lplant_params, s.lplant_expr, s.field_params);
                upload_mesh(r.allocator, s.mesh, m);
            }
        } else if (s.species_kind == 4) {
            if (s.view_mode == 0) {
                auto m = bestiary::generate_clump(s.reed_params);
                upload_mesh(r.allocator, s.mesh, m);
            } else {
                auto m = bestiary::generate_field(s.reed_params, s.reed_expr, s.field_params);
                upload_mesh(r.allocator, s.mesh, m);
            }
        } else if (s.species_kind == 5) {
            if (s.view_mode == 0) {
                auto m = bestiary::generate_wildflower(s.wildflower_params);
                upload_mesh(r.allocator, s.mesh, m);
            } else {
                auto m = bestiary::generate_wildflower_field(s.wildflower_params, s.wildflower_expr, s.field_params);
                upload_mesh(r.allocator, s.mesh, m);
            }
        }
        s.last_snapshot = current;
    }

    // Draw the ImGui panel (calls NewFrame and Render).
    // Returns true if the Back button was pressed (embedded mode only).
    bool back_pressed = draw_lab_panel(s, dt);

    return !back_pressed;
}

void plant_lab_render(PlantLabState& s, Renderer& r,
                      FrameData& frame, uint32_t image_index, VkExtent2D extent)
{
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
    proj[1][1] *= -1.0f;
    glm::mat4 view = orbit_view(s.camera);

    float wind_rad = glm::radians(s.wind.angle);
    ClumpPC pc{};
    pc.mvp        = proj * view;
    pc.wind_dir[0] = std::cos(wind_rad);
    pc.wind_dir[1] = std::sin(wind_rad);
    pc.wind_speed  = s.wind.speed;
    pc.time        = static_cast<float>(glfwGetTime());

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

        if (s.mesh.index_count > 0) {
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

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.cpipe.pipeline);

            vkCmdPushConstants(cmd, s.cpipe.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &s.mesh.vertex_buf.buffer, &offset);
            vkCmdBindIndexBuffer(cmd, s.mesh.index_buf.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, s.mesh.index_count, 1, 0, 0, 0);
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

void plant_lab_shutdown(PlantLabState& s, Renderer& r)
{
    vkDeviceWaitIdle(r.device);
    destroy_mesh(r.allocator, s.mesh);
    destroy_clump_pipeline(r.device, s.cpipe);
    s.initialized = false;
}

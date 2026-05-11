// plant_lab.h — embeddable Plant Lab module.
//
// Provides init/tick/render/shutdown so the lab can be driven by either
// a standalone executable or a unified launcher.
#pragma once

#include "renderer.h"
#include "shared/lab_common.h"

#include "morphology/clump.h"
#include "morphology/bush.h"
#include "morphology/tree.h"
#include "morphology/lplant.h"
#include "morphology/wildflower.h"
#include "environment.h"
#include "distribution.h"

// ---------------------------------------------------------------------------
// Dirty-check snapshot (memcmp'd each tick to detect param changes)
// ---------------------------------------------------------------------------
struct PlantLabSnapshot {
    int                        species_kind;
    int                        view_mode;
    bestiary::ClumpParams      clump_params;
    bestiary::ClumpExpression  clump_expr;
    bestiary::BushParams       bush_params;
    bestiary::BushExpression   bush_expr;
    bestiary::TreeParams       tree_params;
    bestiary::TreeExpression   tree_expr;
    bestiary::LPlantParams          lplant_params;
    bestiary::LPlantExpression      lplant_expr;
    bestiary::ClumpParams           reed_params;
    bestiary::ClumpExpression       reed_expr;
    bestiary::WildflowerParams      wildflower_params;
    bestiary::WildflowerExpression  wildflower_expr;
    bestiary::FieldParams           field;
    bestiary::EcosystemParams       eco_params;
    bestiary::NoiseFieldParams      noise_params;
};

// ---------------------------------------------------------------------------
// All plant-lab state, bundled into a single struct.
// ---------------------------------------------------------------------------
struct PlantLabState {
    bool embedded     = false;  // true when driven by launcher (shows Back button)
    bool initialized  = false;

    // Pipeline + mesh
    ClumpPipeline cpipe{};
    MeshState     mesh{};

    // Params
    int species_kind = 0;
    int view_mode    = 0;
    int last_view_mode = -1;

    bestiary::ClumpParams      clump_params{};
    bestiary::ClumpExpression  clump_expr{};
    bestiary::BushParams       bush_params{};
    bestiary::BushExpression   bush_expr{};
    bestiary::TreeParams       tree_params{};
    bestiary::TreeExpression   tree_expr{};
    bestiary::LPlantParams          lplant_params{};
    bestiary::LPlantExpression      lplant_expr{};
    bestiary::ClumpParams           reed_params{};
    bestiary::ClumpExpression       reed_expr{};
    bestiary::WildflowerParams      wildflower_params{};
    bestiary::WildflowerExpression  wildflower_expr{};
    bestiary::FieldParams           field_params{};
    bestiary::EcosystemParams       eco_params{};
    bestiary::NoiseFieldParams      noise_params{};

    // Snapshot for dirty-checking
    PlantLabSnapshot last_snapshot{};

    // Camera, file I/O, wind
    OrbitCamera camera{};
    FileIOState fio{};
    WindState   wind{};
};

// Module API ---------------------------------------------------------------

/// Create pipeline, set default params, init file list.
void plant_lab_init(PlantLabState& s, Renderer& r);

/// Poll camera, dirty-check -> regen mesh, draw ImGui panel.
/// Returns false if the user pressed "Back" (only possible when s.embedded).
bool plant_lab_tick(PlantLabState& s, Renderer& r, float dt);

/// Record the command buffer: barriers, clump draw, ImGui draw, present barrier.
void plant_lab_render(PlantLabState& s, Renderer& r,
                      FrameData& frame, uint32_t image_index, VkExtent2D extent);

/// Destroy pipeline + mesh.  Must be called before renderer_shutdown.
void plant_lab_shutdown(PlantLabState& s, Renderer& r);

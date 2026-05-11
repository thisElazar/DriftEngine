# Handoff: Engine Core Library Extraction (in progress)

## What was built (2026-05-10)

### Ecosystem v0.2 features (first half of session)

Added three systems to the creature simulation in world_lab:

1. **Spatial grid** (`bestiary/creature/spatial_grid.h`) — template uniform grid with `query_radius()`. 10m cells for agents, 5m for plants. All O(n²) queries (herd, food, mate) now use it.

2. **Terrain-aware movement** (`herbivore.cpp`) — 5-probe fan samples water depth + slope ahead of each agent. Water avoidance steering, hard rejection of movement into standing water, slope speed penalty (linear ramp, full block above `max_slope`). `CreatureWorldView` gained a `water_depth` query function.

3. **Caloric energy model** — replaces flat `energy_drain`/`idle_drain` with:
   - `basal_rate` (resting cost)
   - `locomotion_cost × speed²` (movement scales quadratically)
   - Slope multiplier on locomotion
   - Per-plant-species caloric values (grass=0.6, bush=1.0, tree=0.3)

4. **3 herbivore species** — Sprinter (small/fast/grass-specialist/180s lifespan), Grazer (medium/balanced/300s), Browser (large/slow/bush-specialist/450s). Per-species model cache in `creature_mesh.cpp`. ImGui species combo for per-species tuning.

5. **Dead agent compaction** — `erase-remove` every ~128 ticks.

### Engine core library extraction (second half of session)

**Created `drift_engine_core` static library** in root CMakeLists.txt. All apps now link it instead of recompiling renderer/resources/vma individually.

**Created `src/vk_util.h` / `src/vk_util.cpp`** with:
- `GpuBuffer` struct + `create_host_buffer()` / `create_readback_buffer()` / `destroy_buffer()`
- `load_spirv()` / `make_shader()` (authoritative copies; removed from pipeline.cpp)
- `OneShot` struct + `oneshot_begin()` / `oneshot_end()`
- `image_barrier()` / `compute_memory_barrier()`
- `ComputePipeline` struct + `make_compute_pipeline()` / `destroy_compute_pipeline()`
- `half_to_float()`
- `update_r32_image()`

**Simplified all 4 app CMakeLists** from ~40 lines each to 2-3 lines:
```cmake
add_executable(world_lab main.cpp)
target_link_libraries(world_lab PRIVATE drift_engine_core bestiary)
```

**Stripped world_lab** of ~290 lines of local utility code (now uses `#include "vk_util.h"`).

## Current state

- All 6 targets (drift_engine, drift_engine_core, plant_lab, animals_lab, world_lab, sim_couple_test) build and run
- world_lab uses the shared `vk_util.h`; other 3 apps still have local copies of the same functions (in anonymous namespaces, so no linker conflict — just redundancy)
- PC structs (SweInitPC, SweStepPC, etc.) are still locally redefined in world_lab + simCoupleTesting instead of using `pipeline.h`

## Remaining extraction steps

Full plan: `/Users/fields/.claude/plans/ticklish-bubbling-jellyfish.md`

### Step 5: Delete local PC struct redefinitions
- world_lab: remove SweInitPC, SweStepPC, TerrainBrushPC, ClumpPC; `#include "pipeline.h"`
- simCoupleTesting: remove SweInitPC, SweStepPC, TerrainBrushPC, ErosionPC, Atmo3DPC, RaymarchPC; `#include "pipeline.h"`
- Add `ClumpPC` to `pipeline.h` (currently only defined in world_lab + plant_lab)
- world_lab uses `_pad0` where pipeline.h has `k_rain` — just set `k_rain = 0.0f` (binary-compatible)
- Keep `WorldTerrainPC` local (render-level struct, slight variants per app)
- Add `static_assert(sizeof(SweStepPC) == 56)` to verify layout correctness

### Step 6: Create `src/grid_util.h` / `src/grid_util.cpp`
Extract from world_lab + simCoupleTesting, parameterize with `uint32_t grid_w, grid_h`:
- `build_grid_mesh()` — terrain grid mesh for rendering
- `cpu_apply_brush()` — CPU-side heightmap brush
- `sample_hm_bilinear()` — bilinear heightmap lookup
- `readback_water_depth()` — GPU water state readback
- `build_moisture_grid()` — water depth to moisture field with capillary blur

### Step 7: Strip local utility copies from remaining apps
- simCoupleTesting: delete local load_spirv, make_shader, GpuBuffer, create_host/readback_buffer, destroy_buffer, oneshot_begin/end, image_barrier, compute_memory_barrier, half_to_float, make_compute_pipeline, destroy_compute_pipeline; add `#include "vk_util.h"`
- animals_lab: delete local load_spirv, make_shader, GpuBuffer, buffer helpers; add `#include "vk_util.h"`
- plant_lab: delete local load_spirv, make_shader, GpuBuffer, buffer helpers, ClumpPC; add `#include "vk_util.h"` and `#include "pipeline.h"`

## Files touched this session

### New files
- `src/vk_util.h`
- `src/vk_util.cpp`
- `bestiary/creature/spatial_grid.h`

### Modified files
- `CMakeLists.txt` (root) — added `drift_engine_core` library target
- `src/pipeline.h` — removed `load_spirv` declaration
- `src/pipeline.cpp` — removed `load_spirv` definition, added `#include "vk_util.h"`
- `apps/world_lab/CMakeLists.txt` — simplified to 3 lines + shader block
- `apps/world_lab/main.cpp` — stripped ~290 lines, added `#include "vk_util.h"`, 3 species, caloric model, terrain awareness, spatial grid, water_depth query
- `apps/plant_lab/CMakeLists.txt` — simplified
- `apps/animals_lab/CMakeLists.txt` — simplified
- `apps/simCoupleTesting/CMakeLists.txt` — simplified
- `bestiary/creature/herbivore.h` — caloric energy model fields, terrain params, `water_depth` in CreatureWorldView
- `bestiary/creature/herbivore.cpp` — spatial grid queries, terrain steering, caloric drain, dead agent compaction
- `bestiary/creature/creature_mesh.cpp` — per-species model cache (vector instead of single static)
- `bestiary/species_file.cpp` — updated TOML serialization for new energy fields
- `species/meadow_grazer.toml` — updated energy section

## Build & run

```bash
cd "Drift Game Engine"
cmake -B build && cmake --build build
cd build && ./apps/world_lab/world_lab
```

## What this enables next

- Finishing extraction (Steps 5-7) removes another ~500 lines of duplication
- After `grid_util.h` exists, a new test app (predator arena, biome stress test) needs only its unique logic + a 3-line CMakeLists
- The ecosystem is ready for predators: 3 prey species with different speeds/sizes/niches, spatial grid for efficient hunt queries, terrain that creates natural refuges

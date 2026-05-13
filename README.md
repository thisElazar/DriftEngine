# Drift Engine

A simulation-first runtime where GPU compute is the primary citizen.

See [`ENGINE_V0.md`](ENGINE_V0.md) for the founding design doc — what this is, what it isn't, and why.

## Status

v0.5.0 — simulation engine with procedural ecosystem.

The engine runs a planet-scale shallow water simulation, hydraulic erosion, volumetric atmosphere, procedural terrain, and a full creature ecosystem with six animal archetypes.

## What's built

**Core engine** (`src/`): Vulkan 1.3 via MoltenVK, VMA, cube-map LOD quadtree planet, HLL Riemann SWE water solver, capacity-based erosion, 3D volumetric atmosphere with cloud raymarching, hot-reloadable HLSL shaders, ImGui UI.

**Bestiary** (`bestiary/`): Parametric phenotype library shared across all apps.
- 5 plant morphology types: clump, bush, tree (space-colonization), L-system, wildflower
- Skeletal animation: joints, inverse-bind skinning, keyframe SLERP
- 6 animal archetypes: herbivore, predator, rabbit, bird, raptor, snake — each with hand-keyed walk cycles
- Creature AI: energy-based metabolism, spatial grid queries, terrain-aware steering, herding, reproduction

**Applications** (`apps/`):
- `drift_engine` — planet simulation: terrain, water, erosion, atmosphere, clouds
- `plant_lab` — morphology editor for all plant types
- `animals_lab` — skeleton/animation editor for all 6 creature archetypes
- `world_lab` — ecosystem simulation: 3 herbivore species, wolf, rabbit, bird, raptor, snake on a terrain tile with water, plants, and atmosphere
- `launcher` — tabbed hub combining all four apps in one window
- `simCoupleTesting` — SWE + erosion coupling test harness

**Species data** (`species/`): 17 TOML profiles defining creature morphology, locomotion, energy models, and plant parameters.

## Build (macOS)

Prerequisites:
- Xcode command line tools (`xcode-select --install`)
- CMake >= 3.24 (`brew install cmake`)
- Vulkan SDK >= 1.3 — download from https://vulkan.lunarg.com/sdk/home#mac, run the installer, then `source ~/VulkanSDK/<version>/setup-env.sh` (or add to your shell rc)

Build:
```bash
cmake -S . -B build
cmake --build build -j$(sysctl -n hw.ncpu)
```

Run:
```bash
./build/drift_engine                    # planet simulation
./build/apps/plant_lab/plant_lab        # plant morphology editor
./build/apps/animals_lab/animals_lab    # skeleton/animation editor
./build/apps/world_lab/world_lab        # ecosystem simulation
./build/launcher                        # all-in-one tabbed launcher
```

## Layout

```
src/           engine core (renderer, resources, pipeline, planet, terrain, camera, UI)
src/vk_util.*  shared Vulkan utilities (buffers, shaders, compute pipelines, barriers)
src/grid_util.*  shared grid/tile helpers (mesh, brush, bilinear sampling, water readback)
bestiary/      phenotype library (plants, animals, skeleton, creature AI)
apps/          standalone lab applications + launcher
shaders/       HLSL -> SPIR-V via glslc (23 shaders: graphics + compute)
species/       TOML species/plant profiles
data/          test inputs
```

## Architecture

- **Simulation-first**: GPU compute dispatches (SWE, erosion, atmosphere) are the primary workload; graphics pipelines consume compute outputs.
- **No singletons**: state is owned and passed explicitly.
- **Shared core library**: `drift_engine_core` (static) provides Vulkan infrastructure to all apps.
- **Parametric procedural generation**: no external art imports; all morphology driven by parameters.
- **CPU skinning v0**: easier to debug; GPU skinning is a mechanical migration when perf demands it.
- **Hand-keyed animation in code**: walk cycles authored in C++ for fast iteration; file importers deferred.

## Dependencies (fetched via CMake FetchContent)

- GLFW 3.4
- vk-bootstrap v1.4.349
- VMA v3.2.1
- GLM 1.0.1
- Dear ImGui v1.91.5
- toml++ v3.4.0

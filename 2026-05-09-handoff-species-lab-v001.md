# Handoff: Species Lab v0.0.1 — Build Verification + v0.0.2 Plan

## Context

You are taking over solo gamedev work on the **Drift Game Engine**, a custom Vulkan engine (C++20, GLFW, vk-bootstrap, VMA, ImGui, glslc-compiled HLSL→SPIR-V). Engine source lives at `/Users/fields/Documents/driftEngine/Drift Game Engine/` (this is the user's workspace folder; it is also the cmake source dir).

The engine is paired with a sibling subsystem called **Bestiary** — a TOML-driven phenotype-expression system for plants and animals. Plan is in `ALPHA_V0.md`. A new editor app called **Species Lab** is being built first to tune morphology parameters interactively, before deploying species into the planet engine. Plan rationale: the user wants to *see* plants and tune them with sliders before committing to deploying them on a full heightmap.

Engine direction is in `ENGINE_V0.md`. Most recent planet-engine state is in `2026-05-08-handoff-camera-and-persistence.md` (cube-sphere planet with quadtree LOD, terrain stamp system, water reintroduction queued).

## What was just done

Scaffolded `bestiary/` static library and `apps/species_lab/` standalone exe. Three new files plus a 4-line edit to top-level `CMakeLists.txt`. The planet engine target (`drift_engine`) was not modified — it must still build identically.

**Files added:**
```
bestiary/
├── CMakeLists.txt                  static library, no Vulkan deps
└── morphology/
    ├── clump.h                     ClumpParams (5 morphology fields + color), generate_clump() decl
    └── clump.cpp                   STUB — returns empty ClumpMesh, real impl is v0.0.2

apps/species_lab/
├── CMakeLists.txt                  links bestiary + Vulkan/GLFW/imgui/etc., compiles src/renderer.cpp etc. directly
└── main.cpp                        window + ImGui panel with 5 sliders + base color picker, slate clear color
```

**CMakeLists.txt change** (top-level): added `add_subdirectory(bestiary)` and `add_subdirectory(apps/species_lab)` after the existing `shaders` target setup.

The Lab shares engine Vulkan setup by compiling `src/renderer.cpp`, `src/resources.cpp`, and `src/vma_impl.cpp` directly into the species_lab target. When the engine eventually becomes a CMake library target, those entries get replaced with `target_link_libraries(species_lab PRIVATE drift_engine)`.

## Your first task — verify the build

The sandbox where the scaffolding was done has no cmake/clang, so the build was not run. Run it now:

```bash
cd "/Users/fields/Documents/driftEngine/Drift Game Engine/build"
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu) --target species_lab
./apps/species_lab/species_lab
```

Then run the existing engine to confirm nothing regressed:
```bash
cmake --build . -j$(sysctl -n hw.ncpu) --target drift_engine
./drift_engine
```

**v0.0.1 success criteria** (eyeball-test only):
1. `species_lab` builds without warnings escalated to errors.
2. Running it opens a 1280×800 window titled "Species Lab".
3. Background is dark slate (~`{0.07, 0.08, 0.10}`).
4. ImGui panel "Species Lab — Grass Clump v0.0.1" is visible top-left, draggable, with sliders: `blade_count` (1–60), `blade_height` (0.05–2.0 m), `blade_width` (0.005–0.05 m), `splay_angle` (0–80°), `clump_radius` (0.0–0.5 m), and a `base_color` picker. Sliders move; nothing else renders yet.
5. Closing the window shuts down cleanly (no validation layer errors, no crash).
6. `drift_engine` still builds and runs as before.

If anything fails, fix it before proceeding to v0.0.2. Likely failure modes: missing include paths, a typo in the new CMakeLists.txt files, or a Vulkan validation complaint about the dynamic-rendering setup in `apps/species_lab/main.cpp` (which copies the engine's pattern from `src/main.cpp` near line 2560–2790).

## v0.0.2 — make sliders drive a real mesh

This is the next session's main work. Three pieces:

### Piece 1: implement `bestiary::generate_clump()`

In `bestiary/morphology/clump.cpp`, fill in the stub. Build a CPU mesh from the params:

- For each blade (count = `params.blade_count`):
  - Pick a base position offset within `clump_radius` (uniform-disk sampling, deterministic from `seed` + blade index).
  - Choose a tilt direction radially outward from clump center.
  - Tilt angle = `splay_angle` × per-blade jitter in [0.6, 1.0] (so blades are not all at max splay).
  - Build a tapered triangle strip from base to tip: 4–6 height segments, width tapers linearly from `blade_width` at base to ~10% at tip.
  - Compute per-vertex normals from the blade's plane (normal = cross of up × radial direction; orient consistent for lighting).
  - Color = `params.base_color` for now; phenotype-driven tinting is later.

The mesh is one big indexed triangle list. Don't bother with strip primitives. Aim for ~20–60 verts per blade, ~30–180 indices per blade. Total clump ≈ 1k–10k verts at the high end; trivial.

Add a small ground quad to the mesh too (or render it as a separate primitive — your call). The user wants to see the clump sitting *on* something so the geometry reads.

### Piece 2: graphics pipeline in `apps/species_lab/main.cpp`

Pattern to copy: the planet engine's `src/pipeline.cpp` builds graphics pipelines via vk-bootstrap-style configuration. The Lab needs a much simpler one:

- Vertex input: position (vec3), normal (vec3), color (vec3) — matching `bestiary::ClumpVertex`.
- Vertex shader: transforms `pos` by an MVP push constant, passes world-space normal and color to fragment.
- Fragment shader: simple Lambertian against a fixed light direction (e.g., `normalize(vec3(0.4, 1.0, 0.3))`), output `color * (0.3 + 0.7 * max(0, dot(N, L)))`.
- Push constants: a `mat4 mvp` is enough for v0.0.2. **HLSL note:** Drift uses glslc (not DXC), so push constants must be declared as `[[vk::push_constant]] cbuffer PC { float4x4 mvp; };` — see existing engine shaders for examples.
- Depth test: enable. Use `VK_FORMAT_D32_SFLOAT` (engine's `create_depth_buffer` already provides one via `Renderer::depth_buffer`).
- Single rendering pass: clear + clump draw + ImGui all inside one `vkCmdBeginRendering` block (with depth attachment) is fine. Or two passes mirroring the planet engine's separation. Either is OK.

Place the new shaders at `apps/species_lab/shaders/clump.vs.hlsl` and `clump.fs.hlsl`. The top-level CMakeLists.txt has a shader-compilation rule that globs `shaders/*.hlsl`; the cleanest extension is to add a similar rule scoped to `apps/species_lab/shaders/`. (Or: temporarily put them in the engine's `shaders/` dir and rely on the existing rule. User's call.)

### Piece 3: orbit camera + mesh re-upload

- Camera: a fixed orbit camera around (0, 0.2, 0). Drag-rotate with RMB; scroll-zoom. The user wants to see the clump from multiple angles to judge morphology.
- Mesh upload: regenerate when any slider value changes (compare param struct against last frame's). Re-upload to a GPU buffer. Don't worry about double-buffering or staging optimization for v0.0.2 — just `vkDeviceWaitIdle` + destroy + recreate buffer when params dirty. The clump is small.
- Add a `[INFO] mesh: %d verts, %d tris` line in the ImGui panel for sanity.

**v0.0.2 success criteria:** moving any slider visibly changes the clump in real time. Pulling `blade_count` low and `blade_height` high gives "wild prairie grass"; pulling `blade_count` high and `splay_angle` low gives "tight upright bunch". The user wants to feel the morphology spectrum from tight bunches to big wild grasses.

## What v0.0.2 is NOT

Hard non-goals to defend if scope creeps mid-session:

- **No L-systems.** No recursion, no procedural growth simulation. Parametric geometry only.
- **No textures.** Solid color + Lambertian only. Detail textures and bark are a much later concern.
- **No wind animation.** Blades are static for v0.0.2. Wind is a v0.0.4-ish concern.
- **No TOML loading yet.** v0.0.2 is geometry-from-sliders. TOML save/load is v0.0.3.
- **No phenotype expression yet.** Sliders set morphology directly. Wiring sliders for environmental fields → response curves → morphology comes after TOML lands.
- **No additional plant kinds.** Bush and tree are v0.0.4+. v0.0.2 is grass clump only.

## User context

Solo gamedev. Long-term goal is a full custom engine, with Drift as the playground. The game shape is god-game (stewardship + sandbox) with a first-person walkable toggle — references From Dust and Black & White 2. Engine destination is full 3D with a flyable camera (current top-down planet is a stepping stone). Treat the user as a peer; honest pushback lands well. Avoid recommending UE5 / Godot / etc. as substitutes — the custom engine is not a means to an end, it is the goal.

## Build conventions

- CMake 3.24+, C++20, RelWithDebInfo by default.
- Shaders: HLSL via glslc, NOT DXC. Push constants: `[[vk::push_constant]] cbuffer PC { ... };`.
- Output naming: `foo.vs.hlsl` → `foo_vs.spv` (dots replaced with underscores in spv name to avoid CMake .rule conflicts; see top-level CMakeLists.txt around the `string(REPLACE "." "_"` line).
- macOS: Vulkan via MoltenVK (Homebrew). The top-level cmake embeds an rpath to `${Vulkan_LIBRARIES}`'s directory — replicate this in any new exe target (already done for species_lab).
- Warnings: `-Wall -Wextra -Wpedantic` on non-MSVC. Don't disable; fix.

## Key reference files for v0.0.2

| What you need to look at | Why |
|---|---|
| `src/renderer.{h,cpp}` | Renderer abstraction the Lab reuses — already initializes Vulkan, GLFW, ImGui, swapchain, depth buffer, frame data |
| `src/main.cpp` lines 2560–2790 | Reference pattern for swapchain barriers + dynamic rendering + ImGui pass — lab's `record_frame` already mirrors this |
| `src/pipeline.cpp` | Reference for building graphics pipelines (vk-bootstrap-flavored). Find a small pipeline (e.g. the water or terrain one) and adapt |
| `shaders/*.hlsl` | Reference for HLSL with vk:: attributes, push_constant cbuffer syntax, vertex input layouts |
| `bestiary/morphology/clump.h` | The data structures you'll be filling in |
| `apps/species_lab/main.cpp` | The host program — `record_frame` is where the clump draw call goes, before the ImGui pass |
| `ALPHA_V0.md` | The bigger plan this is part of |

## When you finish v0.0.2

Update this handoff doc with what you actually built, what surprised you, and what's pending. Save as `2026-05-09-handoff-species-lab-v002.md` (or `2026-05-10-...` if work spans midnight). Do not delete this v001 doc — it's the historical record.

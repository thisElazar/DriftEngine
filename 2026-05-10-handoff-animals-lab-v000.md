# Handoff: Animals Lab v0.0.0 — Scaffold + Build Verification

## Context

You are taking over solo gamedev work on the **Drift Game Engine**, a custom Vulkan engine (C++20, GLFW, vk-bootstrap, VMA, ImGui, glslc-compiled HLSL→SPIR-V). Engine source lives at `/Users/fields/Documents/driftEngine/Drift Game Engine/` (this is the user's workspace folder; it is also the CMake source dir).

The engine is paired with a sibling subsystem called **Bestiary** — a TOML-driven parametric-morphology system for plants and animals. The plants side has a working editor app at `apps/species_lab/` (currently at v0.0.1, geometry-from-sliders is the in-flight v0.0.2 milestone). This handoff opens a **second** editor app, `apps/animals_lab/`, that will host the equivalent workflow for animals — starting with a single ground-walking quadruped herbivore archetype (valley_deer-shaped).

**Read `ANIMALS_LAB_V0.md` first.** It is the founding design doc for this lab — module layout, data model, stages, non-goals. Everything in this handoff assumes you have read it. The plan doc sits next to `ENGINE_V0.md` and is structured the same way.

Engine direction is in `ENGINE_V0.md`. Most recent species_lab handoff is `2026-05-09-handoff-species-lab-v001.md` — that doc is the template this one mirrors.

## What was just done in the previous session

Authored `ANIMALS_LAB_V0.md` — 14-section design doc covering scope, module layout, data types, skinning approach, walk-cycle authoring, open questions, and validation criteria. No code yet.

A few open questions in §11 of that doc were pre-resolved by inspecting the existing engine. You don't need to re-investigate these:

| Open question | Resolution | Source |
|---|---|---|
| Coordinate convention (Y-up vs Z-up, handedness) | **Y-up, right-handed, Vulkan depth `[0, 1]`** | `src/camera.h:3` uses `GLM_FORCE_DEPTH_ZERO_TO_ONE`; `apps/species_lab/main.cpp:135` uses `glm::lookAt(eye, target, vec3(0, 1, 0))` |
| Quaternion library | **`glm::quat` with `glm::slerp`** | `bestiary` already links `glm::glm` (see `bestiary/CMakeLists.txt:18`) |
| Joint count for v0 | **20** — one segment per limb (no knee/ankle split) | Plan doc §11 question 1, recommended choice |
| Save/load timing | **Deferred** to after Stage 2, mirroring species_lab progression | Plan doc §11 question 6 |
| Walk-cycle authoring format | **In C++ source** at `bestiary/animals/walks/herbivore_walk.cpp` | Plan doc §8 |

The remaining open question worth flagging mid-implementation: whether to compile shaders via the existing species_lab pattern (custom rule scoped to `apps/animals_lab/shaders/`) or temporarily share `shaders/` with the engine. Recommendation: **mirror species_lab's per-app shader rule** — keeps targets clean, no cross-contamination if the engine adds an `animal_*.hlsl` later.

## Your first task — v0.0.0: scaffold and verify the build

This session is **scaffolding only**. No skeleton math, no mesh generation, no rendering of bestiary content. The goal is a clean build with three targets (`drift_engine`, `species_lab`, `animals_lab`) all compiling, and `animals_lab` opening a blank ImGui window. If you finish v0.0.0 fast, you may begin v0.0.1 below — but only after v0.0.0's success criteria are all green.

### Files to create

```
bestiary/
├── skeleton/
│   ├── joint.h                     ← Joint struct (parent, local_bind, name[32]) — header only for v0.0.0
│   ├── skeleton.h                  ← Skeleton struct (joints vector, inverse_bind vector) — header only for v0.0.0
│   ├── skeleton.cpp                ← empty translation unit; placeholder for bind-pose math (v0.0.1)
│   ├── animation.h                 ← WalkCycle, JointKeyframe — header only for v0.0.0 (used in v0.0.x)
│   └── animation.cpp               ← empty translation unit
└── animals/
    ├── herbivore.h                 ← HerbivoreParams, AnimalVertex, AnimalMesh, build_herbivore_skeleton() decl, generate_herbivore_mesh() decl
    └── herbivore.cpp               ← STUB — both functions return empty/default. Real impl in v0.0.1+.

apps/animals_lab/
├── CMakeLists.txt                  ← lifted from apps/species_lab/CMakeLists.txt with target renamed
├── main.cpp                        ← lifted from apps/species_lab/main.cpp, stripped of clump-specific UI
└── shaders/
    └── (empty for v0.0.0 — no shaders compiled until v0.0.1)
```

### CMakeLists.txt edits

**`bestiary/CMakeLists.txt`** — add the new source files. Current list (lines 6-14):

```cmake
add_library(bestiary STATIC
    morphology/clump.cpp
    morphology/bush.cpp
    morphology/tree.cpp
    morphology/lplant.cpp
    species_file.cpp
    environment.cpp
    distribution.cpp
)
```

Append the four new sources:

```cmake
add_library(bestiary STATIC
    morphology/clump.cpp
    morphology/bush.cpp
    morphology/tree.cpp
    morphology/lplant.cpp
    species_file.cpp
    environment.cpp
    distribution.cpp
    skeleton/skeleton.cpp
    skeleton/animation.cpp
    animals/herbivore.cpp
)
```

**`apps/animals_lab/CMakeLists.txt`** — start from `apps/species_lab/CMakeLists.txt`, replace `species_lab` → `animals_lab` throughout, replace `SL_` prefix → `AL_` in the shader compilation block. The shader directory (`shaders/`) will be empty for v0.0.0; that is OK — the `file(GLOB AL_HLSL_SHADERS ...)` will return an empty list and the custom target will have no dependencies, which is fine.

**Top-level `CMakeLists.txt`** — at line 185 (just below `add_subdirectory(apps/species_lab)`), insert:

```cmake
add_subdirectory(apps/animals_lab)
```

### `apps/animals_lab/main.cpp` shape

Strip species_lab's `main.cpp` down to:

- All the Vulkan/GLFW/ImGui/swapchain init (identical to species_lab).
- An orbit camera at default position (eye at `(2.5, 1.2, 2.5)`, target at `(0, 1.0, 0)`, up `(0, 1, 0)`). The deer's belly will eventually be at ~`y=1.0` so anchor the orbit there.
- A single ImGui panel titled `"Animals Lab — Herbivore v0.0.0"` (top-left, draggable) that shows a placeholder text line: `"Scaffold OK. Skeleton rendering lands in v0.0.1."` No sliders yet.
- A slate clear color `{0.07, 0.08, 0.10}` matching species_lab.
- No mesh upload, no draw call beyond the clear + ImGui pass.
- Window title `"Animals Lab"`, dimensions 1280×800.
- Window close → clean shutdown (no validation layer errors).

### Build + run commands

```bash
cd "/Users/fields/Documents/driftEngine/Drift Game Engine/build"
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu) --target animals_lab
./apps/animals_lab/animals_lab
```

Then verify nothing regressed:
```bash
cmake --build . -j$(sysctl -n hw.ncpu) --target drift_engine
./drift_engine
# Ctrl-C / close window
cmake --build . -j$(sysctl -n hw.ncpu) --target species_lab
./apps/species_lab/species_lab
```

### v0.0.0 success criteria (all must hold)

1. `animals_lab` builds clean: no warnings escalated to errors under `-Wall -Wextra -Wpedantic`.
2. Running it opens a 1280×800 window titled **"Animals Lab"**.
3. Background is dark slate (~`{0.07, 0.08, 0.10}`).
4. ImGui panel **"Animals Lab — Herbivore v0.0.0"** is visible top-left, draggable, with the placeholder text line.
5. Closing the window shuts down cleanly — no Vulkan validation layer errors, no crash.
6. `drift_engine` still builds and runs as before — no regressions to planet engine.
7. `species_lab` still builds and runs as before — no regressions to plant lab.
8. `bestiary` library compiles clean with the three new stub sources linked in.

If anything fails: likely causes are (a) typo in `bestiary/CMakeLists.txt` source list, (b) missing include path in `apps/animals_lab/CMakeLists.txt`, (c) you forgot to add `apps/animals_lab` as a subdirectory in the top-level CMakeLists, (d) the `AL_` shader-rule rename missed a reference, leaving a CMake target-name collision with species_lab's `SL_` rule. Fix before proceeding.

## v0.0.1 — render the herbivore skeleton as joint spheres

This is the next session's main work (or the back half of this one if v0.0.0 went smooth). Four pieces.

### Piece 1: implement `build_herbivore_skeleton(params)`

In `bestiary/animals/herbivore.cpp`, fill in the function. Build a 20-joint hierarchy with this layout (parent index in brackets, all coordinates in meters, Y-up, deer standing facing +Z):

```
 0  pelvis              [-1] root, at (0, params.leg_length_back, 0)
 1  spine_1             [0]  forward along +Z by torso_length * 0.33
 2  spine_2             [1]  forward by torso_length * 0.33
 3  spine_3             [2]  forward by torso_length * 0.33 (shoulder line)
 4  neck_1              [3]  up+forward at ~45° by neck_length * 0.5
 5  neck_2              [4]  up+forward at ~45° by neck_length * 0.5
 6  head                [5]  forward by head_length
 7  shoulder_L          [3]  left (+X) by torso_girth * 0.5
 8  upper_front_L       [7]  down (-Y) by leg_length_front * 0.5
 9  lower_front_L       [8]  down by leg_length_front * 0.5
10  hoof_front_L        [9]  down by hoof_size
11  shoulder_R          [3]  right (-X) by torso_girth * 0.5
12  upper_front_R       [11] (mirror of left)
13  lower_front_R       [12]
14  hoof_front_R        [13]
15  hip_L               [0]  left (+X) by torso_girth * 0.5
16  upper_back_L        [15] down by leg_length_back * 0.5
17  lower_back_L        [16] down by leg_length_back * 0.5
18  hoof_back_L         [17] down by hoof_size
19  hip_R               [0]  right (-X) by torso_girth * 0.5
... etc — back-right leg gets joints 20-23, OR you can use 24 joints total to keep mirror symmetry.
```

**Wait, that's 24 joints, not 20.** Reconciling: §6 of the plan doc said ~20 as a recommendation; the symmetric quadruped lower-bound is 24 (4 limbs × 4 joints + spine 3 + neck 2 + head 1 + pelvis 1 = 23, plus shoulder/hip joints differs from upper-leg joints depending on parameterization). **Use 24 joints, keep the layout symmetric.** Update the plan doc §11 question 1 if you want to record this.

Each joint stores:
- `parent` (int, -1 for pelvis root)
- `local_bind` (`glm::mat4`, the joint's transform in *parent space*) — compute from the relative offsets above; rotation = identity in bind pose (T-pose, all legs straight down).
- `name[32]` (`char` array, null-terminated)

After populating `joints`, precompute `inverse_bind`:

```cpp
std::vector<glm::mat4> world(joints.size());
for (size_t i = 0; i < joints.size(); ++i) {
    world[i] = (joints[i].parent == -1)
        ? joints[i].local_bind
        : world[joints[i].parent] * joints[i].local_bind;
}
inverse_bind.resize(joints.size());
for (size_t i = 0; i < joints.size(); ++i) {
    inverse_bind[i] = glm::inverse(world[i]);
}
```

### Piece 2: minimal graphics pipeline for joint spheres

Two new shaders at `apps/animals_lab/shaders/`:

- `joint.vs.hlsl`: vertex shader for instanced sphere rendering. Push constant `mat4 mvp`. Per-vertex `position` + `normal` (small unit sphere mesh, ~32 verts is enough). Per-instance `mat4 model` from a uniform/SSBO array, indexed by `gl_InstanceIndex`. Output world-space normal + a fixed "bone color" (white for now).
- `joint.fs.hlsl`: simple Lambertian, same lighting as species_lab.

**HLSL note:** glslc, not DXC. Push constants:
```hlsl
[[vk::push_constant]] cbuffer PC { float4x4 mvp; };
```

Per-bone transforms come from a uniform/SSBO array, not per-instance vertex inputs — that's the convention skinning will reuse later. For v0.0.1 keep it simple: upload the world-space joint transforms to a uniform buffer once per frame, vertex shader reads `g_joints[gl_InstanceIndex]` to position each sphere.

Also draw lines from each joint to its parent. Either (a) a separate line pipeline that takes the same joint array and indexes pairs, or (b) extra cylinder instances with computed transforms. (a) is simpler — go (a).

### Piece 3: ImGui sliders for body proportions

The `HerbivoreParams` fields (plan doc §6):

- `torso_length` (0.5–2.5 m)
- `torso_girth` (0.2–0.8 m)
- `neck_length` (0.1–1.0 m)
- `head_length` (0.1–0.6 m)
- `leg_length_front` (0.3–1.5 m)
- `leg_length_back` (0.3–1.5 m)
- `leg_thickness` (0.02–0.15 m) — visualized as sphere radius for v0.0.1
- `hoof_size` (0.02–0.12 m)
- `coat_color` (color picker)

Walk-cycle fields (`walk_period_seconds`, `foot_lift_height`) — defer to v0.0.x. Don't show those sliders in v0.0.1.

Mesh upload: when any slider changes, regenerate skeleton, recompute world-space joint transforms, re-upload the joint uniform buffer. `vkDeviceWaitIdle` + destroy + recreate is fine; this happens at human-input rate.

Add an Info readout: `joints: 24`, current frame time.

### Piece 4: orbit camera

Lift `apps/species_lab/main.cpp:135` orbit camera verbatim. Target `(0, 1.0, 0)`. RMB-drag rotates, scroll zooms. Identical UX to species_lab.

### v0.0.1 success criteria

1. Moving any body-proportion slider visibly reshapes the joint constellation in real time.
2. Pulling `leg_length_front` to 1.5× while leaving `leg_length_back` at default produces a slope-shouldered "deer-tilted-forward" silhouette.
3. Pulling `neck_length` to 0.2× collapses the head onto the spine.
4. Lines between joints and parents read as a recognizable deer skeleton (4 legs, spine, neck, head) from orbit camera.
5. Sliders move smoothly — no perceptible hitch on re-upload.
6. `drift_engine` and `species_lab` still build and run identically.

## What v0.0.0 / v0.0.1 are NOT

Hard non-goals to defend if scope creeps mid-session:

- **No skinned mesh.** v0.0.1 renders joint spheres + parent lines only. The swept-ellipsoid body and tapered-cylinder legs are v0.0.2.
- **No skinning math.** No vertex bone weights, no linear blend skinning. CPU skinning lands once there is a mesh to skin (v0.0.2+).
- **No walk cycle.** No keyframes, no animation playback, no phase scrubber. v0.0.1 shows static T-pose only. Walk authoring is v0.0.x (post-mesh).
- **No IK.** Forward kinematics only. No foot targets, no raycasts.
- **No TOML loading.** Sliders set params directly. Save/load mirrors species_lab's v0.0.3 timing.
- **No phenotype response / `HerbivoreExpression`.** Defer until after walk cycle lands.
- **No multiple species.** One archetype: herbivore. Goat / sheep / antelope share the schema but aren't tested yet.
- **No second gait, no biped, no flier, no swimmer.** See plan doc §12 "what to push back on later."
- **No FBX/GLTF importer.** Walk cycle (when it lands) is authored in C++.
- **No GPU skinning.** When skinning arrives, it is CPU.

## User context

Solo gamedev. Long-term goal is a full custom engine, with Drift as the playground. The game shape is god-game (stewardship + sandbox) with a first-person walkable toggle — references From Dust and Black & White 2. Engine destination is full 3D with a flyable camera. Treat the user as a peer; honest pushback lands well. Do not recommend UE5 / Godot / etc. as substitutes — the custom engine is not a means to an end, it is the goal.

The user has explicitly chosen, via clarifying questions on 2026-05-10:
- Module layout: `bestiary/animals/` (single nested module, not split into `locomotion/` + `animals/`).
- v0 reach: **Stages 1-2** (skeleton + skinning + hand-keyed walk cycle, ~2 weeks).
- Doc-before-code: planning doc was written before scaffolding.

## Build conventions

- CMake 3.24+, C++20, RelWithDebInfo by default.
- Shaders: HLSL via **glslc**, NOT DXC. Push constants: `[[vk::push_constant]] cbuffer PC { ... };`.
- Output naming: `foo.vs.hlsl` → `foo_vs.spv` (dots replaced with underscores in spv name to avoid CMake .rule conflicts; see top-level CMakeLists.txt around the `string(REPLACE "." "_"` line).
- macOS: Vulkan via MoltenVK (Homebrew). The top-level CMake embeds an rpath to `${Vulkan_LIBRARIES}`'s directory — replicate this in any new exe target (already done in species_lab; mirror in animals_lab).
- Warnings: `-Wall -Wextra -Wpedantic` on non-MSVC. Don't disable; fix.
- No singletons. State is owned, passed explicitly. MasterController-pattern from TerrAI Legacy is not how we do things.
- Coordinate convention: **Y-up, right-handed, Vulkan depth `[0, 1]`** (`GLM_FORCE_DEPTH_ZERO_TO_ONE`).

## Key reference files

| What you need to look at | Why |
|---|---|
| `ANIMALS_LAB_V0.md` | The founding design doc. Read first. |
| `apps/species_lab/CMakeLists.txt` | Direct template for `apps/animals_lab/CMakeLists.txt` — copy then rename `species_lab`→`animals_lab` and `SL_`→`AL_` |
| `apps/species_lab/main.cpp` | Direct template for `apps/animals_lab/main.cpp`. Lift Vulkan/GLFW/ImGui setup; strip clump-specific UI |
| `bestiary/CMakeLists.txt` | The line you'll be editing to register the new sources |
| `bestiary/morphology/clump.h` | Pattern for parametric types — mirror this shape for `HerbivoreParams`, `AnimalVertex`, `AnimalMesh` |
| `bestiary/morphology/clump.cpp` | Pattern for `generate_*` functions — mirror for `generate_herbivore_mesh` (v0.0.2) |
| `src/renderer.{h,cpp}` | Renderer abstraction the lab reuses — already initializes Vulkan, GLFW, ImGui, swapchain, depth buffer |
| `src/main.cpp` lines 2560–2790 | Reference pattern for swapchain barriers + dynamic rendering + ImGui pass |
| `src/pipeline.cpp` | Reference for building graphics pipelines (vk-bootstrap-flavored). Small terrain or water pipeline is a good template |
| `shaders/*.hlsl` | Reference for HLSL with `vk::` attributes, push_constant cbuffer syntax, vertex input layouts |
| `2026-05-09-handoff-species-lab-v001.md` | The handoff doc this one mirrors; reference for any unfamiliar convention |

## When you finish v0.0.0 (and optionally v0.0.1)

Update this handoff doc with:
1. What you actually built (file paths, line counts, anything that diverged from the plan).
2. What surprised you (CMake quirks, validation-layer warnings, build-system gotchas).
3. What's pending (which success criteria are green, which are red).

Save as `2026-05-10-handoff-animals-lab-v001.md` (or `2026-05-11-...` if work spans midnight). Do not delete this v000 doc — it is the historical record.

If you complete only v0.0.0, the next handoff doc plans v0.0.1 (skeleton math + joint sphere render). If you complete v0.0.1 as well, the next handoff plans v0.0.2 (parametric skinned mesh — the swept-ellipsoid body and tapered-cylinder legs from plan doc §6, with CPU linear blend skinning per §7).

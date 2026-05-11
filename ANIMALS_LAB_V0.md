# Animals Lab v0 — Founding Design Doc

> Drafted 2026-05-10. Sibling to `ENGINE_V0.md` and the species_lab handoff line. This locks in the calls that determine what the rigging+animation subsystem feels like for the next few months, and deliberately leaves everything else open. When in doubt, re-read §1.

---

## 1. What this lab is, in one paragraph

Animals Lab is a standalone editor app, sibling to Species Lab, for tuning **animal morphology** and **hand-keyed locomotion** before any animal touches the planet engine. v0 builds one creature archetype — a ground-walking quadrupedal herbivore, valley_deer-shaped — end-to-end: TOML genome → parametric skeleton → skinned mesh → looping hand-keyed walk cycle, played in an orbit-camera viewport with sliders for body proportions. Same architectural spine as species_lab; the new dimension is *bones*. Plants are static geometry; animals are posed geometry that updates every frame.

This is **not** an AI lab. It is **not** a behavior lab. It has no IK, no procedural foot placement, no terrain raycasts, no pathfinding, no drives, no perception, no multi-species variety. Those are deliberate non-goals for v0. They become goals once the lab proves that the *modeling+movement* slice works at all.

The load-bearing claim this lab is testing: **the bestiary phenotype pipeline generalizes to skeletal organisms with no architectural special-casing.** If implementing the deer requires bolting on a parallel TOML schema, a parallel save/load path, or a parallel rendering pipeline, the architecture is wrong and we find that out in 2-3 weeks instead of 2-3 months.

---

## 2. Why a separate lab (not just adding to species_lab)

Three reasons.

First, the rendering pipeline diverges. A clump is `VegetationMesh` — static vertices, uploaded once, drawn forever. A deer is a skinned mesh — vertex positions are a function of joint transforms, recomputed every frame. The vertex layout adds bone indices and weights, the draw-side adds a joint-palette uniform, and the shader does linear blend skinning. Shoving both into one app means a mode switch that complicates the UI and the codepath without adding capability.

Second, the slider set diverges. Plant sliders are morphology (blade count, height, splay). Deer sliders are morphology *and* skeleton geometry (leg length, torso length, neck length, hoof size, pelvis-to-shoulder spacing) *and* animation parameters (stride length, gait cycle period, foot lift height). Putting all of these into one panel produces noise; separating them into two apps keeps each focused.

Third, iteration tempo. species_lab is mid-development (v0.0.2 in flight). Forking animals work into a clean app lets the two evolve independently without one stepping on the other's build or UI. Both link `bestiary` as a static library — that's where the *shared* code lives.

What stays shared: the TOML schema vocabulary, the `species_file` save/load pattern, the `ParamRange`-style expression layer, the `bestiary/` library itself. What's new: `bestiary/animals/` and `bestiary/skeleton/` subdirectories, the `SkinnedMesh` vertex type, the joint palette, and the walk-cycle authoring data.

---

## 3. v0 scope

The proof-of-thesis slice. Nothing else gets built until this is running. Two stages, sequenced.

### Stage 1 — Skeleton + posed static mesh (~1 week)

Goal: a deer in T-pose, generated from sliders, rendered correctly.

1. **Joint definitions.** Hard-coded list of joints for a quadruped herbivore. ~20 joints: pelvis, spine_1..3, neck_1..2, head, plus four legs (shoulder/hip, upper, lower, hoof). Indexable; each joint stores parent index, local-bind transform, and a name.
2. **Parameter-driven skeleton.** `HerbivoreParams` defines body proportions (torso_length, neck_length, leg_length_front, leg_length_back, etc.). A `build_skeleton(params)` function emits the bind-pose joint array. Same pattern as `generate_clump` but for transforms instead of vertices.
3. **Parametric skinned mesh.** `generate_herbivore_mesh(params)` produces an `AnimalMesh` — vertices include position, normal, **bone indices (4)**, **bone weights (4)**. Body is a swept ellipsoid along the spine; legs are tapered cylinders. Skin weights are assigned procedurally based on vertex proximity to bones — no rigging tool, no manual paint, this is the whole point.
4. **CPU skinning, GPU upload.** v0 skins on CPU: for each vertex, compute the final position from current joint transforms and bone weights, upload the result as a regular position buffer every frame. Slow but trivial to debug. GPU skinning (joint palette in a uniform, vertex shader does the matrix blend) is a Stage 3 concern.
5. **Render.** Lambertian against a fixed light, same shader pipeline as species_lab's clump renderer. No textures. Solid color from `params.coat_color`.

**Stage 1 success criteria:** moving body-proportion sliders visibly reshapes the deer in T-pose in real time. Pulling `leg_length` to 2× gives "giraffe-deer"; pulling `neck_length` to 0.5× gives "stubby-deer". The full morphology spectrum reads as legible silhouettes.

### Stage 2 — Hand-keyed walk cycle (~1 week)

Goal: the deer in Stage 1 plays a looping walk.

1. **Walk-cycle data.** A `WalkCycle` struct: per-joint keyframe arrays of local-space rotations (quaternions), normalized phase [0, 1). Hand-authored in code initially (~8-12 keyframes per moving joint, ~6 joints animated: four upper-legs and two spine-bends). NOT loaded from a file — this is authoring in C++ for v0; FBX/GLTF importers are a v2+ concern.
2. **Sampler.** `sample_walk(cycle, phase) → joint_pose_array`. Linear interpolation between keyframes; SLERP for quaternions. Output overrides bind-pose local rotations for animated joints, identity for the rest.
3. **Playback.** Phase advances at `cycle.period * playback_speed * dt`. ImGui controls: play/pause, scrub, speed slider, loop toggle.
4. **Foot contact validation.** Visualize foot-joint world positions over the gait cycle as a small overlay. A correct walk shows feet roughly stationary relative to ground during stance phase. This is the bullshit-detector for whether your hand-authored keyframes are coherent.

**Stage 2 success criteria:** play the cycle and the deer walks in place legibly — alternating diagonal pairs of legs (trot) or proper four-beat gait (walk). Pick one gait. Pausing freezes the pose. Slowing to 0.1× speed reads as clear individual leg motions. Resizing body proportions via Stage 1 sliders while playing still works — the same cycle drives differently-sized deer.

---

## 4. Explicit non-goals for v0

Hard non-goals. If scope creeps mid-session, point at this list.

- **No IK.** All animation is forward kinematics from authored keyframes. Foot targets, raycasts to ground, two-bone solvers — all Stage 3.
- **No procedural locomotion.** No "speed determines stride length." Walk cycle is a fixed loop. Variable gait is Stage 3.
- **No physics.** No gravity, no ground collision, no balance. The deer floats in space.
- **No terrain.** No heightmap, no slope, no ground plane interaction beyond a flat reference quad to anchor the eye.
- **No AI / behavior / drives.** The deer does not decide anything. It walks because we pressed play.
- **No multiple gaits.** One gait (walk *or* trot, not both). Adding gallop is Stage 3.
- **No multiple species.** One archetype: ground-walking quadruped herbivore. Bipeds, climbers, swimmers, fliers — all later.
- **No TOML loading yet.** Sliders set parameters directly. TOML save/load wires up after Stage 2, mirroring species_lab's progression.
- **No GPU skinning.** CPU skinning is the v0 default. Move to GPU when frame time demands it.
- **No file-loaded animation.** Walk cycle is authored in code. FBX/GLTF parsers are out of scope.
- **No phenotype field response.** No "the deer is leaner in dry climates." That's the expression layer's job, deferred until after Stage 2.

---

## 5. Module layout

Following the user's chosen path (option 1 in the scoping question): everything new lives under `bestiary/`, reinforcing the claim that plants and animals share one phenotype path.

```
bestiary/
├── CMakeLists.txt                  ← updated, adds animals/ and skeleton/ sources
├── morphology/                     ← existing, unchanged (clump, bush, tree, lplant)
├── species_file.{h,cpp}            ← existing, gains save_herbivore/load_herbivore
├── environment.{h,cpp}             ← existing, unchanged
├── distribution.{h,cpp}            ← existing, unchanged
├── skeleton/                       ← NEW. Bones, joints, skinning. Engine-agnostic.
│   ├── joint.h                     ← Joint struct (parent index, local bind transform, name)
│   ├── skeleton.{h,cpp}            ← Skeleton struct + bind-pose math + joint-palette computation
│   ├── animation.{h,cpp}           ← WalkCycle struct, keyframe sampler, SLERP helpers
│   └── skinning.{h,cpp}            ← CPU linear blend skinning (vertex × bone weights → final positions)
└── animals/                        ← NEW. Animal-specific morphology + walk authoring.
    ├── herbivore.{h,cpp}           ← HerbivoreParams, build_skeleton(), generate_herbivore_mesh()
    └── walks/
        └── herbivore_walk.cpp      ← hand-keyed WalkCycle data for the quadruped walk

apps/animals_lab/                   ← NEW. Mirrors apps/species_lab/.
├── CMakeLists.txt                  ← links bestiary + shared Vulkan/GLFW/imgui machinery
├── main.cpp                        ← window, ImGui panels, orbit camera, mesh re-upload on slider change
└── shaders/
    ├── animal.vs.hlsl              ← MVP transform, normal pass-through (skinning on CPU for v0)
    └── animal.fs.hlsl              ← Lambertian
```

Naming convention: `herbivore` not `deer`. The deer is the *test instance* of the herbivore archetype, in the same way the clump is one *kind* of plant. When goat / sheep / antelope follow, they share the herbivore skeleton and differ only by `HerbivoreParams`.

---

## 6. The data model (the load-bearing types)

These are the types the doc commits to. Naming and exact field lists can shift, but the shape is fixed.

```cpp
// bestiary/skeleton/joint.h
namespace bestiary {
struct Joint {
    int       parent;        // index into Skeleton::joints; -1 for root
    glm::mat4 local_bind;    // local transform in bind pose
    char      name[32];
};

// bestiary/skeleton/skeleton.h
struct Skeleton {
    std::vector<Joint>     joints;
    std::vector<glm::mat4> inverse_bind;  // precomputed for skinning
};

// bestiary/skeleton/animation.h
struct JointKeyframe {
    float       phase;       // [0, 1)
    glm::quat   rotation;    // local-space rotation override
};

struct WalkCycle {
    float                                       period_seconds = 1.0f;
    std::vector<std::vector<JointKeyframe>>     tracks;   // tracks[joint_index] = keyframes
};

// bestiary/animals/herbivore.h
struct HerbivoreParams {
    // Body proportions (meters)
    float torso_length        = 1.20f;
    float torso_girth         = 0.45f;
    float neck_length         = 0.45f;
    float head_length         = 0.30f;
    float leg_length_front    = 0.80f;
    float leg_length_back     = 0.85f;
    float leg_thickness       = 0.06f;
    float hoof_size           = 0.05f;

    // Animation tuning (Stage 2)
    float walk_period_seconds = 0.9f;
    float foot_lift_height    = 0.08f;

    // Aesthetic
    float coat_color[3]       = {0.55f, 0.40f, 0.28f};
};

struct AnimalVertex {
    float position[3];
    float normal[3];
    float color[3];
    uint8_t bone_indices[4];
    uint8_t bone_weights[4];   // normalized: 255 = 1.0
};

struct AnimalMesh {
    std::vector<AnimalVertex>  vertices;
    std::vector<uint32_t>      indices;
    Skeleton                   skeleton;     // bind-pose skeleton matched to mesh
};

Skeleton    build_herbivore_skeleton(const HerbivoreParams& p);
AnimalMesh  generate_herbivore_mesh(const HerbivoreParams& p, uint32_t seed = 0);
WalkCycle   make_herbivore_walk(const HerbivoreParams& p);
} // namespace bestiary
```

**Why `uint8_t` weights:** 1-byte weights keep `AnimalVertex` at 48 bytes (vs. 60+ with float weights), which matters at field scale even though it doesn't matter in the lab. Establish the layout early; the lab and the eventual planet integration share this vertex format.

**Why a fixed 4 bones per vertex:** standard linear blend skinning convention. Tools, engines, and shader patterns all assume 4. Going higher is exotic; going lower needs special-casing.

---

## 7. Skinning approach — CPU first, GPU later

CPU linear blend skinning for v0:

```
for each vertex v:
    final_pos = (0,0,0)
    for i in 0..3:
        bone = joint_palette[v.bone_indices[i]]   // bone = current_joint * inverse_bind
        weight = v.bone_weights[i] / 255.0
        final_pos += weight * (bone * v.position)
    upload final_pos to per-frame vertex buffer
```

Per-frame cost at deer scale (~2000 vertices): negligible. CPU skinning makes debugging trivial — `printf` the joint palette, the vertex's bone indices, the contribution per bone. When the deer's leg detaches from its body, you can step through the math.

GPU skinning becomes mandatory at field scale (100+ deer in view). The migration is mechanical: move the loop into the vertex shader, upload the joint palette as a uniform buffer, vertices stay in their bind-pose buffer. Vertex layout doesn't change. v0 establishes the layout; v0.x or v1 does the GPU move.

---

## 8. Walk-cycle authoring — in code, not in a file

Stage 2 hand-keyed walk lives in `bestiary/animals/walks/herbivore_walk.cpp`. Why in C++ source instead of a data file:

1. **No importer to write.** FBX/GLTF parsers are weeks of work for a one-creature lab.
2. **Hot iteration via shader-style reload.** The walk function takes `HerbivoreParams` and returns a `WalkCycle`. Tune constants in the function, recompile, see the difference. Not as fast as a file watcher but acceptable at lab scale.
3. **The data shape is dirt simple.** ~6 animated joints × ~8 keyframes × (phase + quaternion) ≈ 200 lines of explicit `make_keyframe(...)` calls. Readable, diffable, version-controllable.

The structure to author:

- **Front leg track:** shoulder rotates back during stance, forward during swing. Upper-leg lifts during swing.
- **Back leg track:** mirrors front with a phase offset (0.5 for trot, 0.25/0.75 for walk).
- **Spine bend:** small lateral bend opposing the leading leg, locks the silhouette into "alive" reading.

Reference: Eadweard Muybridge's *Animals in Motion* (1887) plates 90-110 are the canonical quadruped gait reference. Cite for self when authoring.

A v2 path: write a tiny TOML walk-cycle format and import. Worth considering after one cycle is authored in code — that's when you actually know what fields the format needs.

---

## 9. Lab UI (the panels)

The app window has three ImGui panels.

**Panel 1 — Morphology** (top-left, ~300px wide). Sliders for `HerbivoreParams` body fields. Drag-to-tune; mesh re-uploads on change (same pattern as species_lab Piece 3). A color picker for coat. A "Reset to default deer" button.

**Panel 2 — Animation** (top-right, ~250px wide, Stage 2). Play/pause toggle, scrub slider (current phase [0, 1)), playback-speed slider [0.0..3.0], "loop" checkbox. A "Show skeleton" toggle that draws joint positions as small spheres + lines between parent/child. A "Show foot trails" toggle that draws each hoof's recent world positions — visual gait sanity check.

**Panel 3 — Info** (bottom-left, ~250px wide). `mesh: %d verts, %d tris`, `skeleton: %d joints`, current playback phase, current frame time. Sanity readouts; same convention as species_lab.

Camera: orbit camera around (0, 1.0, 0) (roughly deer-belly height), drag-rotate RMB, scroll-zoom. Identical to species_lab's, lifted verbatim from `apps/species_lab/main.cpp`.

---

## 10. Build conventions (carried from species_lab)

These are not new decisions; they're reminders of the existing project rules so the lab matches.

- C++20, CMake 3.24+, RelWithDebInfo default.
- Shaders: HLSL via **glslc** (not DXC). Push constants are `[[vk::push_constant]] cbuffer PC { float4x4 mvp; };` — DXC syntax does *not* work here.
- Shader naming: `foo.vs.hlsl` → `foo_vs.spv` (dots replaced with underscores in spv name; see top-level CMakeLists).
- macOS: Vulkan via MoltenVK; embed rpath to `${Vulkan_LIBRARIES}` directory in any new exe target.
- Warnings: `-Wall -Wextra -Wpedantic` on non-MSVC. Don't disable, fix.
- No singletons. Pass state explicitly. MasterController-pattern stays in TerrAI Legacy.

---

## 11. Open questions (decide before Stage 1 lands)

These don't block writing this doc but block writing code. Flagging so we hit them deliberately.

1. **Joint count: ~20 or ~30?** ~20 (one segment per limb) is fastest to author; ~30 (knee + ankle separately) lets the walk read better but doubles keyframe authoring. Recommend 20 for v0, knee/ankle split deferred to v0.x.
2. **Coordinate convention.** Engine is already committed to a handedness via the planet code — match it. Verify by reading `src/main.cpp` view-matrix construction before writing the orbit camera.
3. **Quaternion library.** GLM has `glm::quat` with SLERP. Use it; don't write your own. `bestiary` already links GLM.
4. **One walk authored in code now, or pure procedural Stage 2?** Hand-keyed wins on legibility. Pure procedural (sine-wave joint angles) is faster to write but reads as "robot deer." Stay hand-keyed for v0.
5. **Z-up vs Y-up for the herbivore's bind pose.** Engine convention applies, but document explicitly in `herbivore.h` because skeleton math is easy to get wrong by an axis.
6. **Save/load: defer until after Stage 2 or land alongside Stage 1?** Recommend defer; species_lab proved that the sliders-first pattern keeps focus on geometry, not file formats.

---

## 12. What I'd push back on later

Reasonable thoughts that I'd argue against if they come up:

- "Let's add IK to Stage 2." No. IK is its own subsystem with its own bugs; lock down forward kinematics first or you can't tell whether breakage is in the keyframes or the solver.
- "Let's add a biped at the same time." No. The biped is enough morphologically different (no front legs, different gait timing, balance assumptions) that it shadows the herbivore work. One archetype, end-to-end, before generalizing.
- "Let's import a real deer model from Sketchfab." That's the opposite of in-house parametric. The whole *point* is owning the morphology pipeline. Refuse politely.
- "Let's GPU-skin from day one." No. CPU skinning is faster to debug; the migration is mechanical. Pay the cost when it actually matters.
- "Let's blend two walk cycles." No. One cycle, one gait, then ship Stage 2. Blend trees come after at least two cycles exist.
- "Let's bind sliders to live phenotype response now." No. Save the expression layer for after Stage 2 lands, then add it the way species_lab will — as a parallel `HerbivoreExpression` struct, not as a rewrite.

---

## 13. What "session 2" looks like

Once this doc is approved:

1. Create the directory skeleton in §5 with stub `.h`/`.cpp` files. Empty implementations.
2. Update `bestiary/CMakeLists.txt` to compile the new sources.
3. Create `apps/animals_lab/` with `CMakeLists.txt` and `main.cpp` lifted from `apps/species_lab/main.cpp`, minus the clump-specific UI. Verify it builds and opens a blank window.
4. Implement `build_herbivore_skeleton(params)` — the joint hierarchy and bind-pose math. Render joints as small spheres connected by lines to confirm the rig looks like a deer.
5. Stop. Session 2 is done. No mesh yet.

Sessions 3 onward each tackle one piece: skinned mesh generator → CPU skinning → Stage 1 sliders → walk-cycle data → sampler → playback UI.

Suggested handoff doc naming when work spans days: `YYYY-MM-DD-handoff-animals-lab-v00X.md` mirroring the species_lab convention.

---

## 14. Validation: when is v0 "done"

v0 is done when all of the following are true simultaneously:

1. `animals_lab` builds clean with `-Wall -Wextra -Wpedantic`, no validation-layer errors.
2. Moving any morphology slider reshapes the rendered deer in real time, without restart.
3. Pressing Play animates a looping walk that reads as a deer walking (not a glitching mess).
4. Pressing Pause freezes the pose; scrubbing the phase slider moves through the cycle.
5. The `drift_engine` and `species_lab` targets still build and run identically (no regressions).
6. The bestiary library compiles in under 10 seconds on a clean build (sanity check that we haven't accidentally pulled in heavy headers).

If 1-6 hold, the modeling+movement slice is proven and the next planning conversation is about either (a) IK + procedural locomotion (Stage 3, the natural continuation) or (b) starting the behavior subsystem as a parallel module, since the rig now has the API surface behavior would need.

---

*This doc is a contract with the next version of you. When in doubt, re-read §1.*

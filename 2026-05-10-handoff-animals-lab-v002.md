# Handoff: Animals Lab v0.0.4 — Gaits, Behaviors, Root Motion

## What was built (2026-05-10)

Starting from v0.0.3 (walk cycle only), this session added root motion, three locomotion gaits with blending, idle and graze behaviors, and several skeleton/mesh fixes.

### v0.0.4 — Root motion + gaits + behaviors

**Root motion**
- Stride-matched forward translation on pelvis (root joint) along +Z
- Stride length derived from gait parameters: `2 * leg * sin(hip_swing) / stance_fraction`
- Camera orbit target follows the animal
- Checkerboard ground plane at Y=0 for visual reference
- UI: root motion toggle, stride/speed readout

**Walk cycle fixes**
- Hip pitch convention fixed: negative = foot ahead, positive = foot behind (was inverted)
- Swing phase restructured: lift → carry → place (4 keyframes instead of 3)
- Hip swing increased ±10° → ±14° for more visible leg reach
- Knee lift increased to ±38° with hold phase for deliberate step feel
- Pelvis bob made asymmetric: heavier dip at front-leg contacts
- Tail sway added (±4° roll)
- Spine sway boosted to ±2.5°, head bob to ±3°

**Trot gait** (new)
- Diagonal pairs: LH+RF together, RH+LF together, 0.5 phase apart
- ±18° hip swing, 70% stance fraction, period = 65% of walk
- ±45° knee lift during swing
- Minimal spine sway (diagonal pairs cancel lateral forces)

**Run/bound gait** (new)
- Front pair + back pair, 0.5 phase apart
- ±22° hip swing, 55% stance fraction, period = 45% of walk
- ±55° knee lift, pronounced spinal flexion-extension (±4-5° pitch)
- Tail bounces vertically

**Gait blending**
- Single `gait` slider (0..1): walk → trot → run
- Per-frame SLERP of joint rotations between adjacent gait pairs
- Period, stride, bob amplitude all lerped smoothly
- `WalkCycle` struct stores `hip_swing_deg`, `stance_fraction`, `pelvis_drop` for self-describing gaits

**Idle animation** (new)
- 5s period, subtle weight shifting (back hip rocking ±1.5°)
- Head looking around: asymmetric keyframes on neck pitch + roll for non-repetitive motion
- Spine sway follows weight shift, lazy tail sway
- Breathing bob (3mm, slow cosine)

**Graze animation** (new)
- Spine_3 hinge: rotation at joint 3 (the ball between front legs) pivots entire neck as rigid beam
- Front shoulders counter-rotated to keep legs vertical
- Feed height parameter (0 = grass, max = canopy) drives hinge angle via lerp
- Neck stays straight (joints 4, 5 get only 3-5° fine adjustment)
- Head adds 30° to aim nose at food
- Reach envelope computed from morphology; niche label (ground grazer / low browser / canopy browser)

**Skeleton fixes**
- Spine tilt: each spine segment gets Y offset `(leg_length_front - leg_length_back) / 3` so all four feet reach ground regardless of leg length ratio
- Generic `add_leg_keys()` helper replaces per-gait leg functions
- Leg keyframe tables extracted as static `const LegKey[]` arrays

**UI**
- Mode combo: Walk, Trot, Run, Move (blended), Idle, Graze
- Gait slider only visible in Move mode
- Feed height slider + reach/niche display only in Graze mode
- Stride/speed readout for locomotion modes

## File inventory

```
bestiary/
├── skeleton/
│   ├── joint.h              — Joint struct
│   ├── skeleton.h           — Skeleton struct
│   ├── skeleton.cpp
│   ├── animation.h          — WalkCycle (period, hip_swing, stance_fraction, pelvis_drop), sample_walk()
│   ├── animation.cpp        — SLERP sampler
│   ├── skinning.h           — SkinnedVertex, cpu_skin()
│   └── skinning.cpp         — CPU linear blend skinning
├── animals/
│   ├── herbivore.h          — HerbivoreParams, AnimalVertex, AnimalMesh, all gait/behavior decls
│   ├── herbivore.cpp        — build_herbivore_skeleton() (with spine tilt), generate_herbivore_mesh()
│   └── walks/
│       └── herbivore_walk.cpp — walk, trot, run, idle, graze (369 lines)

apps/animals_lab/
├── CMakeLists.txt
├── main.cpp                 — ~1158 lines: Vulkan rendering, gait blending, ground plane, all UI
└── shaders/                 — joint/bone/mesh vertex+fragment shaders (unchanged)
```

## Key patterns established

### Gait authoring
Each gait is a `WalkCycle` with metadata (`hip_swing_deg`, `stance_fraction`, `pelvis_drop`) plus per-joint keyframe tracks. Gaits are built by `make_herbivore_<gait>(params)` functions that use `add_leg_keys()` with a static keyframe table and phase offsets to define leg pairing.

### Gait blending
Two adjacent gaits are sampled at the same phase, then SLERP'd per-joint by a blend factor. Period, stride, and bob are lerped from the gait metadata. This lets a single speed slider smoothly transition between any number of gaits.

### Graze hinge (hard-won)
The neck pivots at **spine_3 (joint 3)**, not at a neck joint. Front shoulders get equal-and-opposite counter-rotation to keep legs vertical. The neck stays straight. See `memory/feedback_graze_hinge.md` for the full rationale — this took several iterations to get right.

### Reach envelope
An animal's feeding niche is determined by its morphology: `min_reach` and `max_reach` computed from leg length, neck length, and head length. Tall legs + short neck = can't reach ground = canopy browser. Short legs + long neck = ground grazer.

## Known issues / rough edges

1. **Backface culling disabled** — CULL_MODE_NONE on mesh pipeline. Triangle winding in `append_tube`, `append_cap`, `append_sphere` needs audit.

2. **Mesh is tubes + spheres** — functional but geometric. Torso should be swept ellipsoid with varying cross-section. Legs should taper more naturally.

3. **Walk foot slide** — root motion uses constant-rate advance but hip sweep is non-linear (SLERP through keyframes). Small residual foot slide during stance.

4. **No TOML save/load** — species_lab has this; animals_lab defers it.

5. **Graze mesh deformation** — 95° rotation at spine_3 creates a sharp bend in the torso tube mesh at the chest. Joint sphere partially covers it but it's visible. Better mesh topology (more segments near the joint, or blended skinning weights) would improve this.

## What's next

1. **Winding fix** — audit `append_tube`, `append_cap`, `append_sphere` triangle winding, re-enable `VK_CULL_MODE_BACK_BIT`.

2. **Mesh quality** — swept ellipsoid torso (elliptical cross-sections varying along spine), tapered legs, smoother joint transitions.

3. **Rabbit-type herbivore** — compact body, hopping gait, different skeletal proportions. Second herbivore archetype to test the generality of the animation system.

4. **Wolf-type predator** — first carnivore. Different skeleton (longer snout, deeper chest), stalking/pursuit gaits. Introduces predator behaviors.

5. **Planet integration** — deferred until creature archetypes are understood. Bridge lab creatures to `creature/` behavioral agents (wander/graze/flee/herd).

## Build

```bash
cd "/Users/fields/Documents/driftEngine/Drift Game Engine/build"
cmake .. && cmake --build . -j$(sysctl -n hw.ncpu) --target animals_lab
./apps/animals_lab/animals_lab
```

## Key reference files

| File | Why |
|------|-----|
| `ANIMALS_LAB_V0.md` | Founding design doc — scope, stages, data model |
| `2026-05-10-handoff-animals-lab-v001.md` | Previous handoff (v0.0.3 walk cycle) |
| `apps/species_lab/main.cpp` | Template/peer for rendering patterns |
| `bestiary/creature/herbivore.{h,cpp}` | Planet-side behavioral agent (wander/graze/flee) |
| `memory/feedback_graze_hinge.md` | How the graze hinge works and why |

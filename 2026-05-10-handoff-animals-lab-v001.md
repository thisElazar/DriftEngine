# Handoff: Animals Lab v0.0.3 — Skeleton + Mesh + Walk Cycle

## What was built (2026-05-10)

Starting from zero, this session scaffolded and implemented the full Animals Lab through Stage 2 of ANIMALS_LAB_V0.md.

### v0.0.0 — Scaffold
- `apps/animals_lab/` with CMakeLists, main.cpp, empty shaders dir
- `bestiary/skeleton/` — joint.h, skeleton.h, animation.h (stub types)
- `bestiary/animals/` — herbivore.h/cpp (stub functions)
- Three targets (`drift_engine`, `species_lab`, `animals_lab`) all building clean

### v0.0.1 — Skeleton rendering
- `build_herbivore_skeleton(params)` — 24-joint hierarchy (pelvis, 3 spine, 2 neck, head, 4×4 legs, tail)
- Instanced sphere rendering via SSBO + bone line pipeline
- ImGui sliders for all body proportions
- Orbit camera (RMB drag, scroll zoom)

### v0.0.2 — Parametric skinned mesh
- `generate_herbivore_mesh(params)` — tubes + caps composing torso, neck, head, 4 legs, tail
- `append_tube()`, `append_cap()`, `append_sphere()` geometry primitives
- Facial features: eyes (dark hemispheres), ears (tapered tubes), nose
- Joint spheres baked into mesh at every joint to cover tube seams
- CPU linear blend skinning (`bestiary/skeleton/skinning.{h,cpp}`)
- Mesh pipeline (SkinnedVertex: pos+normal+color, push constant MVP)
- Skeleton overlay toggle ("show skeleton" checkbox)
- Rigid tube segments (100% single-bone weight per tube, no blending)
- Backface culling disabled (CULL_MODE_NONE) as workaround for winding issues

### v0.0.3 — Walk cycle
- `sample_walk(cycle, phase, joint_count)` — SLERP sampler with phase wrap-around (`animation.cpp`)
- `make_herbivore_walk(params)` — hand-keyed lateral-sequence walk (`bestiary/animals/walks/herbivore_walk.cpp`)
- Gait: LH → LF → RH → RF, 0.25 phase spacing, 83% stance / 17% swing
- 10 animated joints: 4 hip/shoulder (±10° pitch), 4 upper-leg knee bend (±22°), 2 spine (±1.5° roll)
- Head bob on neck joint, pelvis vertical bob (1.5cm cosine at 2× cycle freq)
- Coupled leg animation: `add_full_leg()` drives hip swing and knee bend from shared keyframe table
- Per-frame CPU skinning when animation plays
- ImGui: Play/Pause, Reset, phase scrubber, speed slider, walk_period and foot_lift_height sliders

## File inventory

```
bestiary/
├── skeleton/
│   ├── joint.h              — Joint struct
│   ├── skeleton.h           — Skeleton struct
│   ├── skeleton.cpp         — placeholder
│   ├── animation.h          — JointKeyframe, WalkCycle, sample_walk()
│   ├── animation.cpp        — SLERP sampler implementation
│   ├── skinning.h           — SkinnedVertex, cpu_skin()
│   └── skinning.cpp         — CPU linear blend skinning
├── animals/
│   ├── herbivore.h          — HerbivoreParams, AnimalVertex, AnimalMesh, function decls
│   ├── herbivore.cpp        — build_herbivore_skeleton(), generate_herbivore_mesh()
│   └── walks/
│       └── herbivore_walk.cpp — make_herbivore_walk()

apps/animals_lab/
├── CMakeLists.txt
├── main.cpp                 — ~920 lines, full Vulkan rendering + animation + ImGui
└── shaders/
    ├── joint.vs.hlsl        — instanced sphere (SSBO joint transforms)
    ├── joint.fs.hlsl        — Lambertian, hardcoded bone color
    ├── bone.vs.hlsl         — line list (SSBO joint index lookup)
    ├── bone.fs.hlsl         — flat gray
    ├── mesh.vs.hlsl         — per-vertex color pass-through
    └── mesh.fs.hlsl         — Lambertian with per-vertex color
```

## Known issues / rough edges

1. **Walk-in-place foot slide** — no root motion, so feet appear to slide backward during stance. Proper fix is contact-driven root translation (foot-plant detection → body moves forward while planted foot stays fixed in world space). This needs IK or at minimum a foot-contact phase detector.

2. **Backface culling disabled** — CULL_MODE_NONE on mesh pipeline. The tube/cap winding was partially fixed but some caps (rump) still show wrong faces. Should audit winding in `append_tube`, `append_cap`, `append_sphere` and re-enable CULL_MODE_BACK_BIT.

3. **Mesh is tubes + spheres** — functional but geometric. The plan doc §6 describes swept-ellipsoid body and tapered-cylinder legs with smooth cross-section variation. The current torso is 3 cylindrical tubes with sphere joints — works but reads as "robot deer."

4. **No TOML save/load** — species_lab has save/load at v0.0.3; animals_lab defers this, matching the plan doc.

## What's next

Candidates for the next session, roughly in priority order:

1. **Root motion** — contact-driven forward translation so the walk looks grounded. Requires detecting when each foot is in stance (planted) and computing how much the body should advance per frame based on the planted foot's world position.

2. **Winding fix** — audit all geometry primitives, fix triangle winding, re-enable backface culling.

3. **Mesh quality** — smooth the torso into a proper swept ellipsoid (elliptical cross-sections varying along the spine path, not just circular tubes). Taper legs more naturally.

4. **Additional gaits** — trot (diagonal pairs in sync), run (front pair / back pair in sync as user described). The animation system supports this — just author new keyframe sets in `make_herbivore_walk()` or add `make_herbivore_trot()` / `make_herbivore_run()`.

5. **Planet integration** — bridge from lab deer to planet-engine creature. The `creature/` subsystem already has `herbivore.{h,cpp}` with wander/graze behaviors and `VegetationDensityField` — connecting the parametric mesh + walk cycle to the behavioral agent.

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
| `2026-05-10-handoff-animals-lab-v000.md` | Original scaffold handoff (v0.0.0/v0.0.1 spec) |
| `apps/species_lab/main.cpp` | Template/peer for rendering patterns |
| `bestiary/morphology/clump.h` | Pattern for parametric types |

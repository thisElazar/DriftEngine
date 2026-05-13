# Handoff: Animals Lab — Bird, Raptor, Snake + Rendering Upgrades

## What was built (2026-05-11)

Starting from v0.0.5 (deer, wolf, rabbit), this session added three new creature archetypes, rendering improvements, and fixed the triangle winding bug.

### Winding fix

All mesh primitives (`append_tube`, `append_tube_ellipse`, `append_cap`, `append_sphere`, `append_ellipsoid`, `append_path_tube`) had their triangle winding audited and corrected for CCW front-face convention. Backface culling (`VK_CULL_MODE_BACK_BIT`) re-enabled on the mesh pipeline. The UV sphere in the skeleton overlay was also fixed.

### Two-tone countershading

New `countershade(AnimalMesh&, belly_color)` function in `animal_mesh.cpp` blends vertex colors toward `belly_color` based on bind-pose normal Y. All creatures now have `belly_color` params with natural defaults. The blend uses `clamp(-normal.y * 1.5 + 0.5, 0, 1)` for a slightly sharp dorsal/ventral transition.

### Hemisphere ambient lighting

Fragment shader (`mesh.fs.hlsl`) upgraded from flat ambient to hemisphere lighting: warm sky color from above, cool ground color from below, blended by normal Y. Combined with the existing directional sun light.

### Bird (CREATURE_BIRD)

**Skeleton** — 23 joints, first non-quadruped body plan:
- Pelvis → spine_1 → chest → 2-segment neck → head
- Wings: 4 joints each (shoulder → elbow → wrist → wingtip), extended laterally in bind pose
- Legs at spine_1 (mid-body): hip → tibiotarsus → tarsometatarsus → foot with anisodactyl toes
- Short tail fan

**Mesh** — body ellipsoid + swept path tube, elliptical wing segments driven by `wing_width`/`wing_taper` params, thin digitigrade legs with yellow beak/feet, eyes.

**9 animation modes:**
- Walk: alternating bipedal with head-bob (thrust-hold-retract)
- Hop: sparrow-style, both feet together
- Fly: wing flapping with configurable amplitude, period, and sweep
- Move: blended walk → hop → fly with altitude ramp
- Idle: looking around, head tilts, wing ruffle
- Peck: dramatic head-down pecking with side glances
- Takeoff: explosive flapping from ground, legs tuck
- Landing: braking flaps, legs extend, settle
- Perch: upright body, legs gripping, calm lookabout

**Flight-specific params:** `flap_period`, `flap_amplitude`, `flap_sweep` (0=up-down, 1=circular), `fly_height`. The bob system elevates the bird during flight modes and tracks altitude for takeoff/landing transitions.

### Raptor (CREATURE_RAPTOR)

Reuses BirdParams and bird skeleton/mesh. Distinct from Bird by:
- Proportions: bigger body (0.38m), long wings (0.55m), broader chord (0.12m), hooked beak (0.045m), large talons (0.035m), pointed taper (0.20)
- **Soar** gait (replaces Fly): slow lazy flaps with long glides, head scanning for prey, subtle body banking, wingtip dihedral
- **Dive** gait (replaces Peck): wings tucked tight, body pitched steeply down, talons extending forward at pull-up, dramatic altitude drop
- **Circular wing motion**: `flap_sweep=0.6` adds yaw 90° out of phase with roll, creating elliptical wingtip path (raptor-style vs songbird up-down)

### Snake (CREATURE_SNAKE)

**Skeleton** — 15 joints, pure spine chain with no limbs:
- Root (anchor) at 1/3 from head (center of mass)
- Forward chain (5 joints → head)
- Backward chain (9 joints → tail tip)
- Branching design so root motion advances the whole body naturally

**Mesh** — continuous path tube from head to tail, radius profile: wide flat head (ellipsoid) → neck taper → peak girth at 30% → tail taper. Yellow snake eyes. Countershaded olive/brown over yellowish belly.

**4 gaits:**
- Slither: sine wave propagating head→tail via per-segment yaw keyframes. Forward chain uses positive yaw, backward chain negated (accounts for reversed chain direction). Head counter-steers (40%), tail whips (140%). Configurable: `slither_period`, `slither_amplitude`, `slither_waves`.
- Fast: half period, 30% more amplitude, extra half-wave for tighter curves
- Idle: static S-curve held by per-segment yaw, head scanning with tongue flick
- Strike: S-coil buildup (neck curves up, body counter-curves), explosive head lunge, tension ripple down body, retract

### Bone buffer regeneration

Fixed a bug where the skeleton overlay's bone vertex buffer wasn't regenerated when switching creature types, causing stray lines to old joint positions.

### Species file + creature mesh updates

- `Archetype::Raptor` and `Archetype::Snake` added to `agent.h`
- `archetype_string()` / `parse_archetype()` handle all 6 archetypes
- `save_animal()` / `load_animal()` accept `SnakeParams` and serialize all raptor/snake morphology
- `creature_mesh.cpp` `ensure_model()` generates meshes and gait cycles for Raptor and Snake

## File inventory

```
bestiary/
├── animals/
│   ├── animal_mesh.h/cpp      — countershade() added; all winding fixed
│   ├── herbivore.h/cpp        — belly_color added
│   ├── predator.h/cpp         — belly_color added
│   ├── rabbit.h/cpp           — belly_color added
│   ├── bird.h/cpp             — NEW: BirdParams, 23-joint skeleton, mesh gen
│   ├── snake.h/cpp            — NEW: SnakeParams, 15-joint branching skeleton, mesh gen
│   └── walks/
│       ├── bird_walk.cpp      — NEW: walk, hop, fly, run, idle, peck, takeoff, land, perch, soar, dive
│       └── snake_walk.cpp     — NEW: slither, fast, idle, strike
├── creature/
│   ├── agent.h                — Raptor + Snake archetypes
│   ├── creature_mesh.cpp      — Raptor + Snake ensure_model() cases
│   └── creature_profile.h     — unchanged
├── species_file.h/cpp         — Raptor + Snake save/load/archetype strings; SnakeParams in API

apps/animals_lab/
├── animals_lab.h              — CREATURE_RAPTOR + CREATURE_SNAKE enums; gaits[8]; raptor/snake params
├── animals_lab.cpp            — full UI for all 6 creatures; bone buffer regen on creature switch
└── shaders/mesh.fs.hlsl       — hemisphere ambient lighting
```

## What's next

### 1. World lab creature integration
The world lab currently spawns 5 species (3 herbivores, wolf, rabbit). To add birds/raptors/snakes:
- Add `CreatureProfile` entries in `world_lab_init()` with appropriate energy models, speeds, diet
- Bird: low body mass, fast move speed, seed-eater (grass/wildflower caloric values)
- Raptor: medium mass, fast chase speed, hunts rabbits and snakes (needs `hunt_radius`, `attack_range`)
- Snake: low mass, slow move speed, hunts rabbits (ambush predator — low `hunt_radius`, high `attack_range`)
- Add spawn logic and population tracking graphs
- The creature mesh system already handles all 6 archetypes

### 2. Flight behavioral agent
Current agents are 2D ground-based. Flight needs:
- Altitude component in Agent struct (`float altitude = 0.0f`)
- New agent states: `Fly`, `Perch`, `Dive` (for raptor)
- Takeoff/landing state transitions
- Tree positions accessible from agent system (for perch targets)
- Modified steering that works in 3D for flying agents

### 3. Seed dispersal
Birds eating from plants → depositing seeds at new locations:
- New action: "deposit seed" creates a PlantInstance at the agent's position
- Triggered probabilistically during flight based on time since last forage
- Links the plant population system to creature behavior (currently one-directional: plants → creatures)

### 4. Snake locomotion in world
The 2D agent system works for snake ground movement, but:
- Snake has no legs — `move_speed` drives forward motion from undulation
- Ambush behavior: snake coils and waits near prey paths (new idle-with-intent state)
- Strike should be a short-range lunge, not a chase

### 5. Mesh quality refinements
- Hooked beak for raptor (currently straight like songbird)
- Scale/pattern texture on snake body
- Feather detail on wing trailing edges
- Better joint blending at sharp bends (graze hinge, snake coil)

## Build

```bash
cd "/Users/fields/Documents/driftEngine/Drift Game Engine/build"
cmake .. && cmake --build . -j$(sysctl -n hw.ncpu) --target animals_lab
./apps/animals_lab/animals_lab   # run from build/ for shader paths
```

## Key design decisions

### Bird leg placement
Legs attach at spine_1 (mid-body) not pelvis (rear). Matches real bird center of gravity. Changed from initial implementation based on user feedback.

### Snake anchor point
Root joint at 1/3 from head, with forward chain (→head) and backward chain (→tail). Forward chain yaw is positive, backward chain negated to create continuous S-curve. Previous head-as-root made the snake look like it was being dragged by the head.

### Raptor vs Bird
Both use BirdParams and generate_bird_mesh(). Raptor is a separate creature type (not a preset) because it needs different gaits (soar/dive vs fly/peck) and will have different behavioral profiles (predator vs herbivore). The `flap_sweep` parameter on BirdParams controls circular vs linear wing stroke.

### Countershading approach
Post-process all vertices rather than modifying mesh gen functions. Uses bind-pose normal Y, so the blend is stable regardless of animation pose. Applied globally including to accent-colored parts (beak, eyes, legs) — the effect is subtle on those since their normals mostly point sideways.

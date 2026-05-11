# Handoff: Ecosystem v0.1 — Herbivore Creatures in World Lab

## What was built (2026-05-10)

Starting from a plant-only bestiary and World Lab (terrain + water + plants), this session added the first creature system and iterated it to a self-sustaining ecosystem.

### Creature architecture (`bestiary/creature/`)
- **agent.h** — `Agent` POD struct: position, velocity, energy, age, heading, sex (m/f), behavioral state (Wander/Graze/Flee/SeekMate), flee direction/timer, mate cooldown, animation blend state
- **herbivore.h/.cpp** — `HerbivoreProfile` (species tuning knobs), `CreatureWorldView`, `update_creatures()`, `spawn_creatures()`, stats functions
- **creature_mesh.h/.cpp** — generates skinned herbivore meshes from `animals/herbivore.h` model, CPU-skinned per frame, animation selection by state+speed with crossfade blending
- **vegetation_field.h/.cpp** — still exists in code but removed from the active creature/plant loop (vestigial, candidate for deletion)

### Behaviors implemented
- **Wander** — wander-circle steering with jitter for naturalistic paths
- **Food-seeking** — scans nearby `PlantInstance` health, steers toward best target weighted by health/distance
- **Graze** — finds nearest healthy plant, reduces its health directly, gains energy at 10% trophic efficiency
- **Herd** — boids (cohesion + separation + alignment), same-species only, tunable weights
- **Flee** — triggered by terrain brush proximity, steers away from threat point, timed duration, redirectable on repeated brushing
- **SeekMate** — when energy > threshold and off cooldown, finds opposite-sex same-species mate above threshold, both steer toward each other, spawn offspring on meeting

### Reproduction
- M/F sex flag assigned randomly at spawn
- Both parents must exceed `reproduce_threshold` energy and be off cooldown
- On meeting: offspring spawns at midpoint with `offspring_energy`, random sex
- Both parents pay `reproduce_cost` energy, enter `reproduce_cooldown` period
- Population dynamics: eat → energy → reproduce → population grows → food pressure → starvation → equilibrium

### Aging
- `max_age` on HerbivoreProfile (default 300s), agents die when age exceeds it
- Creates generational turnover, prevents immortal herds

### Plant population system (`bestiary/distribution.h/.cpp`)
- **PlantInstance** — persistent individuals with position, species (grass/bush/tree), health (0..1), seed
- **place_ecosystem()** — Poisson disk placement, returns persistent instances starting at health=0.01
- **tick_plant_population()** — per-frame health growth/decay based on species suitability at plant's location
- **sprout_plants()** — species-specific patterns: grass spreads from edges (1-3m runners), trees form stands (3-8m seed shadow), bushes scatter randomly
- **generate_mesh_from_population()** — builds VegetationMesh from persistent instances, scaled by health

### Ecosystem automation
- Watershed terrain with ridgeline, hills, and pool depression
- Persistent spring (continuous SWE water pulse on hillside)
- Auto environment refresh every 8s (water readback → moisture field update)
- Auto creature spawn at t=3s
- Plant sprouting on each refresh cycle

### Procedural herbivore model integration
- `animals/herbivore.h/.cpp` model (24-joint skeleton, tube+cap mesh with eyes/ears/nose) renders in World Lab
- 5 animations auto-selected: idle (<0.1 m/s), walk (0.1-2.0), trot (2.0-4.0), run (>4.0 or fleeing), graze (Graze state)
- Animation crossfading: 0.2s slerp blend on state transitions
- Heading-oriented rendering with correct rotation convention

### World Lab ImGui controls
- Energy balance: energy use, idle use, nutrient gain, graze consume, reproduce threshold, max age sliders
- Plant: growth rate, decay rate, plant count display
- Ecosystem: autorun toggle, spring toggle + rate slider
- Population stats: alive/total, plant count, plant:animal ratio, energy min/avg/max
- Buttons: Spawn, Clear, Replant now, Reset terrain
- Escape key to quit

## Current state

- Single species (box/grazer profile) — multi-species infrastructure exists but disabled for equilibrium testing
- 3 mesh shapes (box/wedge/disc) in creature_mesh.cpp but currently bypassed by the procedural herbivore model
- Energy balance requires manual slider tuning to achieve reproduction (trophic efficiency ~0.10 is tight)
- Plant population grows from ~500 to ~600+ over 10 minutes with spring active
- Population of ~7-20 creatures sustainable depending on energy tuning

## Known issues / rough edges

1. **Energy balance fragile** — ad-hoc knobs, no principled caloric model. Need: body-mass metabolic rate, plant caloric value, distance-based foraging cost
2. **O(n²) queries** — herd scans all agents, food-finding scans all plants, sprouting checks all plants. Blocks scaling past ~100 agents / ~1000 plants
3. **No terrain awareness** — creatures walk into water, ignore slopes, clip through steep terrain
4. **VegetationDensityField still in codebase** — removed from active loop but files remain. Can delete or repurpose
5. **Static model cache** — only one model cached. Multi-species would need per-species cache
6. **Dead agents accumulate** — removed from alive count but not from vector. `agents` vector grows unbounded with dead entries

## Files touched this session

### New files
- `bestiary/creature/agent.h`
- `bestiary/creature/herbivore.h`, `herbivore.cpp`
- `bestiary/creature/vegetation_field.h`, `vegetation_field.cpp`
- `bestiary/creature/creature_mesh.h`, `creature_mesh.cpp`
- `species/meadow_grazer.toml`

### Modified files
- `bestiary/distribution.h`, `distribution.cpp` — PlantInstance, place_ecosystem, sprout_plants, generate_mesh_from_population, density-modulated variant
- `bestiary/species_file.h`, `species_file.cpp` — herbivore TOML support
- `bestiary/CMakeLists.txt` — creature sources added
- `bestiary/animals/walks/herbivore_walk.cpp` — idle + graze cycles added
- `apps/world_lab/main.cpp` — full creature integration, watershed terrain, spring, autorun, ImGui controls

## Next session priorities

1. **Energy model overhaul** — principled caloric system replacing slider tuning
2. **Spatial partitioning** — grid bucketing for O(n) creature/plant queries
3. **Terrain-aware movement** — water avoidance, slope cost
4. **Multi-species return** — bring back 3 herbivore species with the procedural model
5. **Predators** — carnivore class with hunt/chase behaviors

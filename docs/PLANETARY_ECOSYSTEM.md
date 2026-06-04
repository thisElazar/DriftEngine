# Planetary Ecosystem — integrated plan (Globe gets life)

Status: **designed, not started.** Written 2026-06-01. Verified against current
code (HEAD after the input-unification commit `93e8b92`).

## Why this doc exists

Two threads were floated as "next": (1) World Lab creature integration — flight,
seed dispersal, snake ambush; (2) environment fields — watersheds as ground
truth, replacing interim noise. **Both are already done at World Lab / single-tile
scale.** What they actually point at is one bigger thing: **the Globe (the planet)
has no ecosystem.** It is terrain + water simulation only — no vegetation, no
creatures. Bringing the ecosystem to planetary scale is the real integration of
the two threads, and it's where "watersheds as ground truth" finally matters.

## Current state (verified, with file refs)

**World Lab (`apps/world_lab/world_lab.cpp`) — full ecosystem on one flat 1024²
tile:**
- Environment field is **water-derived**, not noise: `refresh_env` (~790) reads
  the SWE water sim, `build_moisture_grid` (`src/grid_util.cpp:95`) turns water
  depth into a moisture grid (capillary-blur heuristic), temperature is
  `0.5 + 0.05*height`. Packaged as a `bestiary::EnvironmentField` (~804).
- Plants: `place_ecosystem` / `tick_plant_population` / `sprout_plants`
  (`bestiary/distribution.*`) consume that field via `SpeciesSuitability`
  moisture/temp curves.
- Creatures: `spawn_creatures` + `update_creatures` (`bestiary/creature/
  creature_profile.cpp`, state machine ~454–801). **Flight, snake ambush, and
  seed dispersal are fully implemented** (Fly/Perch/Dive + altitude; Ambush→
  Strike; flying birds drop `PlantInstance` seeds on a forage cooldown).
- Animation **crossfade** is done: `creature_mesh.cpp:267–295` slerps between
  walk cycles over a 0.2 s blend.

**Globe (`apps/globe/globe.cpp`) — no ecosystem:** zero references to
vegetation / plants / ecosystem / creatures. Pure terrain + hydrology, with LOD
tile streaming and a planet-scale SWE water sim.

**Noise:** the only `make_noise_field` call is `plant_lab.cpp:726` — a design
sandbox for tuning suitability curves. Noise never reaches a real world.

**Watershed reality check:** there is **no flow-accumulation** anywhere. The
terrain "drainage basins" in `src/terrain.cpp:133` are Voronoi-shaped *visuals*,
not hydrology. World Lab moisture is local standing-water blur, not upslope/
catchment area. So "watersheds as ground truth" is still aspirational even at
World Lab scale — see Phase 4.

## The engine-agnostic asset

The entire `bestiary` ecosystem stack takes an `EnvironmentField` (a
`std::function<EnvironmentSample(x,z)>`) and a height function — nothing
GLFW/Vulkan/tile specific. So **the planet reuses all of it.** The work is:
build a planet-scale field, place plants/creatures per region, render instanced,
and solve the one coordinate-system fork below.

---

## The design fork (decide first): agents are 2D, the planet is a sphere

`bestiary::Agent` stores `pos[2]` / `vel[2]` — a flat 2D world. World Lab is a
flat tile, so that's fine. The planet is a sphere. Options:

- **(A) Per-region tangent frame (recommended first):** pick an active region
  around the camera, define a local tangent plane (origin + east/north basis on
  the sphere), and run the *existing* 2D agent sim inside it. Plants too. Convert
  local (x,z) → sphere position only at render. Creatures stay regional (they
  don't roam the whole globe) — which is realistic and dodges the hard problem.
- **(B) Geodesic agents:** rework Agent to 3D/sphere-native steering. Correct for
  globe-spanning migration, much more work, not needed yet.

Go with **(A)**. It keeps `update_creatures` untouched and is plenty for "life
appears where you're looking."

---

## Phases

### Phase 1 — Planet-scale `EnvironmentField` (no rendering yet)
Mirror `refresh_env` at planet scale. Inputs already exist on the Globe: the SWE
water-state images + the CPU terrain height function (`cpu_terrain_height_with_
stamps`). For an active region:
- moisture: read planet water depth for the region → `build_moisture_grid` (reuse
  as-is) → sample in the region's local frame.
- temperature: latitude (`sphere_dir.y`) + altitude lapse, instead of World Lab's
  flat `0.5+0.05*h`.
Expose as `EnvironmentField` over local (x,z). Acceptance: log/inspect samples;
no visual yet.

### Phase 2 — Vegetation on the Globe (first visible slice)
For the highest-LOD tile(s) under the camera: `place_ecosystem(eco, env)` →
`collect_plant_instances` → `PlantGPUInstance` with `y = planet height`. Port
World Lab's instanced clump/plant pipeline + shaders. Hook rebuild into the
existing tile stream (populate on tile activation, drop on eviction; persist the
`PlantInstance` population per tile so it doesn't churn). **This is the
recommended first deliverable** — grass/trees appearing on the planet under the
camera, driven by real water-derived moisture. Lowest risk, highest motivation.

### Phase 3 — Creatures on the Globe (depends on the fork)
With a per-region tangent frame (Option A): `spawn_creatures` in the region,
`update_creatures(dt, env, height_fn_in_local_frame)`, render via `creature_mesh`
with `terrain_y` from the planet height fn. Flight/ambush/dispersal come for free.
Handle region recentering as the camera moves (carry agents, rebase the frame).

### Phase 4 — Watershed ground truth (the real "fields" upgrade)
Replace the capillary-blur moisture with actual hydrology: D8/D-infinity flow
routing + upslope-accumulation area on the heightfield, combined with standing
water from the SWE sim. This is what makes valleys/catchments wetter than ridges
independent of current rainfall. Lives near `src/terrain.cpp`/`grid_util.cpp` and
**benefits World Lab too** (shared `build_moisture_grid` replacement) — so it's a
genuine cross-scale win, not Globe-only.

---

## Files

**Reuse unchanged (engine-agnostic):** `bestiary/environment.{h,cpp}`,
`bestiary/distribution.{h,cpp}`, `bestiary/creature/creature_profile.{h,cpp}`,
`bestiary/creature/creature_mesh.{h,cpp}`, `bestiary/animals/*`,
`src/grid_util.cpp` (`build_moisture_grid`).

**New / changed:** `apps/globe/globe.{h,cpp}` (region frame, env builder, plant +
creature layers, instanced rendering); likely a small shared header extracted
from `world_lab.cpp` for "ecosystem on a tile" so Globe and World Lab don't fork
the placement/upload logic; `src/terrain.cpp` + `grid_util.cpp` for Phase 4.

## Risks / open questions
- **Coordinate fork (Phase 0)** is the one real design decision — settle it before
  Phase 3.
- **Scope of life:** plants/creatures should exist only on near-camera high-LOD
  tiles (like grass in most planet renderers), not the whole globe. Confirm.
- **Persistence** across tile eviction/recenter (don't re-randomize on every
  camera nudge).
- **Cost:** per-region `place_ecosystem` + water readback — keep it to active
  tiles and amortize.

## Suggested first slice
Phase 1 + a minimal Phase 2: grass + trees on a single high-LOD Globe tile under
the camera, placed from the Globe water sim's moisture, reusing `place_ecosystem`
and World Lab's instanced veg renderer. Defer creatures and flow-accumulation.
Proves the planet-scale `EnvironmentField` + instanced-veg path with least risk.

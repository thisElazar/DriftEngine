# Handoff: Planet water → first-person redirect

## TL;DR

Planet-scale SWE water has been built out (selective per-tile sim, persistence
across camera moves, cross-tile flow with auto-expanding domain, water stamps
for LOD persistence, h-adjust on terrain stamps). It functions but is **visually
unsatisfying** because at level-8 cell sizes (~800 m) wave propagation takes
tens of seconds — humans don't read that as "flow." The next agent should pivot
to **first-person / local-scale water** (the existing `!g_ui.ocean_enabled`
flat-grid path at `bp.cell_spacing`, typically 5–50 m), polish brush + flow +
terrain interaction there, and only then revisit the planet-scale port.

## What was completed this session

### Planet-scale SWE plumbing (built on commit 121fe49 baseline)
- **Stable per-tile slot allocator** (`main.cpp:1632-1641, 2153-2237`):
  `tile_slot_map: QuadNode → pool_index`, `slot_to_tile` reverse, `free_slots`
  LIFO. A QuadNode keeps the same pool slot for as long as it remains visible
  *or* disturbed. Disturbed tiles are anchored across camera moves / LOD
  transitions — this is what makes brushed water persist.
- **Selective SWE step** (`main.cpp:2487-2620`): only disturbed tiles step;
  step iterates `disturbed_tiles` (not visible tiles), so anchored-but-invisible
  tiles still simulate.
- **`pending_init` queue** funnels init dispatches: visible-tile allocator and
  cross-tile auto-anchor both feed it, init pass drains it.
- **Cross-tile flow Phase 1** (same-face same-level only):
  - Step shader (`shaders/planet_swe_step.cs.hlsl`) takes 4 neighbor pool
    slots; off-edge samples read the neighbor's edge cells instead of clamping.
  - GPU edge-flag buffer (`edge_flags_buf`, `main.cpp:292-313`): each step
    `InterlockedOr`s a 4-bit flag (L/R/D/U) per pool slot when water exceeds
    `static + 0.25 m` at any edge.
  - CPU auto-anchor (`main.cpp:2153-2186`): each frame, reads previous-frame
    edge flags, calls `planet_neighbor_same_face` to find missing neighbors,
    allocates slots and queues them in `pending_init`.
  - `planet_neighbor_same_face` (`src/planet.{h,cpp}`): same-face same-level
    only. **Returns invalid for face seams; reflective fallback there.**
- **Water stamps for LOD persistence** (`pipeline.h` `WaterStamp`, `main.cpp`
  `g_water_stamps`, `shaders/planet_swe_init.cs.hlsl`): mirror of terrain
  stamps; brushing accumulates one every 0.1 s; SWE init applies them at
  every LOD's first init so a brushed lake shows up at any zoom (when the
  ancestor tile gets allocated).
- **h-adjust on terrain stamps** (`shaders/planet_swe_h_adjust.cs.hlsl`,
  `main.cpp:2424-2487`): when a terrain stamp adds Δz at a disturbed tile,
  `h_new = max(0, h - Δz)` on previously-wet cells. Preserves water surface
  across bed changes; lateral gradient drives downhill flow via HLL.
- **Brush radius is world-angular** (`main.cpp:~2566`): matches the cursor
  ring (`brush_world.w` is in radians, world-arc = `r * PLANET_RADIUS`).
- **Friction / damping defaults lowered** (`ui.h`): `0.01 → 0.002`,
  `0.001 → 0.0002`. Helps a bit, doesn't fix the scale issue.

### Earlier from same commit baseline
- Re-enabled the planet SWE block (was `if (false && g_ui.ocean_enabled)`).
- SWE init writes all three textures (state_a, state_b, water_output) so a
  freshly-allocated pool slot is fully consistent (fixed a vertex-explosion
  bug on zoom).

## Why redirect

At level 8, `cell_dx ≈ PLANET_RADIUS / (PLANET_TILE_RES - 1) / 256 ≈ 800 m`.
With sea_level ≈ 800 m of water, wave speed `c = √(g h) ≈ 90 m/s`, so a wave
crosses one cell in ~9 s. **Brushed perturbations are visible-but-slow.** At
level 12 the cells are 50 m and the timescale is ~0.5 s; that level is only
visible up close.

The user can see that the planet "works" but it doesn't *feel* like water,
because the visual rate of change is wrong for the spatial scale. This is a
fundamental SWE-at-planetary-scale problem, not a bug.

The flat-grid SWE in `!g_ui.ocean_enabled` (`main.cpp:2789+`) runs on
`bp.cell_spacing` cells of 5–50 m and is the right place to dial in the
"feel" before re-attacking planet scale.

## Recommended starting point for the next agent

1. **Toggle `Ocean enabled` off** in the UI to switch to the flat-grid SWE.
   The water brush is already wired there (`main.cpp:~2837`, `pulse_pending`
   path). This path uses `swe_step.spv` (the original flat-grid solver), not
   the planet variant.
2. **Add a first-person / surface camera mode** alongside the existing
   orbital planet camera. `src/camera.{h,cpp}` is the camera module; the
   existing camera does sphere-relative orbit. A walk/fly mode that keeps the
   eye at terrain-height + small offset and consumes WASD as tangent-plane
   velocity is the minimal addition.
3. **Tune brush + visualization at 1:1 scale.** The flat-grid water already
   has foam, surface gradients, Froude-based wave-breaking — none of which
   read at planet scale. They will read at local scale.
4. **Once water *feels* right at local scale**, decide what of it ports up
   to planet:
   - Multi-LOD dynamic SWE (downsample disturbed children into ancestor
     ocean). Probably the right answer but expensive.
   - Hybrid: dynamic at the brushed level, hierarchical static water-stamp
     summaries above it (current Phase B is the seed of this idea).

## Files / line references

- `src/main.cpp`
  - Slot allocator: `1632-1641, 2153-2237`
  - Auto-anchor from edge flags: `2153-2186`
  - SWE init (planet, with water stamps): `2393-2415`
  - h-adjust dispatch (terrain stamps): `2424-2487`
  - Step pass (per-disturbed-tile, neighbor-aware): `2487-2620`
  - Brush water stamp accumulation: `~2140`
  - Flat-grid SWE block (the local-scale starting point): `2789+`
- `shaders/`
  - `planet_swe_init.cs.hlsl` — applies water stamps at init
  - `planet_swe_step.cs.hlsl` — neighbor-aware step + edge flags
  - `planet_swe_h_adjust.cs.hlsl` — h_new = max(0, h - Δz) on terrain stamps
  - `swe_step.hlsl` — the **flat-grid** solver, used in `!ocean_enabled`
- `src/planet.{h,cpp}` — `QuadNode`, `QuadNodeHash`, `planet_pick_tile`,
  `planet_neighbor_same_face`, `PlanetTilePick`
- `src/pipeline.h` — `WaterStamp`, `TerrainStamp`, `PlanetSweInitPC`,
  `PlanetSweStepPC` (with `neighbor_*`), `PlanetSweHAdjustPC`
- `src/ui.h` — physics tunables; ocean toggle drives the planet/flat split

## Deferred — planet-scale work to NOT pick up first

- **Cube-face seams** at tile boundaries crossing cube faces: the user picked
  "full system: faces + LOD-mismatch" earlier but only Phase 1 (same-face
  same-level) was built. `planet_neighbor_same_face` returns invalid at
  seams, so those edges currently reflect. Implementing this needs a
  per-(face, dir) table of (neighbor_face, edge_orientation) and a coord-flip
  in the step shader for the off-edge sample.
- **LOD-mismatch dynamic flow**: when a disturbed tile is adjacent to a
  visible tile at a different level, no flux flows between them. Needs 2:1
  sampling (average / replicate) at the seam.
- **h_apply_water_stamp pass**: water stamps only affect *fresh* SWE init.
  A tile that was already initialized before a stamp was added doesn't
  retroactively pick up the new stamp. Common path (zoom-out reveals new
  tiles → fresh init reads stamps) is fine; the edge case is a tile that
  stays visible through the brushing. A delta-apply compute (mirror of
  h_adjust but adding instead of subtracting, no wet-cell gate) would fix
  this. Skipped for now to keep scope contained.
- **Pool exhaustion**: with anchored disturbed tiles never freed, brushing
  500+ unique tiles will exhaust the 2048-slot pool. Falls back silently to
  slot 0 today (broken). Not a problem for normal play; would need an
  eviction policy at scale.

## Open questions for the next agent

- Should the first-person camera be wired into the existing `world_lab` /
  `species_lab` apps, or stay in the main `drift_engine` binary?
- Is the "lake painter" intent (water stamps as a static deposit layer
  underneath dynamic waves) the right long-term abstraction? Or should
  brushed water always be 100% dynamic and persistence be solved differently
  (e.g., volumetric water-level field on the planet, separate from SWE)?
- Tuning: at local scale, what are the right defaults for brush_strength,
  pulse_radius, friction, damping? The current planet-scale defaults
  (`brush_strength=1.5`, `friction=0.002`, `damping=0.0002`) are probably
  too gentle for sub-second timescales.

## Building / running

```sh
cd "Drift Game Engine/build"
cmake --build . -j8
./drift_engine
```

`./apps/world_lab/world_lab` is a separate executable that may be more
focused for water work — worth checking what it currently renders.

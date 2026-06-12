# Handoff: PLANET (DEV) — multi-tile world sandbox

Status as of 2026-06-11, branch `globe-water-field`, HEAD `5486c1e`.
Commits: `dbfaf34` (module) → `3b3a4dd` (creatures + moisture) → `6861a46`
(brush modes) → `5486c1e` (CONFIGURE_DEPENDS build fix).

## What this module is, and why it exists

PLANET (DEV) (`apps/planet_dev/`) is the flat proving ground for scaling
World Lab's water + ecosystem across multiple tiles — the path to the full
planet (and eventually multiple planets), validated at a scale where every
behavior is visible. It exists because a previous attempt put vegetation
directly on the Globe's LOD tile stream and failed repeatedly (that work is
parked on branch `veg-experiments`; lesson recorded in
`docs/PLANETARY_ECOSYSTEM.md`'s addendum on that branch). The rule this
module enforces: **tiles are storage; the world model lives in one
continuous world coordinate frame.**

A 2×2 grid of 256 m tiles (`PD_GRID 256`, 1 m cells, world = ±256 m²).
Orbit camera (RMB drag, wheel zoom) with WASD panning the target across
seams. Launcher button "Planet (Dev)" in the World group, plus standalone
`build/apps/planet_dev/planet_dev`.

## The architectural core (read this before changing anything)

**Water runs the planet's own compute shader, unmodified.**
`shaders/planet_swe_step.cs.hlsl` — the same SPIR-V the Globe dispatches —
is reused verbatim: it takes all geometry from push constants
(`PlanetSweStepPC`, `src/pipeline.h`), reads tile state from
`Texture2DArray` layers, and resolves off-edge samples through the
`neighbor_left/right/down/up` slot indices (`ResolveSample`, shader lines
59–81; `0xFFFFFFFF` = reflective wall). PLANET (DEV) supplies a fixed
neighbor table (`PD_NEIGHBORS`, planet_dev.h) instead of the Globe's
quadtree lookup. `sea_level = -10000` disables the ocean coupling (World
Lab's convention). The SWE init shader is NOT used — clearing the state
images is equivalent with the ocean off.

**Everything CPU-side is world coordinates.** Plants are one population
with world (x, z); creatures roam the whole ±256 m world
(`CreatureWorldView.tile_half_* = PD_WORLD_HALF`); both route height/water
queries to the owning tile via `pd_tile_of()` + `pd_world_to_local_*()`.
The brush (1 = raise, 2 = lower, 3 = water; LMB applies) is applied in
world space to ALL tiles, each receiving the brush in its own local frame —
edits and pours straddling a seam are symmetric by construction.

**Moisture is seam-free by stitching, not halos.** `refresh_env` reads back
all 4 water layers, assembles a 512² world grid, runs `build_moisture_grid`
ONCE globally, slices back per tile for the GPU overlay. The env field and
creature water queries read the world grids (`water_world`,
`moisture_world`) directly. (At N×N streaming scale this becomes a halo
exchange; stitching is correct and simple while all tiles are resident.)

**Terrain is one continuous analytic function.** `pd_height(wx, wz)`
(planet_dev.cpp) is pure world-space — tiles agree at seams by
construction. The central depression at (0,0) sits on the 4-corner point
deliberately: it is the canonical seam test (pour water there → symmetric
pooling across all four tiles). CPU heightmaps `hm_cpu[4]` mirror every
brush edit via `cpu_apply_brush` so picking/plants/creatures track edits.

GPU layout: 5 array images ×4 layers (terrain R32F, SWE state A/B + output
RGBA16F, moisture R32F), 2 SWE descriptor sets total (ping/pong — tile +
neighbors ride push constants, the Globe's pattern), `edge_flags` buffer
(4 uints, host-visible) feeding the per-tile L/R/D/U debug readout in the
UI. Rendering: forward-Z LESS_OR_EQUAL (World Lab convention, NOT the
Globe's reverse-Z), one shared 256-grid mesh drawn 4× with per-tile
`PlanetDevTerrainPC` (tile_origin + layer + seam_highlight), then instanced
clumps (existing `world_clump_*` shaders — instances are world-space), then
the creature mesh.

## Verified working

Builds all targets; liveness-checked headless run (10 s, zero
aborts/validation errors, 10 plant + 8 creature species). The user has
driven it interactively: water crossing seams, creatures wandering,
brush modes. Worth re-verifying after any change: the (0,0) pour test,
a raised ridge straddling x=0, moisture overlay continuity at seams,
creatures crossing the highlighted borders.

## Known warts (deliberate, inherited from World Lab)

1. **Per-frame `vkDeviceWaitIdle`** in the creature mesh re-upload
   (planet_dev_tick) and in `upload_plant_instances` (0.5 s cadence).
   The creature one runs EVERY frame while creatures are enabled — the
   single biggest perf cleanup available. Fix pattern: per-frame-in-flight
   buffers or the Globe-vegetation retire-queue pattern (see
   `veg-experiments` branch, globe_region.cpp).
2. **Blocking readbacks**: `refresh_env` does 4 × oneshot
   `readback_water_depth` every 8 s (one visible hitch). Async candidates:
   copy-to-buffer in the frame command buffer + fence-deferred CPU read.
3. **Half-texel height step at seam lines** (per-tile edge texels under
   linear clamp). Cosmetic; visible only with seam highlight off and low
   sun. Fix: 1-texel border duplication or shared edge rows.
4. **Brush pulse vs. solver**: water brush amount is per-substep
   (`strength × swe_dt`), so perceived pour rate scales with substeps.

## Gotchas that already bit once

- **`file(GLOB)` shader rules are configure-time.** Fixed with
  `CONFIGURE_DEPENDS` everywhere (`5486c1e`) — but any NEW glob added
  without it reintroduces silently-missing .spv files (the launcher crashed
  on a missing brush shader).
- **Headless smoke tests must check liveness** (`kill -0` after sleep) and
  grep for `abort|Failed|error`, not just validation strings — this repo
  fails via abort() with an fprintf, and an empty log reads as a pass.
- The plant roster builder is **duplicated** from world_lab.cpp
  (`build_plant_roster`/`load_plant_profiles` in planet_dev.cpp). Extract
  to `apps/shared/` (or bestiary) once the seam settles — there's a clean
  extraction already designed on the `veg-experiments` branch
  (`bestiary/plant_roster.{h,cpp}`) that can be cherry-picked.

## Next steps, in rough order of leverage

1. **Kill the per-frame waitIdle** (creature mesh upload) — see wart #1.
2. **N×N + tile streaming** — the reason this module exists. Grow the
   neighbor table beyond 2×2, activate/deactivate tiles by camera distance
   (pool slots + free list, exactly the Globe's `tile_slot_map` pattern),
   and let the edge-flags readback drive activation the way the Globe's
   auto-anchor does. The SWE shader needs zero changes. Moisture stitching
   becomes a halo exchange at that point.
3. **Erosion / atmosphere across tiles** — World Lab has both as single-
   tile compute passes; the erosion shader reads/writes terrain + water and
   would need the same array-layer + neighbor treatment as the SWE step
   (terrain is where cross-tile erosion gets interesting).
4. **Sea level / ocean toggle** — bind `planet_swe_init` (it needs a dummy
   water-stamps buffer at binding 4) and pass a real `sea_level` to
   init + step; the shader already supports it.
5. **Back to the sphere** — once N×N streaming works flat, the same
   world-frame discipline maps onto the Globe: the hard part (which
   coordinate frame owns the ecosystem) is what this module proves.
   Multiple planets = multiple pool sets + per-planet world frames over
   the same machinery.

## Build / run

```sh
cd "Drift Game Engine/build"
cmake --build . -j8
./apps/planet_dev/planet_dev        # standalone
./apps/launcher/launcher            # or via "Planet (Dev)" button
```

Controls: RMB-drag orbit, wheel zoom, WASD pan, 1/2/3 brush modes,
LMB apply, ` toggles nothing here (menu is always on), ESC/Back exits.
UI panels: Brush, Water (substeps/physics), Plants (density/replant),
Creatures (count/speed/respawn), Tiles (cursor→tile/cell readout,
per-tile edge flags, seam highlight, frame ms).

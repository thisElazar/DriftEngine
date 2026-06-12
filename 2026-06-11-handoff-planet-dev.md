# Handoff: PLANET (DEV) — multi-tile world sandbox

Status as of 2026-06-11 (evening), branch `globe-water-field`, HEAD `e999761`.
Commits: `dbfaf34` (module) → `3b3a4dd` (creatures + moisture) → `6861a46`
(brush modes) → `5486c1e` (CONFIGURE_DEPENDS build fix) → `eccfb68` (retire
queue kills per-frame waitIdle) → `e999761` (8x8 world, 16-slot streaming).

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

An 8×8 world of 256 m tiles (`PD_GRID 256`, 1 m cells, world = ±1024 m²)
STREAMED through a 16-slot GPU pool (`PD_POOL`). Orbit camera (RMB drag,
wheel zoom) with WASD panning the target across seams — panning streams
tiles in/out. Launcher button "Planet (Dev)" in the World group, plus
standalone `build/apps/planet_dev/planet_dev`.

**Streaming (new, `e999761`).** `tile_slot[]`/`slot_tile[]`/free list =
the Globe's `tile_slot_map` pattern, flat. Per tick (`pd_update_streaming`):
recycle slots past their FRAMES_IN_FLIGHT cooldown → deactivate resident
tiles beyond `ui_stream_radius` + one tile of hysteresis IF their edge
flags are quiet → auto-anchor neighbors of edge-flagged tiles (water pulls
tiles in before it arrives, the Globe's auto-anchor) → activate by camera
distance, nearest first. The SWE shader is untouched: neighbor slots are
resolved through the map each substep; a non-resident neighbor is a
reflective wall for at most the frame the anchor takes to fire. State
survives streaming: all 64 CPU heightmaps persist (brush edits to
streamed-out tiles upload on activation) and water depth saves to the
persistent `water_world` grid on deactivate / restores on activate
(velocity dropped — pooled water is near-static). Edge-flagged tiles never
deactivate, so flowing water pins its tiles.

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
the RESIDENT water layers into the persistent 2048² world grid, runs
`build_moisture_grid` over the resident tiles' bounding box (a whole-world
blur would be a 16× hitch for tiles nobody simulates), slices back per
slot for the GPU overlay. The env field and creature water queries read
the world grids (`water_world`, `moisture_world`) directly — including
saved water under streamed-out tiles.

**Terrain is one continuous analytic function.** `pd_height(wx, wz)`
(planet_dev.cpp) is pure world-space — tiles agree at seams by
construction. The central depression at (0,0) sits on the 4-corner point
deliberately: it is the canonical seam test (pour water there → symmetric
pooling across all four tiles). CPU heightmaps `hm_cpu` (all 64 tiles,
persistent) mirror every brush edit via `cpu_apply_brush` so
picking/plants/creatures track edits everywhere, resident or not.

GPU layout: 5 array images ×16 POOL layers (terrain R32F, SWE state A/B +
output RGBA16F, moisture R32F), 2 SWE descriptor sets total (ping/pong —
slot + neighbors ride push constants, the Globe's pattern), `edge_flags`
buffer (16 uints, one per SLOT, host-visible) latched each tick — it
drives streaming AND the per-slot L/R/D/U debug readout in the UI.
Rendering: forward-Z LESS_OR_EQUAL (World Lab convention, NOT the
Globe's reverse-Z), one shared 256-grid mesh drawn 4× with per-tile
`PlanetDevTerrainPC` (tile_origin + layer + seam_highlight), then instanced
clumps (existing `world_clump_*` shaders — instances are world-space), then
the creature mesh.

## Verified working

Builds all targets; liveness-checked headless runs (12 s, zero
aborts/validation errors, 10 plant + 8 creature species, 4 center tiles
activate at init and the resident set stays stable while idle). NOT yet
driven interactively since the streaming change — the things to verify by
hand: WASD-pan past the loaded frontier (tiles stream in ahead, far quiet
tiles stream out), pour at (0,0) (canonical 4-corner seam test), pour at
a resident/non-resident edge and watch the auto-anchor pull the neighbor
in, pan away from a pool and back (water depth restored), brush a seam,
moisture overlay continuity.

## Known warts (deliberate)

1. ~~Per-frame `vkDeviceWaitIdle`~~ FIXED (`eccfb68`): retire queue
   (`s.retire` + `frame_counter`), drained in tick, destroys buffers
   FRAMES_IN_FLIGHT frames after replacement.
2. **Blocking readbacks**: `refresh_env` does one oneshot
   `readback_water_depth` PER RESIDENT SLOT every 8 s (now up to 16, was
   4 — the hitch grew). Batch into one oneshot over all resident layers,
   or go async: copy-to-buffer in the frame command buffer +
   fence-deferred CPU read. Deactivation also does one blocking readback
   (rare, fine).
3. **Half-texel height step at seam lines** (per-tile edge texels under
   linear clamp). Cosmetic; visible only with seam highlight off and low
   sun. Fix: 1-texel border duplication or shared edge rows.
4. **Brush pulse vs. solver**: water brush amount is per-substep
   (`strength × swe_dt`), so perceived pour rate scales with substeps.
5. **Activation hitch** (new): `pd_activate_tile` is synchronous —
   2–3 oneshot uploads + poisson plant placement on the tick. Small for
   dry tiles; moist tiles cost more (placement density scales with
   moisture). If it bothers, stage activations over frames or move
   placement off-thread.
6. **Plant health resets on stream-out** (new, deliberate): placement is
   deterministic per tile (seed = eco.seed + t*7919) so the layout
   returns, but growth/decay history doesn't. Creatures roam the whole
   ±1024 world including unloaded tiles (CPU mirrors answer their
   queries); they render floating over void out there. Both fine for a
   sandbox; revisit when the sphere needs persistence.

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

1. ~~Kill the per-frame waitIdle~~ DONE (`eccfb68`).
2. ~~N×N + tile streaming~~ DONE (`e999761`) — needs an interactive
   verification pass (see Verified working).
3. **Erosion / atmosphere across tiles** — World Lab has both as single-
   tile compute passes; the erosion shader reads/writes terrain + water and
   would need the same array-layer + neighbor treatment as the SWE step
   (terrain is where cross-tile erosion gets interesting). Note: erosion
   writes terrain on the GPU, so the CPU heightmap mirror needs a
   readback path on deactivation, like water already has.
4. **Sea level / ocean toggle** — bind `planet_swe_init` (it needs a dummy
   water-stamps buffer at binding 4) and pass a real `sea_level` to
   init + step; the shader already supports it. With streaming this gets
   interesting: an ocean tile activating must init at sea level, not dry.
5. **Back to the sphere** — the streaming machinery this module now has
   (slot pool, edge-flag anchoring, persistent world store) is exactly
   what the Globe's quadtree needs; the coordinate-frame discipline is
   proven. Multiple planets = multiple pool sets + per-planet world
   frames over the same machinery.

## Build / run

```sh
cd "Drift Game Engine/build"
cmake --build . -j8
./apps/planet_dev/planet_dev        # standalone
./apps/launcher/launcher            # or via "Planet (Dev)" button
```

Controls: RMB-drag orbit, wheel zoom, WASD pan (pan to stream tiles),
1/2/3 brush modes, LMB apply, ` toggles nothing here (menu is always on),
ESC/Back exits. UI panels: Brush, Water (substeps/physics), Plants
(density/replant), Creatures (count/speed/respawn), Streaming (stream
radius, resident/pool counts, per-slot tile + edge flags, seam highlight,
frame ms).

# Handoff: Water Brush on Planet

## Current state (commit 121fe49)
- Static sea-level ocean renders cleanly in terrain shaders via `sea_level` push constant
- Per-tile SWE infrastructure is fully built but **disabled** (`if (false && ...)` in main.cpp ~line 2168)
- SWE at equilibrium duplicates the static ocean but with tile-boundary artifacts, so it's off
- The `max(static_depth, swe_depth)` logic in terrain.vs.hlsl masks SWE jitter below the static floor

## What's built
- `shaders/planet_swe_init.cs.hlsl` — initializes water per-tile from sea_level + heightmap array
- `shaders/planet_swe_step.cs.hlsl` — HLL Riemann solver on Texture2DArray (no sponge layer)
- GPU resources: water_state_a/b + water_output (Texture2DArray RGBA16F 64×64×2048 each)
- Pipeline: `planet_swe_init_pipeline`, `planet_swe_step_pipeline` with descriptor sets + ping-pong
- Tile change tracking: `prev_visible_tiles` + sorted tile ordering for stable pool indices
- Push constants: `PlanetSweInitPC`, `PlanetSweStepPC` (includes pulse_x/y/radius/amount per tile)

## What needs to happen for the water brush

### Core problem
The SWE simulation adds no visual value for equilibrium ocean — it just creates tile-boundary artifacts.
The brush should inject water ABOVE equilibrium, making SWE meaningful only where disturbed.

### Recommended approach
1. **Don't run SWE globally.** Track which tiles have been disturbed (brushed/stamped) in a set.
2. **Only dispatch SWE for disturbed tiles.** Undisturbed tiles use the static sea_level (zero cost, smooth).
3. **Map cursor to tile:** Reverse the cube-to-sphere projection to find face UV from `stamp_sphere_dir`, then find the finest visible tile containing that UV. Convert to tile-local grid coords (0–63).
4. **Inject via pulse push constants:** Set `pulse_x/y/radius/amount` on the target tile's SWE step dispatch.
5. **Handle tile identity changes:** When a disturbed tile's pool_index changes (camera moved), the water state is lost. Options: accept the reset, or copy state between slots.

### Key challenges encountered
- **Tile ordering instability:** Solved by sorting visible_tiles by (face, level, x, y) after selection.
- **SWE equilibrium ≠ static sea_level:** Due to RGBA16F precision (half-float) vs R32F terrain, clamped tile boundaries, and small numeric drift. The `max(static_depth, swe_depth)` approach in terrain.vs.hlsl handles this.
- **LOD transitions:** When tiles split/merge during zoom, water state for new tiles is initialized from sea_level. Water accumulated by the brush resets. This is acceptable for now.

### Files to modify
- `src/main.cpp` — Enable SWE selectively, add brush→tile mapping, track disturbed tiles
- `shaders/terrain.vs.hlsl` — Already has `max(static_depth, swe_depth)` + foam/normal logic

### Key line references
- SWE dispatch block: main.cpp ~line 2168 (`if (false && g_ui.ocean_enabled)`)
- Brush mode check: main.cpp ~line 1969 (`brush_hit`)
- Cursor ray-pick result: main.cpp ~line 1963 (`stamp_sphere_dir`)
- Water output sampling in VS: terrain.vs.hlsl ~line 95
- face_uv_to_cube mapping (for reverse): terrain.vs.hlsl line 47 and planet_gen.cs.hlsl line 91

# Handoff: Planet LOD Tile Disappearance Bug

## What This Engine Is
A Vulkan 1.3 terrain/sand simulation engine (`src/main.cpp`, ~5000 lines, monolithic). HLSL shaders compiled with glslc. macOS/MoltenVK on Apple M1 Pro.

## What Was Built This Session (in order)
1. **3D volumetric atmosphere** — 128x128x32 compute shader with wind, clouds, orographic lift
2. **Cloud raymarch rendering** — fullscreen volume raymarching with Beer-Lambert scattering
3. **Sand density field** — stored in wind texture .a channel, advected by wind
4. **GPU sand particle system** — 128K particles with saltation, velocity-streak rendering
5. **Sand brush (key 4)** — paints sand deposits on terrain, particles loft from deposits
6. **Flat infinite terrain (clipmap)** — geometry clipmap with GPU procedural heightmaps — **this worked well**
7. **Cube-sphere planet** — replaced clipmap with quadtree-per-face on a sphere — **this is where issues started**

## Current Architecture: Cube-Sphere Planet

### How it works
- 6 cube faces, each a quadtree. CPU-side tile selection per frame.
- Each tile = 64x64 vertex grid mesh (shared VBO/IBO)
- Tile pool: `Texture2DArray<float>` with 2048 layers of 64x64 R32_SFLOAT
- GPU compute shader (`planet_gen.cs.hlsl`) generates height per tile into pool slot
- Terrain VS (`terrain.vs.hlsl`) projects vertices onto sphere using cube-to-sphere tangent mapping
- Camera-relative rendering: double-precision camera position on CPU, float offsets to GPU
- Reversed-Z depth buffer (GREATER_OR_EQUAL, clear 0.0, infinite far plane)
- Priority queue tile selection (replaced earlier broken DFS)

### Key constants
```
PLANET_TILE_RES = 64       (grid vertices per tile)
PLANET_TILE_POOL = 2048    (max simultaneous tiles, Metal limit)
PLANET_MAX_LEVEL = 14      (finest LOD ~10m cells)
PLANET_RADIUS = 6371000.0  (Earth scale)
PLANET_MAX_ELEVATION = 9000
TILE_SUBDIVIDE_PX = 512    (subdivide when tile > this many screen pixels)
```

### Key files
- `src/main.cpp` — everything (structs, Vulkan setup, frame loop, tile selection)
- `shaders/planet_gen.cs.hlsl` — procedural height generation on sphere surface
- `shaders/terrain.vs.hlsl` — cube-sphere vertex projection with camera-relative coords
- `shaders/terrain.fs.hlsl` — elevation ramp + fog
- `shaders/cloud_raymarch.vs.hlsl` / `.fs.hlsl` — disabled for planet mode
- `shaders/sand_sim.cs.hlsl`, `sand_render.vs/fs.hlsl` — disabled for planet mode

### Simulation systems (still compile/run but rendering disabled)
SWE water, erosion, 3D atmosphere, sand particles — all tied to the original 10km flat domain. Their compute dispatches still run. Their rendering is wrapped in `if (false)` blocks. They'll be reintegrated on the sphere in Phase 2.

## THE BUG: Tiles Disappear at Ground Level

### Symptoms
- From orbit (100km+), the planet sphere is visible and colored
- As you fly closer, LOD levels increase (tile count goes up)  
- At some point near the surface, tiles stop rendering — screen goes black/empty
- The flat clipmap system (before sphere conversion) worked perfectly at all distances

### What's been tried
1. DFS tile selection → replaced with priority queue (fixed uneven budget distribution)
2. Added frustum culling (FOV-based cone test)
3. Added horizon culling
4. Increased pool from 256 → 512 → 1024 → 2048
5. Adjusted subdivision threshold from 192 → 512
6. Face sorting by distance
7. Various starting altitudes and pitch angles

### Likely root causes to investigate

**1. Camera-relative precision at fine LOD levels**
The VS computes: `displacement = (sphere_dir - center_dir) * planet_radius + sphere_dir * h`
At fine levels, `sphere_dir` and `center_dir` are nearly identical (tile is tiny on the sphere). Their difference `delta_dir` is very small. Multiplied by `planet_radius` (6.4M), this could lose precision. Need to verify the displacement values aren't degenerating.

**2. Tile generation compute might produce flat/zero heights at fine levels**
At fine LOD levels, all texels in a tile map to nearly the same sphere direction. The noise function evaluates at nearly the same 3D coordinate for all 64x64 texels, producing a uniform heightmap. This is correct behavior (terrain is locally flat at that scale), but the height VALUE should still be correct and non-zero.

**3. Pool index / layer mismatch**
Each visible tile gets `pool_index = i` (its position in the visible_tiles vector). The compute writes to `tile_pool[uint3(dtid.xy, pool_index)]` and the VS reads from `heightmap.SampleLevel(heightmap_sampler, float3(tile_uv, float(pool_index)), 0)`. If pool_index exceeds the array layer count, writes/reads silently fail.

**4. The priority queue might not produce enough fine-level tiles**
The priority queue always subdivides the largest-error tile first. But with 2048 pool slots and many visible tiles from orbit, by the time fine-level tiles are needed near the camera, the pool might already be full with medium-level tiles covering the whole visible hemisphere.

**5. Reversed-Z depth issues at close range**
At near-plane 0.5m, vertices closer than 0.5m clip. But terrain at ground level should be >10m away. Unless the camera is inside the terrain mesh somehow.

**6. GRID_MAX mismatch**
The VS has `GRID_MAX = 63.0`. The mesh was created with `PLANET_TILE_RES = 64` (vertices 0-63). `grid_pos / 63.0` gives UV [0, 1]. This should match the heightmap texel layout. Verify the mesh creation actually uses 64 vertices.

### How to debug
1. Add ImGui readout of: tile count, min/max level in visible set, tiles at each level
2. Reduce PLANET_MAX_LEVEL to 5 and verify rendering works at coarse LOD from close up (ugly but visible)
3. Set planet_gen to output `tile_pool[...] = 500.0` (constant height) — if visible, the issue is noise; if not, pipeline is broken
4. Check if `pool_index` ever exceeds `PLANET_TILE_POOL` by adding an assert
5. Print `visible_tiles.size()` and level distribution to stderr each frame

### Build & run
```bash
cd "Drift Game Engine/build"
cmake .. && cmake --build . -j$(sysctl -n hw.ncpu)
./drift_engine
```
Camera: WASD move, RMB+drag look, Q/E up/down, Shift fast, Alt slow. F5 hot-reloads all shaders. Backtick toggles ImGui.

### Reference document
`/Users/fields/Documents/driftEngine/GRAY_ENGINE_ANALYSIS.md` — analysis of GRay 2.0 engine with techniques applicable to this project (reversed-Z, double-precision camera, async compute, pipeline builder).

### What the user wants ultimately
- Earth-scale planet viewable from orbital to ground level (sand grain scale)
- Mixed biomes (desert, mountains, plains)
- Full simulation on the sphere: SWE water, erosion, 3D atmosphere, sand particles
- Sand brush painting + wind-driven sand transport
- This is a "drift engine" focused on sand/desert simulation at planetary scale

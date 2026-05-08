# Handoff: Camera Overhaul, Stamp Debugging, and Water Reintroduction

## Current State (v0.5.0+)

Cube-sphere planet rendering is working. Terrain editable via stamp system (may need debugging). Key systems:

### What Works
- 6-face cube-sphere with quadtree LOD, priority queue tile selection
- Dynamic max LOD level based on altitude (limits LOD range)
- Tile edge skirts hide cross-LOD gaps (10km drop)
- Heightmap texel alignment at tile boundaries (same-level seams fixed)
- Procedural terrain: continental base + biome noise (mountains, desert, plains, polar)
- CPU terrain height matching GPU (for camera clamping above terrain)
- Terrain stamp system: GPU SSBO of edit stamps applied in planet_gen compute shader
- CPU stamp application for camera clamping
- Ray-sphere intersection for cursor picking (just fixed Retina HiDPI coordinate scaling)
- Brush cursor ring rendered via sphere direction comparison in fragment shader
- ImGui: stamp count, undo, clear, angular scale slider

### What Needs Work

## Issue 1: Camera is Confusing on a Sphere

**The core problem**: The camera uses a fixed `(0, 1, 0)` world-up vector everywhere on the planet. This works at the north pole but is wrong at the equator — the horizon tilts, movement directions become unintuitive.

**What needs to change**:
1. **View matrix up vector**: should be `normalize(cam_pos)` (radial direction from planet center), not `(0,1,0)`
2. **Movement directions**: Q/E (up/down) should move radially. Forward/back should follow the sphere surface. Left/right should be tangent to the surface.
3. **Yaw/pitch**: pitch should rotate around the local east axis, yaw around the local radial axis. The current Euler angle approach (`g_camera.yaw`, `g_camera.pitch`) works at the pole but breaks elsewhere.

**Suggested approach**: Replace the Euler angle camera with a **quaternion camera** or a **local frame camera**:
```
struct Camera {
    double pos_x, pos_y, pos_z;      // world position (double)
    glm::dvec3 local_up;              // = normalize(pos), radial direction
    glm::dvec3 local_forward;         // tangent to sphere, initially toward equator
    float pitch_angle;                // rotation from tangent plane (looking up/down)
    float fov_y, near_plane;
};
```

On each frame:
- Recompute `local_up = normalize(pos)`
- `local_east = normalize(cross(some_ref, local_up))`
- `local_north = cross(local_up, local_east)`
- `forward = rotate(local_north, pitch_angle, local_east)` (after yaw rotation)
- `cam_view = lookAtRH(vec3(0), forward, vec3(local_up))`

The tricky part: maintaining a stable `local_forward` direction as the camera moves around the sphere. A quaternion approach avoids gimbal lock.

**Files**: `src/main.cpp` — Camera struct (line ~339), camera movement (line ~4530), view matrix (line ~4610)

## Issue 2: Stamp System May Not Be Working

The stamp system was just built. Ray picking was broken due to Retina HiDPI cursor coordinate scaling (fixed: using `glfwGetWindowSize` instead of framebuffer `extent`). The cursor ring and stamp placement need testing.

**If the cursor ring still doesn't appear**:
- Check that `g_cursor_on_world` is being set to true (add `ImGui::Text("Cursor on world: %d", g_cursor_on_world)` to debug)
- The ray-sphere intersection uses `PLANET_RADIUS + g_terrain_height_at_cam` as the sphere radius. If the terrain where you're pointing is much higher/lower than at the camera, the ray misses. Try using a range of radii or iterating.
- Check that `brush_world.w > 0` is reaching the fragment shader (the angular radius might be too small to see)

**If stamps place but terrain doesn't deform**:
- Verify `stamp_count` in push constants is non-zero
- Check that the stamp buffer upload (`memcpy` + `vmaFlushAllocation`) runs
- In planet_gen.cs.hlsl, the stamp loop uses `dot(sphere_dir, stamps[i].pos)` — verify the stamp positions are normalized unit vectors

**Stamp angular scale**: default 0.0001 = ~640m diameter. At 40km altitude this is tiny. Try 0.001-0.01 for visible stamps from high up.

## Issue 3: Remaining LOD Artifacts

- **Cross-LOD height steps**: tiles at different LOD levels still show height discontinuities. The skirts hide the gaps but not the visual steps. Proper fix: either restrict adjacent tiles to max 1-level difference (quadtree balancing) or implement geomorphing.
- **Float32 precision jitter at fine LOD**: `sphere_dir - center_dir` cancellation in VS at levels 12+. Fix: tangent-frame VS that computes vertex displacement from local (tile_uv - 0.5) offsets instead of global sphere direction subtraction.
- **Skirt corners**: 4-way tile intersections still have small gaps. Need corner skirt triangles.

## Issue 4: Save/Load

The stamp vector IS the save format. Implementation:
```cpp
void save_world(const char* path) {
    FILE* f = fopen(path, "wb");
    uint32_t seed = 42, count = g_stamps.size();
    fwrite(&seed, 4, 1, f);
    fwrite(&count, 4, 1, f);
    fwrite(g_stamps.data(), sizeof(TerrainStamp), count, f);
    fclose(f);
}
void load_world(const char* path) {
    FILE* f = fopen(path, "rb");
    uint32_t seed, count;
    fread(&seed, 4, 1, f);
    fread(&count, 4, 1, f);
    g_stamps.resize(count);
    fread(g_stamps.data(), sizeof(TerrainStamp), count, f);
    fclose(f);
    g_stamps_dirty = true;
}
```
Add ImGui buttons for save/load. Add keyboard shortcut (Ctrl+S/Ctrl+O).

## Next Phase: Water Reintroduction

After editable terrain + save/load work:
1. **Per-tile water depth**: add a second channel (RG32F or separate texture array) for water depth per tile
2. **SWE compute per tile**: adapt the existing SWE solver to run per-tile
3. **Rain + accumulation**: add precipitation, water pools in terrain depressions
4. **Tile boundary flow**: exchange water at tile edges between neighbors (hard problem)
5. **Water rendering**: transparent second pass per tile, reusing the water shader

## Key Files

| File | What |
|------|------|
| `src/main.cpp` | Everything — camera, tile selection, compute dispatch, rendering |
| `shaders/planet_gen.cs.hlsl` | Procedural terrain + stamp application |
| `shaders/terrain.vs.hlsl` | Cube-sphere vertex projection, skirts |
| `shaders/terrain.fs.hlsl` | Elevation ramp, lighting, brush cursor ring |
| `shaders/swe_step.cs.hlsl` | Existing SWE solver (flat domain, needs adaptation) |

## Build & Run
```bash
cd "Drift Game Engine/build"
cmake .. && cmake --build . -j$(sysctl -n hw.ncpu)
./drift_engine
```
Camera: WASD move, RMB+drag look, Q/E up/down, Shift fast, Alt slow. F5 hot-reloads shaders. Backtick toggles ImGui. Keys 1/2 = Raise/Lower mode, LMB = paint stamps, Z/X = brush size.

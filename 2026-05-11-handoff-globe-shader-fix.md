# Globe MoltenVK Shader Fix — Handoff

**Date:** 2026-05-11 (updated)
**Status:** Blocked — drift_engine crashes on launch due to MoltenVK descriptor binding collisions

## Problem

MoltenVK translates Vulkan descriptor bindings to Metal resource indices by type. When a shader mixes plain `Texture2D` bindings with `[[vk::combinedImageSampler]]` bindings, MoltenVK's SPIRV-Cross translation assigns overlapping Metal resource indices — the combined image sampler's texture component gets index 0, colliding with any plain `Texture2D` already at binding 0.

**Error:**
```
cannot reserve 'texture' resource location at index 0
kernel void main0(..., texture2d<float> terrain [[texture(0)]], texture2d<float> ground_cond [[texture(0)]], ...)
```

## What actually uses `[[vk::combinedImageSampler]]`

**Verified** — only these 5 graphics shaders:

| Shader | Combined Image Sampler bindings |
|--------|-------------------------------|
| `shaders/terrain.vs.hlsl` | bindings 1, 2 (heightmap, water_output) |
| `shaders/water.vs.hlsl` | bindings 1, 2 (heightmap, swe_output) |
| `shaders/water.fs.hlsl` | bindings 2, 3, 4 (swe_output, sediment_tex, atmo_render) |
| `shaders/cloud.fs.hlsl` | binding 4 (atmo_render) |
| `shaders/cloud_raymarch.fs.hlsl` | needs checking |

**Compute shaders are already fixed** — `swe_step.hlsl` and `erosion.hlsl` already use separate `Texture2D` + `SamplerState` at different binding numbers (4 and 5). The `planet_swe_step.cs.hlsl` doesn't use ground_cond at all (uses SSBO instead). `atmosphere3d.cs.hlsl` and `sand_sim.cs.hlsl` need checking.

## Pipeline mapping

The graphics shaders share a single descriptor set layout (`gfx_bindings[]` at pipeline.cpp:160-194) with 7 `COMBINED_IMAGE_SAMPLER` bindings (1-7) plus a push constant at binding 0. The collision occurs when MoltenVK maps these to Metal resource indices alongside any plain texture bindings in the same pipeline.

| Pipeline.cpp line | Layout | Shaders |
|---|---|---|
| 160-194 | `gfx_desc_layout` | terrain.vs + water.vs + water.fs + cloud.fs |
| 80-109 | `swe_step_desc_layout` | swe_step.hlsl (already separate bindings — OK) |
| 550-583 | `ero_desc_layout` | erosion.hlsl (already separate bindings — OK) |
| 647+ | `atmo_desc_layout` | atmosphere3d.cs.hlsl (needs checking) |

## Fix Pattern

For each affected **graphics** shader:

1. **Shader:** Replace `[[vk::combinedImageSampler]]` pairs with separate bindings:
   ```hlsl
   // Before:
   [[vk::combinedImageSampler]][[vk::binding(2, 0)]] Texture2D<float4> swe_output;
   [[vk::combinedImageSampler]][[vk::binding(2, 0)]] SamplerState swe_sampler;

   // After:
   [[vk::binding(2, 0)]] Texture2D<float4> swe_output;
   [[vk::binding(N, 0)]] SamplerState      swe_sampler;   // new binding number
   ```
   Assign new binding numbers for the split-out samplers. Keep texture bindings stable.

2. **Descriptor layout** (`src/pipeline.cpp` ~line 160): Split each `COMBINED_IMAGE_SAMPLER` entry into a `SAMPLED_IMAGE` + `SAMPLER` pair. Increase `bindingCount`.

3. **Descriptor writes** (`src/main.cpp`): Update `VkWriteDescriptorSet` arrays — write the image info to the texture binding and sampler to the sampler binding separately. Increase descriptor pool sizes to add `VK_DESCRIPTOR_TYPE_SAMPLER` entries.

4. **Descriptor pool** (`src/main.cpp`): Add pool size for `VK_DESCRIPTOR_TYPE_SAMPLER`. Bump `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE` count if needed.

## Key Files

- `src/pipeline.cpp` — descriptor set layouts and pipeline creation
- `src/pipeline.h` — `Pipelines` struct
- `src/main.cpp` — descriptor pool, set allocation, descriptor writes
- `shaders/terrain.vs.hlsl`, `water.vs.hlsl`, `water.fs.hlsl`, `cloud.fs.hlsl`, `cloud_raymarch.fs.hlsl` — the affected shaders

## Estimated Scope

- ~5 graphics shaders to fix
- ~1 graphics descriptor layout to expand (gfx_desc_layout)
- ~1 set of descriptor writes to update
- 1 descriptor pool size to add (SAMPLER type)
- Total: ~2 hours mechanical work, no design decisions

## What's NOT affected

- **World Lab** — uses its own shader copies compiled into separate pipelines in `world_lab.cpp`. These are already working on MoltenVK.
- **Compute shaders** (swe_step, erosion) — already use separate bindings (SAMPLED_IMAGE + SAMPLER), no combinedImageSampler.
- **Animals Lab**, **Plant Lab** — no combined image samplers.

## Verification

After fixing each shader:
1. `cmake --build build` — clean compile
2. `./build/drift_engine` — launches without Vulkan validation errors
3. Planet renders with terrain, water, atmosphere, erosion all functioning

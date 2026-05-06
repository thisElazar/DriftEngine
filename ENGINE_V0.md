# Engine v0 — Founding Design Doc

> Drafted 2026-05-05. This is the doc the project is missing on Day 0. It locks in the calls that determine what the next year of work feels like, and deliberately leaves everything else open.

---

## 1. What this engine is, in one paragraph

A simulation-first runtime where GPU compute is the primary citizen and the renderer is a consumer of compute textures. The first non-trivial program it runs is a port of Drift V2's shallow water solver. Drift V2 (UE 5.5) remains the reference implementation — every architectural call here gets measured against the question: "could this run Drift's water sim faster, cleaner, or simpler than Unreal does?"

This is **not** a general-purpose game engine. It has no editor, no asset pipeline, no scene graph, no actor system, no level streaming, no input mapping system, no audio, no physics. Those are deliberate non-goals for v0 and v1. They may be goals later. They are not goals now.

---

## 2. Language and graphics API

**Decision: C++20 + Vulkan 1.3 (via MoltenVK on macOS), HLSL shaders compiled to SPIR-V via DXC.**

The priorities you named are performance and accessibility. Walking through the alternatives:

| Stack | Performance | Accessibility | Familiarity cost |
|---|---|---|---|
| **C++20 + Vulkan** | Top tier, lowest overhead | Win / Linux / macOS (MoltenVK) / Android | Low — your daily language; Vulkan is verbose but well-documented |
| C++ + Metal | Top tier on Apple | Apple-only | Low |
| C++ + wgpu-native (Dawn) | ~5–15% overhead vs raw | Everything Vulkan covers + WebGPU | Medium — less mature ecosystem |
| Rust + wgpu | Same as wgpu-native | Same | High — Rust is a new language for you, borrow checker is a real headwind in graphics code |

For a *simulation* engine the per-frame cost is dominated by the compute shaders themselves, which run identically across APIs. The 5–15% wgpu overhead is in dispatch/barrier management, which is a small fraction of total frame time. So the performance gap is smaller than the chart suggests.

The reason I'm landing on Vulkan over wgpu anyway:

1. **HLSL ports cleanly.** You have a working HLSL pipeline in Drift. DXC → SPIR-V is the well-trodden path. wgpu's first-class shader language is WGSL; HLSL via Tint works but is less mature and you'd hit edge cases.
2. **Mac dev via MoltenVK is genuinely fine.** Same Vulkan code runs everywhere; on macOS it's translated to Metal. The translation layer is mature (used by Dolphin, Doom Eternal port, many others).
3. **Ecosystem maturity.** Vulkan tutorials, validation layers, RenderDoc, profiling tools, books — all of it is more complete than wgpu's.
4. **Apple-platform escape hatch.** If MoltenVK ever becomes a problem you can write a Metal backend later; the engine should be backend-pluggable from the start anyway.

The cost: you give up a "compile to web" target almost for free. If shipping a browser demo of the engine matters to you (the way Drift's WebDemo does), revisit this — wgpu gets you that, Vulkan does not.

**Verdict: go Vulkan unless web export is in your top three goals.**

---

## 3. v0.1 surface area

The proof-of-thesis slice. Nothing else gets built until this is running.

1. **Window** — open a window, show frames, handle close events. GLFW.
2. **Vulkan device** — instance, physical device, logical device, queues, swapchain. Use `vk-bootstrap` to skip 800 lines of boilerplate without losing control.
3. **Memory allocator** — Vulkan Memory Allocator (VMA). Non-negotiable; you don't write your own.
4. **Compute dispatch loop** — bind a compute pipeline, dispatch it, get the result back into a presentable image.
5. **Fullscreen blit** — present a 2D texture to the swapchain. Simplest possible graphics pipeline (full-screen triangle vertex shader + sample-and-output fragment shader).
6. **Heightmap loader** — read a single `.r16` or `.tif` file into a GPU texture. No DEM pyramid yet, no streaming, no tiles.
7. **One real compute shader** — visualize the heightmap with elevation-based shading. Proves the compute → present pipeline end-to-end.

That is the whole engine in v0.1. If it can do this in noticeably fewer lines than the equivalent Unreal scaffolding, the thesis is on track.

**Explicit non-goals for v0.1:** any tile system, any sim, any UI, any input handling beyond ESC-to-quit, any cross-platform testing, any optimization.

---

## 4. v0.2: the actual test

Port Drift V2's shallow water solver into the engine.

This is the first thing that proves the engine isn't just a Vulkan tutorial. It's also the first thing that *might* be measurably better than Unreal's version. What we're looking for, in order:

1. **Correctness.** Same physical behavior — flow conservation, settling, depression filling.
2. **Frame time.** Compare against Drift V2's GPU SWE perf at matching grid size.
3. **Code volume.** How many lines does the engine version take vs. Drift's? If it's not dramatically shorter, the simplification thesis is wrong.
4. **CPU/GPU coupling.** Drift currently does CPU/GPU sponge-zone coupling for cross-tile flow. v0.2 stays single-tile, so this is deferred. Note it as a v0.3 question.

If v0.2 is faster *and* leaner than Drift V2's solver, the engine is justified. If it's neither, we learn that and decide whether to push further or fold the work back into Drift.

---

## 5. Architecture principles (kept short)

- **Backend-pluggable from day one.** Even if Vulkan is the only backend in v0, the GPU abstraction lives behind an interface (`IRenderBackend`) so a Metal or wgpu backend is a 2-week port, not a rewrite.
- **No singletons.** State is owned, passed explicitly. The MasterController pain from TerrAI Legacy stays in TerrAI Legacy.
- **Compute is the protagonist.** Render passes exist to *show* what compute produced, not the other way around. The frame graph (when one exists) is built around compute first.
- **No reflection, no scripting, no GC.** This is a C++ codebase. If you want to script it later, embed Lua. Don't build your own.
- **Hot-reloadable shaders.** Watch the shader directory, recompile on change, swap pipelines mid-frame. Cheap to build early, painful to retrofit later.
- **Fail fast.** Validation layers always on in debug. Every Vulkan return code checked. Asserts are load-bearing, not decorative.

---

## 6. Project layout (proposed)

```
DriftEngine/
├── CMakeLists.txt
├── README.md
├── ENGINE_V0.md                  ← this doc
├── src/
│   ├── core/                     ← logging, allocators, math
│   ├── window/                   ← GLFW wrapper
│   ├── gpu/                      ← Vulkan abstraction (IRenderBackend, devices, queues)
│   ├── compute/                  ← compute pipeline + dispatch helpers
│   ├── render/                   ← swapchain, fullscreen blit, present
│   ├── io/                       ← heightmap loader, file I/O
│   └── main.cpp
├── shaders/
│   ├── compute/
│   │   └── visualize_heightmap.hlsl
│   └── present/
│       └── fullscreen.hlsl
├── third_party/                  ← vk-bootstrap, VMA, GLFW (vendored or via CMake fetch)
└── data/
    └── heightmaps/               ← test data (one DEM tile from Drift's pyramid)
```

---

## 7. Open questions (decide before v0.1)

These don't block writing this doc but block writing code. Flagging so we hit them deliberately, not by accident.

1. **Build system: CMake or something newer (xmake, Meson)?** CMake is the boring right answer. Pick boring on day one.
2. **Coordinate convention: Y-up or Z-up? Right-handed or left-handed?** Vulkan is right-handed Y-down NDC; world coordinates are your call. Pick once, document, move on.
3. **Floating-point precision policy.** Drift hit float32 precision walls at planetary scale. The engine should plan for this from the start — relative-to-camera rendering, doubles where it matters. Worth a separate note in v0.2.
4. **Logging.** Simple printf macros for v0; revisit when it hurts.
5. **Testing.** Catch2 or doctest, set up early. Compute shaders especially benefit from unit tests on small inputs.

---

## 8. What "session 2" looks like

Once this doc is approved:

1. `git init` in this folder. `.gitignore` for build/, third_party/build/, .DS_Store.
2. Create the directory skeleton above with empty files.
3. Write `CMakeLists.txt` that fetches GLFW, vk-bootstrap, and VMA via CPM or FetchContent.
4. Verify it builds an empty `main.cpp` on macOS with the Vulkan SDK installed.
5. Stop. That's the whole session. No graphics yet.

Sessions 3–6 each tackle one item from §3 (window → device → memory → compute → blit → heightmap), in order.

---

## 9. What I'd push back on later

These are reasonable thoughts that I'd argue against if they come up:

- "Let's add a scene graph before the sim port" — no. v0.2 doesn't need one.
- "Let's support OpenGL too" — no. Vulkan-first, Metal-later, nothing else.
- "Let's write an editor" — not before v0.5.
- "Let's switch to Rust for safety" — talk to me again after v0.3 if you still want to. Switching languages mid-engine is a project killer.
- "Let's build a frame graph system" — only when you have ≥3 distinct render passes that need ordering. Premature.

---

*This doc is a contract with the next version of you. When in doubt, re-read §1.*

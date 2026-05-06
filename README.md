# Drift Engine

A simulation-first runtime where GPU compute is the primary citizen.

See [`ENGINE_V0.md`](ENGINE_V0.md) for the founding design doc — what this is, what it isn't, and why.

## Status

v0.0 — toolchain bring-up. The build target prints physical device info and exits. No graphics yet.

## Build (macOS)

Prerequisites:
- Xcode command line tools (`xcode-select --install`)
- CMake ≥ 3.24 (`brew install cmake`)
- Vulkan SDK ≥ 1.3 — download from <https://vulkan.lunarg.com/sdk/home#mac>, run the installer, then `source ~/VulkanSDK/<version>/setup-env.sh` (or add to your shell rc)

Build:
```bash
cmake -S . -B build
cmake --build build
./build/drift_engine
```

Expected output: a list of physical devices Vulkan can see (your GPU, plus any others).

## Layout

```
src/        engine code (currently just main.cpp)
shaders/    HLSL → SPIR-V via DXC (none yet)
data/       test inputs (heightmaps etc.)
```

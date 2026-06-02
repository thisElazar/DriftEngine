#pragma once

#include <cstdint>

// Brush tool selection, shared by globe + UI. (Stage 2 of the input
// unification removed InputState/CallbackContext and the GLFW input callbacks;
// all per-frame input now flows through InputFrame — see input_frame.h /
// input_poll.h / docs/INPUT_UNIFICATION.md. This enum is all that remains.)
enum class BrushMode : uint32_t { Raise, Lower, Water, Sand };

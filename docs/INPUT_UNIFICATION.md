# Stage 2 — Unified Input Architecture (design)

Status: **designed, not yet implemented.** Capture written 2026-05-31 because
the editing session's read I/O degraded; execute this after a fresh start when
reads are reliable. Edits/Writes were reliable; only Bash/Read output corrupted.

## Problem

The launcher hosts every lab through one `Lab` vtable (init/tick/render/shutdown
+ `embedded`), but input is handled two incompatible ways:

- **Globe** — callback-based. `input_install_callbacks()` (src/input.cpp) registers
  GLFW `key/mouse_button/cursor_pos/scroll` callbacks that write into `InputState`
  and *directly* mutate the camera (`camera_apply_mouse_look`, `camera_zoom`). It
  also drives cursor capture via `glfwSetInputMode`. On shutdown it nulls the
  callbacks and reinstalls only ImGui's — which is why scroll-zoom leaked (fixed
  in Stage 0 by `install_launcher_input`, but that's a patch over the split).
- **World / Plant / Animals** — polling. `update_orbit()` (apps/shared/lab_common.h)
  reads the global `g_scroll_accum` (set by the launcher's `lab_scroll_cb`) and
  polls `glfwGetMouseButton`/`glfwGetKey` directly inside each `*_tick`.

Consequences: ownership of GLFW callbacks ping-pongs between launcher and Globe;
two zoom mechanisms; ESC/back logic lives in three places; adding a lab means
picking a paradigm. This is the last thing keeping Globe "special."

## Target

One input source. The launcher polls once per frame into an immutable
`InputFrame` and passes it to each `*_tick`. No lab touches GLFW directly; cursor
capture becomes a *request* the launcher honors.

```cpp
// new: src/input_frame.h  (in drift_engine_core)
struct InputFrame {
    // pointer (framebuffer pixels)
    double mouse_x = 0, mouse_y = 0;
    float  mouse_dx = 0, mouse_dy = 0;   // delta since last frame
    float  scroll = 0;                    // accumulated wheel this frame
    // buttons: held + edge
    bool lmb = false, rmb = false, mmb = false;
    bool lmb_pressed = false, rmb_pressed = false;   // went down this frame
    // keys the labs actually use (extend as needed)
    bool key_shift = false, key_r = false, key_space = false,
         key_f = false, key_c = false, key_esc_pressed = false;
    // ImGui capture flags (labs must respect these for world interaction)
    bool ui_wants_mouse = false, ui_wants_keyboard = false;
    // framebuffer size (saves every lab re-querying)
    int fb_w = 0, fb_h = 0;
};

// labs gain a cursor-capture *request* in their tick return or state:
//   s.want_cursor_capture = true;   // launcher applies glfwSetInputMode
```

### New tick signature

```cpp
bool xxx_tick(XxxState& s, Renderer& r, const InputFrame& in, float dt);
```

The launcher builds one `InputFrame` per frame (mouse delta from last pos, scroll
from the existing `g_scroll_accum`, buttons/keys via `glfwGet*`, ImGui flags from
`ImGui::GetIO()`), then calls `labs[idx].tick(in, dt)`.

## Migration plan (smallest-risk order)

1. **Add `InputFrame` + a `poll_input(Renderer&, InputFrame& prev)` helper** in
   drift_engine_core. Launcher fills it each frame. No behavior change yet.
2. **Thread `const InputFrame&` through the `Lab::tick` lambda** (vtable already
   isolates this — only the 4 lambda bodies + 4 `*_tick` signatures change).
3. **Port World/Plant/Animals**: replace `update_orbit(cam, window)` with
   `update_orbit(cam, in)` (overload reading `in.scroll`/`in.mouse_dx`), and
   replace their `glfwGetMouseButton/Key` polls with `in.*`. Mechanical.
4. **Port Globe**: delete `input_install_callbacks`/`cb_ctx` usage; rebuild
   `InputState`-equivalent reads from `InputFrame`. Camera look: feed `in.mouse_dx/dy`
   to `camera_apply_mouse_look` when `in.rmb && !in.ui_wants_mouse`. Zoom: `in.scroll`.
   Cursor capture: set `s.want_cursor_capture` instead of calling `glfwSetInputMode`.
   ESC: `in.key_esc_pressed` → return false (drop the wants_back/key_callback path).
5. **Delete the now-dead code**: `input_install_callbacks`, the GLFW callbacks in
   src/input.cpp, `CallbackContext`, `lab_scroll_cb`/`g_scroll_accum` global,
   `install_launcher_input` (Stage 0 patch — no longer needed once nobody steals
   callbacks). The launcher installs ImGui's GLFW callbacks once at startup and
   never touches them again.
6. **Cursor capture in launcher**: after tick, if `state.want_cursor_capture`
   differs from current, call `glfwSetInputMode(window, GLFW_CURSOR, ...)`.

## Verification (needs a human — no screenshots)

After each of steps 3 and 4, run the launcher and confirm in the ported lab:
orbit/look drag, scroll-zoom, LMB brush/paint, ESC→menu, and (Globe) RMB cursor
capture + fly-to-cursor (C). Then confirm scroll-zoom survives a Globe round-trip
*without* `install_launcher_input` once step 5 removes it.

## Files touched

- new `src/input_frame.h`, `src/input_frame.cpp` (poll helper) — in core lib
- `src/input.h` / `src/input.cpp` — gut callbacks (keep `InputState` only if still
  referenced; likely deletable)
- `apps/shared/lab_common.h` — `update_orbit` overload on `InputFrame`; drop
  `g_scroll_accum`/`lab_scroll_cb`
- `apps/{world_lab,plant_lab,animals_lab,globe}/*.{h,cpp}` — tick signature + body
- `apps/launcher/main.cpp` — build `InputFrame`, pass to tick, honor cursor request,
  delete `install_launcher_input`
- `src/main.cpp` (standalone) — build a minimal `InputFrame` too (or retire; see
  Stage 3)

---

## Increment B — Globe off callbacks (VERIFIED SCOPE, not yet done)

Increment A (InputFrame threaded through every tick; orbit labs fully ported)
landed in commit 36cd189. Globe takes `const InputFrame&` but still `(void)in`s
it and uses its GLFW callbacks. Removing those is bigger than a globe.cpp edit:

**The camera module is the real coupling.** `camera_update(Camera&, GLFWwindow*,
dt, ...)` (src/camera.cpp:84) polls movement *itself*:
`glfwGetKey(window, W/A/S/D/Q/E/LEFT_SHIFT/LEFT_CONTROL)` at lines 99-105. Mouse-
look enters via `camera_apply_mouse_look(cam, dx, dy)` (camera.cpp:53) called from
globe's `cursor_pos_callback` on RMB-drag; zoom via `camera_zoom()` from
`scroll_callback`. So input reaches the camera through callbacks, not InputState.

**globe.cpp s.input touchpoints (ground-truth lines, HEAD 36cd189):**
- 1958/1960 reload_shaders (F5)
- 2263/2268/2269 toggle_camera_mode (F) + reset rmb_held
- 2290-2296 cursor-visibility state machine (rmb_held → GLFW_CURSOR_DISABLED;
  first_mouse reset) via glfwSetInputMode
- 2316 cursor_x/cursor_y → ray-pick NDC
- 2421/2431 warp_to_cursor (C)
- 2437 lmb_held && !rmb_held → brush gate
- 3151/3156 pulse_pending (Space)
- brush_mode (1/2/3/4) used throughout stamp/water/sand dispatch
- 3580 shutdown restores GLFW_CURSOR_NORMAL

**Required changes (do with reliable file I/O + interactive testing):**
1. Extend InputFrame with movement (w/a/s/d/q/e, ctrl) and *edge-triggered*
   keys (f_pressed, c_pressed, space_pressed, f5_pressed) — the callbacks fire on
   GLFW_PRESS edges, so held-state booleans would retrigger every frame.
2. Change `camera_update` to take movement + look-delta from InputFrame instead
   of `GLFWwindow*` (core-lib change; standalone src/main.cpp must build the
   frame too, or keep a window-polling overload for standalone).
3. Replace globe's 15 s.input.* sites with in.*; feed in.mouse_dx/dy to
   camera_apply_mouse_look when in.rmb && !in.ui_wants_mouse; in.scroll to zoom.
4. Cursor capture: globe sets s.want_cursor_capture (InputFrame out-field already
   exists); launcher applies glfwSetInputMode after tick. Remove globe's direct
   glfwSetInputMode calls + the init input_install_callbacks/cb_ctx.
5. Delete dead code: input_install_callbacks + GLFW callbacks (src/input.cpp),
   CallbackContext (src/input.h), g_scroll_accum/lab_scroll_cb (lab_common.h),
   install_launcher_input (launcher) — once nobody steals callbacks.

**Verification (human, no screenshots):** orbital RMB-orbit + scroll-zoom; F to
first-person; WASD+QE walk; RMB mouse-look with cursor captured; C fly-to-cursor;
1/2/3/4 brush modes + LMB paint; Space pulse; ESC→menu; scroll-zoom survives a
Globe round-trip with install_launcher_input removed.

Status note: paused before starting — session file-read I/O was fabricating
buffer content, unsafe for a precise shared-camera rewrite. Resume with clean I/O.

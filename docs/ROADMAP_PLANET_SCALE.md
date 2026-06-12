# Roadmap: from tile sandbox to planet scale

Written 2026-06-12, after PLANET (DEV) proved streaming flat (branch
`globe-water-field`, `5b92efb`). The question this answers: **how do we get
from here to No Man's Sky in terms of scale, beauty, and loading times —
without giving up the living world?**

## The framing

NMS achieves scale / beauty / zero loading from one decision: **the universe
is a pure function, not data.** Terrain, plants, creatures, palettes — all
deterministically derived from coordinates + seed. Nothing is stored except
player edits (sparse deltas re-applied after regeneration). Storage never
grows with world size; "loading" is "compute the function where the camera
is, prioritized and budgeted so the frame never waits."

The flip side: NMS is *shallow*. Static water shell, no erosion, ambient
creatures. The world never changes — that is exactly WHY it can be a pure
function.

Drift's bet is the opposite: real water, real moisture, a real ecosystem.
So the goal is NOT "become NMS." It is:

> **Adopt NMS's scale architecture for everything outside the simulation
> frontier; keep the living world as the inner ring.**

The far-field / sim-pool split PLANET (DEV) just built is this idea at toy
scale. NMS is "all far field, beautifully rendered." Drift is "NMS's far
field, plus an inner ring that is actually alive."

## What already exists

- **Sphere-scale geometry**: the Globe's cube-sphere quadtree LOD stream
  renders a planet orbit-to-ground. The hard skeleton exists.
- **Procedural species**: the bestiary IS the NMS creature/plant
  architecture — archetypes + parameter morphing from TOML profiles,
  expression-evaluated by environment. Species Lab is the editor.
- **Streaming discipline** (PLANET (DEV), flat): slot pool + free list +
  cooldown, edge-flag auto-anchor, hitch-free activation (frame-cmd staged
  uploads), async generation jobs, far-field backdrop so terrain never
  visually disappears.
- **Real simulation**: SWE water / moisture / ecosystem — the thing NMS
  doesn't have.

## The gaps, in order of leverage

1. **Determinism + delta store** (scale unlock AND the tile→globe
   connector). PLANET (DEV) persists full state in flat world arrays —
   fine for 64 tiles, impossible for millions. The move: tiles are
   REGENERATED from seed on activation; only edits persist, as sparse
   per-tile deltas keyed by tile ID (flat key `(tx,ty)`, sphere key
   `QuadNode`). "Regenerate + apply delta" replaces "save + restore
   everything."
2. **Terrain variety** — scale of interest, not just kilometers. A layered
   noise stack whose biome parameters vary across the sphere
   (`terrain_gen.cs.hlsl` is the seed of this). Curation effort goes into
   parameter RANGES that keep outputs in the interesting band.
   Heightfields only — skip voxel caves (a year of work, not needed yet).
3. **Atmosphere, lighting, palettes** — the beauty multiplier, and cheap.
   NMS's look is ~80% sky scattering + fog/aerial perspective + curated
   per-biome palettes + HDR tonemapping, not asset quality. We render
   flat-lit LDR with a constant sky. This track is nearly decoupled from
   streaming work.
4. **Distant life = impostors.** Real vegetation meshes don't belong on
   far tiles (the veg-experiments lesson). Billboards/low-poly impostors
   to the horizon; real instanced geometry near.
5. **The marriage**: PLANET (DEV)'s sim pool mounted on the Globe's LOD
   stream as the innermost ring. Water crossing LOD boundaries (the SWE
   shader's deferred "Phase 2/3") is mostly SIDESTEPPED by the delta-store
   model: far water is frozen state rendered cheaply, like far terrain.
   Simulation only ever runs at one resolution, near the player.

## Milestones

- **M1 — "one beautiful planet, orbit to ground."** Globe + procedural
  biome terrain + atmosphere/sky/palettes + vegetation impostors. No
  simulation. The pure NMS demo; mostly rendering + generation work on
  infrastructure that exists.
  - M1a: biome-varied procedural terrain on the Globe (noise stack,
    seed-driven biome parameters).
  - M1b: sky scattering + aerial perspective + HDR tonemap.
  - M1c: per-biome palettes sampled by terrain + species shaders.
  - M1d: vegetation impostors out to the horizon.
- **M2 — "the living ring."** Sim pool (water/plants/creatures) mounted on
  the Globe near the player; delta store replaces flat persistence. The
  world-speed knob (speed up / slow down the changing world) lives here —
  bounded cost by construction, since only the inner ring simulates.
- **M3 — "many planets."** Seeds + system view. Nearly free once M1/M2
  hold: a planet costs nothing until visited.

## What NOT to copy from NMS

- Voxel terrain / caves (cost >> value at this stage).
- Galaxy scale before one planet is compelling.
- Their shallow simulation — the living ring is the differentiator.

## Status

- [x] M1a biome terrain (`a3a3a79`, 2026-06-12) — shared climate fn
      (planet_climate.hlsli) drives both generator character and surface
      palette; needs a visual pass from low orbit / ground
- [ ] M1a follow-ups: mesa/badlands terracing, palette tuning per seed
- [ ] M1b atmosphere/tonemap
- [ ] M1c palettes
- [ ] M1d impostors
- [ ] M2 delta store + sim ring on Globe
- [ ] M3 seeds/system view

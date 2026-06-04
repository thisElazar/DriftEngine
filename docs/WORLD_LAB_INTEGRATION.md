# World Lab — full pipeline integration (design → species files → living world)

Status: **in progress.** Started 2026-06-02. Goal: make World Lab the fully
realized game sandbox — every organism designed in the labs, saved as a species
file, automatically eligible to live in the world. Globe is the eventual canvas
but is deferred (needs rendering/scale work first); the `bestiary` stack is
engine-agnostic so this work ports to it later.

## Chosen model: "whole library is the palette"

World Lab's roster = **every species file in `species/`**. Designing + saving any
organism in a lab makes it eligible to appear in the world. No curation file, no
per-session selection — the library *is* the world's cast. (Decided 2026-06-02.)

## Current state (verified 2026-06-02)

The design→save→load round trip already works with full param fidelity for the
5 plant kinds (grass/bush/tree/reed/wildflower) and all 6 creature archetypes.
What's missing is the *connective tissue*:

- **Launcher → World Lab is a dead end.** Species double-click routes only to
  Plant/Animals Lab (`launcher/main.cpp:363–375`); World Lab is a fixed cast you
  can't feed.
- **World Lab loads by fixed filename, not the library.** It reads exactly
  `world_grass.toml`/`world_bush.toml`/`world_tree.toml`/`world_reed.toml`/
  `world_wildflower.toml` and 8 named creatures (`load_plant_profiles` ~943,
  `load_creature_profiles` ~893 in `world_lab.cpp`). Arbitrary library names
  don't flow in.
- **L-Plant is orphaned.** Fully designed/serialized in Plant Lab, never loaded
  or placed in World Lab.
- **No multiple-species-per-kind.** `distribution.{h,cpp}` is hardwired to
  `PLANT_KIND_COUNT = 5`, one species per kind; `PlantInstance.kind` is a 0–4
  enum; `EcosystemParams`/`PlantMeshParams` carry one entry per kind.

## The architecture change — a PlantSpecies roster (decided 2026-06-02)

The per-kind pattern is the root pain: `[PLANT_KIND_COUNT]` arrays + parallel
`switch (kind)` blocks in ~8 sites (`density_at`, `select_species`,
`tick_plant_population`, `sprout_plants`, `generate_mesh_from_population`,
`build_canonical_meshes`, `upload_plant_instances`, render). Adding L-Plant as a
6th hardcoded kind would touch all 8 and then be deleted by the N-species work —
throwaway. So we abstract directly; L-Plant becomes just another roster entry.

```cpp
enum class PlantKind { Grass, Bush, Tree, Reed, Wildflower, LPlant }; // semantic tag

struct PlantSpecies {
    std::string        name;
    PlantKind          kind;   // for trophic links + growth semantics ONLY, not dispatch
    SpeciesSuitability suit;   // moisture/temp tolerance + base_density
    // type-erased: closes over the loaded params+expression
    std::function<VegetationMesh(float moisture, uint32_t seed, float x, float z)> gen;
};
```

- Roster = `std::vector<PlantSpecies>`, built by scanning `species/`.
- `PlantInstance` gains a `species` index (keep `kind` for trophic/growth).
- `place_ecosystem` / `tick_plant_population` / `sprout_plants` /
  `collect_plant_instances` / `density_at` / `select_species` take the roster;
  every `switch (kind)` collapses to a roster lookup / `sp.gen(...)`. Adding a new
  plant type = register one generator, zero switch edits.
- World Lab's `plant_canonical[PLANT_KIND_COUNT]` / `plant_inst[...]` become
  `std::vector` sized to the roster — one canonical mesh + instance buffer per
  species. Multiple trees, multiple grasses, L-Plants: all just entries.
- These (`place_ecosystem`/`tick`/`sprout`/`collect`) are World-Lab-only callers,
  so their signatures change freely. The **baked-mesh preview path**
  (`generate_ecosystem`/`generate_mesh_from_population`, used only by Plant Lab's
  noise preview) stays on the old 5-kind switch — separate concern, left alone.

Creatures are closer — `load_creature_dir` already returns every creature file —
but spawning is a fixed 8-id cast that must become a dynamic roster.

**Two facts that keep creature code untouched in Increment 1:**
- Caloric value is chosen by an `if/else if` on plant kind
  (`creature_profile.cpp:508–513`), not an array index — a 6th kind (LPlant) is
  safe; it falls through to grass-caloric until an lplant value is added.
- Seed dispersal makes a `PlantInstance` by *kind* with no species index
  (`creature_profile.cpp:790–799`). So `PlantInstance.species` defaults to `-1`,
  and World Lab resolves any `species < 0` to the first roster entry of that kind
  each frame (drop if none). No change to `creature_profile.cpp`.

## Staged increments (smallest-risk order)

1. **PlantSpecies roster abstraction** — DONE (builds green 2026-06-02; runtime
   verification by the user pending, no screenshots). `PlantSpecies` +
   `std::vector<PlantSpecies>` roster (type-erased `gen` closure) landed in
   `distribution.{h,cpp}`; `place_ecosystem`/`tick_plant_population`/
   `sprout_plants`/`collect_plant_instances` are roster-driven (internal
   `density_at_roster`/`select_species_roster`; `poisson_disk` now takes a
   density callback shared with the baked path). `PlantInstance` gained a
   `species` index; `resolve_plant_species` maps creature-dispersed seeds
   (`species < 0`) to a roster entry by kind. World Lab: `plant_canonical`/
   `plant_inst`/`plant_inst_count` are now `std::vector` sized to the roster;
   `build_plant_roster()` registers the six built-in "house" species
   (grass/bush/tree/reed/wildflower/**lplant**) plus every *other* plant `.toml`
   in `species/` — so the whole library is the palette. L-Plant uses static bulk
   instancing for now (incremental growth is a later enhancement — see note).
   Plant Lab's baked preview path was left untouched.
3. **Creature roster.** Spawn every creature file found as its own species_id
   with a per-profile (or default) population count; retire the hardcoded 8-cast
   and fixed-name matching.
4. **Launcher wiring.** "Open World Lab" loads the whole current library; add a
   refresh path so species saved during the session appear (rescan on enter, and
   a manual "Reload library" button). Single-species routing is unnecessary under
   the palette model.
5. **World persistence (later).** Save/load a composed world (seed + roster
   snapshot + climate), making a "world" a first-class artifact.

## Deferred / open
- **Globe as the canvas** — after its rendering/scale work. Keep all of the above
  flat-tile-agnostic where cheap (it mostly is — the bestiary stack only needs an
  `EnvironmentField` + height fn).
- **L-Plant real-time growth vs static instancing** — [[project_lplant_vs_static]]
  says L-Plants are meant for incremental growth, bush/tree for static bulk. Stage
  1 does static placement to get them visible; growing them individually in the
  world is a follow-up, not a one-way door.

## Key files
- `bestiary/distribution.{h,cpp}` — the N-species generalization (core).
- `bestiary/environment.h` — `SpeciesSuitability` per roster entry.
- `bestiary/morphology/lplant.*` — already complete; wire its generator in.
- `apps/world_lab/world_lab.{h,cpp}` — roster scan/load, placement, per-species
  render batches, creature roster, ImGui.
- `apps/launcher/main.cpp` — enter World Lab loading the library; reload button.
- `bestiary/species_file.cpp` — already covers all plant + creature params.

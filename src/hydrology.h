#pragma once

// Planet hydrology — a coarse, global, deterministic drainage field.
//
// Sampled over a 6×R×R cube-face grid from the deterministic terrain height
// function, so it is a pure function of (seed, stamps) and never needs saving —
// the same rivers reappear at every zoom. One field yields two outputs: visible
// rivers (river_strength) and a watershed moisture field (moisture).
//
// This is the GLOBAL coarse layer. Near-camera fine refinement is a later phase.

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <atomic>

struct TerrainStamp;  // fwd (src/pipeline.h)

struct PlanetHydrology {
    uint32_t res = 0;  // R: cells per cube-face edge

    // Per-cell payload, 6 faces laid out contiguously.
    // Index: face*res*res + j*res + i.
    //   .r = river_strength (0..1, log-normalized accumulation; >threshold = river)
    //   .g = moisture       (0..1, watershed wetness for the ecosystem)
    //   .b = flow_dir east  (local tangent frame)
    //   .a = flow_dir north (local tangent frame)
    std::vector<glm::vec4> cells;

    bool valid() const {
        return res > 0 && cells.size() == static_cast<size_t>(6) * res * res;
    }
    size_t index(uint32_t face, uint32_t i, uint32_t j) const {
        return static_cast<size_t>(face) * res * res + static_cast<size_t>(j) * res + i;
    }
};

// Build the coarse global hydrology field. `stamps`/`stamp_count` are the terrain
// edit stamps (may be null/0). Runs on the CPU; safe to call off the main thread
// as long as `stamps` points at a stable snapshot for the call's duration.
//
PlanetHydrology build_planet_hydrology(uint32_t res, float sea_level,
                                       const TerrainStamp* stamps, uint32_t stamp_count);

// ---------------------------------------------------------------------------
// CPU sampling of a baked hydrology field (the worker's published RGBA payload).
// This is the bridge the ecosystem reads: given a sphere direction, return the
// watershed values at that point. No GPU readback — the field is already CPU-side.
// ---------------------------------------------------------------------------
struct HydroSample {
    float     river_strength = 0.0f;  // 0..1 (>threshold = river channel)
    float     moisture       = 1.0f;  // 0..1 watershed wetness (ocean defaults wet)
    glm::vec2 flow{0.0f};             // downstream direction, local east/north (unit)
    float     lake_surface   = 0.0f;  // live water-surface ELEVATION (m); standing
                                      // water exists where terrain sits below it
};

// Sample the field (6*res*res RGBA cells, as produced by bake/build) at a unit
// sphere direction. Returns defaults if the field is empty/mismatched.
HydroSample sample_planet_field(const std::vector<glm::vec4>& cells, uint32_t res,
                                glm::vec3 sphere_dir);

// Surface temperature proxy in [0,1]: warm at the equator, cold at the poles,
// with an altitude lapse. Pairs with moisture to form a bestiary EnvironmentSample.
float planet_temperature(glm::vec3 sphere_dir, float height_m);

// ---------------------------------------------------------------------------
// LiveHydrology — the field as a live, evolving ground-truth water sim.
//
// Holds the static drainage *structure* (terrain, filled surface, downstream
// pointers, lakes, flow directions) plus a live water volume that is routed
// downstream every step. Rivers are derived from the moving water, so they flow
// continuously and — crucially — when the terrain is edited and the structure is
// rebuilt, the existing water RE-ROUTES along the new paths over a few seconds
// instead of snapping to a new static solution.
//
// rebuild_structure() is the expensive part (terrain sampling + Priority-Flood +
// D8); step()/bake() are cheap. Intended to be driven by a persistent worker
// thread (see HydroAsync in globe.h).
// ---------------------------------------------------------------------------
struct LiveHydrology {
    uint32_t res = 0;
    float    sea_level = 800.0f;

    std::vector<glm::vec3> dir;         // cell directions (cached, computed once)
    std::vector<float> base_height;     // noise-only height (cached; the expensive part)
    uint32_t           applied_stamps = 0; // # stamps already folded into `height`
    std::vector<float> height;      // base_height + applied terrain stamps
    std::vector<float> filled;      // Priority-Flood surface
    std::vector<int>   downstream;  // D8 downstream cell index (-1 = ocean/sink)
    std::vector<float> lake01;      // normalized lake depth (static per structure)
    std::vector<float> flow_angle;  // downstream direction, local frame, [0,1] (static)
    std::vector<float> moisture;    // watershed wetness (static per structure)
    std::vector<float> cap_w;       // depression storage capacity in water units
                                    // (filled-height in metres → water units): water up
                                    // to this stays as a standing lake, the rest spills.
    std::vector<float> water;       // LIVE water volume (≤cap_w = lake, excess = river flow)

    // --- Climate / ocean circulation (static current field + advected SST) ---
    std::vector<glm::vec2> current;   // surface current, local tangent frame (east, north)
    std::vector<float>     sst_base;  // solar baseline temperature (static, per structure)
    std::vector<float>     sst;       // LIVE sea-surface temperature (advected by current)
    std::vector<int>       nE, nW, nN, nS;  // cached cardinal neighbour indices (-1 = none)

    // Recompute the drainage structure for the given terrain stamps. Preserves
    // the current `water` (so it re-routes); seeds water to steady state the
    // first time. Expensive — call off the main thread. If `cancel` is set true
    // mid-rebuild, returns early (structure left partial) for a fast shutdown.
    void rebuild_structure(const TerrainStamp* stamps, uint32_t stamp_count,
                           const std::atomic<bool>* cancel = nullptr);
    // Inject a one-shot slug of water mass at a sphere direction, gaussian-falloff
    // within the angular radius (cos_radius = cos(angular_radius)). Land cells only
    // (ocean is a sink). Subsequent step()s route it downstream / drain it to sea —
    // i.e. a coarse flood pulse. Cheap: one O(N) scan of the field.
    void add_water_deposit(glm::vec3 dir, float cos_radius, float amount);
    // Bake-back from the fine SWE sim: set one cell's water from an observed
    // water-surface elevation (m). Depression cells are SET to the matching fill
    // (the lake freezes at the level the live sim left it); slope cells (no
    // storage) receive mean_depth as an ADDED slug that routes downstream on
    // subsequent steps. Ocean cells are ignored (the SWE ocean is not field water).
    void apply_surface_set(int cell, float surf, float mean_depth);
    // Advance the live water by one step: rain in, route downstream, ocean sink.
    void step();
    // Advance the ocean climate by one step: advect SST along the current field,
    // relax toward the solar baseline. Cheap O(N) pass; call after step().
    void step_climate();
    // Bake the climate state into an RGBA payload:
    //   r = sst, g = sst_base (reserved for humidity), b = current angle [0,1],
    //   a = current speed.
    void bake_climate(std::vector<glm::vec4>& out) const;
    // Bake the current state into the RGBA payload
    // (r=river strength from live water, g=moisture, b=flow angle,
    //  a=live water-surface elevation in metres).
    void bake(std::vector<glm::vec4>& out) const;
};

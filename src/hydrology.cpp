#include "hydrology.h"
#include "planet.h"
#include "terrain.h"
#include "pipeline.h"  // TerrainStamp

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

// --- Tunables (river extraction). Exposed here for easy iteration. -----------
// Minimum upstream cell count for a channel to register as a river. The upper
// end of the strength ramp is the actual max accumulation on the planet, so
// trunk rivers always reach ~1.0 regardless of grid resolution.
static constexpr float RIVER_ACCUM_MIN = 4.0f;
// Moisture spreads more broadly than visible rivers.
static constexpr float MOIST_ACCUM_REF = 60.0f;
// Lake depth (metres) that reads as "full" depth in the normalized channel.
static constexpr float LAKE_DEPTH_NORM = 150.0f;
static constexpr float TWO_PI = 6.28318530718f;

// --- Live water routing tunables ---
static constexpr float LIVE_RAIN   = 1.0f;   // water added per land cell per step
static constexpr float LIVE_FLOW_F = 0.5f;   // fraction of a cell's water routed downstream per step
// Steady-state water = (RAIN/FLOW_F) * accumulation, so seed/compare against that.
// --- Fill-and-spill (lakes) tunables ---
static constexpr float LAKE_FILL_K = 1.0f;   // metres of depression depth → water units of storage
static constexpr float LIVE_SEEP   = 0.05f;  // water lost per land cell per step (settles transients)
static constexpr float LAKE_FLOOD   = 1.30f; // how far above "full" a brushed overfill shows as a lake

static const int OFF[8][2] = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};

// Unit sphere direction of a hydrology cell center on face f at grid (i, j).
static glm::vec3 cell_dir(uint32_t f, int i, int j, uint32_t res)
{
    float ts = 2.0f / static_cast<float>(res);
    float u = -1.0f + (static_cast<float>(i) + 0.5f) * ts;
    float v = -1.0f + (static_cast<float>(j) + 0.5f) * ts;
    return glm::normalize(planet_cube_to_sphere(planet_face_uv_to_cube(u, v, f)));
}

// Linear index of the cell offset (di, dj) from (f, i, j). In-face neighbors are
// direct; off-edge neighbors route across the cube-face seam via the sphere
// round-trip (handles all 6×4 edges and the 8 corners by construction).
static int neighbor_index(uint32_t f, int i, int j, int di, int dj, uint32_t res)
{
    int ni = i + di, nj = j + dj;
    int R = static_cast<int>(res);
    bool in_u = (ni >= 0 && ni < R);
    bool in_v = (nj >= 0 && nj < R);
    if (in_u && in_v)
        return static_cast<int>(f * res * res) + nj * R + ni;

    float ts = 2.0f / static_cast<float>(res);
    float u = -1.0f + (static_cast<float>(ni) + 0.5f) * ts;  // may fall outside [-1,1]
    float v = -1.0f + (static_cast<float>(nj) + 0.5f) * ts;
    glm::vec3 dir = glm::normalize(planet_cube_to_sphere(planet_face_uv_to_cube(u, v, f)));

    uint32_t nf; float ou, ov;
    planet_sphere_to_face_uv(dir, nf, ou, ov);
    int oi = std::clamp(static_cast<int>(std::floor((ou + 1.0f) / ts)), 0, R - 1);
    int oj = std::clamp(static_cast<int>(std::floor((ov + 1.0f) / ts)), 0, R - 1);
    return static_cast<int>(nf * res * res) + oj * R + oi;
}

static void idx_to_fij(size_t c, uint32_t res, uint32_t& f, int& i, int& j)
{
    f = static_cast<uint32_t>(c / (static_cast<size_t>(res) * res));
    size_t rem = c % (static_cast<size_t>(res) * res);
    j = static_cast<int>(rem / res);
    i = static_cast<int>(rem % res);
}

static float flow_angle01(glm::vec3 n, glm::vec3 down_dir)
{
    glm::vec3 ref = (std::fabs(n.y) > 0.999f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 east = glm::normalize(glm::cross(ref, n));
    glm::vec3 north = glm::cross(n, east);
    glm::vec3 fw = down_dir - n;
    glm::vec2 flow(glm::dot(fw, east), glm::dot(fw, north));
    if (glm::length(flow) > 1e-6f)
        return std::atan2(flow.y, flow.x) / TWO_PI + 0.5f;
    return 0.5f;
}

// Terrain-stamp contribution at a direction (matches cpu_terrain_height_with_stamps's
// stamp loop). Cheap — lets us re-apply edits without re-evaluating the noise.
static float stamp_delta(glm::vec3 dir, const TerrainStamp* stamps, uint32_t n)
{
    float h = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        const TerrainStamp& s = stamps[i];
        glm::vec3 sp(s.pos_x, s.pos_y, s.pos_z);
        float d = glm::dot(dir, sp);
        if (d < s.cos_radius) continue;
        float t = (d - s.cos_radius) / std::max(1.0f - s.cos_radius, 1e-7f);
        h += s.delta_h * std::exp(-4.0f * (1.0f - t) * (1.0f - t));
    }
    return h;
}

// Sample cell directions + base (noise-only) heights over the whole 6×R×R grid.
// This is the EXPENSIVE pass; it depends only on resolution, so it's computed
// once and cached — terrain edits then only re-apply cheap stamp deltas.
static void sample_base(uint32_t res, std::vector<glm::vec3>& dir,
                        std::vector<float>& base, const std::atomic<bool>* cancel)
{
    const size_t N = static_cast<size_t>(6) * res * res;
    dir.resize(N);
    base.resize(N);
    for (uint32_t f = 0; f < 6; ++f)
        for (uint32_t j = 0; j < res; ++j) {
            if (cancel && cancel->load(std::memory_order_acquire)) return;
            for (uint32_t i = 0; i < res; ++i) {
                size_t idx = static_cast<size_t>(f) * res * res + static_cast<size_t>(j) * res + i;
                glm::vec3 d = cell_dir(f, static_cast<int>(i), static_cast<int>(j), res);
                dir[idx] = d;
                base[idx] = cpu_terrain_height(d);   // noise only, no stamps
            }
        }
}

// Priority-Flood depression fill + D8 downstream pointers on a height field.
// Cheap (no noise) — runs every edit. Flood inward from the ocean so every land
// cell drains to the sea (no interior pits → connected rivers).
static void route_drainage(uint32_t res, float sea_level, const std::vector<float>& height,
                           std::vector<float>& filled, std::vector<int>& downstream)
{
    const size_t N = height.size();
    filled = height;
    {
        const float EPS = 0.5f;
        std::vector<char> closed(N, 0);
        struct Node { float h; uint32_t idx; };
        struct Cmp { bool operator()(const Node& a, const Node& b) const { return a.h > b.h; } };
        std::priority_queue<Node, std::vector<Node>, Cmp> pq;
        for (size_t c = 0; c < N; ++c)
            if (height[c] < sea_level) { closed[c] = 1; pq.push({height[c], static_cast<uint32_t>(c)}); }
        while (!pq.empty()) {
            Node n = pq.top(); pq.pop();
            uint32_t f; int i, j; idx_to_fij(n.idx, res, f, i, j);
            for (auto& o : OFF) {
                int nb = neighbor_index(f, i, j, o[0], o[1], res);
                if (closed[nb]) continue;
                closed[nb] = 1;
                filled[nb] = std::max(height[nb], n.h + EPS);
                pq.push({filled[nb], static_cast<uint32_t>(nb)});
            }
        }
    }

    downstream.assign(N, -1);
    for (uint32_t f = 0; f < 6; ++f)
        for (uint32_t j = 0; j < res; ++j)
            for (uint32_t i = 0; i < res; ++i) {
                size_t c = static_cast<size_t>(f) * res * res + static_cast<size_t>(j) * res + i;
                if (height[c] < sea_level) continue;
                float hc = filled[c];
                float best_drop = 0.0f;
                int   best = -1;
                for (auto& o : OFF) {
                    int nb = neighbor_index(f, static_cast<int>(i), static_cast<int>(j),
                                            o[0], o[1], res);
                    float drop = hc - filled[nb];
                    if (drop > best_drop) { best_drop = drop; best = nb; }
                }
                downstream[c] = best;
            }
}

// Flow accumulation: unit rain per cell, pushed downstream in descending
// filled-height order (each cell processed after everything draining into it).
static std::vector<double> compute_accumulation(const std::vector<float>& filled,
                                                const std::vector<int>& downstream)
{
    const size_t N = filled.size();
    std::vector<double> accum(N, 1.0);
    std::vector<uint32_t> order(N);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return filled[a] > filled[b]; });
    for (uint32_t c : order) {
        int d = downstream[c];
        if (d >= 0) accum[d] += accum[c];
    }
    return accum;
}

PlanetHydrology build_planet_hydrology(uint32_t res, float sea_level,
                                       const TerrainStamp* stamps, uint32_t stamp_count)
{
    PlanetHydrology h;
    h.res = res;
    const size_t N = static_cast<size_t>(6) * res * res;
    h.cells.assign(N, glm::vec4(0.0f));

    std::vector<glm::vec3> dir;
    std::vector<float> base;
    sample_base(res, dir, base, nullptr);
    std::vector<float> height(N);
    for (size_t c = 0; c < N; ++c)
        height[c] = base[c] + stamp_delta(dir[c], stamps, stamp_count);

    std::vector<float> filled;
    std::vector<int> downstream;
    route_drainage(res, sea_level, height, filled, downstream);
    std::vector<double> accum = compute_accumulation(filled, downstream);

    double max_accum = RIVER_ACCUM_MIN + 1.0;
    for (size_t c = 0; c < N; ++c)
        if (height[c] >= sea_level) max_accum = std::max(max_accum, accum[c]);
    const float lr_min = std::log(1.0f + RIVER_ACCUM_MIN);
    const float lr_max = std::log(1.0f + static_cast<float>(max_accum));
    const float lm_ref = std::log(1.0f + MOIST_ACCUM_REF);

    for (size_t c = 0; c < N; ++c) {
        bool land = height[c] >= sea_level;
        float la = std::log(1.0f + static_cast<float>(accum[c]));

        float strength = 0.0f, moisture = 1.0f, flow_a = 0.5f;
        float lake = std::max(0.0f, filled[c] - height[c]);
        float lake01 = (height[c] < sea_level) ? 0.0f : std::clamp(lake / LAKE_DEPTH_NORM, 0.0f, 1.0f);

        if (land) {
            strength = std::clamp((la - lr_min) / std::max(lr_max - lr_min, 1e-4f), 0.0f, 1.0f);
            moisture = std::clamp(la / std::max(lm_ref, 1e-4f), 0.0f, 1.0f);
            int d = downstream[c];
            if (d >= 0) flow_a = flow_angle01(dir[c], dir[d]);
        }
        h.cells[c] = glm::vec4(strength, moisture, flow_a, lake01);
    }
    return h;
}

HydroSample sample_planet_field(const std::vector<glm::vec4>& cells, uint32_t res,
                                glm::vec3 sphere_dir)
{
    HydroSample s;
    if (res == 0 || cells.size() != static_cast<size_t>(6) * res * res)
        return s;
    uint32_t face; float u, v;
    planet_sphere_to_face_uv(glm::normalize(sphere_dir), face, u, v);
    float ts = 2.0f / static_cast<float>(res);
    int i = std::clamp(static_cast<int>(std::floor((u + 1.0f) / ts)), 0, static_cast<int>(res) - 1);
    int j = std::clamp(static_cast<int>(std::floor((v + 1.0f) / ts)), 0, static_cast<int>(res) - 1);
    glm::vec4 c = cells[static_cast<size_t>(face) * res * res + static_cast<size_t>(j) * res + i];

    s.river_strength = c.x;
    s.moisture       = c.y;
    float ang        = (c.z - 0.5f) * TWO_PI;
    s.flow           = glm::vec2(std::cos(ang), std::sin(ang));
    s.lake_depth     = c.w;
    return s;
}

float planet_temperature(glm::vec3 sphere_dir, float height_m)
{
    float lat = std::fabs(glm::normalize(sphere_dir).y);   // 0 at equator, 1 at poles
    float t = 1.0f - lat;                                   // warm equator, cold poles
    t -= std::clamp(height_m, 0.0f, 8000.0f) / 8000.0f * 0.5f;  // altitude lapse
    return std::clamp(t, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// LiveHydrology
// ---------------------------------------------------------------------------

void LiveHydrology::rebuild_structure(const TerrainStamp* stamps, uint32_t stamp_count,
                                     const std::atomic<bool>* cancel)
{
    const size_t N = static_cast<size_t>(6) * res * res;

    // Sample the noise once (the expensive part); cached across all later edits.
    if (base_height.size() != N) {
        sample_base(res, dir, base_height, cancel);
        if (cancel && cancel->load(std::memory_order_acquire)) return;
        height.clear();          // force a full stamp apply below
        applied_stamps = 0;
    }

    // Apply terrain stamps onto the cached base. Incremental when stamps were
    // only added (the brush case) — just fold in the new ones; full re-apply on
    // undo/clear. This is what makes edits cost ~routing time, not a re-sample.
    if (height.size() != N || stamp_count < applied_stamps) {
        height.resize(N);
        for (size_t c = 0; c < N; ++c)
            height[c] = base_height[c] + stamp_delta(dir[c], stamps, stamp_count);
    } else if (stamp_count > applied_stamps) {
        for (size_t c = 0; c < N; ++c)
            height[c] += stamp_delta(dir[c], stamps + applied_stamps, stamp_count - applied_stamps);
    }
    applied_stamps = stamp_count;

    route_drainage(res, sea_level, height, filled, downstream);
    std::vector<double> accum = compute_accumulation(filled, downstream);

    lake01.assign(N, 0.0f);
    flow_angle.assign(N, 0.5f);
    moisture.assign(N, 1.0f);
    cap_w.assign(N, 0.0f);
    const float lm_ref = std::log(1.0f + MOIST_ACCUM_REF);
    for (size_t c = 0; c < N; ++c) {
        bool land = height[c] >= sea_level;
        float lake = std::max(0.0f, filled[c] - height[c]);   // depression depth, metres
        lake01[c] = land ? std::clamp(lake / LAKE_DEPTH_NORM, 0.0f, 1.0f) : 0.0f;
        cap_w[c]  = land ? LAKE_FILL_K * lake : 0.0f;          // storage capacity, water units
        if (land) {
            moisture[c] = std::clamp(std::log(1.0f + static_cast<float>(accum[c])) /
                                     std::max(lm_ref, 1e-4f), 0.0f, 1.0f);
            int d = downstream[c];
            if (d >= 0) flow_angle[c] = flow_angle01(dir[c], dir[d]);
        }
    }

    // Seed water the first time so rivers AND lakes are instantly present: flux
    // from accumulation, plus enough to fill each depression to its spill level.
    // Otherwise keep existing water so an edit re-routes it rather than snapping.
    if (water.size() != N) {
        water.assign(N, 0.0f);
        for (size_t c = 0; c < N; ++c)
            if (height[c] >= sea_level)
                water[c] = std::max((LIVE_RAIN / LIVE_FLOW_F) * static_cast<float>(accum[c]),
                                    cap_w[c]);
    }
}

void LiveHydrology::add_water_deposit(glm::vec3 dir, float cos_radius, float amount)
{
    const size_t N = water.size();
    if (N == 0 || N != this->dir.size()) return;
    const float denom = std::max(1.0f - cos_radius, 1e-5f);
    glm::vec3 c = glm::normalize(dir);
    for (size_t i = 0; i < N; ++i) {
        if (height[i] < sea_level) continue;            // ocean is a sink
        float cosang = glm::dot(this->dir[i], c);
        if (cosang <= cos_radius) continue;             // outside the brush cap
        float t = (cosang - cos_radius) / denom;        // 0 at rim, 1 at center
        water[i] += amount * (t * t * (3.0f - 2.0f * t)); // smoothstep falloff
    }
}

void LiveHydrology::step()
{
    const size_t N = water.size();
    if (N == 0 || cap_w.size() != N) return;
    std::vector<float> Wn = water;
    for (size_t c = 0; c < N; ++c)
        if (height[c] >= sea_level) Wn[c] += LIVE_RAIN;
    for (size_t c = 0; c < N; ++c) {
        if (height[c] < sea_level) continue;
        // Fill-and-spill: water up to the depression's capacity STAYS (a standing
        // lake); only the excess above cap_w spills downstream. On slopes cap_w≈0,
        // so they route as before → rivers unchanged. Basins fill from rain and a
        // brushed flood pushes them over → the surplus overflows and drains away.
        float spill = std::max(0.0f, water[c] - cap_w[c]);
        float out   = spill * LIVE_FLOW_F;
        Wn[c] -= out;
        int d = downstream[c];
        if (d >= 0) Wn[d] += out;       // ocean cells receive then get zeroed (sink)
    }
    for (size_t c = 0; c < N; ++c) {
        if (height[c] < sea_level) { Wn[c] = 0.0f; continue; }   // ocean sink
        Wn[c] -= std::min(Wn[c], LIVE_SEEP);                     // gentle seepage settles transients
    }
    water.swap(Wn);
}

void LiveHydrology::bake(std::vector<glm::vec4>& out) const
{
    const size_t N = water.size();
    out.resize(N);
    const bool have_cap = (cap_w.size() == N);

    // Rivers come from the FLOWING water (the part above storage that spills);
    // standing lake water is excluded so a full basin reads as a lake, not a river.
    double maxf = 1e-6;
    for (size_t c = 0; c < N; ++c)
        if (height[c] >= sea_level) {
            float flow = have_cap ? std::max(0.0f, water[c] - cap_w[c]) : water[c];
            maxf = std::max(maxf, static_cast<double>(flow));
        }
    const float lr_min = std::log(1.0f + (LIVE_RAIN / LIVE_FLOW_F) * RIVER_ACCUM_MIN);
    const float lr_max = std::log(1.0f + static_cast<float>(maxf));

    for (size_t c = 0; c < N; ++c) {
        float strength = 0.0f, lake = 0.0f;
        if (height[c] >= sea_level) {
            float flow = have_cap ? std::max(0.0f, water[c] - cap_w[c]) : water[c];
            float lf = std::log(1.0f + flow);
            strength = std::clamp((lf - lr_min) / std::max(lr_max - lr_min, 1e-4f), 0.0f, 1.0f);
            // Live lake depth: how full the depression is (water/cap), scaled by its
            // geometric depth. A brushed overfill briefly reads above "full" (LAKE_FLOOD)
            // before it drains, so the lake level visibly responds.
            if (have_cap && cap_w[c] > 1e-3f) {
                float fill = std::clamp(water[c] / cap_w[c], 0.0f, LAKE_FLOOD);
                lake = lake01[c] * fill;
            } else {
                lake = lake01[c];
            }
        }
        out[c] = glm::vec4(strength, moisture[c], flow_angle[c], lake);
    }
}

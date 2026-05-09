#include "tree.h"
#include <algorithm>
#include <cmath>

namespace bestiary {

static uint32_t thash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

static float thash_float(uint32_t seed, uint32_t index)
{
    return static_cast<float>(thash(seed ^ (index * 2654435761u))) /
           static_cast<float>(0xFFFFFFFFu);
}

static float tlerp(float a, float b, float t) { return a + (b - a) * t; }

// -----------------------------------------------------------------------
// Space colonization (shared structure with bush, adapted for trees)
// -----------------------------------------------------------------------

struct TreeNode {
    float pos[3];
    int   parent;
    float width;
    int   child_count;
    int   depth;
    bool  is_trunk;
};

struct TreeAttractor {
    float pos[3];
    bool  alive;
};

static bool inside_crown(int shape, float lx, float ly, float lz,
                         float R, float H)
{
    float rx = lx / R;
    float rz = lz / R;
    float r2 = rx * rx + rz * rz;

    switch (shape) {
    case 0: { // ellipsoid
        float hy = (ly - H * 0.5f) / (H * 0.5f);
        return r2 + hy * hy <= 1.0f;
    }
    case 1: { // hemisphere
        float hy = ly / H;
        return r2 + hy * hy <= 1.0f && ly >= 0.0f;
    }
    case 2: // cylinder
        return r2 <= 1.0f && ly >= 0.0f && ly <= H;
    case 3: { // cone
        float t = (H - ly) / H;
        return r2 <= t * t && ly >= 0.0f && ly <= H;
    }
    default:
        return r2 <= 1.0f;
    }
}

static float dist3(const float* a, const float* b)
{
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static void seed_crown_attractors(std::vector<TreeAttractor>& attractors,
                                  int count, int shape,
                                  float R, float H,
                                  float surface_bias,
                                  float cx, float cy, float cz,
                                  uint32_t seed)
{
    attractors.reserve(static_cast<size_t>(count));
    uint32_t attempts = 0;
    int placed = 0;

    while (placed < count && attempts < static_cast<uint32_t>(count) * 20u) {
        float u = thash_float(seed, attempts * 3u + 1000u);
        float v = thash_float(seed, attempts * 3u + 1001u);
        float w = thash_float(seed, attempts * 3u + 1002u);
        ++attempts;

        float lx = (u * 2.0f - 1.0f) * R;
        float ly = v * H;
        float lz = (w * 2.0f - 1.0f) * R;

        if (!inside_crown(shape, lx, ly, lz, R, H))
            continue;

        if (surface_bias > 0.001f) {
            float dist = std::sqrt(lx * lx + lz * lz);
            float max_r = R;
            if (shape == 3) {
                float t = (H - ly) / H;
                max_r = R * t;
            }
            if (max_r > 0.001f && dist > 0.001f) {
                float norm_r = dist / max_r;
                float biased = std::pow(norm_r, 1.0f / (1.0f + surface_bias * 3.0f));
                float scale = biased / norm_r;
                lx *= scale;
                lz *= scale;
                if (!inside_crown(shape, lx, ly, lz, R, H))
                    continue;
            }
        }

        TreeAttractor a{};
        a.pos[0] = cx + lx;
        a.pos[1] = cy + ly;
        a.pos[2] = cz + lz;
        a.alive  = true;
        attractors.push_back(a);
        ++placed;
    }
}

static void build_trunk(std::vector<TreeNode>& nodes,
                        float ox, float oz,
                        float trunk_height, float trunk_width,
                        float step_d, uint32_t seed)
{
    int trunk_steps = std::max(static_cast<int>(trunk_height / step_d), 1);
    float actual_step = trunk_height / static_cast<float>(trunk_steps);

    TreeNode root{};
    root.pos[0] = ox;
    root.pos[1] = 0.0f;
    root.pos[2] = oz;
    root.parent = -1;
    root.width  = trunk_width;
    root.child_count = 0;
    root.depth  = 0;
    root.is_trunk = true;
    nodes.push_back(root);

    for (int i = 1; i <= trunk_steps; ++i) {
        float wobble_x = (thash_float(seed, static_cast<uint32_t>(i) * 2u + 300u) - 0.5f) * step_d * 0.1f;
        float wobble_z = (thash_float(seed, static_cast<uint32_t>(i) * 2u + 301u) - 0.5f) * step_d * 0.1f;

        TreeNode n{};
        n.pos[0] = ox + wobble_x;
        n.pos[1] = static_cast<float>(i) * actual_step;
        n.pos[2] = oz + wobble_z;
        n.parent = static_cast<int>(nodes.size()) - 1;
        n.width  = trunk_width * (1.0f - 0.3f * static_cast<float>(i) / static_cast<float>(trunk_steps));
        n.child_count = 0;
        n.depth  = i;
        n.is_trunk = true;
        nodes.back().child_count += 1;
        nodes.push_back(n);
    }
}

static void grow_crown(std::vector<TreeNode>& nodes,
                       std::vector<TreeAttractor>& attractors,
                       float D, float d_k, float d_i,
                       float tropism, float gravity, float wobble,
                       int shape, float R, float H,
                       float cx, float cy, float cz,
                       uint32_t seed)
{
    struct GrowDir { float dx, dy, dz; int count; };
    std::vector<GrowDir> dirs;
    float min_sep = D * 0.5f;

    for (int iter = 0; iter < 200; ++iter) {
        dirs.assign(nodes.size(), {0.0f, 0.0f, 0.0f, 0});
        bool any_active = false;

        for (auto& a : attractors) {
            if (!a.alive) continue;

            float best_dist = d_i;
            int best_node = -1;
            bool killed = false;

            for (int ni = 0; ni < static_cast<int>(nodes.size()); ++ni) {
                float d = dist3(a.pos, nodes[static_cast<size_t>(ni)].pos);
                if (d < d_k) {
                    a.alive = false;
                    killed = true;
                    break;
                }
                if (d < best_dist) {
                    best_dist = d;
                    best_node = ni;
                }
            }

            if (!killed && best_node >= 0) {
                auto& g = dirs[static_cast<size_t>(best_node)];
                float d = best_dist;
                if (d < 0.0001f) d = 0.0001f;
                g.dx += (a.pos[0] - nodes[static_cast<size_t>(best_node)].pos[0]) / d;
                g.dy += (a.pos[1] - nodes[static_cast<size_t>(best_node)].pos[1]) / d;
                g.dz += (a.pos[2] - nodes[static_cast<size_t>(best_node)].pos[2]) / d;
                g.count += 1;
                any_active = true;
            }
        }

        if (!any_active) break;

        size_t prev_count = nodes.size();
        for (size_t ni = 0; ni < prev_count; ++ni) {
            auto& g = dirs[ni];
            if (g.count == 0) continue;

            float dx = g.dx;
            float dy = g.dy + tropism * 0.3f;
            float dz = g.dz;

            float depth_frac = static_cast<float>(nodes[ni].depth) /
                               std::max(static_cast<float>(iter + 5), 1.0f);
            dy -= gravity * depth_frac * 0.5f;

            if (wobble > 0.001f) {
                uint32_t wi = static_cast<uint32_t>(iter * 1000 + ni);
                dx += (thash_float(seed, wi * 3u + 700u) - 0.5f) * wobble;
                dy += (thash_float(seed, wi * 3u + 701u) - 0.5f) * wobble * 0.5f;
                dz += (thash_float(seed, wi * 3u + 702u) - 0.5f) * wobble;
            }

            float len = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (len < 0.0001f) continue;
            dx /= len; dy /= len; dz /= len;

            float np[3] = {
                nodes[ni].pos[0] + dx * D,
                nodes[ni].pos[1] + dy * D,
                nodes[ni].pos[2] + dz * D
            };

            float lx = np[0] - cx, ly = np[1] - cy, lz_local = np[2] - cz;
            if (!inside_crown(shape, lx, ly, lz_local, R, H))
                continue;

            bool too_close = false;
            for (size_t mi = 0; mi < nodes.size(); ++mi) {
                if (dist3(np, nodes[mi].pos) < min_sep) {
                    too_close = true;
                    break;
                }
            }
            if (too_close) continue;

            TreeNode cn{};
            cn.pos[0] = np[0]; cn.pos[1] = np[1]; cn.pos[2] = np[2];
            cn.parent = static_cast<int>(ni);
            cn.width  = 0.0f;
            cn.child_count = 0;
            cn.depth  = nodes[ni].depth + 1;
            cn.is_trunk = false;
            nodes[ni].child_count += 1;
            nodes.push_back(cn);
        }

        if (nodes.size() == prev_count) break;
    }
}

static void compute_tree_widths(std::vector<TreeNode>& nodes, float taper_exp,
                                float min_w, float max_w)
{

    for (auto& n : nodes) {
        if (n.is_trunk) continue;
        n.width = (n.child_count == 0) ? min_w : 0.0f;
    }

    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
        auto& n = nodes[static_cast<size_t>(i)];
        if (n.parent >= 0 && !n.is_trunk) {
            auto& p = nodes[static_cast<size_t>(n.parent)];
            if (!p.is_trunk) {
                p.width = std::pow(
                    std::pow(p.width, taper_exp) + std::pow(n.width, taper_exp),
                    1.0f / taper_exp);
            }
        }
    }

    for (auto& n : nodes)
        n.width = std::min(n.width, max_w);
}

// -----------------------------------------------------------------------
// Mesh emission
// -----------------------------------------------------------------------

static void emit_trunk_and_branches(VegetationMesh& mesh,
                                    const std::vector<TreeNode>& nodes,
                                    const float trunk_color[3])
{
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        if (n.parent < 0) continue;
        const auto& p = nodes[static_cast<size_t>(n.parent)];

        float dx = n.pos[0] - p.pos[0];
        float dy = n.pos[1] - p.pos[1];
        float dz = n.pos[2] - p.pos[2];
        float len = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 0.0001f) continue;
        dx /= len; dy /= len; dz /= len;

        float ref_x = 0.0f, ref_y = 1.0f, ref_z = 0.0f;
        if (std::abs(dy) > 0.95f) { ref_x = 1.0f; ref_y = 0.0f; }

        float rx = ref_y * dz - ref_z * dy;
        float ry = ref_z * dx - ref_x * dz;
        float rz = ref_x * dy - ref_y * dx;
        float rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (rlen < 0.0001f) continue;
        rx /= rlen; ry /= rlen; rz /= rlen;

        float ux = dy * rz - dz * ry;
        float uy = dz * rx - dx * rz;
        float uz = dx * ry - dy * rx;

        float w0 = p.width;
        float w1 = n.width;

        // Darken branch color for smaller branches
        float branch_dark = n.is_trunk ? 1.0f : 0.8f;
        float bc[3] = {
            trunk_color[0] * branch_dark,
            trunk_color[1] * branch_dark,
            trunk_color[2] * branch_dark
        };

        for (int q = 0; q < 2; ++q) {
            float px = (q == 0) ? rx : ux;
            float py = (q == 0) ? ry : uy;
            float pz = (q == 0) ? rz : uz;

            auto si = static_cast<uint32_t>(mesh.vertices.size());
            float corners[4][3] = {
                {p.pos[0] - px * w0, p.pos[1] - py * w0, p.pos[2] - pz * w0},
                {p.pos[0] + px * w0, p.pos[1] + py * w0, p.pos[2] + pz * w0},
                {n.pos[0] + px * w1, n.pos[1] + py * w1, n.pos[2] + pz * w1},
                {n.pos[0] - px * w1, n.pos[1] - py * w1, n.pos[2] - pz * w1},
            };

            float nx = (q == 0) ? ux : rx;
            float ny = (q == 0) ? uy : ry;
            float nz = (q == 0) ? uz : rz;

            for (auto& c : corners) {
                VegetationVertex v{};
                v.position[0] = c[0]; v.position[1] = c[1]; v.position[2] = c[2];
                v.normal[0] = nx; v.normal[1] = ny; v.normal[2] = nz;
                v.color[0] = bc[0]; v.color[1] = bc[1]; v.color[2] = bc[2];
                mesh.vertices.push_back(v);
            }
            mesh.indices.push_back(si);     mesh.indices.push_back(si + 1); mesh.indices.push_back(si + 2);
            mesh.indices.push_back(si);     mesh.indices.push_back(si + 2); mesh.indices.push_back(si + 3);
            mesh.indices.push_back(si);     mesh.indices.push_back(si + 2); mesh.indices.push_back(si + 1);
            mesh.indices.push_back(si);     mesh.indices.push_back(si + 3); mesh.indices.push_back(si + 2);
        }
    }
}

static void place_tree_leaves(VegetationMesh& mesh,
                              const std::vector<TreeNode>& nodes,
                              const TreeParams& params,
                              float /*crown_base_y*/,
                              uint32_t seed)
{
    constexpr float pi = 3.14159265f;
    constexpr float golden_angle = 2.39996323f;

    if (nodes.empty() || params.leaf_count <= 0) return;

    int max_depth = 0;
    for (auto& n : nodes)
        if (n.depth > max_depth) max_depth = n.depth;
    if (max_depth == 0) max_depth = 1;

    std::vector<size_t> tip_sites;
    std::vector<size_t> interior_sites;
    int depth_thresh = max_depth / 2;

    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].is_trunk) continue;
        if (nodes[i].child_count == 0)
            tip_sites.push_back(i);
        else if (nodes[i].depth >= depth_thresh)
            interior_sites.push_back(i);
    }

    if (tip_sites.empty() && interior_sites.empty()) return;

    int n_tip = static_cast<int>(std::round(
        static_cast<float>(params.leaf_count) * params.tip_leaf_bias));
    int n_interior = params.leaf_count - n_tip;

    float total_height = params.trunk_height + params.crown_height;

    auto emit_leaves_at_sites = [&](const std::vector<size_t>& sites, int total_leaves, int leaf_offset) {
        if (sites.empty() || total_leaves <= 0) return;

        int per_site = std::max(total_leaves / static_cast<int>(sites.size()), 1);
        int remaining = total_leaves;
        int global_i = leaf_offset;

        for (size_t si = 0; si < sites.size() && remaining > 0; ++si) {
            const auto& node = nodes[sites[si]];
            int count = std::min(per_site, remaining);
            if (si == sites.size() - 1) count = remaining;

            for (int li = 0; li < count; ++li) {
                auto ui = static_cast<uint32_t>(global_i);

                float depth_frac = static_cast<float>(node.depth) / static_cast<float>(max_depth);
                float size_scale = 1.0f - 0.4f * depth_frac;

                float leaf_azimuth = static_cast<float>(global_i) * golden_angle;
                float spread = params.crown_radius * 0.1f;
                float jitter_r = spread * (0.3f + 0.7f * thash_float(seed, ui * 4u + 200u));

                float lx = node.pos[0] + std::cos(leaf_azimuth) * jitter_r;
                float lz = node.pos[2] + std::sin(leaf_azimuth) * jitter_r;
                float ly = node.pos[1] + (thash_float(seed, ui * 4u + 201u) - 0.3f) * params.crown_height * 0.05f;

                float out_x = lx - node.pos[0];
                float out_y = 0.3f;
                float out_z = lz - node.pos[2];
                out_y -= params.leaf_droop * 0.5f;

                float olen = std::sqrt(out_x * out_x + out_y * out_y + out_z * out_z);
                if (olen < 0.001f) { out_x = 0.0f; out_y = 1.0f; out_z = 0.0f; olen = 1.0f; }
                out_x /= olen; out_y /= olen; out_z /= olen;

                float ref_x = 0.0f, ref_y = 1.0f, ref_z = 0.0f;
                if (std::abs(out_y) > 0.95f) { ref_x = 1.0f; ref_y = 0.0f; }

                float rrx = ref_y * out_z - ref_z * out_y;
                float rry = ref_z * out_x - ref_x * out_z;
                float rrz = ref_x * out_y - ref_y * out_x;
                float rlen = std::sqrt(rrx * rrx + rry * rry + rrz * rrz);
                if (rlen < 0.001f) continue;
                rrx /= rlen; rry /= rlen; rrz /= rlen;

                float uux = out_y * rrz - out_z * rry;
                float uuy = out_z * rrx - out_x * rrz;
                float uuz = out_x * rry - out_y * rrx;

                float twist = thash_float(seed, ui * 4u + 202u) * pi;
                float cos_t = std::cos(twist);
                float sin_t = std::sin(twist);
                float tr_x = rrx * cos_t + uux * sin_t;
                float tr_y = rry * cos_t + uuy * sin_t;
                float tr_z = rrz * cos_t + uuz * sin_t;
                float tu_x = -rrx * sin_t + uux * cos_t;
                float tu_y = -rry * sin_t + uuy * cos_t;
                float tu_z = -rrz * sin_t + uuz * cos_t;

                float hl = params.leaf_length * 0.5f * size_scale;
                float hw = params.leaf_width * 0.5f * size_scale;

                float height_frac = ly / std::max(total_height, 0.01f);
                float base_ht = std::clamp(height_frac, 0.0f, 1.0f);

                auto vi_base = static_cast<uint32_t>(mesh.vertices.size());

                struct LeafPt { float dx_r, dx_u, ht; };
                LeafPt pts[4] = {
                    { 0.0f, -hl, 0.0f},
                    {  hw,  0.0f, 0.5f},
                    { 0.0f,  hl, 1.0f},
                    { -hw,  0.0f, 0.5f},
                };

                float color_var = 0.9f + 0.2f * thash_float(seed, ui * 4u + 203u);

                for (auto& pt : pts) {
                    VegetationVertex v{};
                    v.position[0] = lx + tr_x * pt.dx_r + tu_x * pt.dx_u;
                    v.position[1] = ly + tr_y * pt.dx_r + tu_y * pt.dx_u;
                    v.position[2] = lz + tr_z * pt.dx_r + tu_z * pt.dx_u;
                    v.normal[0] = out_x; v.normal[1] = out_y; v.normal[2] = out_z;
                    v.color[0] = params.base_color[0] * color_var;
                    v.color[1] = params.base_color[1] * color_var;
                    v.color[2] = params.base_color[2] * color_var;
                    v.height_t = base_ht + pt.ht * (1.0f - base_ht) * 0.3f;
                    mesh.vertices.push_back(v);
                }

                mesh.indices.push_back(vi_base + 0); mesh.indices.push_back(vi_base + 1); mesh.indices.push_back(vi_base + 2);
                mesh.indices.push_back(vi_base + 0); mesh.indices.push_back(vi_base + 2); mesh.indices.push_back(vi_base + 3);
                mesh.indices.push_back(vi_base + 0); mesh.indices.push_back(vi_base + 2); mesh.indices.push_back(vi_base + 1);
                mesh.indices.push_back(vi_base + 0); mesh.indices.push_back(vi_base + 3); mesh.indices.push_back(vi_base + 2);

                ++global_i;
                --remaining;
            }
        }
    };

    emit_leaves_at_sites(tip_sites, n_tip, 0);
    emit_leaves_at_sites(interior_sites, n_interior, n_tip);
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

TreeParams evaluate_tree_expression(const TreeParams& base,
                                    const TreeExpression& expr,
                                    float moisture)
{
    float t = std::clamp(moisture, 0.0f, 1.0f);
    TreeParams out = base;

    if (expr.tree_height.enabled)
        out.tree_height = tlerp(expr.tree_height.low, expr.tree_height.high, t);
    if (expr.trunk_height.enabled)
        out.trunk_height = tlerp(expr.trunk_height.low, expr.trunk_height.high, t);
    if (expr.crown_radius.enabled)
        out.crown_radius = tlerp(expr.crown_radius.low, expr.crown_radius.high, t);
    if (expr.crown_height.enabled)
        out.crown_height = tlerp(expr.crown_height.low, expr.crown_height.high, t);
    if (expr.trunk_width.enabled)
        out.trunk_width = tlerp(expr.trunk_width.low, expr.trunk_width.high, t);
    if (expr.leaf_count.enabled)
        out.leaf_count = std::clamp(static_cast<int>(std::round(tlerp(expr.leaf_count.low, expr.leaf_count.high, t))), 1, 2000);
    if (expr.leaf_length.enabled)
        out.leaf_length = tlerp(expr.leaf_length.low, expr.leaf_length.high, t);
    if (expr.leaf_width.enabled)
        out.leaf_width = tlerp(expr.leaf_width.low, expr.leaf_width.high, t);
    if (expr.attractor_count.enabled)
        out.attractor_count = std::clamp(static_cast<int>(std::round(tlerp(expr.attractor_count.low, expr.attractor_count.high, t))), 100, 3000);
    if (expr.kill_ratio.enabled)
        out.kill_ratio = tlerp(expr.kill_ratio.low, expr.kill_ratio.high, t);
    if (expr.influence_ratio.enabled)
        out.influence_ratio = tlerp(expr.influence_ratio.low, expr.influence_ratio.high, t);
    if (expr.tropism.enabled)
        out.tropism = tlerp(expr.tropism.low, expr.tropism.high, t);
    if (expr.surface_bias.enabled)
        out.surface_bias = tlerp(expr.surface_bias.low, expr.surface_bias.high, t);
    if (expr.branch_width_min.enabled)
        out.branch_width_min = tlerp(expr.branch_width_min.low, expr.branch_width_min.high, t);
    if (expr.branch_width_max.enabled)
        out.branch_width_max = tlerp(expr.branch_width_max.low, expr.branch_width_max.high, t);
    if (expr.branch_gravity.enabled)
        out.branch_gravity = tlerp(expr.branch_gravity.low, expr.branch_gravity.high, t);
    if (expr.branch_wobble.enabled)
        out.branch_wobble = tlerp(expr.branch_wobble.low, expr.branch_wobble.high, t);
    if (expr.branch_taper.enabled)
        out.branch_taper = tlerp(expr.branch_taper.low, expr.branch_taper.high, t);
    if (expr.tip_leaf_bias.enabled)
        out.tip_leaf_bias = tlerp(expr.tip_leaf_bias.low, expr.tip_leaf_bias.high, t);
    if (expr.leaf_droop.enabled)
        out.leaf_droop = tlerp(expr.leaf_droop.low, expr.leaf_droop.high, t);

    if (expr.vary_color) {
        for (int i = 0; i < 3; ++i)
            out.base_color[i] = tlerp(expr.dry_color[i], expr.wet_color[i], t);
    }

    return out;
}

VegetationMesh generate_tree(const TreeParams& params, uint32_t seed,
                             bool include_ground,
                             float offset_x, float offset_z)
{
    VegetationMesh mesh;

    float R = params.crown_radius;
    float H = params.crown_height;
    float D = std::max(std::max(H, R * 2.0f) / 20.0f, 0.01f);
    float d_k = D * params.kill_ratio;
    float d_i = D * params.influence_ratio;

    // Crown center is at the top of the trunk
    float crown_base_y = params.trunk_height;

    // 1. Build trunk
    std::vector<TreeNode> nodes;
    build_trunk(nodes, offset_x, offset_z,
                params.trunk_height, params.trunk_width, D, seed);

    // 2. Seed attractors in crown volume
    std::vector<TreeAttractor> attractors;
    seed_crown_attractors(attractors, params.attractor_count, params.crown_shape,
                          R, H, params.surface_bias,
                          offset_x, crown_base_y, offset_z, seed);

    // 3. Grow crown branches from trunk top
    grow_crown(nodes, attractors, D, d_k, d_i,
               params.tropism, params.branch_gravity, params.branch_wobble,
               params.crown_shape,
               R, H, offset_x, crown_base_y, offset_z, seed);

    // 4. Compute branch widths (crown only; trunk widths are set during build)
    compute_tree_widths(nodes, params.branch_taper,
                        params.branch_width_min, params.branch_width_max);

    // 5. Emit geometry
    emit_trunk_and_branches(mesh, nodes, params.trunk_color);
    place_tree_leaves(mesh, nodes, params, crown_base_y, seed);

    if (include_ground) {
        float gs = std::max(params.crown_radius * 2.0f, 1.0f) + 0.5f;
        auto gi = static_cast<uint32_t>(mesh.vertices.size());

        const float gc[3] = {0.25f, 0.20f, 0.15f};
        float corners[4][3] = {
            {offset_x - gs, 0.0f, offset_z - gs},
            {offset_x + gs, 0.0f, offset_z - gs},
            {offset_x + gs, 0.0f, offset_z + gs},
            {offset_x - gs, 0.0f, offset_z + gs}
        };
        for (auto& c : corners) {
            VegetationVertex v{};
            v.position[0] = c[0]; v.position[1] = c[1]; v.position[2] = c[2];
            v.normal[0] = 0.0f; v.normal[1] = 1.0f; v.normal[2] = 0.0f;
            v.color[0] = gc[0]; v.color[1] = gc[1]; v.color[2] = gc[2];
            mesh.vertices.push_back(v);
        }
        mesh.indices.push_back(gi);     mesh.indices.push_back(gi + 1); mesh.indices.push_back(gi + 2);
        mesh.indices.push_back(gi);     mesh.indices.push_back(gi + 2); mesh.indices.push_back(gi + 3);
    }

    return mesh;
}

VegetationMesh generate_tree_field(const TreeParams& base,
                                   const TreeExpression& expr,
                                   const FieldParams& field,
                                   uint32_t base_seed)
{
    VegetationMesh mesh;
    float extent = static_cast<float>(field.grid_n - 1) * field.spacing;

    for (int iz = 0; iz < field.grid_n; ++iz) {
        for (int ix = 0; ix < field.grid_n; ++ix) {
            float x = static_cast<float>(ix) * field.spacing - extent * 0.5f;
            float z = static_cast<float>(iz) * field.spacing - extent * 0.5f;
            float moisture = (field.grid_n > 1)
                ? static_cast<float>(ix) / static_cast<float>(field.grid_n - 1)
                : 0.5f;

            TreeParams resolved = evaluate_tree_expression(base, expr, moisture);
            uint32_t seed = base_seed + static_cast<uint32_t>(iz * field.grid_n + ix);

            auto cell = generate_tree(resolved, seed, false, x, z);

            auto vert_offset = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(),
                                 cell.vertices.begin(), cell.vertices.end());
            for (uint32_t idx : cell.indices)
                mesh.indices.push_back(idx + vert_offset);
        }
    }

    float half = extent * 0.5f + field.spacing;
    auto gi = static_cast<uint32_t>(mesh.vertices.size());

    const float gc[3] = {0.25f, 0.20f, 0.15f};
    float corners[4][3] = {
        {-half, 0.0f, -half}, { half, 0.0f, -half},
        { half, 0.0f,  half}, {-half, 0.0f,  half}
    };
    for (auto& c : corners) {
        VegetationVertex v{};
        v.position[0] = c[0]; v.position[1] = c[1]; v.position[2] = c[2];
        v.normal[0] = 0.0f; v.normal[1] = 1.0f; v.normal[2] = 0.0f;
        v.color[0] = gc[0]; v.color[1] = gc[1]; v.color[2] = gc[2];
        mesh.vertices.push_back(v);
    }
    mesh.indices.push_back(gi);     mesh.indices.push_back(gi + 1); mesh.indices.push_back(gi + 2);
    mesh.indices.push_back(gi);     mesh.indices.push_back(gi + 2); mesh.indices.push_back(gi + 3);

    return mesh;
}

} // namespace bestiary

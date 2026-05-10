#include "lplant.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace bestiary {

static uint32_t lhash(uint32_t x)
{
    x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16;
    return x;
}

static float lhash_float(uint32_t seed, uint32_t index)
{
    return static_cast<float>(lhash(seed ^ (index * 2654435761u))) /
           static_cast<float>(0xFFFFFFFFu);
}

static float llerp(float a, float b, float t) { return a + (b - a) * t; }

constexpr float PI = 3.14159265f;

// -----------------------------------------------------------------------
// Vector math
// -----------------------------------------------------------------------

static void vec_normalize(float v[3])
{
    float len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len < 0.0001f) return;
    v[0] /= len; v[1] /= len; v[2] /= len;
}

static void vec_cross(float out[3], const float a[3], const float b[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static float vec_dot(const float a[3], const float b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static float vec_len(const float v[3])
{
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static void vec_rotate(float out[3], const float v[3], const float axis[3], float angle)
{
    float c = std::cos(angle), s = std::sin(angle);
    float d = vec_dot(v, axis);
    float cross[3]; vec_cross(cross, axis, v);
    out[0] = v[0]*c + cross[0]*s + axis[0]*d*(1.0f - c);
    out[1] = v[1]*c + cross[1]*s + axis[1]*d*(1.0f - c);
    out[2] = v[2]*c + cross[2]*s + axis[2]*d*(1.0f - c);
}

static void vec_copy(float dst[3], const float src[3])
{
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
}

// -----------------------------------------------------------------------
// Space colonization
// -----------------------------------------------------------------------

struct LAttractor { float pos[3]; bool alive; };

static bool inside_envelope(int shape, float lx, float ly, float lz, float R, float H)
{
    float rx = lx/R, rz = lz/R, r2 = rx*rx + rz*rz;
    switch (shape) {
    case 0: { float hy = (ly-H*0.5f)/(H*0.5f); return r2+hy*hy<=1.0f; }
    case 1: { float hy = ly/H; return r2+hy*hy<=1.0f && ly>=0.0f; }
    case 2: return r2<=1.0f && ly>=0.0f && ly<=H;
    case 3: { float t=(H-ly)/H; return r2<=t*t && ly>=0.0f && ly<=H; }
    default: return r2<=1.0f;
    }
}

static void seed_attractors(std::vector<LAttractor>& attractors, int count, int shape,
                            float R, float H, float surface_bias,
                            float cx, float cy, float cz, uint32_t seed)
{
    attractors.reserve(static_cast<size_t>(count));
    uint32_t att = 0; int placed = 0;
    while (placed < count && att < static_cast<uint32_t>(count) * 20u) {
        float u = lhash_float(seed, att*3u+1000u);
        float v = lhash_float(seed, att*3u+1001u);
        float w = lhash_float(seed, att*3u+1002u);
        ++att;
        float lx=(u*2-1)*R, ly=v*H, lz=(w*2-1)*R;
        if (!inside_envelope(shape, lx, ly, lz, R, H)) continue;
        if (surface_bias > 0.001f) {
            float dist = std::sqrt(lx*lx+lz*lz);
            float max_r = R; if (shape==3) max_r = R*((H-ly)/H);
            if (max_r>0.001f && dist>0.001f) {
                float biased = std::pow(dist/max_r, 1.0f/(1.0f+surface_bias*3.0f));
                lx *= biased/(dist/max_r); lz *= biased/(dist/max_r);
                if (!inside_envelope(shape, lx, ly, lz, R, H)) continue;
            }
        }
        LAttractor a{}; a.pos[0]=cx+lx; a.pos[1]=cy+ly; a.pos[2]=cz+lz; a.alive=true;
        attractors.push_back(a); ++placed;
    }
}

static bool query_attractor_dir(float dir[3], const float pos[3],
                                std::vector<LAttractor>& attractors, float d_k, float d_i)
{
    float dx=0, dy=0, dz=0; int count=0;
    for (auto& a : attractors) {
        if (!a.alive) continue;
        float ddx=a.pos[0]-pos[0], ddy=a.pos[1]-pos[1], ddz=a.pos[2]-pos[2];
        float d = std::sqrt(ddx*ddx+ddy*ddy+ddz*ddz);
        if (d < d_k) { a.alive = false; continue; }
        if (d < d_i) { if (d<0.0001f) d=0.0001f; dx+=ddx/d; dy+=ddy/d; dz+=ddz/d; ++count; }
    }
    if (count==0) return false;
    float len = std::sqrt(dx*dx+dy*dy+dz*dz);
    if (len<0.0001f) return false;
    dir[0]=dx/len; dir[1]=dy/len; dir[2]=dz/len;
    return true;
}

// -----------------------------------------------------------------------
// Module tree helpers
// -----------------------------------------------------------------------

static int add_child(std::vector<PlantModule>& tree, int parent_idx, PlantModule child)
{
    int idx = static_cast<int>(tree.size());
    child.parent = parent_idx;
    if (parent_idx >= 0) {
        auto& p = tree[static_cast<size_t>(parent_idx)];
        if (p.first_child < 0) { p.first_child = idx; }
        else {
            int sib = p.first_child;
            while (tree[static_cast<size_t>(sib)].next_sibling >= 0)
                sib = tree[static_cast<size_t>(sib)].next_sibling;
            tree[static_cast<size_t>(sib)].next_sibling = idx;
        }
        p.child_count++;
    }
    tree.push_back(child);
    return idx;
}

// Compute growth direction with attractor influence, tropism, gravity, wobble
static void compute_grow_dir(float dir[3], const PlantModule& apex,
                             float inode_len,
                             std::vector<LAttractor>& attractors,
                             float d_k, float d_i,
                             const LPlantParams& params,
                             uint32_t seed, int step, int idx)
{
    vec_copy(dir, apex.heading);

    float probe[3] = {
        apex.pos[0] + apex.heading[0] * inode_len,
        apex.pos[1] + apex.heading[1] * inode_len,
        apex.pos[2] + apex.heading[2] * inode_len
    };
    float attr_dir[3];
    if (query_attractor_dir(attr_dir, probe, attractors, d_k, d_i)) {
        float blend = 0.3f;
        dir[0] = dir[0]*(1-blend) + attr_dir[0]*blend;
        dir[1] = dir[1]*(1-blend) + attr_dir[1]*blend;
        dir[2] = dir[2]*(1-blend) + attr_dir[2]*blend;
    }

    dir[1] += params.tropism * 0.3f;
    float depth_frac = static_cast<float>(apex.order) / std::max(static_cast<float>(params.growth_steps), 1.0f);
    dir[1] -= params.branch_gravity * depth_frac * 0.5f;

    if (params.branch_wobble > 0.001f) {
        uint32_t wi = static_cast<uint32_t>(step * 1000 + idx);
        dir[0] += (lhash_float(seed, wi*3u+700u) - 0.5f) * params.branch_wobble;
        dir[1] += (lhash_float(seed, wi*3u+701u) - 0.5f) * params.branch_wobble * 0.5f;
        dir[2] += (lhash_float(seed, wi*3u+702u) - 0.5f) * params.branch_wobble;
    }
    vec_normalize(dir);
}

// Build turtle frame from heading
static void frame_from_heading(float left[3], float up[3], const float heading[3], const float parent_up[3])
{
    vec_cross(left, parent_up, heading);
    float ll = vec_len(left);
    if (ll < 0.001f) { left[0]=1; left[1]=0; left[2]=0; }
    else { left[0]/=ll; left[1]/=ll; left[2]/=ll; }
    vec_cross(up, heading, left);
    vec_normalize(up);
}

// Create a child apex at end_pos with a rotated heading
static void make_branch_apex(std::vector<PlantModule>& tree, int parent_idx,
                             const float end_pos[3], const float grow_dir[3],
                             const float parent_up[3],
                             float branch_angle, float phyl_rot,
                             int order, float resource_v)
{
    float lat_heading[3];
    vec_rotate(lat_heading, grow_dir, parent_up, branch_angle);
    float lat_rotated[3];
    vec_rotate(lat_rotated, lat_heading, grow_dir, phyl_rot);
    vec_normalize(lat_rotated);

    PlantModule child{};
    child.type = ModuleType::Apex;
    vec_copy(child.pos, end_pos);
    vec_copy(child.heading, lat_rotated);
    frame_from_heading(child.left, child.up, child.heading, parent_up);
    child.order = order;
    child.resource_v = resource_v;
    child.age = 0;
    add_child(tree, parent_idx, child);
}

// -----------------------------------------------------------------------
// Unified growth rule with maturity model
// -----------------------------------------------------------------------

// How many internodes an apex grows before branching
static int maturity_for_order(int order, const LPlantParams& params)
{
    // Order 0: grows many internodes (main axis). Higher orders: fewer before forking.
    int base = std::max(params.growth_steps / 3, 2);
    int mat = base - order;
    return std::max(mat, 1);
}

static void apply_growth_rule(std::vector<PlantModule>& tree, int apex_idx,
                              const LPlantParams& params,
                              std::vector<LAttractor>& attractors,
                              float d_k, float d_i,
                              uint32_t seed, int step)
{
    auto& apex = tree[static_cast<size_t>(apex_idx)];
    if (apex.resource_v < params.v_threshold) {
        apex.type = ModuleType::Dormant;
        return;
    }

    float inode_len = params.internode_length *
                      std::pow(params.length_decay, static_cast<float>(apex.order));

    int maturity = maturity_for_order(apex.order, params);
    bool should_branch = (apex.age >= maturity);

    // Turn apex into internode (extend the branch)
    apex.type = ModuleType::Internode;
    apex.length = inode_len;

    float grow_dir[3];
    compute_grow_dir(grow_dir, apex, inode_len, attractors, d_k, d_i,
                     params, seed, step, apex_idx);
    vec_copy(apex.heading, grow_dir);

    float end_pos[3] = {
        apex.pos[0] + grow_dir[0] * inode_len,
        apex.pos[1] + grow_dir[1] * inode_len,
        apex.pos[2] + grow_dir[2] * inode_len
    };

    float phyl_angle = params.phyllotaxis_angle * PI / 180.0f;
    uint32_t hash_base = static_cast<uint32_t>(step * 100 + apex_idx);
    float angle_var = (lhash_float(seed, hash_base + 400u) - 0.5f) * params.branch_angle_var * 2.0f;

    if (!should_branch) {
        // Just extend: create a new apex continuing in the same direction
        PlantModule continuation{};
        continuation.type = ModuleType::Apex;
        vec_copy(continuation.pos, end_pos);
        vec_copy(continuation.heading, grow_dir);
        frame_from_heading(continuation.left, continuation.up, grow_dir, apex.up);
        continuation.order = apex.order;
        continuation.resource_v = apex.resource_v;
        continuation.age = apex.age + 1;
        add_child(tree, apex_idx, continuation);
        return;
    }

    // Branch! The archetype determines how.
    float rot = phyl_angle * static_cast<float>(step);

    switch (params.archetype) {
    case GrowthArchetype::Monopodial:
    case GrowthArchetype::MultistemBush: {
        // Main axis continues, one lateral branches off
        make_branch_apex(tree, apex_idx, end_pos, grow_dir, apex.up,
                         params.branch_angle + angle_var, rot,
                         apex.order + 1, apex.resource_v * 0.5f);

        PlantModule main_c{};
        main_c.type = ModuleType::Apex;
        vec_copy(main_c.pos, end_pos);
        vec_copy(main_c.heading, grow_dir);
        frame_from_heading(main_c.left, main_c.up, grow_dir, apex.up);
        main_c.order = apex.order;
        main_c.resource_v = apex.resource_v;
        main_c.age = 0;
        add_child(tree, apex_idx, main_c);
        break;
    }
    case GrowthArchetype::Sympodial:
    case GrowthArchetype::Dichotomous: {
        // Main axis terminates, two laterals fork
        make_branch_apex(tree, apex_idx, end_pos, grow_dir, apex.up,
                         params.branch_angle + angle_var, rot,
                         apex.order + 1, apex.resource_v * 0.5f);
        make_branch_apex(tree, apex_idx, end_pos, grow_dir, apex.up,
                         -(params.branch_angle + angle_var), rot + PI,
                         apex.order + 1, apex.resource_v * 0.5f);
        break;
    }
    case GrowthArchetype::Whorled: {
        int n_whorl = std::max(params.whorl_count, 2);
        for (int w = 0; w < n_whorl; ++w) {
            float wrot = 2.0f * PI * static_cast<float>(w) / static_cast<float>(n_whorl);
            make_branch_apex(tree, apex_idx, end_pos, grow_dir, apex.up,
                             params.branch_angle, wrot,
                             apex.order + 1, apex.resource_v * 0.3f);
        }
        // Main axis continuation
        PlantModule main_c{};
        main_c.type = ModuleType::Apex;
        vec_copy(main_c.pos, end_pos);
        vec_copy(main_c.heading, grow_dir);
        frame_from_heading(main_c.left, main_c.up, grow_dir, apex.up);
        main_c.order = apex.order;
        main_c.resource_v = apex.resource_v;
        main_c.age = 0;
        add_child(tree, apex_idx, main_c);
        break;
    }
    }
}

// -----------------------------------------------------------------------
// Borchert-Honda resource model
// -----------------------------------------------------------------------

static void resource_basipetal(std::vector<PlantModule>& tree)
{
    for (auto& m : tree) m.light_q = 0.0f;
    for (int i = static_cast<int>(tree.size())-1; i >= 0; --i) {
        auto& m = tree[static_cast<size_t>(i)];
        if (m.type == ModuleType::Dead) continue;
        if (m.child_count == 0 && m.type != ModuleType::Dormant) m.light_q = 1.0f;
        if (m.parent >= 0) tree[static_cast<size_t>(m.parent)].light_q += m.light_q;
    }
}

static void resource_acropetal(std::vector<PlantModule>& tree, float lambda)
{
    for (auto& m : tree) { if (m.parent < 0) m.resource_v = 1.0f; }

    for (size_t i = 0; i < tree.size(); ++i) {
        auto& m = tree[i];
        if (m.type == ModuleType::Dead || m.child_count == 0) continue;

        int main_c = -1;
        float total_q = 0.0f;
        int live = 0;
        int c = m.first_child;
        while (c >= 0) {
            if (tree[static_cast<size_t>(c)].type != ModuleType::Dead) {
                total_q += tree[static_cast<size_t>(c)].light_q;
                ++live;
                if (main_c < 0 && tree[static_cast<size_t>(c)].order == m.order)
                    main_c = c;
            }
            c = tree[static_cast<size_t>(c)].next_sibling;
        }
        if (live == 0) continue;

        if (main_c >= 0 && live > 1) {
            float Q_main = tree[static_cast<size_t>(main_c)].light_q;
            float Q_lat = std::max(total_q - Q_main, 0.001f);
            float denom = lambda * Q_main + (1.0f - lambda) * Q_lat;
            float v_main_frac = (denom > 0.0001f) ? (lambda * Q_main) / denom : 0.5f;

            c = m.first_child;
            while (c >= 0) {
                if (tree[static_cast<size_t>(c)].type != ModuleType::Dead) {
                    if (c == main_c) {
                        tree[static_cast<size_t>(c)].resource_v = m.resource_v * v_main_frac;
                    } else {
                        float lat_share = (Q_lat > 0.001f)
                            ? tree[static_cast<size_t>(c)].light_q / Q_lat
                            : 1.0f / static_cast<float>(live - 1);
                        tree[static_cast<size_t>(c)].resource_v =
                            m.resource_v * (1.0f - v_main_frac) * lat_share;
                    }
                }
                c = tree[static_cast<size_t>(c)].next_sibling;
            }
        } else {
            float share = m.resource_v / static_cast<float>(live);
            c = m.first_child;
            while (c >= 0) {
                if (tree[static_cast<size_t>(c)].type != ModuleType::Dead)
                    tree[static_cast<size_t>(c)].resource_v = share;
                c = tree[static_cast<size_t>(c)].next_sibling;
            }
        }
    }
}

// -----------------------------------------------------------------------
// Pipe model widths
// -----------------------------------------------------------------------

static void compute_lplant_widths(std::vector<PlantModule>& tree, float taper_exp,
                                  float min_w, float max_w)
{
    for (auto& m : tree)
        m.width = (m.child_count == 0 && m.type != ModuleType::Dead) ? min_w : 0.0f;

    for (int i = static_cast<int>(tree.size())-1; i >= 0; --i) {
        auto& m = tree[static_cast<size_t>(i)];
        if (m.type == ModuleType::Dead) continue;
        if (m.parent >= 0) {
            auto& p = tree[static_cast<size_t>(m.parent)];
            p.width = std::pow(std::pow(p.width, taper_exp) + std::pow(m.width, taper_exp),
                               1.0f / taper_exp);
        }
    }
    for (auto& m : tree) m.width = std::min(m.width, max_w);
}

// -----------------------------------------------------------------------
// Mesh emission
// -----------------------------------------------------------------------

static void emit_lplant_branches(VegetationMesh& mesh,
                                 const std::vector<PlantModule>& tree,
                                 const float trunk_color[3])
{
    for (size_t i = 0; i < tree.size(); ++i) {
        const auto& m = tree[i];
        if (m.type != ModuleType::Internode || m.parent < 0) continue;
        const auto& p = tree[static_cast<size_t>(m.parent)];

        float dir[3] = { m.pos[0]-p.pos[0], m.pos[1]-p.pos[1], m.pos[2]-p.pos[2] };
        float len = vec_len(dir);
        if (len < 0.0001f) continue;
        dir[0]/=len; dir[1]/=len; dir[2]/=len;

        float ref[3] = {0.0f, 1.0f, 0.0f};
        if (std::abs(dir[1]) > 0.95f) { ref[0]=1; ref[1]=0; }

        float rx[3]; vec_cross(rx, ref, dir);
        float rl = vec_len(rx);
        if (rl < 0.0001f) continue;
        rx[0]/=rl; rx[1]/=rl; rx[2]/=rl;

        float ux[3]; vec_cross(ux, dir, rx);

        float w0 = p.width, w1 = m.width;
        float dark = (m.order == 0) ? 1.0f : 0.8f;
        float bc[3] = { trunk_color[0]*dark, trunk_color[1]*dark, trunk_color[2]*dark };

        for (int q = 0; q < 2; ++q) {
            float* px = (q==0) ? rx : ux;
            auto si = static_cast<uint32_t>(mesh.vertices.size());
            float corners[4][3] = {
                {p.pos[0]-px[0]*w0, p.pos[1]-px[1]*w0, p.pos[2]-px[2]*w0},
                {p.pos[0]+px[0]*w0, p.pos[1]+px[1]*w0, p.pos[2]+px[2]*w0},
                {m.pos[0]+px[0]*w1, m.pos[1]+px[1]*w1, m.pos[2]+px[2]*w1},
                {m.pos[0]-px[0]*w1, m.pos[1]-px[1]*w1, m.pos[2]-px[2]*w1},
            };
            float* n = (q==0) ? ux : rx;
            for (auto& c : corners) {
                VegetationVertex v{};
                v.position[0]=c[0]; v.position[1]=c[1]; v.position[2]=c[2];
                v.normal[0]=n[0]; v.normal[1]=n[1]; v.normal[2]=n[2];
                v.color[0]=bc[0]; v.color[1]=bc[1]; v.color[2]=bc[2];
                mesh.vertices.push_back(v);
            }
            mesh.indices.push_back(si); mesh.indices.push_back(si+1); mesh.indices.push_back(si+2);
            mesh.indices.push_back(si); mesh.indices.push_back(si+2); mesh.indices.push_back(si+3);
            mesh.indices.push_back(si); mesh.indices.push_back(si+2); mesh.indices.push_back(si+1);
            mesh.indices.push_back(si); mesh.indices.push_back(si+3); mesh.indices.push_back(si+2);
        }
    }
}

static void emit_lplant_leaves(VegetationMesh& mesh,
                               const std::vector<PlantModule>& tree,
                               const LPlantParams& params,
                               uint32_t seed)
{
    constexpr float golden_angle = 2.39996323f;
    if (params.leaf_count <= 0) return;

    std::vector<size_t> tip_sites, interior_sites;
    int max_order = 0;
    for (auto& m : tree) if (m.order > max_order) max_order = m.order;
    int order_thresh = max_order / 2;

    for (size_t i = 0; i < tree.size(); ++i) {
        if (tree[i].type == ModuleType::Dead || tree[i].type == ModuleType::Dormant) continue;
        if (tree[i].child_count == 0) tip_sites.push_back(i);
        else if (tree[i].order >= order_thresh) interior_sites.push_back(i);
    }
    if (tip_sites.empty() && interior_sites.empty()) return;

    int n_tip = static_cast<int>(std::round(
        static_cast<float>(params.leaf_count) * params.tip_leaf_bias));
    int n_interior = params.leaf_count - n_tip;

    float max_y = 0.01f;
    for (auto& m : tree) if (m.pos[1] > max_y) max_y = m.pos[1];

    auto emit_at = [&](const std::vector<size_t>& sites, int total, int offset) {
        if (sites.empty() || total <= 0) return;
        int per_site = std::max(total / static_cast<int>(sites.size()), 1);
        int remaining = total, gi = offset;

        for (size_t si = 0; si < sites.size() && remaining > 0; ++si) {
            const auto& node = tree[sites[si]];
            int count = std::min(per_site, remaining);
            if (si == sites.size()-1) count = remaining;

            for (int li = 0; li < count; ++li) {
                auto ui = static_cast<uint32_t>(gi);
                float order_frac = static_cast<float>(node.order) / static_cast<float>(std::max(max_order,1));
                float size_scale = 1.0f - 0.4f * order_frac;

                float leaf_az = static_cast<float>(gi) * golden_angle;
                float spread = params.crown_radius * 0.1f;
                float jitter_r = spread * (0.3f+0.7f*lhash_float(seed, ui*4u+200u));

                float lx = node.pos[0] + std::cos(leaf_az)*jitter_r;
                float lz = node.pos[2] + std::sin(leaf_az)*jitter_r;
                float ly = node.pos[1] + (lhash_float(seed, ui*4u+201u)-0.3f)*0.05f;

                float out_x=lx-node.pos[0], out_y=0.3f, out_z=lz-node.pos[2];
                out_y -= params.leaf_droop*0.5f;
                float olen = std::sqrt(out_x*out_x+out_y*out_y+out_z*out_z);
                if (olen<0.001f) { out_x=0; out_y=1; out_z=0; olen=1; }
                out_x/=olen; out_y/=olen; out_z/=olen;

                float ref_x=0,ref_y=1,ref_z=0;
                if (std::abs(out_y)>0.95f) { ref_x=1; ref_y=0; }
                float rrx=ref_y*out_z-ref_z*out_y, rry=ref_z*out_x-ref_x*out_z, rrz=ref_x*out_y-ref_y*out_x;
                float rlen=std::sqrt(rrx*rrx+rry*rry+rrz*rrz);
                if (rlen<0.001f) { ++gi; --remaining; continue; }
                rrx/=rlen; rry/=rlen; rrz/=rlen;
                float uux=out_y*rrz-out_z*rry, uuy=out_z*rrx-out_x*rrz, uuz=out_x*rry-out_y*rrx;

                float twist = lhash_float(seed, ui*4u+202u)*PI;
                float ct=std::cos(twist), st=std::sin(twist);
                float tr_x=rrx*ct+uux*st, tr_y=rry*ct+uuy*st, tr_z=rrz*ct+uuz*st;
                float tu_x=-rrx*st+uux*ct, tu_y=-rry*st+uuy*ct, tu_z=-rrz*st+uuz*ct;

                float hl = params.leaf_length*0.5f*size_scale;
                float hw = params.leaf_width*0.5f*size_scale;
                float base_ht = std::clamp(ly/max_y, 0.0f, 1.0f);
                auto vi = static_cast<uint32_t>(mesh.vertices.size());

                struct LP { float dx_r,dx_u,ht; };
                LP pts[4]={{0,-hl,0},{hw,0,0.5f},{0,hl,1},{-hw,0,0.5f}};
                float cv = 0.9f+0.2f*lhash_float(seed, ui*4u+203u);
                for (auto& pt : pts) {
                    VegetationVertex v{};
                    v.position[0]=lx+tr_x*pt.dx_r+tu_x*pt.dx_u;
                    v.position[1]=ly+tr_y*pt.dx_r+tu_y*pt.dx_u;
                    v.position[2]=lz+tr_z*pt.dx_r+tu_z*pt.dx_u;
                    v.normal[0]=out_x; v.normal[1]=out_y; v.normal[2]=out_z;
                    v.color[0]=params.base_color[0]*cv; v.color[1]=params.base_color[1]*cv; v.color[2]=params.base_color[2]*cv;
                    v.height_t = base_ht + pt.ht*(1-base_ht)*0.3f;
                    mesh.vertices.push_back(v);
                }
                mesh.indices.push_back(vi); mesh.indices.push_back(vi+1); mesh.indices.push_back(vi+2);
                mesh.indices.push_back(vi); mesh.indices.push_back(vi+2); mesh.indices.push_back(vi+3);
                mesh.indices.push_back(vi); mesh.indices.push_back(vi+2); mesh.indices.push_back(vi+1);
                mesh.indices.push_back(vi); mesh.indices.push_back(vi+3); mesh.indices.push_back(vi+2);
                ++gi; --remaining;
            }
        }
    };
    emit_at(tip_sites, n_tip, 0);
    emit_at(interior_sites, n_interior, n_tip);
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

LPlantParams evaluate_lplant_expression(const LPlantParams& base,
                                        const LPlantExpression& expr,
                                        float moisture)
{
    float t = std::clamp(moisture, 0.0f, 1.0f);
    LPlantParams out = base;

    if (expr.total_height.enabled)     out.total_height = llerp(expr.total_height.low, expr.total_height.high, t);
    if (expr.trunk_height.enabled)     out.trunk_height = llerp(expr.trunk_height.low, expr.trunk_height.high, t);
    if (expr.crown_radius.enabled)     out.crown_radius = llerp(expr.crown_radius.low, expr.crown_radius.high, t);
    if (expr.crown_height.enabled)     out.crown_height = llerp(expr.crown_height.low, expr.crown_height.high, t);
    if (expr.trunk_width.enabled)      out.trunk_width = llerp(expr.trunk_width.low, expr.trunk_width.high, t);
    if (expr.growth_steps.enabled)
        out.growth_steps = std::clamp(static_cast<int>(std::round(llerp(expr.growth_steps.low, expr.growth_steps.high, t))), 4, 25);
    if (expr.branch_angle.enabled)     out.branch_angle = llerp(expr.branch_angle.low, expr.branch_angle.high, t);
    if (expr.internode_length.enabled) out.internode_length = llerp(expr.internode_length.low, expr.internode_length.high, t);
    if (expr.length_decay.enabled)     out.length_decay = llerp(expr.length_decay.low, expr.length_decay.high, t);
    if (expr.lambda.enabled)           out.lambda = llerp(expr.lambda.low, expr.lambda.high, t);
    if (expr.attractor_count.enabled)
        out.attractor_count = std::clamp(static_cast<int>(std::round(llerp(expr.attractor_count.low, expr.attractor_count.high, t))), 100, 3000);
    if (expr.tropism.enabled)          out.tropism = llerp(expr.tropism.low, expr.tropism.high, t);
    if (expr.leaf_count.enabled)
        out.leaf_count = std::clamp(static_cast<int>(std::round(llerp(expr.leaf_count.low, expr.leaf_count.high, t))), 1, 2000);
    if (expr.leaf_length.enabled)      out.leaf_length = llerp(expr.leaf_length.low, expr.leaf_length.high, t);
    if (expr.leaf_width.enabled)       out.leaf_width = llerp(expr.leaf_width.low, expr.leaf_width.high, t);
    if (expr.leaf_droop.enabled)       out.leaf_droop = llerp(expr.leaf_droop.low, expr.leaf_droop.high, t);
    if (expr.branch_gravity.enabled)   out.branch_gravity = llerp(expr.branch_gravity.low, expr.branch_gravity.high, t);
    if (expr.branch_wobble.enabled)    out.branch_wobble = llerp(expr.branch_wobble.low, expr.branch_wobble.high, t);

    if (expr.vary_color) {
        for (int i = 0; i < 3; ++i)
            out.base_color[i] = llerp(expr.dry_color[i], expr.wet_color[i], t);
    }
    return out;
}

VegetationMesh generate_lplant(const LPlantParams& params, uint32_t seed,
                               bool include_ground,
                               float offset_x, float offset_z)
{
    VegetationMesh mesh;

    // Derive internode length from total_height if not explicitly set
    float inode_base = params.internode_length;
    if (params.total_height > 0.01f) {
        float crown_extent = params.total_height - params.trunk_height;
        if (crown_extent > 0.1f)
            inode_base = crown_extent / static_cast<float>(std::max(params.growth_steps, 4));
    }

    // Use a working copy with the derived internode length
    LPlantParams wp = params;
    wp.internode_length = inode_base;

    float R = wp.crown_radius;
    float H = wp.crown_height;
    float D = std::max(std::max(H, R*2.0f) / 20.0f, 0.01f);
    float d_k = D * wp.kill_ratio;
    float d_i = D * wp.influence_ratio;
    float crown_base_y = wp.trunk_height;

    std::vector<LAttractor> attractors;
    seed_attractors(attractors, wp.attractor_count, wp.envelope_shape,
                    R, H, wp.surface_bias, offset_x, crown_base_y, offset_z, seed);

    std::vector<PlantModule> tree;

    // Build trunk with lateral buds at phyllotaxis intervals
    int trunk_steps = std::max(static_cast<int>(wp.trunk_height / std::max(inode_base, 0.01f)), 1);
    float trunk_step_len = wp.trunk_height / static_cast<float>(trunk_steps);
    float phyl_rad = wp.phyllotaxis_angle * PI / 180.0f;

    {
        PlantModule root{};
        root.type = ModuleType::Internode;
        root.pos[0]=offset_x; root.pos[1]=0; root.pos[2]=offset_z;
        root.heading[0]=0; root.heading[1]=1; root.heading[2]=0;
        root.left[0]=1; root.left[1]=0; root.left[2]=0;
        root.up[0]=0; root.up[1]=0; root.up[2]=1;
        root.length = trunk_step_len;
        root.width = wp.trunk_width;
        root.order = 0;
        root.resource_v = 1.0f;
        tree.push_back(root);

        for (int i = 1; i < trunk_steps; ++i) {
            float y = static_cast<float>(i) * trunk_step_len;
            PlantModule seg{};
            seg.type = ModuleType::Internode;
            seg.pos[0]=offset_x; seg.pos[1]=y; seg.pos[2]=offset_z;
            seg.heading[0]=0; seg.heading[1]=1; seg.heading[2]=0;
            seg.left[0]=1; seg.left[1]=0; seg.left[2]=0;
            seg.up[0]=0; seg.up[1]=0; seg.up[2]=1;
            seg.length = trunk_step_len;
            seg.width = wp.trunk_width * (1.0f - 0.3f*static_cast<float>(i)/static_cast<float>(trunk_steps));
            seg.order = 0;
            seg.resource_v = 1.0f;
            int seg_idx = add_child(tree, i-1, seg);

            // Add lateral bud on upper portion of trunk
            float height_frac = static_cast<float>(i) / static_cast<float>(trunk_steps);
            if (height_frac > 0.4f && i % 2 == 0) {
                float rot = phyl_rad * static_cast<float>(i);
                float lat_h[3];
                float trunk_dir[3] = {0,1,0};
                float trunk_up[3] = {0,0,1};
                vec_rotate(lat_h, trunk_dir, trunk_up, wp.branch_angle * 1.2f);
                float lat_r[3];
                vec_rotate(lat_r, lat_h, trunk_dir, rot);
                vec_normalize(lat_r);

                PlantModule bud{};
                bud.type = ModuleType::Apex;
                bud.pos[0]=offset_x; bud.pos[1]=y; bud.pos[2]=offset_z;
                vec_copy(bud.heading, lat_r);
                frame_from_heading(bud.left, bud.up, bud.heading, trunk_up);
                bud.order = 1;
                bud.resource_v = 0.4f;
                bud.age = 0;
                add_child(tree, seg_idx, bud);
            }
        }

        // Crown apex(es) at trunk top
        int trunk_top = -1;
        for (int i = static_cast<int>(tree.size())-1; i >= 0; --i) {
            if (tree[static_cast<size_t>(i)].order == 0 &&
                tree[static_cast<size_t>(i)].type == ModuleType::Internode) {
                trunk_top = i; break;
            }
        }
        if (trunk_top < 0) trunk_top = 0;

        int n_initial = (wp.archetype == GrowthArchetype::MultistemBush) ? 3 : 1;
        for (int a = 0; a < n_initial; ++a) {
            PlantModule apex{};
            apex.type = ModuleType::Apex;
            apex.pos[0]=offset_x; apex.pos[1]=crown_base_y; apex.pos[2]=offset_z;
            apex.heading[0]=0; apex.heading[1]=1; apex.heading[2]=0;
            apex.left[0]=1; apex.left[1]=0; apex.left[2]=0;
            apex.up[0]=0; apex.up[1]=0; apex.up[2]=1;
            apex.order = 0;
            apex.resource_v = 1.0f;
            apex.age = 0;
            if (n_initial > 1) {
                float sp_ang = 2.0f*PI*static_cast<float>(a)/static_cast<float>(n_initial);
                apex.heading[0] = std::sin(sp_ang)*0.3f;
                apex.heading[1] = 0.9f;
                apex.heading[2] = std::cos(sp_ang)*0.3f;
                vec_normalize(apex.heading);
                frame_from_heading(apex.left, apex.up, apex.heading, tree[0].up);
            }
            add_child(tree, trunk_top, apex);
        }
    }

    // Growth loop
    for (int step = 0; step < wp.growth_steps; ++step) {
        std::vector<int> apices;
        for (size_t i = 0; i < tree.size(); ++i)
            if (tree[i].type == ModuleType::Apex) apices.push_back(static_cast<int>(i));

        for (int ai : apices) {
            if (tree[static_cast<size_t>(ai)].type != ModuleType::Apex) continue;
            apply_growth_rule(tree, ai, wp, attractors, d_k, d_i, seed, step);
        }

        resource_basipetal(tree);
        resource_acropetal(tree, wp.lambda);
    }

    // Pipe model widths (preserve trunk)
    compute_lplant_widths(tree, wp.branch_taper, wp.branch_width_min, wp.branch_width_max);
    for (int i = 0; i < trunk_steps && i < static_cast<int>(tree.size()); ++i) {
        if (tree[static_cast<size_t>(i)].order == 0)
            tree[static_cast<size_t>(i)].width = wp.trunk_width *
                (1.0f - 0.3f * static_cast<float>(i) / static_cast<float>(trunk_steps));
    }

    // Emit geometry
    emit_lplant_branches(mesh, tree, wp.trunk_color);
    emit_lplant_leaves(mesh, tree, wp, seed);

    if (include_ground) {
        float gs = std::max(wp.crown_radius*2.0f, 1.0f) + 0.5f;
        auto gi = static_cast<uint32_t>(mesh.vertices.size());
        const float gc[3]={0.25f,0.20f,0.15f};
        float corners[4][3]={
            {offset_x-gs,0,offset_z-gs},{offset_x+gs,0,offset_z-gs},
            {offset_x+gs,0,offset_z+gs},{offset_x-gs,0,offset_z+gs}};
        for (auto& c : corners) {
            VegetationVertex v{};
            v.position[0]=c[0]; v.position[1]=c[1]; v.position[2]=c[2];
            v.normal[0]=0; v.normal[1]=1; v.normal[2]=0;
            v.color[0]=gc[0]; v.color[1]=gc[1]; v.color[2]=gc[2];
            mesh.vertices.push_back(v);
        }
        mesh.indices.push_back(gi); mesh.indices.push_back(gi+1); mesh.indices.push_back(gi+2);
        mesh.indices.push_back(gi); mesh.indices.push_back(gi+2); mesh.indices.push_back(gi+3);
    }
    return mesh;
}

VegetationMesh generate_lplant_field(const LPlantParams& base,
                                     const LPlantExpression& expr,
                                     const FieldParams& field,
                                     uint32_t base_seed)
{
    VegetationMesh mesh;
    float extent = static_cast<float>(field.grid_n-1) * field.spacing;

    for (int iz = 0; iz < field.grid_n; ++iz) {
        for (int ix = 0; ix < field.grid_n; ++ix) {
            float x = static_cast<float>(ix)*field.spacing - extent*0.5f;
            float z = static_cast<float>(iz)*field.spacing - extent*0.5f;
            float moisture = (field.grid_n>1) ? static_cast<float>(ix)/static_cast<float>(field.grid_n-1) : 0.5f;
            LPlantParams resolved = evaluate_lplant_expression(base, expr, moisture);
            uint32_t seed = base_seed + static_cast<uint32_t>(iz*field.grid_n + ix);
            auto cell = generate_lplant(resolved, seed, false, x, z);
            auto vo = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(), cell.vertices.begin(), cell.vertices.end());
            for (uint32_t idx : cell.indices) mesh.indices.push_back(idx + vo);
        }
    }

    float half = extent*0.5f + field.spacing;
    auto gi = static_cast<uint32_t>(mesh.vertices.size());
    const float gc[3]={0.25f,0.20f,0.15f};
    float corners[4][3]={{-half,0,-half},{half,0,-half},{half,0,half},{-half,0,half}};
    for (auto& c : corners) {
        VegetationVertex v{};
        v.position[0]=c[0]; v.position[1]=c[1]; v.position[2]=c[2];
        v.normal[0]=0; v.normal[1]=1; v.normal[2]=0;
        v.color[0]=gc[0]; v.color[1]=gc[1]; v.color[2]=gc[2];
        mesh.vertices.push_back(v);
    }
    mesh.indices.push_back(gi); mesh.indices.push_back(gi+1); mesh.indices.push_back(gi+2);
    mesh.indices.push_back(gi); mesh.indices.push_back(gi+2); mesh.indices.push_back(gi+3);
    return mesh;
}

} // namespace bestiary

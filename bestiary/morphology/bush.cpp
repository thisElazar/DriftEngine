#include "bush.h"
#include <algorithm>
#include <cmath>

namespace bestiary {

static uint32_t bhash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

static float bhash_float(uint32_t seed, uint32_t index)
{
    return static_cast<float>(bhash(seed ^ (index * 2654435761u))) /
           static_cast<float>(0xFFFFFFFFu);
}

static float blerp(float a, float b, float t) { return a + (b - a) * t; }

BushParams evaluate_bush_expression(const BushParams& base,
                                    const BushExpression& expr,
                                    float moisture)
{
    float t = std::clamp(moisture, 0.0f, 1.0f);
    BushParams out = base;

    if (expr.leaf_count.enabled)
        out.leaf_count = std::clamp(static_cast<int>(std::round(blerp(expr.leaf_count.low, expr.leaf_count.high, t))), 1, 80);
    if (expr.leaf_length.enabled)
        out.leaf_length = blerp(expr.leaf_length.low, expr.leaf_length.high, t);
    if (expr.leaf_width.enabled)
        out.leaf_width = blerp(expr.leaf_width.low, expr.leaf_width.high, t);
    if (expr.bush_height.enabled)
        out.bush_height = blerp(expr.bush_height.low, expr.bush_height.high, t);
    if (expr.bush_radius.enabled)
        out.bush_radius = blerp(expr.bush_radius.low, expr.bush_radius.high, t);
    if (expr.stem_height.enabled)
        out.stem_height = blerp(expr.stem_height.low, expr.stem_height.high, t);

    if (expr.vary_color) {
        for (int i = 0; i < 3; ++i)
            out.base_color[i] = blerp(expr.dry_color[i], expr.wet_color[i], t);
    }

    return out;
}

ClumpMesh generate_bush(const BushParams& params, uint32_t seed,
                        bool include_ground,
                        float offset_x, float offset_z)
{
    ClumpMesh mesh;

    constexpr float pi = 3.14159265f;

    // Stem: two crossed quads forming an X
    if (params.stem_height > 0.001f) {
        float sw = 0.012f;
        float sh = params.stem_height;
        const float sc[3] = {0.35f, 0.25f, 0.15f};

        for (int q = 0; q < 2; ++q) {
            float dx = (q == 0) ? sw : 0.0f;
            float dz = (q == 0) ? 0.0f : sw;

            auto si = static_cast<uint32_t>(mesh.vertices.size());

            float corners[4][3] = {
                {offset_x - dx, 0.0f, offset_z - dz},
                {offset_x + dx, 0.0f, offset_z + dz},
                {offset_x + dx, sh,   offset_z + dz},
                {offset_x - dx, sh,   offset_z - dz},
            };
            float nx = (q == 0) ? 0.0f : 1.0f;
            float nz = (q == 0) ? 1.0f : 0.0f;

            for (auto& c : corners) {
                ClumpVertex v{};
                v.position[0] = c[0]; v.position[1] = c[1]; v.position[2] = c[2];
                v.normal[0] = nx; v.normal[1] = 0.0f; v.normal[2] = nz;
                v.color[0] = sc[0]; v.color[1] = sc[1]; v.color[2] = sc[2];
                mesh.vertices.push_back(v);
            }
            mesh.indices.push_back(si);     mesh.indices.push_back(si + 1); mesh.indices.push_back(si + 2);
            mesh.indices.push_back(si);     mesh.indices.push_back(si + 2); mesh.indices.push_back(si + 3);
            mesh.indices.push_back(si);     mesh.indices.push_back(si + 2); mesh.indices.push_back(si + 1);
            mesh.indices.push_back(si);     mesh.indices.push_back(si + 3); mesh.indices.push_back(si + 2);
        }
    }

    constexpr float golden_angle = 2.39996323f;
    constexpr int n_branches = 5;
    const float branch_width = 0.008f;
    const float bc[3] = {0.35f, 0.25f, 0.15f};
    constexpr int branch_segs = 3;

    float stem_top_y = params.stem_height;
    int leaves_per_branch = std::max(params.leaf_count / n_branches, 1);

    for (int br = 0; br < n_branches; ++br) {
        float br_azimuth = static_cast<float>(br) * (2.0f * pi / static_cast<float>(n_branches))
                         + bhash_float(seed, static_cast<uint32_t>(br) * 2u + 100u) * 0.5f;
        float br_elev = 0.4f + 0.35f * bhash_float(seed, static_cast<uint32_t>(br) * 2u + 101u);

        float br_dir_x = std::cos(br_azimuth) * std::cos(br_elev);
        float br_dir_y = std::sin(br_elev);
        float br_dir_z = std::sin(br_azimuth) * std::cos(br_elev);

        float br_len = params.bush_height * 0.7f + params.bush_radius * 0.5f;

        // Branch tangent frame for crossed-quad rendering
        float br_right_x = -std::sin(br_azimuth);
        float br_right_z =  std::cos(br_azimuth);
        float br_up_x = -br_dir_y * std::cos(br_azimuth);
        float br_up_y =  std::cos(br_elev);
        float br_up_z = -br_dir_y * std::sin(br_azimuth);
        float br_up_len = std::sqrt(br_up_x*br_up_x + br_up_y*br_up_y + br_up_z*br_up_z);
        if (br_up_len > 0.001f) { br_up_x /= br_up_len; br_up_y /= br_up_len; br_up_z /= br_up_len; }

        // Render branch as crossed quads along segments
        for (int q = 0; q < 2; ++q) {
            float perp_x = (q == 0) ? br_right_x : br_up_x;
            float perp_y = (q == 0) ? 0.0f        : br_up_y;
            float perp_z = (q == 0) ? br_right_z  : br_up_z;

            for (int seg = 0; seg < branch_segs; ++seg) {
                float t0 = static_cast<float>(seg)     / static_cast<float>(branch_segs);
                float t1 = static_cast<float>(seg + 1) / static_cast<float>(branch_segs);
                float w0 = branch_width * (1.0f - t0 * 0.6f);
                float w1 = branch_width * (1.0f - t1 * 0.6f);

                float p0x = offset_x + br_dir_x * br_len * t0;
                float p0y = stem_top_y + br_dir_y * br_len * t0;
                float p0z = offset_z + br_dir_z * br_len * t0;
                float p1x = offset_x + br_dir_x * br_len * t1;
                float p1y = stem_top_y + br_dir_y * br_len * t1;
                float p1z = offset_z + br_dir_z * br_len * t1;

                auto si = static_cast<uint32_t>(mesh.vertices.size());
                float seg_corners[4][3] = {
                    {p0x - perp_x*w0, p0y - perp_y*w0, p0z - perp_z*w0},
                    {p0x + perp_x*w0, p0y + perp_y*w0, p0z + perp_z*w0},
                    {p1x + perp_x*w1, p1y + perp_y*w1, p1z + perp_z*w1},
                    {p1x - perp_x*w1, p1y - perp_y*w1, p1z - perp_z*w1},
                };
                float nx = (q == 0) ? br_up_x : br_right_x;
                float ny = (q == 0) ? br_up_y : 0.0f;
                float nz = (q == 0) ? br_up_z : br_right_z;

                for (auto& c : seg_corners) {
                    ClumpVertex v{};
                    v.position[0] = c[0]; v.position[1] = c[1]; v.position[2] = c[2];
                    v.normal[0] = nx; v.normal[1] = ny; v.normal[2] = nz;
                    v.color[0] = bc[0]; v.color[1] = bc[1]; v.color[2] = bc[2];
                    mesh.vertices.push_back(v);
                }
                mesh.indices.push_back(si); mesh.indices.push_back(si+1); mesh.indices.push_back(si+2);
                mesh.indices.push_back(si); mesh.indices.push_back(si+2); mesh.indices.push_back(si+3);
                mesh.indices.push_back(si); mesh.indices.push_back(si+2); mesh.indices.push_back(si+1);
                mesh.indices.push_back(si); mesh.indices.push_back(si+3); mesh.indices.push_back(si+2);
            }
        }

        // Place leaves along this branch using golden angle
        for (int li = 0; li < leaves_per_branch; ++li) {
            int global_i = br * leaves_per_branch + li;
            auto ui = static_cast<uint32_t>(global_i);

            float t_along = 0.2f + 0.8f * static_cast<float>(li) / static_cast<float>(std::max(leaves_per_branch - 1, 1));

            float lx = offset_x + br_dir_x * br_len * t_along;
            float ly = stem_top_y + br_dir_y * br_len * t_along;
            float lz = offset_z + br_dir_z * br_len * t_along;

            float leaf_azimuth = static_cast<float>(global_i) * golden_angle;
            float spread = params.bush_radius * 0.3f * t_along;
            float jitter_r = spread * (0.3f + 0.7f * bhash_float(seed, ui * 4u + 200u));

            lx += std::cos(leaf_azimuth) * jitter_r;
            lz += std::sin(leaf_azimuth) * jitter_r;
            ly += (bhash_float(seed, ui * 4u + 201u) - 0.3f) * params.bush_height * 0.15f;

            // Leaf faces outward from branch axis
            float out_x = lx - offset_x;
            float out_y = ly - stem_top_y;
            float out_z = lz - offset_z;
            float olen = std::sqrt(out_x*out_x + out_y*out_y + out_z*out_z);
            if (olen < 0.001f) olen = 1.0f;
            float ox = out_x / olen;
            float oy = out_y / olen;
            float oz = out_z / olen;

            float ref_x = 0.0f, ref_y = 1.0f, ref_z = 0.0f;
            if (std::abs(oy) > 0.95f) { ref_x = 1.0f; ref_y = 0.0f; }

            float rrx = ref_y * oz - ref_z * oy;
            float rry = ref_z * ox - ref_x * oz;
            float rrz = ref_x * oy - ref_y * ox;
            float rlen = std::sqrt(rrx*rrx + rry*rry + rrz*rrz);
            rrx /= rlen; rry /= rlen; rrz /= rlen;

            float uux = oy * rrz - oz * rry;
            float uuy = oz * rrx - ox * rrz;
            float uuz = ox * rry - oy * rrx;

            float twist = bhash_float(seed, ui * 4u + 202u) * pi;
            float cos_t = std::cos(twist);
            float sin_t = std::sin(twist);
            float tr_x = rrx * cos_t + uux * sin_t;
            float tr_y = rry * cos_t + uuy * sin_t;
            float tr_z = rrz * cos_t + uuz * sin_t;
            float tu_x = -rrx * sin_t + uux * cos_t;
            float tu_y = -rry * sin_t + uuy * cos_t;
            float tu_z = -rrz * sin_t + uuz * cos_t;

            float size_scale = 1.0f - 0.3f * t_along;
            float hl = params.leaf_length * 0.5f * size_scale;
            float hw = params.leaf_width * 0.5f * size_scale;

            float height_frac = (ly - params.stem_height) / std::max(params.bush_height, 0.01f);
            float base_ht = std::clamp(height_frac, 0.0f, 1.0f);

            auto vi_base = static_cast<uint32_t>(mesh.vertices.size());

            struct LeafPt { float dx_r, dx_u, ht; };
            LeafPt pts[4] = {
                { 0.0f, -hl, 0.0f},
                {  hw,  0.0f, 0.5f},
                { 0.0f,  hl, 1.0f},
                { -hw,  0.0f, 0.5f},
            };

            float color_var = 0.9f + 0.2f * bhash_float(seed, ui * 4u + 203u);

            for (auto& p : pts) {
                ClumpVertex v{};
                v.position[0] = lx + tr_x * p.dx_r + tu_x * p.dx_u;
                v.position[1] = ly + tr_y * p.dx_r + tu_y * p.dx_u;
                v.position[2] = lz + tr_z * p.dx_r + tu_z * p.dx_u;
                v.normal[0] = ox; v.normal[1] = oy; v.normal[2] = oz;
                v.color[0] = params.base_color[0] * color_var;
                v.color[1] = params.base_color[1] * color_var;
                v.color[2] = params.base_color[2] * color_var;
                v.height_t = base_ht + p.ht * (1.0f - base_ht) * 0.3f;
                mesh.vertices.push_back(v);
            }

            mesh.indices.push_back(vi_base + 0); mesh.indices.push_back(vi_base + 1); mesh.indices.push_back(vi_base + 2);
            mesh.indices.push_back(vi_base + 0); mesh.indices.push_back(vi_base + 2); mesh.indices.push_back(vi_base + 3);
            mesh.indices.push_back(vi_base + 0); mesh.indices.push_back(vi_base + 2); mesh.indices.push_back(vi_base + 1);
            mesh.indices.push_back(vi_base + 0); mesh.indices.push_back(vi_base + 3); mesh.indices.push_back(vi_base + 2);
        }
    }

    if (include_ground) {
        float gs = std::max(params.bush_radius * 2.0f, 0.2f) + 0.1f;
        auto gi = static_cast<uint32_t>(mesh.vertices.size());

        const float gc[3] = {0.25f, 0.20f, 0.15f};
        float corners[4][3] = {
            {offset_x - gs, 0.0f, offset_z - gs},
            {offset_x + gs, 0.0f, offset_z - gs},
            {offset_x + gs, 0.0f, offset_z + gs},
            {offset_x - gs, 0.0f, offset_z + gs}
        };
        for (auto& c : corners) {
            ClumpVertex v{};
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

ClumpMesh generate_bush_field(const BushParams& base,
                              const BushExpression& expr,
                              const FieldParams& field,
                              uint32_t base_seed)
{
    ClumpMesh mesh;
    float extent = static_cast<float>(field.grid_n - 1) * field.spacing;

    for (int iz = 0; iz < field.grid_n; ++iz) {
        for (int ix = 0; ix < field.grid_n; ++ix) {
            float x = static_cast<float>(ix) * field.spacing - extent * 0.5f;
            float z = static_cast<float>(iz) * field.spacing - extent * 0.5f;
            float moisture = (field.grid_n > 1)
                ? static_cast<float>(ix) / static_cast<float>(field.grid_n - 1)
                : 0.5f;

            BushParams resolved = evaluate_bush_expression(base, expr, moisture);
            uint32_t seed = base_seed + static_cast<uint32_t>(iz * field.grid_n + ix);

            auto cell = generate_bush(resolved, seed, false, x, z);

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
        ClumpVertex v{};
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

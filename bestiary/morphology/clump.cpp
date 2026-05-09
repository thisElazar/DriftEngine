#include "clump.h"
#include <algorithm>
#include <cmath>

namespace bestiary {

static uint32_t hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

static float hash_float(uint32_t seed, uint32_t index)
{
    return static_cast<float>(hash(seed ^ (index * 2654435761u))) /
           static_cast<float>(0xFFFFFFFFu);
}

static float lerp(float a, float b, float t) { return a + (b - a) * t; }

ClumpParams evaluate_expression(const ClumpParams& base,
                                const ClumpExpression& expr,
                                float moisture)
{
    float t = std::clamp(moisture, 0.0f, 1.0f);
    ClumpParams out = base;

    if (expr.blade_count.enabled)
        out.blade_count = std::clamp(static_cast<int>(std::round(lerp(expr.blade_count.low, expr.blade_count.high, t))), 1, 60);
    if (expr.blade_height.enabled)
        out.blade_height = lerp(expr.blade_height.low, expr.blade_height.high, t);
    if (expr.blade_width.enabled)
        out.blade_width = lerp(expr.blade_width.low, expr.blade_width.high, t);
    if (expr.splay_angle.enabled)
        out.splay_angle = lerp(expr.splay_angle.low, expr.splay_angle.high, t);
    if (expr.clump_radius.enabled)
        out.clump_radius = lerp(expr.clump_radius.low, expr.clump_radius.high, t);

    if (expr.vary_color) {
        for (int i = 0; i < 3; ++i)
            out.base_color[i] = lerp(expr.dry_color[i], expr.wet_color[i], t);
    }

    return out;
}

ClumpMesh generate_clump(const ClumpParams& params, uint32_t seed,
                         bool include_ground,
                         float offset_x, float offset_z)
{
    ClumpMesh mesh;

    constexpr int   segments        = 5;
    constexpr float tip_width_ratio = 0.1f;
    constexpr float pi              = 3.14159265f;

    for (int b = 0; b < params.blade_count; ++b) {
        float angle  = hash_float(seed, static_cast<uint32_t>(b) * 4u + 0u) * 2.0f * pi;
        float r      = params.clump_radius * std::sqrt(hash_float(seed, static_cast<uint32_t>(b) * 4u + 1u));
        float base_x = r * std::cos(angle) + offset_x;
        float base_z = r * std::sin(angle) + offset_z;

        float jitter     = 0.6f + 0.4f * hash_float(seed, static_cast<uint32_t>(b) * 4u + 2u);
        float tilt       = params.splay_angle * jitter * pi / 180.0f;

        float rel_x = r * std::cos(angle);
        float rel_z = r * std::sin(angle);
        float rad_len = std::sqrt(rel_x * rel_x + rel_z * rel_z);
        float rad_x, rad_z;
        if (rad_len > 0.001f) {
            rad_x = rel_x / rad_len;
            rad_z = rel_z / rad_len;
        } else {
            float fallback = hash_float(seed, static_cast<uint32_t>(b) * 4u + 3u) * 2.0f * pi;
            rad_x = std::cos(fallback);
            rad_z = std::sin(fallback);
        }

        float blade_up_x = std::sin(tilt) * rad_x;
        float blade_up_y = std::cos(tilt);
        float blade_up_z = std::sin(tilt) * rad_z;

        float tan_x = -rad_z;
        float tan_z =  rad_x;

        float norm_x = -std::cos(tilt) * rad_x;
        float norm_y =  std::sin(tilt);
        float norm_z = -std::cos(tilt) * rad_z;

        auto base_vert = static_cast<uint32_t>(mesh.vertices.size());

        for (int s = 0; s <= segments; ++s) {
            float t      = static_cast<float>(s) / static_cast<float>(segments);
            float height = t * params.blade_height;
            float width  = params.blade_width * (1.0f - t * (1.0f - tip_width_ratio));
            float hw     = width * 0.5f;

            float cx = base_x + blade_up_x * height;
            float cy =          blade_up_y * height;
            float cz = base_z + blade_up_z * height;

            ClumpVertex left{};
            left.position[0] = cx - tan_x * hw;
            left.position[1] = cy;
            left.position[2] = cz - tan_z * hw;
            left.normal[0] = norm_x;  left.normal[1] = norm_y;  left.normal[2] = norm_z;
            left.color[0] = params.base_color[0];
            left.color[1] = params.base_color[1];
            left.color[2] = params.base_color[2];
            left.height_t = t;

            ClumpVertex right{};
            right.position[0] = cx + tan_x * hw;
            right.position[1] = cy;
            right.position[2] = cz + tan_z * hw;
            right.normal[0] = norm_x;  right.normal[1] = norm_y;  right.normal[2] = norm_z;
            right.color[0] = params.base_color[0];
            right.color[1] = params.base_color[1];
            right.color[2] = params.base_color[2];
            right.height_t = t;

            mesh.vertices.push_back(left);
            mesh.vertices.push_back(right);
        }

        for (int s = 0; s < segments; ++s) {
            uint32_t i0 = base_vert + static_cast<uint32_t>(s) * 2u;
            uint32_t i1 = i0 + 1;
            uint32_t i2 = i0 + 2;
            uint32_t i3 = i0 + 3;

            mesh.indices.push_back(i0);
            mesh.indices.push_back(i1);
            mesh.indices.push_back(i2);

            mesh.indices.push_back(i1);
            mesh.indices.push_back(i3);
            mesh.indices.push_back(i2);
        }
    }

    if (include_ground) {
        float gs = std::max(params.clump_radius * 2.0f, 0.2f) + 0.1f;
        auto gi  = static_cast<uint32_t>(mesh.vertices.size());

        const float gc[3] = {0.25f, 0.20f, 0.15f};
        const float gn[3] = {0.0f, 1.0f, 0.0f};

        float corners[4][3] = {
            {offset_x - gs, 0.0f, offset_z - gs},
            {offset_x + gs, 0.0f, offset_z - gs},
            {offset_x + gs, 0.0f, offset_z + gs},
            {offset_x - gs, 0.0f, offset_z + gs}
        };
        for (auto& c : corners) {
            ClumpVertex v{};
            v.position[0] = c[0]; v.position[1] = c[1]; v.position[2] = c[2];
            v.normal[0] = gn[0];  v.normal[1] = gn[1];  v.normal[2] = gn[2];
            v.color[0]  = gc[0];  v.color[1]  = gc[1];  v.color[2]  = gc[2];
            mesh.vertices.push_back(v);
        }
        mesh.indices.push_back(gi);     mesh.indices.push_back(gi + 1); mesh.indices.push_back(gi + 2);
        mesh.indices.push_back(gi);     mesh.indices.push_back(gi + 2); mesh.indices.push_back(gi + 3);
    }

    return mesh;
}

ClumpMesh generate_field(const ClumpParams& base,
                         const ClumpExpression& expr,
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

            ClumpParams resolved = evaluate_expression(base, expr, moisture);
            uint32_t seed = base_seed + static_cast<uint32_t>(iz * field.grid_n + ix);

            auto cell = generate_clump(resolved, seed, false, x, z);

            auto vert_offset = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(),
                                 cell.vertices.begin(), cell.vertices.end());
            for (uint32_t idx : cell.indices)
                mesh.indices.push_back(idx + vert_offset);
        }
    }

    // One big ground plane
    float half = extent * 0.5f + field.spacing;
    auto gi = static_cast<uint32_t>(mesh.vertices.size());

    const float gc[3] = {0.25f, 0.20f, 0.15f};
    const float gn[3] = {0.0f, 1.0f, 0.0f};

    float corners[4][3] = {
        {-half, 0.0f, -half}, { half, 0.0f, -half},
        { half, 0.0f,  half}, {-half, 0.0f,  half}
    };
    for (auto& c : corners) {
        ClumpVertex v{};
        v.position[0] = c[0]; v.position[1] = c[1]; v.position[2] = c[2];
        v.normal[0] = gn[0];  v.normal[1] = gn[1];  v.normal[2] = gn[2];
        v.color[0]  = gc[0];  v.color[1]  = gc[1];  v.color[2]  = gc[2];
        mesh.vertices.push_back(v);
    }
    mesh.indices.push_back(gi);     mesh.indices.push_back(gi + 1); mesh.indices.push_back(gi + 2);
    mesh.indices.push_back(gi);     mesh.indices.push_back(gi + 2); mesh.indices.push_back(gi + 3);

    return mesh;
}

} // namespace bestiary

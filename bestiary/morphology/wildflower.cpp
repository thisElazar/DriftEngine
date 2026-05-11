#include "wildflower.h"
#include <cmath>
#include <algorithm>

namespace bestiary {

static uint32_t wf_hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    x *= 0x45d9f3bu;
    x ^= x >> 16;
    return x;
}

static float wf_hash_float(uint32_t seed, uint32_t index)
{
    return static_cast<float>(wf_hash(seed ^ (index * 2654435761u))) /
           static_cast<float>(0xFFFFFFFFu);
}

static float wf_lerp(float a, float b, float t) { return a + (b - a) * t; }

WildflowerParams evaluate_wildflower_expression(const WildflowerParams& base,
                                                 const WildflowerExpression& expr,
                                                 float moisture)
{
    float t = std::clamp(moisture, 0.0f, 1.0f);
    WildflowerParams out = base;

    if (expr.flower_count.enabled)
        out.flower_count = static_cast<int>(wf_lerp(
            static_cast<float>(expr.flower_count.low),
            static_cast<float>(expr.flower_count.high), t));
    if (expr.stem_height.enabled)
        out.stem_height = wf_lerp(expr.stem_height.low, expr.stem_height.high, t);
    if (expr.petal_radius.enabled)
        out.petal_radius = wf_lerp(expr.petal_radius.low, expr.petal_radius.high, t);
    if (expr.clump_radius.enabled)
        out.clump_radius = wf_lerp(expr.clump_radius.low, expr.clump_radius.high, t);

    if (expr.vary_color) {
        for (int i = 0; i < 3; ++i)
            out.petal_color[i] = wf_lerp(expr.dry_color[i], expr.wet_color[i], t);
    }

    return out;
}

void jitter_wildflower(WildflowerParams& params, float variance, uint32_t seed)
{
    auto jit = [&](float& v, uint32_t idx) {
        float r = wf_hash_float(seed, idx) * 2.0f - 1.0f;
        v *= 1.0f + r * variance;
    };
    jit(params.stem_height, 0);
    jit(params.petal_radius, 1);
    jit(params.clump_radius, 2);
    jit(params.stem_width, 3);

    float hue_shift = (wf_hash_float(seed, 4) * 2.0f - 1.0f) * variance * 0.3f;
    params.petal_color[0] = std::clamp(params.petal_color[0] + hue_shift, 0.0f, 1.0f);
    params.petal_color[1] = std::clamp(params.petal_color[1] - hue_shift * 0.5f, 0.0f, 1.0f);
    params.petal_color[2] = std::clamp(params.petal_color[2] + hue_shift * 0.3f, 0.0f, 1.0f);
}

VegetationMesh generate_wildflower(const WildflowerParams& params, uint32_t seed,
                                    bool include_ground,
                                    float offset_x, float offset_z)
{
    VegetationMesh mesh;
    constexpr float pi = 3.14159265f;

    for (int f = 0; f < params.flower_count; ++f) {
        uint32_t fs = wf_hash(seed ^ static_cast<uint32_t>(f * 7919u));

        float angle = wf_hash_float(fs, 0) * 2.0f * pi;
        float dist = wf_hash_float(fs, 1) * params.clump_radius;
        float base_x = offset_x + std::cos(angle) * dist;
        float base_z = offset_z + std::sin(angle) * dist;

        float lean_angle = (wf_hash_float(fs, 2) * 2.0f - 1.0f) * 0.15f;
        float lean_dir = wf_hash_float(fs, 3) * 2.0f * pi;
        float height_var = 0.85f + wf_hash_float(fs, 4) * 0.3f;
        float stem_h = params.stem_height * height_var;

        float top_x = base_x + std::sin(lean_angle) * std::cos(lean_dir) * stem_h;
        float top_z = base_z + std::sin(lean_angle) * std::sin(lean_dir) * stem_h;
        float top_y = std::cos(lean_angle) * stem_h;

        // Stem — two crossed quads
        float sw = params.stem_width;
        for (int axis = 0; axis < 2; ++axis) {
            float dx = (axis == 0) ? sw : 0.0f;
            float dz = (axis == 0) ? 0.0f : sw;

            VegetationVertex v0{}, v1{}, v2{}, v3{};
            v0.position[0] = base_x - dx; v0.position[1] = 0.0f; v0.position[2] = base_z - dz;
            v1.position[0] = base_x + dx; v1.position[1] = 0.0f; v1.position[2] = base_z + dz;
            v2.position[0] = top_x + dx;  v2.position[1] = top_y; v2.position[2] = top_z + dz;
            v3.position[0] = top_x - dx;  v3.position[1] = top_y; v3.position[2] = top_z - dz;

            float nx = (axis == 0) ? 0.0f : 1.0f;
            float nz = (axis == 0) ? 1.0f : 0.0f;

            for (auto* vp : {&v0, &v1, &v2, &v3}) {
                vp->normal[0] = nx; vp->normal[1] = 0.0f; vp->normal[2] = nz;
                vp->color[0] = params.stem_color[0];
                vp->color[1] = params.stem_color[1];
                vp->color[2] = params.stem_color[2];
            }
            v0.height_t = 0.0f; v1.height_t = 0.0f;
            v2.height_t = 1.0f; v3.height_t = 1.0f;

            auto qi = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(v0);
            mesh.vertices.push_back(v1);
            mesh.vertices.push_back(v2);
            mesh.vertices.push_back(v3);
            mesh.indices.push_back(qi); mesh.indices.push_back(qi+1); mesh.indices.push_back(qi+2);
            mesh.indices.push_back(qi); mesh.indices.push_back(qi+2); mesh.indices.push_back(qi+3);
        }

        // Flower head — flat disc of petal triangles radiating from center
        float pr = params.petal_radius;
        VegetationVertex center{};
        center.position[0] = top_x; center.position[1] = top_y; center.position[2] = top_z;
        center.normal[0] = 0.0f; center.normal[1] = 1.0f; center.normal[2] = 0.0f;
        center.color[0] = params.petal_color[0] * 0.8f;
        center.color[1] = params.petal_color[1] * 0.8f;
        center.color[2] = params.petal_color[2] * 0.3f;
        center.height_t = 1.0f;

        auto ci = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(center);

        for (int p = 0; p < params.petal_count; ++p) {
            float a0 = (static_cast<float>(p) / static_cast<float>(params.petal_count)) * 2.0f * pi;
            float a1 = (static_cast<float>(p + 1) / static_cast<float>(params.petal_count)) * 2.0f * pi;
            float mid = (a0 + a1) * 0.5f;

            VegetationVertex tip{};
            tip.position[0] = top_x + std::cos(mid) * pr;
            tip.position[1] = top_y + 0.002f;
            tip.position[2] = top_z + std::sin(mid) * pr;
            tip.normal[0] = 0.0f; tip.normal[1] = 1.0f; tip.normal[2] = 0.0f;
            tip.color[0] = params.petal_color[0];
            tip.color[1] = params.petal_color[1];
            tip.color[2] = params.petal_color[2];
            tip.height_t = 1.0f;

            VegetationVertex edge0{};
            edge0.position[0] = top_x + std::cos(a0) * pr * 0.4f;
            edge0.position[1] = top_y + 0.001f;
            edge0.position[2] = top_z + std::sin(a0) * pr * 0.4f;
            edge0.normal[0] = 0.0f; edge0.normal[1] = 1.0f; edge0.normal[2] = 0.0f;
            edge0.color[0] = params.petal_color[0] * 0.9f;
            edge0.color[1] = params.petal_color[1] * 0.9f;
            edge0.color[2] = params.petal_color[2] * 0.9f;
            edge0.height_t = 1.0f;

            auto ti = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(edge0);
            mesh.vertices.push_back(tip);

            mesh.indices.push_back(ci);
            mesh.indices.push_back(ti);
            mesh.indices.push_back(ti + 1);
        }
    }

    (void)include_ground;
    return mesh;
}

VegetationMesh generate_wildflower_field(const WildflowerParams& base,
                                          const WildflowerExpression& expr,
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

            WildflowerParams resolved = evaluate_wildflower_expression(base, expr, moisture);
            uint32_t seed = base_seed + static_cast<uint32_t>(iz * field.grid_n + ix);

            auto cell = generate_wildflower(resolved, seed, false, x, z);

            auto vert_offset = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(),
                                 cell.vertices.begin(), cell.vertices.end());
            for (uint32_t idx : cell.indices)
                mesh.indices.push_back(idx + vert_offset);
        }
    }

    return mesh;
}

} // namespace bestiary

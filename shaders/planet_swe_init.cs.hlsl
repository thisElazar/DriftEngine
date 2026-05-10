// planet_swe_init.cs.hlsl — Initialize the SWE state and water_output for a
// freshly-allocated tile. Seeds h with the static sea-level depth at this
// cell's terrain, plus contributions from any persistent WaterStamps. Mirrors
// the terrain-stamp model: water stamps are CPU-side data applied at every
// LOD's init, so a brushed lake stays visible from space.

[[vk::binding(0, 0)]] Texture2DArray<float>    terrain;
[[vk::binding(1, 0)]] RWTexture2DArray<float4> state_a;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> state_b;
[[vk::binding(3, 0)]] RWTexture2DArray<float4> output;

struct WaterStamp {
    float3 pos;
    float  radius;
    float  water_amount;
    float  cos_radius;
    float2 _pad;
};

[[vk::binding(4, 0)]] StructuredBuffer<WaterStamp> water_stamps;

[[vk::push_constant]]
cbuffer PushConstants {
    uint  grid_w;
    uint  grid_h;
    float sea_level;
    uint  pool_index;

    float u_min;
    float v_min;
    float tile_size;
    uint  face;

    uint  water_stamp_count;
    uint  _pad0, _pad1, _pad2;
};

float3 face_uv_to_cube(float2 uv, uint f)
{
    switch (f) {
        case 0: return float3( 1.0, uv.y, -uv.x);
        case 1: return float3(-1.0, uv.y,  uv.x);
        case 2: return float3(uv.x,  1.0, -uv.y);
        case 3: return float3(uv.x, -1.0,  uv.y);
        case 4: return float3(uv.x, uv.y,  1.0);
        case 5: return float3(-uv.x, uv.y, -1.0);
        default: return float3(0, 0, 0);
    }
}

float3 cube_to_sphere(float3 p)
{
    float x2 = p.x * p.x;
    float y2 = p.y * p.y;
    float z2 = p.z * p.z;
    return float3(
        p.x * sqrt(max(0.0, 1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0)),
        p.y * sqrt(max(0.0, 1.0 - x2 * 0.5 - z2 * 0.5 + x2 * z2 / 3.0)),
        p.z * sqrt(max(0.0, 1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0))
    );
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= grid_w || dtid.y >= grid_h) return;

    float t_h = terrain[uint3(dtid.xy, pool_index)];
    float h = max(sea_level - t_h, 0.0);

    // Apply water stamps: any cell within an angular radius of a stamp's center
    // gets water_amount added to its column with the same gaussian falloff used
    // by terrain stamps in planet_gen.cs.hlsl.
    if (water_stamp_count > 0u)
    {
        float2 tile_uv = float2(dtid.xy) / float(grid_w - 1);
        float2 face_uv = float2(u_min + tile_uv.x * tile_size,
                                v_min + tile_uv.y * tile_size);
        float3 sphere_dir = normalize(cube_to_sphere(face_uv_to_cube(face_uv, face)));
        for (uint i = 0; i < water_stamp_count; ++i)
        {
            WaterStamp s = water_stamps[i];
            float d = dot(sphere_dir, s.pos);
            if (d < s.cos_radius) continue;
            float u = (d - s.cos_radius) / max(1.0 - s.cos_radius, 1e-7);
            h += s.water_amount * exp(-4.0 * (1.0 - u) * (1.0 - u));
        }
    }

    float4 eq = float4(h, 0.0, 0.0, 0.0);
    state_a[uint3(dtid.xy, pool_index)] = eq;
    state_b[uint3(dtid.xy, pool_index)] = eq;
    output[uint3(dtid.xy, pool_index)] = eq;
}

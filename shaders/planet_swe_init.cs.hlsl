// planet_swe_init.cs.hlsl — Initialize the SWE state and water_output for a
// tile. Seeds h from ONE water-surface function: the max of the ocean sea
// level and the hydrology field's live water-surface elevation (channel a,
// metres; == terrain height where nothing stands). Ocean tiles get sea-level
// depth, field lakes become real SWE water columns, and a zeroed field (worker
// hasn't published yet) degrades to ocean-only. Re-dispatched for resident
// tiles on every field publish, so far lake levels track the live field.

[[vk::binding(0, 0)]] Texture2DArray<float>    terrain;
[[vk::binding(1, 0)]] RWTexture2DArray<float4> state_a;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> state_b;
[[vk::binding(3, 0)]] RWTexture2DArray<float4> output;
[[vk::binding(4, 0)]] Texture2DArray<float4>   hydrology;  // .a = water-surface elevation (m)
[[vk::binding(5, 0)]] SamplerState             samp;

[[vk::push_constant]]
cbuffer PushConstants {
    uint  grid_w;
    uint  grid_h;
    float sea_level;
    uint  pool_index;

    // Tile → cube-face uv mapping, for sampling the per-face hydrology layers.
    float u_min;
    float v_min;
    float tile_size;
    uint  face;
};

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= grid_w || dtid.y >= grid_h) return;

    float t_h = terrain[uint3(dtid.xy, pool_index)];

    float2 tile_uv = float2(dtid.xy) / float(grid_w - 1);
    float2 face_uv = float2(u_min + tile_uv.x * tile_size,
                            v_min + tile_uv.y * tile_size);
    float2 hyd_uv  = face_uv * 0.5 + 0.5;
    float  surf    = hydrology.SampleLevel(samp, float3(hyd_uv, float(face)), 0).a;

    float h = max(max(sea_level, surf) - t_h, 0.0);

    float4 eq = float4(h, 0.0, 0.0, 0.0);
    state_a[uint3(dtid.xy, pool_index)] = eq;
    state_b[uint3(dtid.xy, pool_index)] = eq;
    output[uint3(dtid.xy, pool_index)] = eq;
}

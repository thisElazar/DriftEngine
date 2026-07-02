// planet_swe_init.cs.hlsl — Initialize the SWE state and water_output for a
// freshly-allocated tile: seed h with the static sea-level depth at this
// cell's terrain. Inland standing water is owned by the hydrology field
// (rivers + lakes, src/hydrology.cpp) — the old per-stamp brushed-lake path
// was removed when the water brush moved to field deposits.

[[vk::binding(0, 0)]] Texture2DArray<float>    terrain;
[[vk::binding(1, 0)]] RWTexture2DArray<float4> state_a;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> state_b;
[[vk::binding(3, 0)]] RWTexture2DArray<float4> output;

[[vk::push_constant]]
cbuffer PushConstants {
    uint  grid_w;
    uint  grid_h;
    float sea_level;
    uint  pool_index;
};

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= grid_w || dtid.y >= grid_h) return;

    float t_h = terrain[uint3(dtid.xy, pool_index)];
    float h = max(sea_level - t_h, 0.0);

    float4 eq = float4(h, 0.0, 0.0, 0.0);
    state_a[uint3(dtid.xy, pool_index)] = eq;
    state_b[uint3(dtid.xy, pool_index)] = eq;
    output[uint3(dtid.xy, pool_index)] = eq;
}

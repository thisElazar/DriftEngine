[[vk::binding(0, 0)]] Texture2DArray<float>    terrain;
[[vk::binding(1, 0)]] RWTexture2DArray<float4> state_write;

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
    float t = terrain[uint3(dtid.xy, pool_index)];
    float h = max(sea_level - t, 0.0);
    state_write[uint3(dtid.xy, pool_index)] = float4(h, 0.0, 0.0, 0.0);
}

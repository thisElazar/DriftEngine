[[vk::binding(0, 0)]] Texture2D<float> terrain;
[[vk::binding(1, 0)]] RWTexture2D<float4> state_write;

[[vk::push_constant]]
cbuffer PushConstants {
    uint  grid_w;
    uint  grid_h;
    float initial_water_level;
    float _pad;
};

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= grid_w || dtid.y >= grid_h) return;
    float t = terrain[dtid.xy];
    float h = max(initial_water_level - t, 0.0);
    state_write[dtid.xy] = float4(h, 0.0, 0.0, 0.0);
}

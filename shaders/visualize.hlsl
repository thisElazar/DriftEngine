// visualize.hlsl — placeholder pattern; will be replaced by
// heightmap visualization in v0.1.7.
RWTexture2D<float4> output : register(u0);

[[vk::push_constant]]
cbuffer PushConstants {
    float pc_time;
    uint pc_width;
    uint pc_height;
    uint pc_pad;
};

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= pc_width || dtid.y >= pc_height) return;
    float2 uv = float2(dtid.xy) / float2(pc_width, pc_height);
    float r = 0.5 + 0.5 * sin(pc_time + uv.x * 6.28);
    float g = 0.5 + 0.5 * sin(pc_time + uv.y * 6.28 + 2.094);
    float b = 0.5 + 0.5 * sin(pc_time + (uv.x + uv.y) * 3.14 + 4.188);
    output[dtid.xy] = float4(r, g, b, 1.0);
}

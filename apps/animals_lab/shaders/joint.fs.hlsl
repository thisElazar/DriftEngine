struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
};

float4 main(PSInput input) : SV_Target
{
    float3 L = normalize(float3(0.4, 1.0, 0.3));
    float3 N = normalize(input.world_normal);
    float lighting = 0.3 + 0.7 * max(0.0, dot(N, L));
    float3 color = float3(0.9, 0.85, 0.8) * lighting;
    return float4(color, 1.0);
}

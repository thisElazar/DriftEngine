struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float3 color : COLOR;
};

float4 main(PSInput input) : SV_Target
{
    float3 L = normalize(float3(0.4, 1.0, 0.3));
    float3 N = normalize(input.world_normal);
    float lighting = 0.3 + 0.7 * max(0.0, dot(N, L));
    return float4(input.color * lighting, 1.0);
}

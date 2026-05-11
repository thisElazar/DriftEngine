struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float3 color : COLOR;
};

float4 main(PSInput input) : SV_Target
{
    float3 L = normalize(float3(0.4, 1.0, 0.3));
    float3 N = normalize(input.world_normal);

    float3 sky = float3(0.48, 0.46, 0.44);
    float3 gnd = float3(0.14, 0.16, 0.20);
    float3 ambient = lerp(gnd, sky, N.y * 0.5 + 0.5);

    float NdotL = max(0.0, dot(N, L));
    float3 sun = float3(0.62, 0.60, 0.55) * NdotL;

    return float4(input.color * (ambient + sun), 1.0);
}

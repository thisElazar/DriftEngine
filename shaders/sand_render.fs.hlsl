struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float alpha : TEXCOORD0;
    [[vk::location(1)]] float dist : TEXCOORD1;
};

float4 main(PSInput input) : SV_Target
{
    if (input.alpha < 0.001) discard;
    float3 sand_color = float3(0.85, 0.70, 0.45);
    return float4(sand_color, input.alpha);
}

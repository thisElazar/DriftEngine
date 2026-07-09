// tonemap.fs.hlsl — HDR scene → display. ACES filmic fit (Narkowicz) with an
// exposure control, then gamma encode (the swapchain is UNORM, not sRGB).

[[vk::binding(0, 0)]] Texture2D<float4> hdr;
[[vk::binding(1, 0)]] SamplerState samp;

[[vk::push_constant]]
cbuffer TonemapPC {
    float exposure;
    float _pad0, _pad1, _pad2;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 ndc : TEXCOORD0;
    [[vk::location(1)]] float2 uv  : TEXCOORD1;
};

float3 aces(float3 x)
{
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

float4 main(PSInput input) : SV_Target
{
    float3 c = hdr.SampleLevel(samp, input.uv, 0).rgb;
    c = aces(c * exposure);
    c = pow(c, 1.0 / 2.2);
    return float4(c, 1.0);
}

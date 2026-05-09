[[vk::combinedImageSampler]][[vk::binding(4, 0)]] Texture2D<float4> atmo_render;
[[vk::combinedImageSampler]][[vk::binding(4, 0)]] SamplerState atmo_sampler;

[[vk::push_constant]]
cbuffer PushConstants {
    float terrain_size;
    float heightmap_texel;
    float max_elevation;
    float cloud_opacity;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

float4 main(PSInput input) : SV_Target
{
    float4 cloud = atmo_render.SampleLevel(atmo_sampler, input.uv, 0);
    if (cloud.a < 0.001) discard;

    float3 cloud_color = cloud.rgb / max(cloud.a, 0.001);
    float alpha = saturate(cloud.a * cloud_opacity);

    return float4(cloud_color, alpha);
}

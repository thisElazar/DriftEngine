[[vk::binding(0, 0)]] RWTexture2D<float4> output;
[[vk::binding(1, 0)]] Texture2D<float> heightmap;
[[vk::binding(3, 0)]] SamplerState heightmap_sampler;
[[vk::binding(2, 0)]] Texture2D<float4> water_state;
[[vk::binding(4, 0)]] SamplerState state_sampler;

[[vk::push_constant]]
cbuffer PushConstants {
    float time;
    uint  width;
    uint  height;
    float max_elevation;
};

float3 elevation_ramp(float h)
{
    float3 c = float3(0.85, 0.78, 0.55);
    c = lerp(c, float3(0.30, 0.55, 0.25), smoothstep(0.00, 0.20, h));
    c = lerp(c, float3(0.20, 0.40, 0.15), smoothstep(0.20, 0.45, h));
    c = lerp(c, float3(0.50, 0.42, 0.32), smoothstep(0.45, 0.70, h));
    c = lerp(c, float3(0.75, 0.72, 0.68), smoothstep(0.70, 0.90, h));
    c = lerp(c, float3(0.98, 0.98, 1.00), smoothstep(0.90, 1.00, h));
    return c;
}

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= width || dtid.y >= height) return;

    float2 uv = (float2(dtid.xy) + 0.5) / float2(width, height);

    float h = heightmap.SampleLevel(heightmap_sampler, uv, 0).r;
    float h_norm = saturate(h / max_elevation);

    float3 color = elevation_ramp(h_norm);

    float4 water = water_state.SampleLevel(state_sampler, uv, 0);
    float depth = water.r;
    if (depth > 0.05)
    {
        float3 shallow = float3(0.50, 0.80, 0.85);
        float3 mid     = float3(0.10, 0.40, 0.65);
        float3 deep_c  = float3(0.02, 0.10, 0.25);
        float t1 = smoothstep(0.0, 20.0, depth);
        float t2 = smoothstep(20.0, 200.0, depth);
        float3 water_color = lerp(lerp(shallow, mid, t1), deep_c, t2);

        float opacity = smoothstep(0.05, 2.0, depth);
        color = lerp(color, water_color, opacity);

        float foam = water.a;
        if (foam > 0.1)
            color = lerp(color, float3(0.9, 0.95, 1.0), saturate(foam * 0.5));
    }

    output[dtid.xy] = float4(color, 1.0);
}

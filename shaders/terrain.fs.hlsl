[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float _pad1;
    float3 cam_pos;   float _pad2;
    float4 brush_world;
    float4 brush_color;
};

[[vk::push_constant]]
cbuffer PushConstants {
    float terrain_size;
    float heightmap_texel;
    float max_elevation;
    float _pc_pad;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float height_normalized : TEXCOORD1;
    [[vk::location(3)]] float3 world_pos : TEXCOORD2;
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

float4 main(PSInput input) : SV_Target
{
    float3 color = elevation_ramp(input.height_normalized);

    float3 L = normalize(sun_dir);
    float3 N = normalize(input.world_normal);
    float NdotL = max(dot(N, L), 0.0);
    float ambient = 0.3;
    float lighting = ambient + (1.0 - ambient) * NdotL;
    color *= lighting;

    if (brush_world.w > 0.5) {
        float2 to_cursor = input.world_pos.xz - brush_world.xy;
        float d = length(to_cursor);
        float ring_thickness = 4.0;
        float ring = exp(-pow((d - brush_world.z) / ring_thickness, 2.0));
        color = lerp(color, brush_color.rgb, saturate(ring) * 0.85);
    }

    return float4(color, 1.0);
}

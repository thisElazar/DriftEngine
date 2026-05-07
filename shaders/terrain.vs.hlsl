[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float _pad1;
    float3 cam_pos;   float _pad2;
    float4 brush_world;
    float4 brush_color;
};

[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2D<float> heightmap;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState heightmap_sampler;

[[vk::push_constant]]
cbuffer PushConstants {
    float terrain_size;
    float heightmap_texel;
    float max_elevation;
    float _pc_pad;
};

struct VSInput {
    [[vk::location(0)]] float2 grid_pos : POSITION;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float height_normalized : TEXCOORD1;
    [[vk::location(3)]] float3 world_pos : TEXCOORD2;
};

VSOutput main(VSInput input)
{
    float2 uv = input.grid_pos / 511.0;

    float h = heightmap.SampleLevel(heightmap_sampler, uv, 0).r;

    float3 world_pos;
    world_pos.x = (uv.x - 0.5) * terrain_size;
    world_pos.y = h;
    world_pos.z = (uv.y - 0.5) * terrain_size;

    float hL = heightmap.SampleLevel(heightmap_sampler, uv + float2(-heightmap_texel, 0), 0).r;
    float hR = heightmap.SampleLevel(heightmap_sampler, uv + float2( heightmap_texel, 0), 0).r;
    float hD = heightmap.SampleLevel(heightmap_sampler, uv + float2(0, -heightmap_texel), 0).r;
    float hU = heightmap.SampleLevel(heightmap_sampler, uv + float2(0,  heightmap_texel), 0).r;

    float dx = terrain_size * heightmap_texel;
    float3 normal = normalize(float3(hL - hR, 2.0 * dx, hD - hU));

    VSOutput o;
    o.position = mul(proj, mul(view, float4(world_pos, 1.0)));
    o.world_normal = normal;
    o.uv = uv;
    o.height_normalized = saturate(h / max_elevation);
    o.world_pos = world_pos;
    return o;
}

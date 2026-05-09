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
[[vk::combinedImageSampler]][[vk::binding(2, 0)]] Texture2D<float4> swe_output;
[[vk::combinedImageSampler]][[vk::binding(2, 0)]] SamplerState swe_sampler;

[[vk::push_constant]]
cbuffer PushConstants {
    float terrain_size;
    float heightmap_texel;
    float max_elevation;
    float cloud_opacity;
};

struct VSInput {
    [[vk::location(0)]] float2 grid_pos : POSITION;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float2 uv : TEXCOORD2;
    [[vk::location(3)]] float depth : TEXCOORD3;
};

VSOutput main(VSInput input)
{
    float2 uv = input.grid_pos / 511.0;

    float t = heightmap.SampleLevel(heightmap_sampler, uv, 0).r;
    float4 swe = swe_output.SampleLevel(swe_sampler, uv, 0);
    float surface_elev = swe.r;
    float depth = surface_elev - t;

    float3 world_pos;
    world_pos.x = (uv.x - 0.5) * terrain_size;
    world_pos.y = surface_elev;
    world_pos.z = (uv.y - 0.5) * terrain_size;

    float2 px = float2(heightmap_texel, heightmap_texel);
    float sL = swe_output.SampleLevel(swe_sampler, uv + float2(-px.x, 0), 0).r;
    float sR = swe_output.SampleLevel(swe_sampler, uv + float2( px.x, 0), 0).r;
    float sD = swe_output.SampleLevel(swe_sampler, uv + float2(0, -px.y), 0).r;
    float sU = swe_output.SampleLevel(swe_sampler, uv + float2(0,  px.y), 0).r;

    float dx = terrain_size * heightmap_texel;
    float3 normal = normalize(float3(sL - sR, 2.0 * dx, sD - sU));

    VSOutput o;
    o.position = mul(proj, mul(view, float4(world_pos, 1.0)));
    o.world_pos = world_pos;
    o.normal = normal;
    o.uv = uv;
    o.depth = depth;
    return o;
}

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
    float cloud_opacity;
};

struct VSInput {
    [[vk::location(0)]] float2 grid_pos : POSITION;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

VSOutput main(VSInput input)
{
    float2 uv = input.grid_pos / 511.0;

    float cloud_height = max_elevation * 0.65;

    float3 world_pos;
    world_pos.x = (uv.x - 0.5) * terrain_size;
    world_pos.y = cloud_height;
    world_pos.z = (uv.y - 0.5) * terrain_size;

    VSOutput o;
    o.position = mul(proj, mul(view, float4(world_pos, 1.0)));
    o.uv = uv;
    return o;
}

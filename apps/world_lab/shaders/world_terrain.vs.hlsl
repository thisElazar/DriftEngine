[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D<float>  heightmap;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState       heightmap_sampler;

[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2D<float4> water_out;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState      water_sampler;

[[vk::combinedImageSampler]][[vk::binding(2, 0)]] Texture2D<float>  moisture;
[[vk::combinedImageSampler]][[vk::binding(2, 0)]] SamplerState      moisture_sampler;

[[vk::push_constant]]
cbuffer WorldTerrainPC {
    float4x4 mvp;
    float    grid_w_f;
    float    grid_h_f;
    float    cell_size;
    float    sea_level;
    float    brush_x;
    float    brush_y;
    float    brush_radius;
    float    brush_active;
    float    moisture_overlay;
    float    _pad0;
    float    _pad1;
    float    _pad2;
};

struct VSInput {
    [[vk::location(0)]] float2 grid_pos : POSITION;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_pos    : TEXCOORD0;
    [[vk::location(1)]] float3 world_normal : NORMAL;
    [[vk::location(2)]] float  water_depth  : TEXCOORD1;
    [[vk::location(3)]] float2 grid_uv      : TEXCOORD2;
    [[vk::location(4)]] float  moisture_v   : TEXCOORD3;
};

VSOutput main(VSInput input)
{
    float2 uv = (input.grid_pos + 0.5) / float2(grid_w_f, grid_h_f);

    float h  = heightmap.SampleLevel(heightmap_sampler, uv, 0);
    float4 w = water_out.SampleLevel(water_sampler, uv, 0);

    // swe_step writes surface ELEVATION (z + h) to output.r, or SEA_LEVEL
    // (large negative sentinel) when the cell is dry. Convert to depth here.
    float surf_elev = w.r;
    float depth = (surf_elev > -1000.0) ? max(surf_elev - h, 0.0) : 0.0;

    // Sample neighbors for normal
    float2 du = float2(1.0 / grid_w_f, 0.0);
    float2 dv = float2(0.0, 1.0 / grid_h_f);
    float hL = heightmap.SampleLevel(heightmap_sampler, uv - du, 0);
    float hR = heightmap.SampleLevel(heightmap_sampler, uv + du, 0);
    float hD = heightmap.SampleLevel(heightmap_sampler, uv - dv, 0);
    float hU = heightmap.SampleLevel(heightmap_sampler, uv + dv, 0);

    float dx = (hR - hL) / (2.0 * cell_size);
    float dz = (hU - hD) / (2.0 * cell_size);
    float3 normal = normalize(float3(-dx, 1.0, -dz));

    float world_x = (input.grid_pos.x - grid_w_f * 0.5 + 0.5) * cell_size;
    float world_z = (input.grid_pos.y - grid_h_f * 0.5 + 0.5) * cell_size;

    VSOutput o;
    o.position = mul(mvp, float4(world_x, h, world_z, 1.0));
    o.world_pos = float3(world_x, h, world_z);
    o.world_normal = normal;
    o.water_depth = depth;
    o.grid_uv = float2(input.grid_pos.x / grid_w_f, input.grid_pos.y / grid_h_f);
    o.moisture_v = moisture.SampleLevel(moisture_sampler, uv, 0);
    return o;
}

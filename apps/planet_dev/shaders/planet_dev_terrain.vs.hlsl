// planet_dev terrain VS — world_lab's world_terrain.vs adapted to tiled arrays.
//
// Differences from world_terrain.vs:
//  - heightmap / water / moisture are Texture2DArray, sampled at (uv, layer);
//    the layer is the tile index, pushed per draw.
//  - world = tile_origin + grid_pos * cell_size and uv = grid_pos / grid_w
//    (NOT world_lab's center-origin +0.5 mapping) so the 4 tiles tile exactly.
//  - water comes from the planet SWE output convention: output.r is DEPTH
//    already (the flat world_lab swe writes surface elevation) — no conversion.

[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2DArray<float>  heightmap;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState           heightmap_sampler;

[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2DArray<float4> water_out;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState           water_sampler;

[[vk::combinedImageSampler]][[vk::binding(2, 0)]] Texture2DArray<float>  moisture;
[[vk::combinedImageSampler]][[vk::binding(2, 0)]] SamplerState           moisture_sampler;

[[vk::push_constant]]
cbuffer PlanetDevTerrainPC {
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
    float    tile_origin_x;
    float    tile_origin_z;
    float    layer;
    float    seam_highlight;
    float    uv_off_x;       // sub-rect of the layer this draw samples:
    float    uv_off_y;       //   suv = uv * uv_scale + uv_off
    float    uv_scale;       // 1 for pool slots; 1/8 for far-field tiles
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
    float2 uv = input.grid_pos / float2(grid_w_f, grid_h_f);
    // Far-field draws sample a sub-rect of the whole-world far layer; pool
    // draws use the identity transform. grid_uv (seam/brush overlays) stays
    // mesh-local.
    float2 suv = uv * uv_scale + float2(uv_off_x, uv_off_y);
    float3 uvl = float3(suv, layer);

    float h  = heightmap.SampleLevel(heightmap_sampler, uvl, 0);
    float4 w = water_out.SampleLevel(water_sampler, uvl, 0);

    // Planet SWE convention: output.r is water depth (0 when dry).
    float depth = max(w.r, 0.0);

    // Sample neighbors for normal
    float2 du = float2(uv_scale / grid_w_f, 0.0);
    float2 dv = float2(0.0, uv_scale / grid_h_f);
    float hL = heightmap.SampleLevel(heightmap_sampler, float3(suv - du, layer), 0);
    float hR = heightmap.SampleLevel(heightmap_sampler, float3(suv + du, layer), 0);
    float hD = heightmap.SampleLevel(heightmap_sampler, float3(suv - dv, layer), 0);
    float hU = heightmap.SampleLevel(heightmap_sampler, float3(suv + dv, layer), 0);

    float dx = (hR - hL) / (2.0 * cell_size);
    float dz = (hU - hD) / (2.0 * cell_size);
    float3 normal = normalize(float3(-dx, 1.0, -dz));

    float world_x = tile_origin_x + input.grid_pos.x * cell_size;
    float world_z = tile_origin_z + input.grid_pos.y * cell_size;

    VSOutput o;
    o.position = mul(mvp, float4(world_x, h, world_z, 1.0));
    o.world_pos = float3(world_x, h, world_z);
    o.world_normal = normal;
    o.water_depth = depth;
    o.grid_uv = uv;
    o.moisture_v = moisture.SampleLevel(moisture_sampler, uvl, 0);
    return o;
}

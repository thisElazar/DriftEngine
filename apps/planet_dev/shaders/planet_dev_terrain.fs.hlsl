// planet_dev terrain FS — world_lab's world_terrain.fs plus a seam-highlight
// overlay (tile borders tinted when enabled) for inspecting cross-tile flow.

[[vk::push_constant]]
cbuffer PlanetDevTerrainPC {
    float4x4 mvp;
    float    grid_w_f;
    float    grid_h_f;
    float    cell_size;
    float    sea_level;
    float    brush_x;          // cursor-tile grid coords
    float    brush_y;
    float    brush_radius;     // grid cells
    float    brush_active;     // only set on the cursor tile's draw
    float    moisture_overlay;
    float    tile_origin_x;
    float    tile_origin_z;
    float    layer;
    float    seam_highlight;
    float    uv_off_x;
    float    uv_off_y;
    float    uv_scale;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_pos    : TEXCOORD0;
    [[vk::location(1)]] float3 world_normal : NORMAL;
    [[vk::location(2)]] float  water_depth  : TEXCOORD1;
    [[vk::location(3)]] float2 grid_uv      : TEXCOORD2;
    [[vk::location(4)]] float  moisture_v   : TEXCOORD3;
};

float4 main(PSInput input) : SV_Target
{
    float3 L = normalize(float3(0.4, 1.0, 0.3));
    float3 N = normalize(input.world_normal);
    float lambert = 0.3 + 0.7 * max(0.0, dot(N, L));

    // Soil base color (drier on slopes/peaks, darker in low areas)
    float3 soil_dry = float3(0.50, 0.42, 0.28);
    float3 soil_wet = float3(0.22, 0.30, 0.18);

    float slope = saturate(1.0 - N.y);
    float3 ground = lerp(soil_wet, soil_dry, slope);

    // Water tint: blend toward water color as depth grows
    float depth = max(0.0, input.water_depth);
    float water_t = saturate(depth / 1.5);
    float3 shallow = float3(0.20, 0.45, 0.55);
    float3 deep    = float3(0.05, 0.18, 0.30);
    float3 water_color = lerp(shallow, deep, saturate(depth / 4.0));

    float3 surface = lerp(ground, water_color, water_t);

    // Moisture heatmap overlay
    if (moisture_overlay > 0.5)
    {
        float m = saturate(input.moisture_v);
        float3 dry = float3(0.78, 0.68, 0.45);
        float3 wet = float3(0.10, 0.45, 0.55);
        surface = lerp(dry, wet, m);
    }

    float3 lit = surface * lambert;

    // Seam highlight: tint within ~1 texel of the tile border.
    if (seam_highlight > 0.5)
    {
        float texel = 1.0 / grid_w_f;
        float2 edge_d = min(input.grid_uv, 1.0 - input.grid_uv);
        float d = min(edge_d.x, edge_d.y);
        float seam = 1.0 - smoothstep(0.0, texel, d);
        lit = lerp(lit, float3(0.9, 0.3, 0.2), seam * 0.5);
    }

    // Brush ring overlay (cursor tile only)
    if (brush_active > 0.5)
    {
        float2 g = input.grid_uv * float2(grid_w_f, grid_h_f);
        float r = distance(g, float2(brush_x, brush_y));
        float ring = smoothstep(brush_radius - 1.0, brush_radius, r)
                   - smoothstep(brush_radius, brush_radius + 1.0, r);
        lit = lerp(lit, float3(1.0, 0.9, 0.4), ring * 0.7);
    }

    return float4(lit, 1.0);
}

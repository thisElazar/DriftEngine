[[vk::push_constant]]
cbuffer WorldTerrainPC {
    float4x4 mvp;
    float    grid_w_f;
    float    grid_h_f;
    float    cell_size;
    float    sea_level;
    float    brush_x;       // grid coords
    float    brush_y;       // grid coords
    float    brush_radius;  // grid cells
    float    brush_active;  // 0 or 1
    float    moisture_overlay; // 0 = normal, 1 = moisture heatmap
    float    _pad0;
    float    _pad1;
    float    _pad2;
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

    // Use slope (1 - N.y) and elevation for a quick aesthetic mix
    float slope = saturate(1.0 - N.y);
    float3 ground = lerp(soil_wet, soil_dry, slope);

    // Water tint: blend toward water color as depth grows
    float depth = max(0.0, input.water_depth);
    float water_t = saturate(depth / 1.5);
    float3 shallow = float3(0.20, 0.45, 0.55);
    float3 deep    = float3(0.05, 0.18, 0.30);
    float3 water_color = lerp(shallow, deep, saturate(depth / 4.0));

    float3 surface = lerp(ground, water_color, water_t);

    // Moisture heatmap overlay: tan (dry) -> teal (wet). Replaces ground tint
    // when overlay is on, so density/moisture relationship reads cleanly.
    if (moisture_overlay > 0.5)
    {
        float m = saturate(input.moisture_v);
        float3 dry = float3(0.78, 0.68, 0.45);
        float3 wet = float3(0.10, 0.45, 0.55);
        surface = lerp(dry, wet, m);
    }

    float3 lit = surface * lambert;

    // Brush ring overlay
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

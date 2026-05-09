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
cbuffer PlanetTilePC {
    float rel_x, rel_y, rel_z;
    float u_min, v_min, tile_size;
    uint  face;
    uint  pool_index;
    float planet_radius;
    float max_elevation;
    float heightmap_texel;
    float cloud_opacity;
    float sea_level;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float height_normalized : TEXCOORD1;
    [[vk::location(3)]] float3 world_pos : TEXCOORD2;
    [[vk::location(4)]] float3 sphere_direction : TEXCOORD3;
    [[vk::location(5)]] float  water_depth : TEXCOORD4;
    [[vk::location(6)]] float  foam : TEXCOORD5;
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

float3 compute_water_color(float depth, float3 sphere_dir, float3 world_pos)
{
    float3 N = normalize(sphere_dir);
    float3 V = normalize(-world_pos);
    float3 L = normalize(sun_dir);

    float3 shallow = float3(0.55, 0.85, 0.90);
    float3 mid     = float3(0.10, 0.40, 0.65);
    float3 deep_c  = float3(0.02, 0.10, 0.22);
    float t1 = smoothstep(0.0,  200.0, depth);
    float t2 = smoothstep(200.0, 2000.0, depth);
    float3 base = lerp(lerp(shallow, mid, t1), deep_c, t2);

    float NdotV = saturate(dot(N, V));
    float fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);

    float3 R = reflect(-V, N);
    float sky_t = saturate(R.y * 0.5 + 0.5);
    float3 sky = lerp(float3(0.85, 0.88, 0.92), float3(0.30, 0.55, 0.85), sky_t);

    float3 H = normalize(L + V);
    float spec = pow(saturate(dot(N, H)), 64.0);
    float3 specular = sun_color * spec * 0.8;

    float NdotL = saturate(dot(N, L));
    float3 diffuse = base * (0.4 + 0.6 * NdotL);

    return lerp(diffuse, sky, fresnel) + specular;
}

float4 main(PSInput input) : SV_Target
{
    float3 color;

    if (input.water_depth > 0.5) {
        color = compute_water_color(input.water_depth, input.sphere_direction, input.world_pos);
        color = lerp(color, float3(0.95, 0.95, 0.97), saturate(input.foam));
    } else if (input.water_depth > 0.0) {
        float3 terrain_color = elevation_ramp(input.height_normalized);
        float3 L = normalize(sun_dir);
        float3 N = normalize(input.world_normal);
        float NdotL = max(dot(N, L), 0.0);
        terrain_color *= 0.3 + 0.7 * NdotL;

        float3 water_col = compute_water_color(input.water_depth, input.sphere_direction, input.world_pos);
        float blend = smoothstep(0.0, 0.5, input.water_depth);
        color = lerp(terrain_color, water_col, blend);
    } else {
        color = elevation_ramp(input.height_normalized);
        float3 L = normalize(sun_dir);
        float3 N = normalize(input.world_normal);
        float NdotL = max(dot(N, L), 0.0);
        float ambient = 0.3;
        float lighting = ambient + (1.0 - ambient) * NdotL;
        color *= lighting;
    }

    // Brush cursor ring
    float brush_radius = brush_world.w;
    if (brush_radius > 0.0) {
        float3 brush_dir = normalize(brush_world.xyz);
        float3 frag_dir = normalize(input.sphere_direction);
        float angular_dist = acos(clamp(dot(frag_dir, brush_dir), -1.0, 1.0));
        float ring = smoothstep(brush_radius * 0.9, brush_radius, angular_dist)
                   * smoothstep(brush_radius * 1.15, brush_radius, angular_dist);
        color = lerp(color, brush_color.rgb, ring * 0.8);
    }

    // Distance fog
    float dist = length(input.world_pos);
    float fog = saturate(dist / 50000000.0);
    float3 fog_color = float3(0.65, 0.75, 0.90);
    color = lerp(color, fog_color, fog);

    return float4(color, 1.0);
}

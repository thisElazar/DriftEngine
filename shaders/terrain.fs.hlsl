[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float time;
    float3 cam_pos;   float _pad2;
    float4 brush_world;
    float4 brush_color;
};

[[vk::binding(8, 0)]] SamplerState tex_sampler;
[[vk::binding(9, 0)]] Texture2DArray<float4> climate;  // r=sst g=base b=cur angle a=cur speed

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

float3 compute_water_color(float depth, float3 sphere_dir, float3 world_pos, float sst)
{
    float3 baseN = normalize(sphere_dir);
    float3 ref = (abs(baseN.y) > 0.999) ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    float3 east  = normalize(cross(ref, baseN));   // +longitude (zonal)
    float3 north = normalize(cross(baseN, east));   // +latitude

    // --- Zonal ocean swell that WRAPS the planet -----------------------------
    // Built from global latitude/longitude (NOT cube-face uv) so it is seam-free
    // across faces; the longitude wavenumbers are integers so the wave is also
    // continuous across the ±180° meridian. Crests run roughly N–S and drift
    // east/west around the globe, banded by latitude (Coriolis: trade-easterlies
    // vs mid-latitude westerlies) — Earth-like currents going around. Low
    // amplitude + low frequency → calm ocean sheen, no moiré.
    float lat = asin(clamp(baseN.y, -1.0, 1.0));
    float lon = atan2(baseN.z, baseN.x);
    float poleFade = saturate(cos(lat) * 1.3);            // calm the polar convergence
    // Integer longitude wavenumbers → continuous across the ±180° meridian; uniform
    // eastward drift → no band-boundary seams. Crests run ~N–S and circle the globe.
    float phase1 =  9.0 * lon + 2.0 * lat - time * 0.10;
    float phase2 = 14.0 * lon - 4.0 * lat - time * 0.07;
    float w1 = sin(phase1);
    float w2 = sin(phase2);
    float amp = 0.030 * poleFade;
    float3 N = normalize(baseN + amp * (w1 * east + 0.5 * w2 * north));
    float  meta = 0.5 + 0.5 * (0.6 * w1 + 0.4 * w2);

    float3 V = normalize(-world_pos);
    float3 L = normalize(sun_dir);

    // Shared water palette (matches river_overlay.fs), then tinted by SST so warm
    // and cold currents are visible: warm → teal, cold → steel blue.
    float3 shallow = float3(0.35, 0.60, 0.85);
    float3 mid     = float3(0.12, 0.35, 0.62);
    float3 deep_c  = float3(0.03, 0.13, 0.34);
    float depth_scale = max(max_elevation, 100.0);
    float t1 = smoothstep(0.0,  depth_scale * 0.05, depth);
    float t2 = smoothstep(depth_scale * 0.05, depth_scale * 0.3, depth);
    float3 base = lerp(lerp(shallow, mid, t1), deep_c, t2);

    float3 cold = float3(0.10, 0.24, 0.55);
    float3 warm = float3(0.18, 0.52, 0.60);
    base = lerp(base, lerp(cold, warm, saturate(sst)), 0.30);

    float3 crest = float3(0.60, 0.82, 1.00);
    base = lerp(base, crest, 0.10 * (0.5 + 0.5 * w1));

    float NdotV = saturate(dot(N, V));
    float fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);

    float3 R = reflect(-V, N);
    float sky_t = saturate(R.y * 0.5 + 0.5);
    float3 sky = lerp(float3(0.85, 0.88, 0.92), float3(0.30, 0.55, 0.85), sky_t);

    float3 H = normalize(L + V);
    float spec = pow(saturate(dot(N, H)), 64.0);
    float3 specular = sun_color * spec * 0.8;

    float NdotL = saturate(dot(N, L));
    float3 diffuse = base * (0.4 + 0.6 * NdotL) * (0.90 + 0.20 * meta);

    return lerp(diffuse, sky, fresnel) + specular;
}

float4 main(PSInput input) : SV_Target
{
    float3 color;

    // Sample the coarse climate field for sea-surface temperature at this point.
    // The cube-face uv comes from the tile uv + the tile's face/region push constants.
    float2 face_uv = float2(u_min + input.uv.x * tile_size, v_min + input.uv.y * tile_size);
    float2 clim_uv = face_uv * 0.5 + 0.5;
    float  sst     = climate.SampleLevel(tex_sampler, float3(clim_uv, float(face)), 0).r;

    float shore_band = max(max_elevation * 0.005, 0.5);
    if (input.water_depth > shore_band) {
        color = compute_water_color(input.water_depth, input.sphere_direction, input.world_pos, sst);
        color = lerp(color, float3(0.95, 0.95, 0.97), saturate(input.foam));
    } else if (input.water_depth > 0.0) {
        float3 terrain_color = elevation_ramp(input.height_normalized);
        float3 L = normalize(sun_dir);
        float3 N = normalize(input.world_normal);
        float NdotL = max(dot(N, L), 0.0);
        terrain_color *= 0.3 + 0.7 * NdotL;

        float3 water_col = compute_water_color(input.water_depth, input.sphere_direction, input.world_pos, sst);
        float blend = smoothstep(0.0, shore_band, input.water_depth);
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

    // Brush cursor — two protocols selected by brush_color.a:
    //   alpha < 0.5  (orbital): brush_world.xyz = sphere direction (unit),
    //                           brush_world.w   = angular radius (radians).
    //   alpha >= 0.5 (FP):      brush_world.xyz = camera-relative world pos
    //                                             of the cursor hit (meters),
    //                           brush_world.w   = meter radius of the dot.
    //
    // FP uses world-space distance because at planet scale the angular dot
    // product loses fp32 precision (sub-1e-7 rad is unresolvable, so a small
    // dot tints the whole screen). Camera-relative world coords stay precise.
    float brush_radius = brush_world.w;
    if (brush_radius > 0.0) {
        float t;
        if (brush_color.a >= 0.5) {
            float dist = length(input.world_pos - brush_world.xyz);
            t = smoothstep(brush_radius, brush_radius * 0.5, dist);
        } else {
            float3 brush_dir = normalize(brush_world.xyz);
            float3 frag_dir = normalize(input.sphere_direction);
            float angular_dist = acos(clamp(dot(frag_dir, brush_dir), -1.0, 1.0));
            t = smoothstep(brush_radius * 0.9, brush_radius, angular_dist)
              * smoothstep(brush_radius * 1.15, brush_radius, angular_dist);
        }
        color = lerp(color, brush_color.rgb, t * 0.85);
    }

    // Distance fog
    float dist = length(input.world_pos);
    float fog = saturate(dist / 50000000.0);
    float3 fog_color = float3(0.65, 0.75, 0.90);
    color = lerp(color, fog_color, fog);

    return float4(color, 1.0);
}

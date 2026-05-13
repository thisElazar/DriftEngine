[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float _pad1;
    float3 cam_pos;   float mud_visibility;
    float4 brush_world;
    float4 brush_color;
};

[[vk::binding(2, 0)]] Texture2D<float4> swe_output;
[[vk::binding(3, 0)]] Texture2D<float> sediment_tex;
[[vk::binding(4, 0)]] Texture2D<float4> atmo_render;
[[vk::binding(8, 0)]] SamplerState tex_sampler;

[[vk::push_constant]]
cbuffer PushConstants {
    float terrain_size;
    float heightmap_texel;
    float max_elevation;
    float cloud_opacity;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_pos : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float2 uv : TEXCOORD2;
    [[vk::location(3)]] float depth : TEXCOORD3;
};

float3 sky_color(float3 view_dir)
{
    float t = saturate(view_dir.y);
    return lerp(float3(0.85, 0.88, 0.92), float3(0.30, 0.55, 0.85), t);
}

float4 main(PSInput input) : SV_Target
{
    if (input.depth < 0.05) discard;

    float4 swe = swe_output.SampleLevel(tex_sampler, input.uv, 0);
    float depth = max(swe.r - (input.world_pos.y - input.depth), 0.0);
    float foam = saturate(swe.a);
    if (depth < 0.05) discard;

    float3 N = normalize(input.normal);
    float3 V = normalize(cam_pos - input.world_pos);
    float3 L = normalize(sun_dir);
    float3 R = reflect(-V, N);

    float sed = sediment_tex.SampleLevel(tex_sampler, input.uv, 0);
    float3 mud_color = float3(0.45, 0.30, 0.18);
    float mud_blend = saturate(sed * mud_visibility);

    float3 shallow = lerp(float3(0.55, 0.85, 0.90), mud_color, mud_blend);
    float3 mid     = lerp(float3(0.10, 0.40, 0.65), mud_color, mud_blend);
    float3 deep_c  = lerp(float3(0.02, 0.10, 0.22), mud_color, mud_blend);
    float t1 = smoothstep(0.0,  20.0, depth);
    float t2 = smoothstep(20.0, 200.0, depth);
    float3 base = lerp(lerp(shallow, mid, t1), deep_c, t2);

    float NdotV = saturate(dot(N, V));
    float F0 = 0.02;
    float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    float3 sky = sky_color(R);

    float3 H = normalize(L + V);
    float spec = pow(saturate(dot(N, H)), 64.0);
    float3 specular = sun_color * spec * 0.8;

    float NdotL = saturate(dot(N, L));
    float3 diffuse = base * (0.4 + 0.6 * NdotL);

    float3 color = lerp(diffuse, sky, fresnel) + specular;

    color = lerp(color, float3(0.95, 0.95, 0.97), foam);

    float alpha = saturate(smoothstep(0.05, 2.0, depth) * 0.85 + foam * 0.15);

    if (brush_world.w > 0.5) {
        float2 to_cursor = input.world_pos.xz - brush_world.xy;
        float d = length(to_cursor);
        float ring_thickness = 4.0;
        float ring = exp(-pow((d - brush_world.z) / ring_thickness, 2.0));
        color = lerp(color, brush_color.rgb, saturate(ring) * 0.85);
    }

    float cloud_shadow = atmo_render.SampleLevel(tex_sampler, input.uv, 0).r;
    float shadow = 1.0 - saturate(cloud_shadow * cloud_opacity) * 0.6;
    color *= shadow;

    return float4(color, alpha);
}

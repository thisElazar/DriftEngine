// cloud_raymarch.fs.hlsl — Raymarch through 3D cloud volume
// Reads camera UBO (binding 0), heightmap (binding 1), 3D volume (binding 5)

[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float _pad1;
    float3 cam_pos;   float _pad2;
    float4 brush_world;
    float4 brush_color;
    float4x4 inv_view_proj;
};

[[vk::binding(1, 0)]] Texture2D<float>  heightmap;
[[vk::binding(5, 0)]] Texture3D<float4> cloud_vol;
[[vk::binding(8, 0)]] SamplerState tex_sampler;

[[vk::push_constant]]
cbuffer RaymarchPC {
    float terrain_size;
    float max_elevation;
    float cloud_opacity;
    float cloud_base;
    uint  vol_w;
    uint  vol_h;
    uint  vol_d;
    float layer_height;
};

struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

float2 world_to_uv(float3 p)
{
    return float2(p.x / terrain_size + 0.5, p.z / terrain_size + 0.5);
}

float get_terrain_height(float3 p)
{
    float2 uv = world_to_uv(p);
    if (uv.x < 0 || uv.x > 1 || uv.y < 0 || uv.y > 1)
        return 0;
    return heightmap.SampleLevel(tex_sampler, uv, 0).r;
}

float3 world_to_vol_uvw(float3 p)
{
    float2 uv = world_to_uv(p);
    float w = (p.y - cloud_base) / (vol_d * layer_height);
    return float3(uv, w);
}

// Box intersection: AABB defined by the volume extent
float2 intersect_box(float3 ro, float3 rd, float3 bmin, float3 bmax)
{
    float3 inv_rd = 1.0 / rd;
    float3 t0 = (bmin - ro) * inv_rd;
    float3 t1 = (bmax - ro) * inv_rd;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float tn = max(max(tmin.x, tmin.y), tmin.z);
    float tf = min(min(tmax.x, tmax.y), tmax.z);
    return float2(tn, tf);
}

static const int MAX_STEPS = 64;
static const float EXTINCTION = 8.0;
static const float SCATTER_COEFF = 0.7;

float4 main(PSInput input) : SV_Target
{
    // Reconstruct world-space ray
    float2 ndc = float2(input.uv.x, 1.0 - input.uv.y) * 2.0 - 1.0;
    float4 clip_near = float4(ndc, 0.0, 1.0);
    float4 clip_far  = float4(ndc, 1.0, 1.0);
    float4 wn = mul(inv_view_proj, clip_near);
    float4 wf = mul(inv_view_proj, clip_far);
    wn /= wn.w;
    wf /= wf.w;

    float3 ro = wn.xyz;
    float3 rd = normalize(wf.xyz - wn.xyz);

    // Volume bounding box in world space
    float half_ts = terrain_size * 0.5;
    float vol_top = cloud_base + vol_d * layer_height;
    float3 bmin = float3(-half_ts, cloud_base, -half_ts);
    float3 bmax = float3( half_ts, vol_top,     half_ts);

    float2 t_hit = intersect_box(ro, rd, bmin, bmax);
    t_hit.x = max(t_hit.x, 0.0);

    if (t_hit.x >= t_hit.y)
        discard;

    float step_size = (t_hit.y - t_hit.x) / float(MAX_STEPS);
    float3 light_dir = normalize(sun_dir);

    float transmittance = 1.0;
    float3 accumulated = float3(0, 0, 0);

    for (int i = 0; i < MAX_STEPS; i++)
    {
        float t = t_hit.x + (float(i) + 0.5) * step_size;
        float3 pos = ro + rd * t;

        // Skip if below terrain
        float th = get_terrain_height(pos);
        if (pos.y < th)
            continue;

        float3 uvw = world_to_vol_uvw(pos);
        if (uvw.z < 0.0 || uvw.z > 1.0)
            continue;

        float4 sample_val = cloud_vol.SampleLevel(tex_sampler, uvw, 0);
        float density = sample_val.r * cloud_opacity;
        if (density < 0.001)
            continue;

        // Beer-Lambert extinction
        float extinction = density * EXTINCTION * step_size;
        float beer = exp(-extinction);

        // Simple sun scattering: march a few steps toward sun
        float sun_transmit = 1.0;
        float sun_step = layer_height * 0.5;
        for (int s = 1; s <= 4; s++)
        {
            float3 sp = pos + light_dir * sun_step * float(s);
            float3 suvw = world_to_vol_uvw(sp);
            if (suvw.z >= 0.0 && suvw.z <= 1.0) {
                float sd = cloud_vol.SampleLevel(tex_sampler, suvw, 0).r;
                sun_transmit *= exp(-sd * EXTINCTION * sun_step * cloud_opacity * 0.5);
            }
        }

        // Phase function: Henyey-Greenstein with forward scatter
        float cos_theta = dot(rd, light_dir);
        float g = 0.4;
        float phase = (1.0 - g * g) / (4.0 * 3.14159 * pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));

        float3 lighting = sun_color * sun_transmit * phase * SCATTER_COEFF;
        float3 ambient = float3(0.35, 0.4, 0.5);

        float3 mat_color = (lighting + ambient);

        // Precipitation darkening
        float precip_val = sample_val.a;
        mat_color *= lerp(1.0, 0.4, saturate(precip_val * 2.0));

        accumulated += transmittance * (1.0 - beer) * mat_color;
        transmittance *= beer;

        if (transmittance < 0.01)
            break;
    }

    float alpha = 1.0 - transmittance;
    if (alpha < 0.001)
        discard;

    return float4(accumulated, alpha);
}

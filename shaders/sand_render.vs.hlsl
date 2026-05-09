// sand_render.vs.hlsl — Render sand particles as velocity streaks (line list)
// 2 vertices per particle: head position and tail (position - velocity * streak)

struct Particle {
    float px, py, pz;
    float vx, vy, vz;
    float lifetime;
    float size;
};

[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float _pad1;
    float3 cam_pos;   float _pad2;
};

[[vk::binding(1, 0)]] StructuredBuffer<Particle> particles;

[[vk::push_constant]]
cbuffer SandRenderPC {
    float streak_length;
    float particle_alpha;
    uint  max_particles;
    float _pad;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float alpha : TEXCOORD0;
    [[vk::location(1)]] float dist : TEXCOORD1;
};

VSOutput main(uint vid : SV_VertexID)
{
    VSOutput o;

    uint pidx = vid / 2;
    uint end = vid & 1;

    if (pidx >= max_particles)
    {
        o.position = float4(0, 0, -10, 1);
        o.alpha = 0;
        o.dist = 0;
        return o;
    }

    Particle p = particles[pidx];

    if (p.lifetime <= 0.0)
    {
        o.position = float4(0, 0, -10, 1);
        o.alpha = 0;
        o.dist = 0;
        return o;
    }

    float3 pos = float3(p.px, p.py, p.pz);
    float3 vel = float3(p.vx, p.vy, p.vz);

    if (end == 1)
        pos -= vel * streak_length;

    float4 clip = mul(proj, mul(view, float4(pos, 1.0)));

    float fade = saturate(p.lifetime * 0.5);
    float cam_dist = length(pos - cam_pos);
    float dist_fade = saturate(1.0 - cam_dist / 8000.0);

    o.position = clip;
    o.alpha = fade * dist_fade * particle_alpha;
    o.dist = cam_dist;
    return o;
}

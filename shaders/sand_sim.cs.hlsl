// sand_sim.cs.hlsl — Combined emit + simulate for GPU sand particles
// Each thread handles one particle slot in the ring buffer.
// Dead slots in the emit range get new particles; live ones get simulated.

struct Particle {
    float px, py, pz;
    float vx, vy, vz;
    float lifetime;
    float size;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D<float>  terrain_tex;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState      terrain_sampler;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture3D<float4> wind_vol;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState      wind_sampler;
[[vk::binding(2, 0)]] RWStructuredBuffer<Particle> particles;
[[vk::combinedImageSampler]][[vk::binding(3, 0)]] Texture2D<float>  sand_deposit;
[[vk::combinedImageSampler]][[vk::binding(3, 0)]] SamplerState      sand_deposit_sampler;

[[vk::push_constant]]
cbuffer SandSimPC {
    float dt;
    float terrain_size;
    float loft_threshold;
    float loft_rate;
    float gravity;
    float accumulated_time;
    uint  max_particles;
    uint  emit_offset;
    uint  emit_count;
    uint  grid_d;
    float layer_height;
    float bounce_energy;
};

float hash(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float2 world_to_uv(float3 pos)
{
    return float2(pos.x / terrain_size + 0.5, pos.z / terrain_size + 0.5);
}

float3 sample_wind(float3 world_pos)
{
    float2 uv = world_to_uv(world_pos);
    float w = world_pos.y / (grid_d * layer_height);
    float3 uvw = float3(uv, clamp(w, 0.01, 0.99));
    float4 wind = wind_vol.SampleLevel(wind_sampler, uvw, 0);
    // wind layout: (horizontal_x, horizontal_y, vertical, sand_conc)
    // Map to world: wind.x → world X, wind.y → world Z, wind.z → world Y
    return float3(wind.x, wind.z, wind.y);
}

float get_terrain_h(float2 uv)
{
    return terrain_tex.SampleLevel(terrain_sampler, uv, 0).r;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= max_particles) return;

    Particle p = particles[idx];

    // --- EMIT ---
    bool in_range = false;
    if (emit_offset + emit_count <= max_particles) {
        in_range = (idx >= emit_offset && idx < emit_offset + emit_count);
    } else {
        in_range = (idx >= emit_offset || idx < (emit_offset + emit_count) % max_particles);
    }

    if (p.lifetime <= 0.0 && in_range)
    {
        float seed = float(idx);
        float2 rand_uv = float2(
            frac(hash(float2(seed * 1.731, accumulated_time * 7.13))),
            frac(hash(float2(seed * 2.419, accumulated_time * 5.37 + 42.0)))
        );

        float terrain_h = get_terrain_h(rand_uv);
        float3 wind = sample_wind(float3(
            (rand_uv.x - 0.5) * terrain_size,
            terrain_h + layer_height * 0.5,
            (rand_uv.y - 0.5) * terrain_size
        ));

        float ws = length(float2(wind.x, wind.z));

        float supply = sand_deposit.SampleLevel(sand_deposit_sampler, rand_uv, 0).r;

        if (ws > loft_threshold && supply > 0.01)
        {
            float excess = ws - loft_threshold;

            p.px = (rand_uv.x - 0.5) * terrain_size;
            p.py = terrain_h + 0.5;
            p.pz = (rand_uv.y - 0.5) * terrain_size;

            float r1 = hash(float2(seed * 3.17, accumulated_time * 2.1)) - 0.5;
            float r2 = hash(float2(seed * 4.23, accumulated_time * 3.7)) - 0.5;
            float r3 = hash(float2(seed * 5.89, accumulated_time * 1.3));

            p.vx = wind.x * 0.4 + r1 * excess * 0.3;
            p.vy = excess * loft_rate * (0.5 + r3 * 0.8) * saturate(supply);
            p.vz = wind.z * 0.4 + r2 * excess * 0.3;

            p.lifetime = 2.0 + hash(float2(seed * 6.31, accumulated_time)) * 6.0 * saturate(supply);
            p.size = 0.3 + hash(float2(seed * 7.11, accumulated_time + 1.0)) * 0.7;
        }
    }

    // --- SIMULATE ---
    if (p.lifetime > 0.0)
    {
        float2 uv = world_to_uv(float3(p.px, p.py, p.pz));

        // Kill if out of world bounds
        if (uv.x < -0.05 || uv.x > 1.05 || uv.y < -0.05 || uv.y > 1.05)
        {
            p.lifetime = 0;
            particles[idx] = p;
            return;
        }

        // Wind drag
        float3 wind = sample_wind(float3(p.px, p.py, p.pz));
        float drag = 3.0;
        p.vx += (wind.x * 0.8 - p.vx) * drag * dt;
        p.vy += (wind.y * 0.3 - p.vy) * drag * dt;
        p.vz += (wind.z * 0.8 - p.vz) * drag * dt;

        // Gravity
        p.vy -= gravity * dt;

        // Integrate position
        p.px += p.vx * dt;
        p.py += p.vy * dt;
        p.pz += p.vz * dt;

        // Terrain collision — saltation bounce
        float terrain_h = get_terrain_h(uv);
        if (p.py < terrain_h)
        {
            p.py = terrain_h;
            p.vy = abs(p.vy) * bounce_energy;

            float ground_ws = length(float2(p.vx, p.vz));
            if (ground_ws < loft_threshold * 0.4 && abs(p.vy) < 0.5)
                p.lifetime = 0;
        }

        // Altitude cap
        if (p.py > grid_d * layer_height)
            p.vy = min(p.vy, 0.0);

        p.lifetime -= dt;
    }

    particles[idx] = p;
}

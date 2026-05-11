[[vk::binding(0, 0)]] RWTexture2D<float>  terrain;
[[vk::binding(1, 0)]] Texture2D<float4>  swe_state;
[[vk::binding(2, 0)]] Texture2D<float>   sediment_in;
[[vk::binding(3, 0)]] RWTexture2D<float> sediment_out;
[[vk::combinedImageSampler]][[vk::binding(4, 0)]] Texture2D<float4> ground_cond;
[[vk::combinedImageSampler]][[vk::binding(4, 0)]] SamplerState      ground_cond_sampler;
[[vk::combinedImageSampler]][[vk::binding(5, 0)]] Texture2D<float2> ground_wind;
[[vk::combinedImageSampler]][[vk::binding(5, 0)]] SamplerState      ground_wind_sampler;

[[vk::push_constant]]
cbuffer PC {
    float dt;
    float dx;
    uint  grid_w;
    uint  grid_h;

    float k_erosion;
    float k_deposit;
    float k_capacity;
    float min_slope;

    float min_depth;
    float max_change;
    float max_sediment;
    float k_wind;

    float k_thermal;
    float wind_threshold;
    uint  _pad0;
    uint  _pad1;
};

int2 clamp_coord(int2 c, int2 maxc) {
    return clamp(c, int2(0, 0), maxc - 1);
}

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    int2 c = int2(dtid.xy);
    if (c.x >= (int)grid_w || c.y >= (int)grid_h) return;
    int2 mx = int2(grid_w, grid_h);

    float t = terrain[c];
    float4 s = swe_state[c];
    float depth = s.r;
    float u = (depth > 1e-3) ? s.g / depth : 0.0;
    float v = (depth > 1e-3) ? s.b / depth : 0.0;
    float speed = sqrt(u * u + v * v);

    float tL = terrain[clamp_coord(c + int2(-1, 0), mx)];
    float tR = terrain[clamp_coord(c + int2( 1, 0), mx)];
    float tD = terrain[clamp_coord(c + int2(0, -1), mx)];
    float tU = terrain[clamp_coord(c + int2(0,  1), mx)];
    float dzdx = (tR - tL) / (2.0 * dx);
    float dzdy = (tU - tD) / (2.0 * dx);
    float slope = sqrt(dzdx * dzdx + dzdy * dzdy);
    slope = max(slope, min_slope);

    float sin_slope = slope / sqrt(1.0 + slope * slope);
    float capacity = k_capacity * sin_slope * speed * depth;

    float2 distFromEdge = float2(
        min((float)c.x, (float)(grid_w - 1 - c.x)),
        min((float)c.y, (float)(grid_h - 1 - c.y))
    );
    float edge_dist = min(distFromEdge.x, distFromEdge.y);
    float sponge_t = saturate(1.0 - edge_dist / 64.0);
    float edge_factor = 1.0 - sponge_t * sponge_t;

    // Advect sediment FIRST so erosion/deposition compares against what actually arrived
    float2 back_pos = float2(c) - float2(u, v) * dt / dx;
    int2 ba = int2(floor(back_pos));
    int2 bb = ba + int2(1, 0);
    int2 bc_ = ba + int2(0, 1);
    int2 bd_ = ba + int2(1, 1);
    float2 frac = back_pos - float2(ba);

    float sa_ = sediment_in[clamp_coord(ba,  mx)];
    float sb_ = sediment_in[clamp_coord(bb,  mx)];
    float sc_ = sediment_in[clamp_coord(bc_, mx)];
    float sd_ = sediment_in[clamp_coord(bd_, mx)];
    float advected =
        lerp(lerp(sa_, sb_, frac.x), lerp(sc_, sd_, frac.x), frac.y);

    float terrain_delta = 0.0;
    float sediment_delta = 0.0;
    float max_delta = max_change * dt;

    if (depth > min_depth) {
        if (advected < capacity) {
            float lift = k_erosion * (capacity - advected) * dt;
            lift = min(lift, max_delta);
            terrain_delta  -= lift;
            sediment_delta += lift;
        } else {
            float drop = k_deposit * (advected - capacity) * dt;
            drop = min(drop, max_delta);
            drop = min(drop, advected);
            terrain_delta  += drop;
            sediment_delta -= drop;
        }
    } else {
        float dump = min(advected, max_delta);
        terrain_delta  += dump;
        sediment_delta -= dump;
    }

    // --- Wind shear erosion (exposed dry terrain) ---
    if (k_wind > 0.0 && depth < min_depth) {
        float2 uv = (float2(c) + 0.5) / float2(grid_w, grid_h);
        float4 gc = ground_cond.SampleLevel(ground_cond_sampler, uv, 0);
        float wind_speed = gc.a;
        float excess_wind = wind_speed - wind_threshold;
        if (excess_wind > 0.0) {
            float wind_erosion = k_wind * excess_wind * slope * dt;
            wind_erosion = min(wind_erosion, max_change * dt);
            terrain_delta -= wind_erosion;
            sediment_delta += wind_erosion;
        }

        // --- Thermal erosion (freeze-thaw near 273K on steep slopes) ---
        if (k_thermal > 0.0) {
            float temperature = gc.g;
            float freeze_thaw = saturate(1.0 - abs(temperature - 273.15) / 5.0);
            if (freeze_thaw > 0.0 && slope > min_slope * 2.0) {
                float thermal_erosion = k_thermal * freeze_thaw * slope * dt;
                thermal_erosion = min(thermal_erosion, max_change * dt);
                terrain_delta -= thermal_erosion;
                sediment_delta += thermal_erosion;
            }
        }
    }

    terrain_delta *= edge_factor;
    sediment_delta *= edge_factor;

    terrain[c] = t + terrain_delta;

    float new_sediment = clamp(advected + sediment_delta, 0.0, max_sediment);
    sediment_out[c] = new_sediment;
}

// atmosphere3d.cs.hlsl — 3D volumetric atmosphere compute shader
// 128x128x32 voxel grid: each cell = 80m horizontal, 100m vertical
// State RGBA16F: (cloud_density, moisture, temperature/400, precipitation)
// Wind  RGBA16F: (vx, vy, vz, pressure)

#define WG_X 4
#define WG_Y 4
#define WG_Z 4
#define TWO_PI 6.28318530718
#define PI     3.14159265359

[[vk::combinedImageSampler]][[vk::binding(0, 0)]] Texture2D<float>   terrain_tex;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]] SamplerState        terrain_sampler;
[[vk::binding(1, 0)]] Texture3D<float4>   state_read;
[[vk::binding(2, 0)]] RWTexture3D<float4> state_write;
[[vk::binding(3, 0)]] Texture3D<float4>   wind_read;
[[vk::binding(4, 0)]] RWTexture3D<float4> wind_write;
[[vk::binding(5, 0)]] RWTexture2D<float>  shadow_out;
[[vk::binding(6, 0)]] RWTexture2D<float4> ground_cond_out;
[[vk::binding(7, 0)]] RWTexture2D<float2> ground_wind_out;
[[vk::combinedImageSampler]][[vk::binding(8, 0)]] Texture2D<float4> swe_output;
[[vk::combinedImageSampler]][[vk::binding(8, 0)]] SamplerState      swe_sampler;

[[vk::push_constant]]
cbuffer PC {
    float dt;
    float accumulated_time;
    uint  grid_w;
    uint  grid_h;
    uint  grid_d;
    float terrain_scale;
    float layer_height;
    float max_elevation;
    float orographic_lift_coeff;
    float adiabatic_cooling_rate;
    float rain_shadow_intensity;
    uint  force_init;
    float k_pressure;
    float wind_strength;
    float k_evaporation;
};

static const float LAPSE_RATE       = 6.5;  // °C per 1000m
static const float BASE_TEMP        = 288.15;
static const float CONDENSATION_RATE = 0.015;
static const float EVAPORATION_RATE  = 0.005;
static const float PRECIP_THRESHOLD  = 0.2;
static const float CLOUD_DISSIPATION = 0.001;
static const float WIND_MOMENTUM     = 0.97;
static const float WIND_FORCING      = 0.10;
static const float WIND_DIFFUSION    = 0.03;
static const float WIND_DAMPING      = 0.999;
static const float BUOYANCY_COEFF    = 0.008;
static const float LATENT_HEAT       = 4.0;
static const float PRECIP_FALL_RATE  = 0.08;
static const float TURBULENCE_STR    = 0.25;
static const float VORTICITY_STR     = 0.15;
static const float PREVAILING_STRENGTH = 1.5;
static const float PRESSURE_BASE       = 1013.25; // hPa at sea level
static const float PRESSURE_SCALE_H    = 8500.0;  // scale height in meters

float hash(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float noise2d(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + float2(1, 0));
    float c = hash(i + float2(0, 1));
    float d = hash(i + float2(1, 1));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float ComputeSaturationPressure(float T)
{
    float Tc = T - 273.15;
    return 611.0 * exp((17.27 * Tc) / (Tc + 237.3));
}

int3 WrapCoord(int3 c)
{
    int3 sz = int3(grid_w, grid_h, grid_d);
    return ((c % sz) + sz) % sz;
}

int3 WrapXY(int3 c)
{
    c.xy = ((c.xy % int2(grid_w, grid_h)) + int2(grid_w, grid_h)) % int2(grid_w, grid_h);
    c.z = clamp(c.z, 0, (int)grid_d - 1);
    return c;
}

[numthreads(WG_X, WG_Y, WG_Z)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= grid_w || tid.y >= grid_h || tid.z >= grid_d)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(grid_w, grid_h);
    float altitude = (float(tid.z) + 0.5) * layer_height;
    float terrain_h = terrain_tex.SampleLevel(terrain_sampler, uv, 0).r;

    // --- INIT ---
    if (force_init == 1)
    {
        float alt_norm = altitude / (grid_d * layer_height);
        float temp = BASE_TEMP - LAPSE_RATE * (altitude / 1000.0);
        float moisture = 0.5 + noise2d(uv * 5.0) * 0.15;
        moisture *= saturate(1.0 - alt_norm * 1.2);

        float init_pressure = PRESSURE_BASE * exp(-altitude / PRESSURE_SCALE_H);
        if (altitude < terrain_h) {
            state_write[tid] = float4(0, 0, temp / 400.0, 0);
            wind_write[tid] = float4(0, 0, 0, init_pressure);
        } else {
            float2 base_wind = float2(
                cos(uv.y * PI) * PREVAILING_STRENGTH,
                sin(uv.x * PI * 0.5) * PREVAILING_STRENGTH * 0.3
            );
            float shear = 1.0 + alt_norm * 0.8;
            base_wind *= shear;

            float init_cloud = 0.0;
            state_write[tid] = float4(init_cloud, moisture, temp / 400.0, 0);
            wind_write[tid] = float4(base_wind, 0, init_pressure);
        }

        if (tid.z == 0) {
            shadow_out[tid.xy] = 0.0;
            ground_cond_out[tid.xy] = float4(0, temp, moisture, 0);
            ground_wind_out[tid.xy] = float2(0, 0);
        }
        return;
    }

    // --- READ STATE ---
    float4 st = state_read[tid];
    float cloud   = st.r;
    float moisture = st.g;
    float temp     = st.b * 400.0;
    float precip   = st.a;

    float4 w = wind_read[tid];
    float3 wind = w.xyz;

    bool underground = (altitude < terrain_h);
    if (underground) {
        float ground_pressure = PRESSURE_BASE * exp(-altitude / PRESSURE_SCALE_H);
        state_write[tid] = float4(0, 0, temp / 400.0, 0);
        wind_write[tid] = float4(0, 0, 0, ground_pressure);
        if (tid.z == 0) {
            float col_shadow = 0.0;
            for (uint zz = 0; zz < grid_d; zz++)
                col_shadow += state_read[uint3(tid.xy, zz)].r;
            shadow_out[tid.xy] = saturate(col_shadow * 0.15);
            // Ground conditions: column-integrated rainfall + lowest above-ground layer
            float col_rain = 0.0;
            float g_temp = BASE_TEMP;
            float g_moist = 0.0;
            float2 g_wind = float2(0, 0);
            for (uint gz = 0; gz < grid_d; gz++) {
                float4 gs = state_read[uint3(tid.xy, gz)];
                col_rain += gs.a;
                float gz_alt = (float(gz) + 0.5) * layer_height;
                if (gz_alt >= terrain_h && g_moist == 0.0) {
                    g_temp = gs.b * 400.0;
                    g_moist = gs.g;
                    g_wind = wind_read[uint3(tid.xy, gz)].xy;
                }
            }
            ground_cond_out[tid.xy] = float4(saturate(col_rain * 0.1), g_temp, g_moist, length(g_wind));
            ground_wind_out[tid.xy] = g_wind;
        }
        return;
    }

    // --- TEMPERATURE ---
    float target_temp = BASE_TEMP - LAPSE_RATE * (altitude / 1000.0);
    float solar = max(0.0, sin(accumulated_time * 0.0001 * TWO_PI));
    float heating = 0.3 * solar * (1.0 - cloud * 0.5);
    if (tid.z == 0 || altitude < terrain_h + layer_height * 1.5)
        heating *= 1.5;
    temp += heating * dt;
    float cooling = (1.0 - solar) * 0.05 * (1.0 - moisture * 0.3);
    temp -= cooling * dt;
    temp = lerp(temp, target_temp, 0.02 * dt);

    // --- EVAPORATION (surface water -> atmospheric moisture) ---
    if (k_evaporation > 0.0 && altitude < terrain_h + layer_height * 2.0 && altitude >= terrain_h) {
        float4 swe = swe_output.SampleLevel(swe_sampler, uv, 0);
        float water_depth = max(swe.r - terrain_h, 0.0);
        if (water_depth > 0.01) {
            float sat_deficit = saturate(1.0 - moisture);
            float temp_factor = saturate((temp - 273.0) / 30.0);
            float wind_factor = 1.0 + length(wind) * 0.3;
            float evap = k_evaporation * sat_deficit * temp_factor * wind_factor
                       * saturate(water_depth * 2.0) * dt;
            moisture = min(moisture + evap, 0.98);
        }
    }

    // --- WIND DYNAMICS ---
    int3 itid = int3(tid);
    int3 xp = WrapXY(itid + int3(1, 0, 0));
    int3 xm = WrapXY(itid + int3(-1, 0, 0));
    int3 yp = WrapXY(itid + int3(0, 1, 0));
    int3 ym = WrapXY(itid + int3(0, -1, 0));
    int3 zp = WrapXY(itid + int3(0, 0, 1));
    int3 zm = WrapXY(itid + int3(0, 0, -1));

    float3 wXp = wind_read[xp].xyz;
    float3 wXm = wind_read[xm].xyz;
    float3 wYp = wind_read[yp].xyz;
    float3 wYm = wind_read[ym].xyz;
    float3 wZp = wind_read[zp].xyz;
    float3 wZm = wind_read[zm].xyz;

    float3 wind_avg = (wXp + wXm + wYp + wYm + wZp + wZm) / 6.0;

    // Pressure gradient force
    float pXp = wind_read[xp].a;
    float pXm = wind_read[xm].a;
    float pYp = wind_read[yp].a;
    float pYm = wind_read[ym].a;
    float pZp = wind_read[zp].a;
    float pZm = wind_read[zm].a;
    float2 pressure_grad_h = float2(pXp - pXm, pYp - pYm) * 0.5;
    float  pressure_grad_v = (pZp - pZm) * 0.5;

    // Prevailing wind with altitude-dependent shear
    float alt_norm = altitude / (grid_d * layer_height);
    float pw = PREVAILING_STRENGTH * wind_strength;
    float2 prevailing = float2(
        cos(accumulated_time * 0.00003 + uv.y * PI * 0.5) * pw,
        sin(accumulated_time * 0.00002 + uv.x * PI * 0.3) * pw * 0.4
    );
    float shear_angle = alt_norm * 0.8;
    float2 rotated = float2(
        prevailing.x * cos(shear_angle) - prevailing.y * sin(shear_angle),
        prevailing.x * sin(shear_angle) + prevailing.y * cos(shear_angle)
    );
    float shear_speed = 1.0 + alt_norm * 0.6;

    // Buoyancy: warm air rises
    float temp_anomaly = temp - target_temp;
    float buoyancy = temp_anomaly * BUOYANCY_COEFF;

    // Orographic lift: wind hitting terrain slope pushes air upward
    float vert_force = 0.0;
    if (altitude < terrain_h + layer_height * 2.0 && altitude >= terrain_h)
    {
        float2 texel = 1.0 / float2(grid_w, grid_h);
        float hR = terrain_tex.SampleLevel(terrain_sampler, uv + float2(texel.x, 0), 0).r;
        float hL = terrain_tex.SampleLevel(terrain_sampler, uv - float2(texel.x, 0), 0).r;
        float hU = terrain_tex.SampleLevel(terrain_sampler, uv + float2(0, texel.y), 0).r;
        float hD = terrain_tex.SampleLevel(terrain_sampler, uv - float2(0, texel.y), 0).r;

        float2 grad = float2(hR - hL, hU - hD) / (2.0 * terrain_scale);
        float windspeed_h = length(wind.xy);
        if (windspeed_h > 0.01) {
            float2 wind_dir = normalize(wind.xy);
            vert_force = dot(wind_dir, grad) * windspeed_h * orographic_lift_coeff;
        }
    }

    // Vorticity confinement (horizontal)
    float2 gradX = (wXp.xy - wXm.xy) * 0.5;
    float2 gradY = (wYp.xy - wYm.xy) * 0.5;
    float vorticity = gradX.y - gradY.x;
    float2 vort_grad = float2(
        (wXp.y - wXm.y - wXp.x + wXm.x) * 0.5,
        (wYp.y - wYm.y - wYp.x + wYm.x) * 0.5
    );
    float vort_mag = length(vort_grad);
    float2 vort_force = float2(0, 0);
    if (vort_mag > 0.0001)
    {
        float2 vn = normalize(vort_grad);
        vort_force = VORTICITY_STR * float2(-vn.y, vn.x) * vorticity;
    }

    // Turbulence
    float2 turb = float2(0, 0);
    float amp = TURBULENCE_STR;
    float freq = 3.0;
    float2 toff = float2(accumulated_time * 0.008, accumulated_time * 0.011);
    for (int oct = 0; oct < 3; oct++) {
        float2 sp = uv * freq + toff + float2(alt_norm * 2.0, 0);
        turb += float2(noise2d(sp) * 2.0 - 1.0, noise2d(sp + 100.0) * 2.0 - 1.0) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }

    // Integrate wind
    float3 new_wind = wind * WIND_MOMENTUM;
    new_wind.xy += rotated * shear_speed * WIND_FORCING;
    new_wind.xy += vort_force * WIND_FORCING;
    new_wind.xy += turb * WIND_FORCING;
    new_wind.xy -= pressure_grad_h * k_pressure * dt;
    new_wind.z += (buoyancy + vert_force - pressure_grad_v * k_pressure) * dt;
    new_wind = lerp(new_wind, wind_avg, WIND_DIFFUSION);
    new_wind *= WIND_DAMPING;

    // Soft cap (scaled by wind_strength so strong winds can develop)
    float ws_cap = 3.5 * max(wind_strength, 1.0);
    float ws = length(new_wind);
    if (ws > ws_cap)
        new_wind = normalize(new_wind) * (ws_cap + (ws - ws_cap) * 0.2);

    // --- ADVECTION (semi-Lagrangian 3D) ---
    float3 cell_size = float3(
        float(grid_w),
        float(grid_h),
        float(grid_d)
    );
    float3 back_pos = float3(tid) + 0.5 - new_wind * dt * 0.001 * cell_size / float3(terrain_scale * grid_w, terrain_scale * grid_h, grid_d * layer_height) * float3(terrain_scale * grid_w, terrain_scale * grid_h, grid_d * layer_height) / float3(1,1,1);

    // Simpler: advect in grid-space
    float3 vel_grid;
    vel_grid.x = new_wind.x * dt * 0.001 * float(grid_w);
    vel_grid.y = new_wind.y * dt * 0.001 * float(grid_h);
    vel_grid.z = new_wind.z * dt * 0.01;

    float3 src = float3(tid) + 0.5 - vel_grid;
    src.xy = frac(src.xy / float2(grid_w, grid_h) + float2(1, 1)) * float2(grid_w, grid_h);
    src.z = clamp(src.z, 0.5, float(grid_d) - 0.5);

    int3 base = int3(floor(src - 0.5));
    float3 f = frac(src - 0.5);

    float4 s000 = state_read[WrapXY(base + int3(0,0,0))];
    float4 s100 = state_read[WrapXY(base + int3(1,0,0))];
    float4 s010 = state_read[WrapXY(base + int3(0,1,0))];
    float4 s110 = state_read[WrapXY(base + int3(1,1,0))];
    float4 s001 = state_read[WrapXY(base + int3(0,0,1))];
    float4 s101 = state_read[WrapXY(base + int3(1,0,1))];
    float4 s011 = state_read[WrapXY(base + int3(0,1,1))];
    float4 s111 = state_read[WrapXY(base + int3(1,1,1))];

    float4 adv = lerp(
        lerp(lerp(s000, s100, f.x), lerp(s010, s110, f.x), f.y),
        lerp(lerp(s001, s101, f.x), lerp(s011, s111, f.x), f.y),
        f.z
    );

    cloud    = adv.r;
    moisture = adv.g;

    // --- CLOUD PHYSICS ---
    float sat_p = ComputeSaturationPressure(temp);
    float vap_p = moisture * sat_p;

    // Condensation
    if (vap_p > sat_p * 0.92)
    {
        float cond = (vap_p - sat_p) / sat_p;
        cond = min(cond * CONDENSATION_RATE, moisture);
        cloud += cond;
        moisture -= cond;
        temp += cond * LATENT_HEAT * dt;
    }

    // Orographic condensation
    if (vert_force > 0.01)
    {
        float oro_cond = vert_force * moisture * 0.3;
        cloud += oro_cond * dt;
        moisture = max(moisture - oro_cond * dt, 0.0);
        temp -= oro_cond * adiabatic_cooling_rate * dt;
    }

    // Convective clouds
    if (temp_anomaly > 2.0 && moisture > 0.2)
    {
        float conv = (temp_anomaly - 2.0) * moisture * 0.002;
        cloud += conv * dt;
        moisture -= conv * dt * 0.5;
    }

    // Precipitation: forms and falls downward
    if (cloud > PRECIP_THRESHOLD)
    {
        float rain = (cloud - PRECIP_THRESHOLD) * 0.04;
        precip = min(precip + rain, 1.0);
        cloud -= rain * 0.5;
    }
    else
    {
        precip *= 0.95;
    }

    // Precipitation received from layer above
    if (tid.z < grid_d - 1)
    {
        float above_precip = state_read[uint3(tid.xy, tid.z + 1)].a;
        if (above_precip > 0.01)
        {
            float fall = above_precip * PRECIP_FALL_RATE * dt;
            moisture = min(moisture + fall * 0.3, 1.0);
            temp -= fall * 0.2 * dt;
        }
    }

    // Cloud dissipation
    float windspeed = length(new_wind);
    float diss = lerp(0.5, 1.2, saturate(windspeed / 3.0));
    cloud = max(cloud - CLOUD_DISSIPATION * dt * diss, 0.0);

    // --- CLAMP ---
    cloud    = saturate(cloud);
    moisture = saturate(moisture);
    temp     = clamp(temp, 200.0, 350.0);
    precip   = saturate(precip);

    // NaN guard — state
    uint nan = asuint(cloud + moisture + temp + precip);
    if ((nan & 0x7F800000) == 0x7F800000 && (nan & 0x007FFFFF) != 0) {
        cloud = 0; moisture = 0.4; temp = target_temp; precip = 0;
    }

    // NaN guard — wind
    uint wnan = asuint(new_wind.x + new_wind.y + new_wind.z);
    if ((wnan & 0x7F800000) == 0x7F800000 && (wnan & 0x007FFFFF) != 0) {
        new_wind = float3(0, 0, 0);
    }

    // Clamp wind magnitude to prevent runaway
    float ws_final = length(new_wind);
    float ws_limit = 50.0;
    if (ws_final > ws_limit)
        new_wind *= ws_limit / ws_final;

    // --- PRESSURE (equation of state) ---
    float pressure = PRESSURE_BASE * exp(-altitude / PRESSURE_SCALE_H)
                   * (temp / BASE_TEMP) * (1.0 + moisture * 0.61);
    // Wind divergence adjusts pressure (mass continuity)
    float div = (wind_read[xp].x - wind_read[xm].x
               + wind_read[yp].y - wind_read[ym].y) * 0.5;
    pressure += div * 50.0 * dt;
    pressure = clamp(pressure, 100.0, 2000.0);

    state_write[tid] = float4(cloud, moisture, temp / 400.0, precip);
    wind_write[tid] = float4(new_wind, pressure);

    // --- SHADOW MAP + GROUND CONDITIONS (column integration) ---
    if (tid.z == 0)
    {
        float col = 0.0;
        for (uint zz = 0; zz < grid_d; zz++)
            col += state_read[uint3(tid.xy, zz)].r;
        shadow_out[tid.xy] = saturate(col * 0.15);

        // Project ground-level conditions to 2D
        // R=rainfall (column-integrated precip), G=temperature(K), B=humidity, A=wind_speed
        float column_rain = 0.0;
        for (uint rz = 0; rz < grid_d; rz++) {
            float layer_precip = state_read[uint3(tid.xy, rz)].a;
            column_rain += layer_precip;
        }
        column_rain = saturate(column_rain * 0.1);
        ground_cond_out[tid.xy] = float4(column_rain, temp, moisture, length(new_wind.xy));
        ground_wind_out[tid.xy] = new_wind.xy;
    }
}

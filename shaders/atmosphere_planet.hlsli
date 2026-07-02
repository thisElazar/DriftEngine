// atmosphere_planet.hlsli — single-scattering Rayleigh + Mie atmosphere shared
// by the sky pass (full-quality march) and the aerial-perspective term in the
// terrain / river shaders (short march to the fragment).
//
// The public API (atmo_integrate, atmo_ground_hit) takes METRES relative to
// the planet center. ALL INTERNAL MATH RUNS IN KILOMETRES: fp32 ray–sphere
// at planet radii in metres is catastrophic — b² ≈ 4e13 carries ~4e6 of ULP
// error into the discriminant, so intersection distances jitter by kilometres
// as the camera moves, which reads as full-screen flicker. In km the same
// terms sit near 4e7 with ~5 ULP, and the residual error is metre-scale.

static const float ATMO_KM_PER_M = 1.0e-3;

static const float ATMO_HEIGHT = 100.0;   // shell thickness above r_ground (km)
static const float H_RAYLEIGH  = 8.5;     // Rayleigh density scale height (km)
static const float H_MIE       = 1.2;     // Mie density scale height (km)
static const float3 BETA_RAY   = float3(5.802e-3, 13.558e-3, 33.1e-3);  // per km
static const float  BETA_MIE   = 3.996e-3;
static const float  MIE_G      = 0.76;    // Henyey-Greenstein anisotropy
// Width of the soft terminator when testing per-sample sun visibility (km).
// A smooth factor instead of a binary occlusion test — the hard branch made
// whole samples snap on/off under camera-motion jitter (visible flicker).
static const float  SHADOW_SOFT_KM = 12.0;

// Ray vs sphere centred at the origin, KILOMETRE inputs. Uses the
// perpendicular-distance form, which cancels far less than b² - c.
// Returns (t_near, t_far) in km; miss when t_near > t_far.
float2 atmo_ray_sphere_km(float3 ro, float3 rd, float r)
{
    float  b = dot(ro, rd);
    float3 perp = ro - b * rd;
    float  disc = r * r - dot(perp, perp);
    if (disc < 0.0) return float2(1.0, -1.0);
    float s = sqrt(disc);
    return float2(-b - s, -b + s);
}

// Smooth sun visibility at a point (km frame): 1 in full sun, 0 behind the
// planet, with a SHADOW_SOFT_KM-wide ramp across the geometric horizon.
float atmo_sun_visibility(float3 p, float3 sun_dir, float r_ground)
{
    float t_close = -dot(p, sun_dir);        // closest approach along the sun ray
    if (t_close <= 0.0) return 1.0;          // sun is above the local horizon
    float3 perp = p + sun_dir * t_close;
    float  min_r = length(perp);
    return smoothstep(r_ground - SHADOW_SOFT_KM, r_ground + SHADOW_SOFT_KM, min_r);
}

// Optical depth (rayleigh, mie) along a ray to the top of the atmosphere (km frame).
float2 atmo_light_depth(float3 p, float3 sun_dir, float r_ground, int steps)
{
    float r_top = r_ground + ATMO_HEIGHT;
    float2 t = atmo_ray_sphere_km(p, sun_dir, r_top);
    float  seg = max(t.y, 0.0) / float(steps);
    float2 depth = 0.0;
    float  ti = seg * 0.5;
    [loop] for (int i = 0; i < steps; ++i) {
        float3 sp = p + sun_dir * ti;
        float  h = max(length(sp) - r_ground, 0.0);
        depth.x += exp(-h / H_RAYLEIGH) * seg;
        depth.y += exp(-h / H_MIE) * seg;
        ti += seg;
    }
    return depth;
}

// First ground-sphere hit along the ray, in METRES (negative = no hit).
// For the sky pass's t_max; computed in km for the same precision reason.
float atmo_ground_hit(float3 ro_m, float3 rd, float r_ground_m)
{
    float2 g = atmo_ray_sphere_km(ro_m * ATMO_KM_PER_M, rd, r_ground_m * ATMO_KM_PER_M);
    if (g.x < g.y && g.x > 0.0) return g.x / ATMO_KM_PER_M;
    return -1.0;
}

struct AtmoResult {
    float3 inscatter;      // radiance added along the path
    float3 transmittance;  // what survives of the surface behind it
};

// Integrate scattering along ro + t*rd for t in [0, t_max] (clipped to the
// atmosphere shell). METRE inputs: `t_max` = distance to the fragment for
// aerial perspective, or 1e12 for the open sky.
AtmoResult atmo_integrate(float3 ro_m, float3 rd, float t_max_m, float3 sun_dir,
                          float r_ground_m, float density, float sun_intensity,
                          int view_steps, int light_steps)
{
    AtmoResult res;
    res.inscatter = 0.0;
    res.transmittance = 1.0;

    float3 ro = ro_m * ATMO_KM_PER_M;
    float  r_ground = r_ground_m * ATMO_KM_PER_M;
    float  t_max = min(t_max_m, 1.0e9) * ATMO_KM_PER_M;

    float r_top = r_ground + ATMO_HEIGHT;
    float2 shell = atmo_ray_sphere_km(ro, rd, r_top);
    float t0 = max(shell.x, 0.0);
    float t1 = min(shell.y, t_max);
    if (t1 <= t0) return res;   // ray never enters the shell before t_max

    float seg = (t1 - t0) / float(view_steps);
    float2 view_depth = 0.0;              // accumulated (rayleigh, mie) depth, km
    float3 sum_ray = 0.0;
    float3 sum_mie = 0.0;

    float ti = t0 + seg * 0.5;
    [loop] for (int i = 0; i < view_steps; ++i) {
        float3 p = ro + rd * ti;
        float  h = max(length(p) - r_ground, 0.0);
        float  d_ray = exp(-h / H_RAYLEIGH) * seg;
        float  d_mie = exp(-h / H_MIE) * seg;
        view_depth += float2(d_ray, d_mie);

        float sun_vis = atmo_sun_visibility(p, sun_dir, r_ground);
        if (sun_vis > 0.0) {
            float2 light_depth = atmo_light_depth(p, sun_dir, r_ground, light_steps);
            float3 tau = BETA_RAY * (view_depth.x + light_depth.x)
                       + BETA_MIE * 1.1 * (view_depth.y + light_depth.y);
            float3 attn = exp(-tau * density) * sun_vis;
            sum_ray += attn * d_ray;
            sum_mie += attn * d_mie;
        }
        ti += seg;
    }

    float mu = dot(rd, sun_dir);
    float phase_ray = 3.0 / (16.0 * 3.14159265) * (1.0 + mu * mu);
    float g2 = MIE_G * MIE_G;
    float phase_mie = 3.0 / (8.0 * 3.14159265) * ((1.0 - g2) * (1.0 + mu * mu))
                    / ((2.0 + g2) * pow(abs(1.0 + g2 - 2.0 * MIE_G * mu), 1.5));

    res.inscatter = sun_intensity * density
                  * (sum_ray * BETA_RAY * phase_ray + sum_mie * BETA_MIE * phase_mie);
    res.transmittance = exp(-(BETA_RAY * view_depth.x + BETA_MIE * 1.1 * view_depth.y)
                            * density);
    return res;
}

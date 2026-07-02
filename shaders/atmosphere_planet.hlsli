// atmosphere_planet.hlsli — single-scattering Rayleigh + Mie atmosphere shared
// by the sky pass (full-quality march) and the aerial-perspective term in the
// terrain / river shaders (short march to the fragment).
//
// All positions are relative to the PLANET CENTER, in metres. Constants are
// Earth-like; `density` scales both optical depths so one slider moves the
// whole look from vacuum to soup.

static const float ATMO_HEIGHT = 100000.0;  // shell thickness above r_ground (m)
static const float H_RAYLEIGH  = 8500.0;    // Rayleigh density scale height (m)
static const float H_MIE       = 1200.0;    // Mie density scale height (m)
static const float3 BETA_RAY   = float3(5.802e-6, 13.558e-6, 33.1e-6);  // per metre
static const float  BETA_MIE   = 3.996e-6;
static const float  MIE_G      = 0.76;      // Henyey-Greenstein anisotropy

// Ray vs sphere centred at the origin. Returns (t_near, t_far); miss when
// t_near > t_far.
float2 atmo_ray_sphere(float3 ro, float3 rd, float r)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float disc = b * b - c;
    if (disc < 0.0) return float2(1.0, -1.0);
    float s = sqrt(disc);
    return float2(-b - s, -b + s);
}

// Optical depth (rayleigh, mie) along a ray to the top of the atmosphere.
float2 atmo_light_depth(float3 p, float3 sun_dir, float r_ground, int steps)
{
    float r_top = r_ground + ATMO_HEIGHT;
    float2 t = atmo_ray_sphere(p, sun_dir, r_top);
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

struct AtmoResult {
    float3 inscatter;      // radiance added along the path
    float3 transmittance;  // what survives of the surface behind it
};

// Integrate scattering along ro + t*rd for t in [0, t_max] (clipped to the
// atmosphere shell). `t_max` = distance to the fragment for aerial perspective,
// or 1e12 for the open sky. Sun visibility is tested per sample against the
// ground sphere, which is what makes the shadowed band after sunset.
AtmoResult atmo_integrate(float3 ro, float3 rd, float t_max, float3 sun_dir,
                          float r_ground, float density, float sun_intensity,
                          int view_steps, int light_steps)
{
    AtmoResult res;
    res.inscatter = 0.0;
    res.transmittance = 1.0;

    float r_top = r_ground + ATMO_HEIGHT;
    float2 shell = atmo_ray_sphere(ro, rd, r_top);
    float t0 = max(shell.x, 0.0);
    float t1 = min(shell.y, t_max);
    if (t1 <= t0) return res;   // ray never enters the shell before t_max

    float seg = (t1 - t0) / float(view_steps);
    float2 view_depth = 0.0;              // accumulated (rayleigh, mie) depth
    float3 sum_ray = 0.0;
    float3 sum_mie = 0.0;

    float ti = t0 + seg * 0.5;
    [loop] for (int i = 0; i < view_steps; ++i) {
        float3 p = ro + rd * ti;
        float  h = max(length(p) - r_ground, 0.0);
        float  d_ray = exp(-h / H_RAYLEIGH) * seg;
        float  d_mie = exp(-h / H_MIE) * seg;
        view_depth += float2(d_ray, d_mie);

        // Sun below the local horizon → this sample is in the planet's shadow.
        float2 g = atmo_ray_sphere(p, sun_dir, r_ground * 0.9999);
        if (!(g.x < g.y && g.y > 0.0)) {
            float2 light_depth = atmo_light_depth(p, sun_dir, r_ground, light_steps);
            float3 tau = BETA_RAY * (view_depth.x + light_depth.x)
                       + BETA_MIE * 1.1 * (view_depth.y + light_depth.y);
            float3 attn = exp(-tau * density);
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

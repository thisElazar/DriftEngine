// atmosphere_planet.hlsli — single-scattering Rayleigh + Mie atmosphere shared
// by the sky pass (full-quality march) and the aerial-perspective term in the
// terrain / river shaders (short march to the fragment).
//
// The public API (atmo_integrate, atmo_ground_hit) takes METRES relative to
// the planet center. ALL INTERNAL MATH RUNS IN KILOMETRES: fp32 ray–sphere
// at planet radii in metres is catastrophic — b² ≈ 4e13 carries ~4e6 of ULP
// error into the discriminant, so intersection distances jitter by kilometres
// as the camera moves, which reads as full-screen flicker.
//
// Sun transmittance is ANALYTIC (Chapman function approximation), not
// marched. A 2–4 sample light march over a path that is hundreds of km and
// horizon-sensitive is stable when the camera is still but decorrelates into
// visible noise the moment it moves — the closed form is smooth by
// construction and also handles the terminator (below-horizon depths grow
// exponentially, so the shadowed side fades out without a branch).
//
// The shell is deliberately TALLER than Earth's (scale heights ~2.6× real,
// β rescaled to keep the zenith optical depth): the atmosphere occupies more
// of the view and the ground stays visible through less haze at distance.

static const float ATMO_KM_PER_M = 1.0e-3;

static const float ATMO_HEIGHT = 240.0;   // shell thickness above r_ground (km)
static const float H_RAYLEIGH  = 22.0;    // Rayleigh density scale height (km)
static const float H_MIE       = 4.0;     // Mie density scale height (km)
// Per-km scattering, rescaled from Earth's (β·H preserved → same zenith depth,
// spread over a taller shell → thinner horizontal haze).
static const float3 BETA_RAY   = float3(2.242e-3, 5.238e-3, 12.789e-3);
static const float  BETA_MIE   = 1.199e-3;
static const float  MIE_G      = 0.76;    // Henyey-Greenstein anisotropy

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

// Chapman grazing-incidence function, Schüler's approximation: the optical
// depth to space along a ray at cos-zenith `coschi`, in units of (local
// density × H). x = r/H (radius in scale heights). The below-horizon branch
// reflects through the tangent point and explodes exponentially — which is
// exactly the terminator falloff.
float atmo_chapman(float x, float coschi)
{
    float c = sqrt(1.57079632 * x);   // sqrt(pi/2 · x)
    if (coschi >= 0.0)
        return c / ((c - 1.0) * coschi + 1.0);
    float sinchi = sqrt(saturate(1.0 - coschi * coschi));
    float x_h = x * sinchi;           // radius (in H units) of the tangent point
    float c_h = sqrt(1.57079632 * x_h);
    float refl = c / ((c - 1.0) * (-coschi) + 1.0);
    // exp(x - x_h) can overflow deep below the horizon; the clamp keeps the
    // arithmetic finite (transmittance is 0 either way).
    return min(2.0 * exp(min(x - x_h, 60.0)) * c_h - refl, 1.0e16);
}

// Analytic optical depth (rayleigh, mie) from a point to space toward the sun.
// p in the km frame; returns km of equivalent sea-level-density path per species.
float2 atmo_sun_depth(float3 p, float3 sun_dir, float r_ground)
{
    float r = length(p);
    float coschi = dot(p, sun_dir) / max(r, 1e-3);
    float alt = max(r - r_ground, 0.0);
    float xr = r / H_RAYLEIGH;
    float xm = r / H_MIE;
    return float2(
        H_RAYLEIGH * exp(-alt / H_RAYLEIGH) * atmo_chapman(xr, coschi),
        H_MIE      * exp(-alt / H_MIE)      * atmo_chapman(xm, coschi));
}

struct AtmoResult {
    float3 inscatter;      // radiance added along the path
    float3 transmittance;  // what survives of the surface behind it
};

// Integrate scattering along ro + t*rd for t in [0, t_max] (clipped to the
// atmosphere shell). METRE inputs: `t_max` = distance to the fragment for
// aerial perspective, or 1e12 for the open sky. Only the view ray is marched;
// sun transmittance per sample is analytic (see atmo_sun_depth).
AtmoResult atmo_integrate(float3 ro_m, float3 rd, float t_max_m, float3 sun_dir,
                          float r_ground_m, float density, float sun_intensity,
                          int view_steps)
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
        // Underground samples clamp to sea-level density: a ray that grazes
        // into the planet extinguishes smoothly instead of needing a separate
        // (fp32-fragile near the limb) ground-hit clip.
        float  h = max(length(p) - r_ground, 0.0);
        float  d_ray = exp(-h / H_RAYLEIGH) * seg;
        float  d_mie = exp(-h / H_MIE) * seg;
        view_depth += float2(d_ray, d_mie);

        float2 sun_depth = atmo_sun_depth(p, sun_dir, r_ground);
        float3 tau = BETA_RAY * (view_depth.x + sun_depth.x)
                   + BETA_MIE * 1.1 * (view_depth.y + sun_depth.y);
        float3 attn = exp(-tau * density);
        sum_ray += attn * d_ray;
        sum_mie += attn * d_mie;
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

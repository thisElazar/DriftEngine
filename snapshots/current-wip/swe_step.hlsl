// swe_step.hlsl — HLL Riemann SWE solver
// Ported from Drift V2 ShallowWater.usf. Stripped: Common.ush, toroidal
// wrap, BoundaryDataTexture, wind term (zeroed). See PORT NOTE comments.

[[vk::binding(0, 0)]] Texture2D<float>    terrain;
[[vk::binding(1, 0)]] Texture2D<float4>   state_read;
[[vk::binding(2, 0)]] RWTexture2D<float4> state_write;
[[vk::binding(3, 0)]] RWTexture2D<float4> output;

[[vk::push_constant]]
cbuffer PushConstants {
    float time;
    float dt;
    float gravity;
    float friction;
    float dx;
    float sea_level;
    float damping;
    float _pad0;
    uint  grid_w;
    uint  grid_h;
    float pulse_x;
    float pulse_y;
    float pulse_radius;
    float pulse_amount;
};

#define GRAVITY   gravity
#define FRICTION  friction
#define DX        dx
#define SEA_LEVEL sea_level
#define DAMPING   damping
#define GRID_W    ((int)grid_w)
#define GRID_H    ((int)grid_h)
#define MIN_DEPTH 0.001f
#define DRY_TOLERANCE 0.01f

int2 ClampCoord(int2 c)
{
    return clamp(c, int2(0, 0), int2(GRID_W - 1, GRID_H - 1));
}

float4 ReadState(int2 coord)
{
    return state_read[ClampCoord(coord)];
}

float ReadTerrain(int2 coord)
{
    return terrain[ClampCoord(coord)];
}

float3 HLLFlux(float3 UL, float3 UR, float zL, float zR, bool bXDir)
{
    float hL = max(UL.x, 0.0f);
    float hR = max(UR.x, 0.0f);

    float uL = (hL > DRY_TOLERANCE) ? UL.y / hL : 0.0f;
    float vL = (hL > DRY_TOLERANCE) ? UL.z / hL : 0.0f;
    float uR = (hR > DRY_TOLERANCE) ? UR.y / hR : 0.0f;
    float vR = (hR > DRY_TOLERANCE) ? UR.z / hR : 0.0f;

    float cL = sqrt(GRAVITY * hL);
    float cR = sqrt(GRAVITY * hR);

    float vnL = bXDir ? uL : vL;
    float vnR = bXDir ? uR : vR;

    float sL = min(vnL - cL, vnR - cR);
    float sR = max(vnL + cL, vnR + cR);

    if (hL < DRY_TOLERANCE && hR < DRY_TOLERANCE)
        return float3(0, 0, 0);

    float3 FL;
    if (bXDir)
    {
        FL.x = hL * uL;
        FL.y = hL * uL * uL + 0.5f * GRAVITY * hL * hL;
        FL.z = hL * uL * vL;
    }
    else
    {
        FL.x = hL * vL;
        FL.y = hL * vL * uL;
        FL.z = hL * vL * vL + 0.5f * GRAVITY * hL * hL;
    }

    float3 FR;
    if (bXDir)
    {
        FR.x = hR * uR;
        FR.y = hR * uR * uR + 0.5f * GRAVITY * hR * hR;
        FR.z = hR * uR * vR;
    }
    else
    {
        FR.x = hR * vR;
        FR.y = hR * vR * uR;
        FR.z = hR * vR * vR + 0.5f * GRAVITY * hR * hR;
    }

    if (sL >= 0.0f)
        return FL;
    if (sR <= 0.0f)
        return FR;

    return (sR * FL - sL * FR + sL * sR * (float3(UR.x, UR.y, UR.z) - float3(UL.x, UL.y, UL.z)))
           / (sR - sL);
}

[numthreads(8, 8, 1)]
void main(uint3 ThreadId : SV_DispatchThreadID)
{
    int2 coord = int2(ThreadId.xy);
    if (coord.x >= GRID_W || coord.y >= GRID_H)
        return;

    float step_dt = min(dt, 0.033f);

    // === Read current state ===
    float4 state = ReadState(coord);

    if (pulse_amount > 0.0f)
    {
        float2 d = float2(coord) - float2(pulse_x, pulse_y);
        float r2 = dot(d, d);
        float falloff = exp(-r2 / (pulse_radius * pulse_radius));
        state.r += pulse_amount * falloff;
    }

    float h  = max(state.r, 0.0f);
    float hu = state.g;
    float hv = state.b;
    float foam = state.a;

    float z = ReadTerrain(coord);

    // === Read neighbor states ===
    float4 sL = ReadState(coord + int2(-1, 0));
    float4 sR = ReadState(coord + int2( 1, 0));
    float4 sD = ReadState(coord + int2( 0,-1));
    float4 sU = ReadState(coord + int2( 0, 1));

    float zL = ReadTerrain(coord + int2(-1, 0));
    float zR = ReadTerrain(coord + int2( 1, 0));
    float zD = ReadTerrain(coord + int2( 0,-1));
    float zU = ReadTerrain(coord + int2( 0, 1));

    // === Hydrostatic reconstruction (Audusse et al. 2004) ===
    float wC = h + z;

    float zs_xL = max(zL, z);
    float hC_xL = max(0.0f, wC - zs_xL);
    float hN_xL = max(0.0f, sL.r + zL - zs_xL);

    float zs_xR = max(z, zR);
    float hC_xR = max(0.0f, wC - zs_xR);
    float hN_xR = max(0.0f, sR.r + zR - zs_xR);

    float zs_yD = max(zD, z);
    float hC_yD = max(0.0f, wC - zs_yD);
    float hN_yD = max(0.0f, sD.r + zD - zs_yD);

    float zs_yU = max(z, zU);
    float hC_yU = max(0.0f, wC - zs_yU);
    float hN_yU = max(0.0f, sU.r + zU - zs_yU);

    float inv_h = (h > DRY_TOLERANCE) ? 1.0f / h : 0.0f;
    float uC = hu * inv_h;
    float vC = hv * inv_h;

    float inv_hL = (sL.r > DRY_TOLERANCE) ? 1.0f / sL.r : 0.0f;
    float inv_hR = (sR.r > DRY_TOLERANCE) ? 1.0f / sR.r : 0.0f;
    float inv_hD = (sD.r > DRY_TOLERANCE) ? 1.0f / sD.r : 0.0f;
    float inv_hU = (sU.r > DRY_TOLERANCE) ? 1.0f / sU.r : 0.0f;

    // === Compute fluxes (HLL with reconstructed states) ===
    float3 fxL = HLLFlux(
        float3(hN_xL, hN_xL * sL.g * inv_hL, hN_xL * sL.b * inv_hL),
        float3(hC_xL, hC_xL * uC, hC_xL * vC), zL, z, true);
    float3 fxR = HLLFlux(
        float3(hC_xR, hC_xR * uC, hC_xR * vC),
        float3(hN_xR, hN_xR * sR.g * inv_hR, hN_xR * sR.b * inv_hR), z, zR, true);

    float3 fyD = HLLFlux(
        float3(hN_yD, hN_yD * sD.g * inv_hD, hN_yD * sD.b * inv_hD),
        float3(hC_yD, hC_yD * uC, hC_yD * vC), zD, z, false);
    float3 fyU = HLLFlux(
        float3(hC_yU, hC_yU * uC, hC_yU * vC),
        float3(hN_yU, hN_yU * sU.g * inv_hU, hN_yU * sU.b * inv_hU), z, zU, false);

    // === Update conserved variables ===
    float dtdx = step_dt / DX;

    float h_flux = dtdx * (fxR.x - fxL.x) + dtdx * (fyU.x - fyD.x);
    float flux_scale = (h_flux > h && h_flux > 0.0f) ? (h / h_flux) : 1.0f;

    float h_new  = h  - h_flux * flux_scale;
    float hu_new = hu - (dtdx * (fxR.y - fxL.y) + dtdx * (fyU.y - fyD.y)) * flux_scale;
    float hv_new = hv - (dtdx * (fxR.z - fxL.z) + dtdx * (fyU.z - fyD.z)) * flux_scale;

    if (h_new < 0.0f)
    {
        h_new = 0.0f;
        hu_new = 0.0f;
        hv_new = 0.0f;
    }

    // === Source terms ===

    // Well-balanced bed slope via hydrostatic reconstruction pressure correction
    hu_new += step_dt * GRAVITY / (2.0f * DX) * (hC_xR * hC_xR - hC_xL * hC_xL);
    hv_new += step_dt * GRAVITY / (2.0f * DX) * (hC_yU * hC_yU - hC_yD * hC_yD);

    // Friction (Manning-style quadratic drag)
    if (h_new > DRY_TOLERANCE)
    {
        float u_cur = hu_new / h_new;
        float v_cur = hv_new / h_new;
        float speed = sqrt(u_cur * u_cur + v_cur * v_cur);
        float friction_coeff = FRICTION * speed / max(h_new, 0.1f);
        float decay = 1.0f / (1.0f + step_dt * friction_coeff);
        hu_new *= decay;
        hv_new *= decay;
    }

    // PORT NOTE: wind forcing stripped — pass zero WindParams for now

    // === Enforce constraints ===
    h_new = max(h_new, 0.0f);

    if (h_new < DRY_TOLERANCE)
    {
        hu_new = 0.0f;
        hv_new = 0.0f;
    }

    // Velocity limiter
    if (h_new > 1e-6f)
    {
        float u_lim = hu_new / h_new;
        float v_lim = hv_new / h_new;
        float speed_lim = sqrt(u_lim * u_lim + v_lim * v_lim);
        float max_speed = 3.0f * sqrt(GRAVITY * h_new);
        if (speed_lim > max_speed)
        {
            float scale = max_speed / speed_lim;
            hu_new *= scale;
            hv_new *= scale;
        }
    }

    // Sponge layer: absorbing boundary — friction-based drain, no hard zeroing.
    float sponge_width = 64.0f;
    float2 distFromEdge = float2(
        min((float)coord.x, (float)(GRID_W - 1 - coord.x)),
        min((float)coord.y, (float)(GRID_H - 1 - coord.y))
    );
    float edge_dist = min(distFromEdge.x, distFromEdge.y);
    float sponge_t = saturate(1.0f - edge_dist / sponge_width);
    float sponge_str = sponge_t * sponge_t;

    float sponge_decay = 1.0f / (1.0f + step_dt * 20.0f * sponge_str);
    hu_new *= sponge_decay;
    hv_new *= sponge_decay;

    h_new *= max(0.0f, 1.0f - step_dt * 3.0f * sponge_str);

    float edgeFade = 1.0f - sponge_str;

    // Global damping
    hu_new *= (1.0f - DAMPING * step_dt);
    hv_new *= (1.0f - DAMPING * step_dt);

    // === Foam ===
    float speed_final = 0.0f;
    if (h_new > DRY_TOLERANCE)
    {
        float uf = hu_new / h_new;
        float vf = hv_new / h_new;
        speed_final = sqrt(uf * uf + vf * vf);
    }

    float breakingFoam = 0.0f;
    if (h_new > DRY_TOLERANCE && h_new < 5.0f)
    {
        float froude = speed_final / max(sqrt(GRAVITY * h_new), 0.01f);
        breakingFoam = saturate((froude - 0.5f) * 2.0f);
    }

    float shoreFoam = 0.0f;
    if (h_new > DRY_TOLERANCE)
    {
        float minNeighborH = min(min(sL.r, sR.r), min(sD.r, sU.r));
        if (minNeighborH < DRY_TOLERANCE)
            shoreFoam = saturate(speed_final * 0.5f);
    }

    foam = max(breakingFoam, shoreFoam) + foam * 0.95f * edgeFade;
    foam = saturate(foam);

    // === Write state ===
    state_write[coord] = float4(h_new, hu_new, hv_new, foam);

    // === Write output ===
    float surfaceElevation = (h_new > DRY_TOLERANCE) ? (z + h_new) : SEA_LEVEL;

    float surfL = (sL.r > DRY_TOLERANCE) ? (zL + sL.r) : SEA_LEVEL;
    float surfR = (sR.r > DRY_TOLERANCE) ? (zR + sR.r) : SEA_LEVEL;
    float surfD = (sD.r > DRY_TOLERANCE) ? (zD + sD.r) : SEA_LEVEL;
    float surfU = (sU.r > DRY_TOLERANCE) ? (zU + sU.r) : SEA_LEVEL;
    float dhdx = (surfR - surfL) / (2.0f * DX);
    float dhdy = (surfU - surfD) / (2.0f * DX);

    output[coord] = float4(surfaceElevation, dhdx, dhdy, foam);
}

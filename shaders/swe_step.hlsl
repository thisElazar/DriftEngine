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
    uint  _pad1;
    uint  _pad2;
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

    // === Compute fluxes (HLL Riemann solver) ===
    float3 U = float3(h, hu, hv);

    float3 fxL = HLLFlux(float3(sL.r, sL.g, sL.b), U, zL, z, true);
    float3 fxR = HLLFlux(U, float3(sR.r, sR.g, sR.b), z, zR, true);

    float3 fyD = HLLFlux(float3(sD.r, sD.g, sD.b), U, zD, z, false);
    float3 fyU = HLLFlux(U, float3(sU.r, sU.g, sU.b), z, zU, false);

    // === Update conserved variables ===
    float dtdx = step_dt / DX;

    float h_new  = h  - dtdx * (fxR.x - fxL.x) - dtdx * (fyU.x - fyD.x);
    float hu_new = hu - dtdx * (fxR.y - fxL.y) - dtdx * (fyU.y - fyD.y);
    float hv_new = hv - dtdx * (fxR.z - fxL.z) - dtdx * (fyU.z - fyD.z);

    // === Source terms ===

    // Bed slope
    float h_avg = max(h_new, 0.0f);
    if (h_avg > DRY_TOLERANCE)
    {
        float dzx = (zR - zL) / (2.0f * DX);
        float dzy = (zU - zD) / (2.0f * DX);
        hu_new -= step_dt * GRAVITY * h_avg * dzx;
        hv_new -= step_dt * GRAVITY * h_avg * dzy;
    }

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
    if (h_new > DRY_TOLERANCE)
    {
        float u_lim = hu_new / h_new;
        float v_lim = hv_new / h_new;
        float speed_lim = sqrt(u_lim * u_lim + v_lim * v_lim);
        float max_speed = 5.0f * sqrt(GRAVITY * h_new);
        if (speed_lim > max_speed)
        {
            float scale = max_speed / speed_lim;
            hu_new *= scale;
            hv_new *= scale;
        }
    }

    // PORT NOTE: edge damping simplified — no toroidal grid, no boundary texture.
    // Sponge zone at grid edges to absorb outgoing waves.
    float2 distFromEdge = float2(
        min((float)coord.x, (float)(GRID_W - 1 - coord.x)),
        min((float)coord.y, (float)(GRID_H - 1 - coord.y))
    );
    float edgeFade = saturate(min(distFromEdge.x, distFromEdge.y) / 32.0f);
    hu_new *= edgeFade;
    hv_new *= edgeFade;

    // PORT NOTE: boundary relaxation removed — single tile, no CPU boundary coupling.
    // Instead, fade depth at edges toward zero (open boundary absorber).
    h_new = lerp(0.0f, h_new, edgeFade);

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

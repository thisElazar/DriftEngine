// planet_swe_step.cs.hlsl — Per-tile HLL Riemann SWE solver on Texture2DArray.
// Adapted from swe_step.hlsl (Audusse et al. 2004 hydrostatic reconstruction).
// Cross-tile flow: when a sample goes off-edge, read from the same-level
// same-face neighbor's edge cells via its pool slot. Face seams / LOD
// mismatches fall back to reflective (clamp) until Phase 2/3.

[[vk::binding(0, 0)]] Texture2DArray<float>    terrain;
[[vk::binding(1, 0)]] Texture2DArray<float4>   state_read;
[[vk::binding(2, 0)]] RWTexture2DArray<float4> state_write;
[[vk::binding(3, 0)]] RWTexture2DArray<float4> output;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> edge_flags;  // one uint per pool slot

[[vk::push_constant]]
cbuffer PushConstants {
    float time;
    float dt;
    float gravity;
    float friction;
    float dx;
    float sea_level;
    float damping;
    uint  pool_index;
    uint  grid_w;
    uint  grid_h;
    float pulse_x;
    float pulse_y;
    float pulse_radius;
    float pulse_amount;
    uint  neighbor_left;
    uint  neighbor_right;
    uint  neighbor_down;
    uint  neighbor_up;
};

#define GRAVITY   gravity
#define FRICTION  friction
#define DX        dx
#define SEA_LEVEL sea_level
#define DAMPING   damping
#define GRID_W    ((int)grid_w)
#define GRID_H    ((int)grid_h)
#define DRY_TOLERANCE 0.01f
#define NO_NEIGHBOR 0xFFFFFFFFu

#define EDGE_BIT_LEFT  1u
#define EDGE_BIT_RIGHT 2u
#define EDGE_BIT_DOWN  4u
#define EDGE_BIT_UP    8u

int2 ClampCoord(int2 c)
{
    return clamp(c, int2(0, 0), int2(GRID_W - 1, GRID_H - 1));
}

// Resolve a (possibly off-edge) coord to (x, y, layer). For purely diagonal
// off-edge cases (corners), fall back to ClampCoord on this tile — those
// influence only second-order terms and aren't worth a 4-way corner table.
struct Sample { uint3 idx; };
Sample ResolveSample(int2 coord)
{
    Sample s;
    if (coord.x < 0 && coord.y >= 0 && coord.y < GRID_H && neighbor_left != NO_NEIGHBOR) {
        s.idx = uint3(uint(GRID_W - 1), uint(coord.y), neighbor_left);
        return s;
    }
    if (coord.x >= GRID_W && coord.y >= 0 && coord.y < GRID_H && neighbor_right != NO_NEIGHBOR) {
        s.idx = uint3(0u, uint(coord.y), neighbor_right);
        return s;
    }
    if (coord.y < 0 && coord.x >= 0 && coord.x < GRID_W && neighbor_down != NO_NEIGHBOR) {
        s.idx = uint3(uint(coord.x), uint(GRID_H - 1), neighbor_down);
        return s;
    }
    if (coord.y >= GRID_H && coord.x >= 0 && coord.x < GRID_W && neighbor_up != NO_NEIGHBOR) {
        s.idx = uint3(uint(coord.x), 0u, neighbor_up);
        return s;
    }
    int2 c = ClampCoord(coord);
    s.idx = uint3(uint(c.x), uint(c.y), pool_index);
    return s;
}

float4 ReadState(int2 coord)   { return state_read[ResolveSample(coord).idx]; }
float  ReadTerrain(int2 coord) { return terrain[ResolveSample(coord).idx]; }

float3 HLLFlux(float3 UL, float3 UR, bool bXDir)
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

float3 Reconstruct(float h_orig, float hu_orig, float hv_orig, float w, float z_star)
{
    float h_rec = max(0.0f, w - z_star);
    float inv_h = (h_orig > DRY_TOLERANCE) ? 1.0f / h_orig : 0.0f;
    return float3(h_rec, h_rec * hu_orig * inv_h, h_rec * hv_orig * inv_h);
}

[numthreads(8, 8, 1)]
void main(uint3 ThreadId : SV_DispatchThreadID)
{
    int2 coord = int2(ThreadId.xy);
    if (coord.x >= GRID_W || coord.y >= GRID_H)
        return;

    float step_dt = dt;

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

    float4 sL = ReadState(coord + int2(-1, 0));
    float4 sR = ReadState(coord + int2( 1, 0));
    float4 sD = ReadState(coord + int2( 0,-1));
    float4 sU = ReadState(coord + int2( 0, 1));

    float zL = ReadTerrain(coord + int2(-1, 0));
    float zR = ReadTerrain(coord + int2( 1, 0));
    float zD = ReadTerrain(coord + int2( 0,-1));
    float zU = ReadTerrain(coord + int2( 0, 1));

    float wC = h + z;
    float wL = sL.r + zL;
    float wR = sR.r + zR;
    float wD = sD.r + zD;
    float wU = sU.r + zU;

    float zs_xL = max(zL, z);
    float zs_xR = max(z, zR);
    float zs_yD = max(zD, z);
    float zs_yU = max(z, zU);

    float3 cL_rec = Reconstruct(h,    hu,   hv,   wC, zs_xL);
    float3 nL_rec = Reconstruct(sL.r, sL.g, sL.b, wL, zs_xL);
    float3 cR_rec = Reconstruct(h,    hu,   hv,   wC, zs_xR);
    float3 nR_rec = Reconstruct(sR.r, sR.g, sR.b, wR, zs_xR);
    float3 cD_rec = Reconstruct(h,    hu,   hv,   wC, zs_yD);
    float3 nD_rec = Reconstruct(sD.r, sD.g, sD.b, wD, zs_yD);
    float3 cU_rec = Reconstruct(h,    hu,   hv,   wC, zs_yU);
    float3 nU_rec = Reconstruct(sU.r, sU.g, sU.b, wU, zs_yU);

    float3 fxL = HLLFlux(nL_rec, cL_rec, true);
    float3 fxR = HLLFlux(cR_rec, nR_rec, true);
    float3 fyD = HLLFlux(nD_rec, cD_rec, false);
    float3 fyU = HLLFlux(cU_rec, nU_rec, false);

    float dtdx = step_dt / DX;

    float h_flux = dtdx * (fxR.x - fxL.x + fyU.x - fyD.x);
    float flux_scale = (h_flux > h && h_flux > 0.0f) ? (h / h_flux) : 1.0f;

    float h_new  = h  - h_flux * flux_scale;
    float hu_new = hu - dtdx * (fxR.y - fxL.y + fyU.y - fyD.y) * flux_scale;
    float hv_new = hv - dtdx * (fxR.z - fxL.z + fyU.z - fyD.z) * flux_scale;

    if (h_new < 0.0f)
    {
        h_new  = 0.0f;
        hu_new = 0.0f;
        hv_new = 0.0f;
    }

    hu_new += step_dt * GRAVITY / (2.0f * DX) * (cR_rec.x * cR_rec.x - cL_rec.x * cL_rec.x);
    hv_new += step_dt * GRAVITY / (2.0f * DX) * (cU_rec.x * cU_rec.x - cD_rec.x * cD_rec.x);

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

    h_new = max(h_new, 0.0f);

    if (h_new < DRY_TOLERANCE)
    {
        hu_new = 0.0f;
        hv_new = 0.0f;
    }

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

    hu_new *= (1.0f - DAMPING * step_dt);
    hv_new *= (1.0f - DAMPING * step_dt);

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

    foam = max(breakingFoam, shoreFoam) + foam * 0.95f;
    foam = saturate(foam);

    state_write[uint3(coord, pool_index)] = float4(h_new, hu_new, hv_new, foam);

    // Output for rendering: water height, surface gradients, foam
    float surfL = (sL.r > DRY_TOLERANCE) ? (zL + sL.r) : SEA_LEVEL;
    float surfR = (sR.r > DRY_TOLERANCE) ? (zR + sR.r) : SEA_LEVEL;
    float surfD = (sD.r > DRY_TOLERANCE) ? (zD + sD.r) : SEA_LEVEL;
    float surfU = (sU.r > DRY_TOLERANCE) ? (zU + sU.r) : SEA_LEVEL;
    float dhdx = (surfR - surfL) / (2.0f * DX);
    float dhdy = (surfU - surfD) / (2.0f * DX);

    output[uint3(coord, pool_index)] = float4(h_new, dhdx, dhdy, foam);

    // Mark per-edge "water above static reached this edge" so the CPU can
    // anchor the corresponding neighbor next frame. Threshold is generous —
    // slight numerical drift shouldn't trigger expansion.
    float static_h = max(SEA_LEVEL - z, 0.0f);
    if (h_new > static_h + 0.25f)
    {
        if (coord.x == 0)         InterlockedOr(edge_flags[pool_index], EDGE_BIT_LEFT);
        if (coord.x == GRID_W-1)  InterlockedOr(edge_flags[pool_index], EDGE_BIT_RIGHT);
        if (coord.y == 0)         InterlockedOr(edge_flags[pool_index], EDGE_BIT_DOWN);
        if (coord.y == GRID_H-1)  InterlockedOr(edge_flags[pool_index], EDGE_BIT_UP);
    }
}

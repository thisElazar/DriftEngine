// planet_swe_h_adjust.cs.hlsl — When a terrain stamp raises (or lowers) the bed
// of a disturbed tile, adjust the SWE column h to keep the water surface at
// roughly the same elevation. Only applied to previously-wet cells, so dry
// land doesn't spontaneously fill when the user lowers terrain.
//
// Math: surface = z + h. To preserve surface across delta_z = z_new - z_old,
// h_new = max(0, h_old - delta_z). delta_z here is the contribution of a
// single stamp at this cell, computed with the same gaussian as planet_gen.

[[vk::binding(0, 0)]] RWTexture2DArray<float4> state_a;
[[vk::binding(1, 0)]] RWTexture2DArray<float4> state_b;

[[vk::push_constant]]
cbuffer PushConstants {
    float u_min;
    float v_min;
    float tile_size;
    uint  face;

    uint  pool_index;
    uint  grid_w;
    uint  grid_h;
    uint  _pad0;

    float stamp_pos_x;
    float stamp_pos_y;
    float stamp_pos_z;
    float stamp_cos_radius;

    float stamp_delta_h;
    float _pad1, _pad2, _pad3;
};

#define DRY_TOLERANCE 0.01f

float3 face_uv_to_cube(float2 uv, uint f)
{
    switch (f) {
        case 0: return float3( 1.0, uv.y, -uv.x);
        case 1: return float3(-1.0, uv.y,  uv.x);
        case 2: return float3(uv.x,  1.0, -uv.y);
        case 3: return float3(uv.x, -1.0,  uv.y);
        case 4: return float3(uv.x, uv.y,  1.0);
        case 5: return float3(-uv.x, uv.y, -1.0);
        default: return float3(0, 0, 0);
    }
}

float3 cube_to_sphere(float3 p)
{
    float x2 = p.x * p.x;
    float y2 = p.y * p.y;
    float z2 = p.z * p.z;
    return float3(
        p.x * sqrt(max(0.0, 1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0)),
        p.y * sqrt(max(0.0, 1.0 - x2 * 0.5 - z2 * 0.5 + x2 * z2 / 3.0)),
        p.z * sqrt(max(0.0, 1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0))
    );
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= grid_w || dtid.y >= grid_h) return;

    float2 tile_uv = float2(dtid.xy) / float(grid_w - 1);
    float2 face_uv = float2(u_min + tile_uv.x * tile_size,
                            v_min + tile_uv.y * tile_size);
    float3 sphere_dir = normalize(cube_to_sphere(face_uv_to_cube(face_uv, face)));

    float3 stamp_pos = float3(stamp_pos_x, stamp_pos_y, stamp_pos_z);
    float d = dot(sphere_dir, stamp_pos);
    if (d < stamp_cos_radius) return;

    float t = (d - stamp_cos_radius) / max(1.0 - stamp_cos_radius, 1e-7);
    float delta_z = stamp_delta_h * exp(-4.0 * (1.0 - t) * (1.0 - t));

    uint3 idx = uint3(dtid.xy, pool_index);

    float4 sa = state_a[idx];
    if (sa.r > DRY_TOLERANCE) {
        sa.r = max(0.0, sa.r - delta_z);
        state_a[idx] = sa;
    }

    float4 sb = state_b[idx];
    if (sb.r > DRY_TOLERANCE) {
        sb.r = max(0.0, sb.r - delta_z);
        state_b[idx] = sb;
    }
}

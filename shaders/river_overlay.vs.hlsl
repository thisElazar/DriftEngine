// river_overlay.vs.hlsl — draws the global drainage network as an animated
// overlay on the planet tiles. Reuses the terrain tile grid mesh and rebuilds the
// same world-space position as terrain.vs.hlsl (so rivers sit on the surface),
// then forwards the cube-face uv so the FS can sample the global hydrology field.

[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(1, 0)]] Texture2DArray<float> heightmap;   // terrain pool
[[vk::binding(3, 0)]] SamplerState samp;

[[vk::push_constant]]
cbuffer RiverPC {
    float rel_x, rel_y, rel_z;
    float u_min, v_min, tile_size;
    uint  face;
    uint  pool_index;
    float planet_radius;
    float heightmap_texel;
    float time;
    float river_threshold;
};

struct VSInput {
    [[vk::location(0)]] float2 grid_pos : POSITION;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 hyd_uv  : TEXCOORD0;             // [0,1] cube-face uv
    [[vk::location(1)]] nointerpolation uint face_out : TEXCOORD1;
    [[vk::location(2)]] float2 tile_uv : TEXCOORD2;
    [[vk::location(3)]] float3 world_pos : TEXCOORD3;           // camera-relative, for view vector
    [[vk::location(4)]] float3 normal    : TEXCOORD4;           // surface (radial) normal
};

static const float GRID_MAX = 63.0;
static const float RIVER_LIFT = 2.0;  // metres above terrain to avoid z-fighting

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
    float x2 = p.x * p.x, y2 = p.y * p.y, z2 = p.z * p.z;
    return float3(
        p.x * sqrt(max(0.0, 1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0)),
        p.y * sqrt(max(0.0, 1.0 - x2 * 0.5 - z2 * 0.5 + x2 * z2 / 3.0)),
        p.z * sqrt(max(0.0, 1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0)));
}

VSOutput main(VSInput input)
{
    float2 grid = clamp(input.grid_pos, 0.0, GRID_MAX);  // ignore skirts for rivers
    float2 tile_uv = grid / GRID_MAX;

    float2 face_uv = float2(u_min + tile_uv.x * tile_size,
                            v_min + tile_uv.y * tile_size);
    float3 sphere_dir = cube_to_sphere(face_uv_to_cube(face_uv, face));

    float2 center_fuv = float2(u_min + 0.5 * tile_size, v_min + 0.5 * tile_size);
    float3 center_dir = cube_to_sphere(face_uv_to_cube(center_fuv, face));

    float2 hm_uv = (grid + 0.5) * heightmap_texel;
    float terrain_h = heightmap.SampleLevel(samp, float3(hm_uv, float(pool_index)), 0).r;

    float3 delta_dir = sphere_dir - center_dir;
    float3 displacement = delta_dir * planet_radius + sphere_dir * (terrain_h + RIVER_LIFT);
    float3 world_rel = displacement + float3(rel_x, rel_y, rel_z);

    VSOutput o;
    o.position  = mul(proj, mul(view, float4(world_rel, 1.0)));
    o.hyd_uv    = face_uv * 0.5 + 0.5;
    o.face_out  = face;
    o.tile_uv   = tile_uv;
    o.world_pos = world_rel;        // camera-relative (cam at origin)
    o.normal    = sphere_dir;       // radial surface normal
    return o;
}

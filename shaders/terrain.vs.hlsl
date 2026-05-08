[[vk::binding(0, 0)]] cbuffer Camera {
    float4x4 view;
    float4x4 proj;
    float3 sun_dir;   float _pad0;
    float3 sun_color; float _pad1;
    float3 cam_pos;   float _pad2;
    float4 brush_world;
    float4 brush_color;
};

[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2DArray<float> heightmap;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState heightmap_sampler;

[[vk::push_constant]]
cbuffer PlanetTilePC {
    float rel_x, rel_y, rel_z;
    float u_min, v_min, tile_size;
    uint  face;
    uint  pool_index;
    float planet_radius;
    float max_elevation;
    float heightmap_texel;
    float cloud_opacity;
};

struct VSInput {
    [[vk::location(0)]] float2 grid_pos : POSITION;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float height_normalized : TEXCOORD1;
    [[vk::location(3)]] float3 world_pos : TEXCOORD2;
    [[vk::location(4)]] float3 sphere_direction : TEXCOORD3;
};

static const float GRID_MAX = 63.0;
static const float SKIRT_DROP = 10000.0;

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

VSOutput main(VSInput input)
{
    // Detect skirt vertices (grid_pos outside [0, GRID_MAX]) and clamp
    float2 grid = clamp(input.grid_pos, 0.0, GRID_MAX);
    bool is_skirt = any(input.grid_pos != grid);

    float2 tile_uv = grid / GRID_MAX;

    float2 face_uv;
    face_uv.x = u_min + tile_uv.x * tile_size;
    face_uv.y = v_min + tile_uv.y * tile_size;

    float3 sphere_dir = cube_to_sphere(face_uv_to_cube(face_uv, face));

    // Tile center direction on sphere (for camera-relative precision)
    float2 center_fuv = float2(u_min + 0.5 * tile_size, v_min + 0.5 * tile_size);
    float3 center_dir = cube_to_sphere(face_uv_to_cube(center_fuv, face));

    // Sample heightmap at texel centers aligned with vertex positions
    float2 hm_uv = (grid + 0.5) * heightmap_texel;

    float3 uvw = float3(hm_uv, float(pool_index));
    float h = heightmap.SampleLevel(heightmap_sampler, uvw, 0).r;

    if (is_skirt) h -= SKIRT_DROP;

    // Vertex displacement from tile center (small float, preserves precision)
    float3 delta_dir = sphere_dir - center_dir;
    float3 displacement = delta_dir * planet_radius + sphere_dir * h;

    // Camera-relative position: displacement + (tile_center - camera)
    float3 world_rel = displacement + float3(rel_x, rel_y, rel_z);

    // Normal: perturb sphere normal by height gradient in tangent space
    float3 ref = abs(sphere_dir.y) > 0.999 ? float3(1, 0, 0) : float3(0, 1, 0);
    float3 east = normalize(cross(ref, sphere_dir));
    float3 north = normalize(cross(sphere_dir, east));

    float3 uvw_L = float3(hm_uv + float2(-heightmap_texel, 0), float(pool_index));
    float3 uvw_R = float3(hm_uv + float2( heightmap_texel, 0), float(pool_index));
    float3 uvw_D = float3(hm_uv + float2(0, -heightmap_texel), float(pool_index));
    float3 uvw_U = float3(hm_uv + float2(0,  heightmap_texel), float(pool_index));

    float hL = heightmap.SampleLevel(heightmap_sampler, uvw_L, 0).r;
    float hR = heightmap.SampleLevel(heightmap_sampler, uvw_R, 0).r;
    float hD = heightmap.SampleLevel(heightmap_sampler, uvw_D, 0).r;
    float hU = heightmap.SampleLevel(heightmap_sampler, uvw_U, 0).r;

    float cell_world = tile_size * planet_radius / GRID_MAX;
    float3 normal = normalize(sphere_dir - ((hR - hL) / (2.0 * cell_world)) * east
                                         - ((hU - hD) / (2.0 * cell_world)) * north);

    VSOutput o;
    o.position = mul(proj, mul(view, float4(world_rel, 1.0)));
    o.world_normal = normal;
    o.uv = tile_uv;
    o.height_normalized = saturate(h / max_elevation);
    o.world_pos = world_rel;
    o.sphere_direction = sphere_dir;
    return o;
}

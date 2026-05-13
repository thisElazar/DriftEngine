[[vk::push_constant]]
cbuffer ClumpPC {
    float4x4 mvp;
    float2 wind_dir;
    float wind_speed;
    float time;
};

struct VSInput {
    // Per-vertex (binding 0)
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float3 color    : COLOR;
    [[vk::location(3)]] float  height_t : TEXCOORD0;
    // Per-instance (binding 1)
    [[vk::location(4)]] float3 inst_pos    : TEXCOORD1;  // world xyz (y = terrain height)
    [[vk::location(5)]] float  inst_health : TEXCOORD2;  // 0..1
    // inst_seed (loc 6) wired up once we switch to DXC for uint vertex attrs
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float3 color : COLOR;
};

VSOutput main(VSInput input)
{
    float3 pos = input.position;

    // Growth: scale height so plants emerge from ground as health rises.
    // Reaches full size at health = 0.5 so mature plants always look complete.
    float growth = saturate(input.inst_health * 2.0);
    pos.y *= growth;

    // Translate canonical mesh to instance world position (y = terrain height).
    pos.x += input.inst_pos.x;
    pos.y += input.inst_pos.y;
    pos.z += input.inst_pos.z;

    // Wind sway — world position varies per instance so phase is naturally staggered.
    // Sway is also gated by growth so seedlings barely move.
    float sway = input.height_t * input.height_t * growth;
    float phase = time * 3.0 + dot(pos.xz, wind_dir) * 8.0;
    float wave = sin(phase) + 0.3 * sin(phase * 2.3 + 1.7);
    pos.xz += wind_dir * sway * wind_speed * 0.15 * wave;

    VSOutput o;
    o.position = mul(mvp, float4(pos, 1.0));
    o.world_normal = input.normal;
    o.color = input.color;
    return o;
}

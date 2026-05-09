[[vk::push_constant]]
cbuffer PC {
    float4x4 mvp;
    float2 wind_dir;
    float wind_speed;
    float time;
};

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float3 color    : COLOR;
    [[vk::location(3)]] float  height_t : TEXCOORD0;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float3 color : COLOR;
};

VSOutput main(VSInput input)
{
    float3 pos = input.position;

    float sway = input.height_t * input.height_t;
    float phase = time * 3.0 + dot(pos.xz, wind_dir) * 8.0;
    float wave = sin(phase) + 0.3 * sin(phase * 2.3 + 1.7);
    pos.xz += wind_dir * sway * wind_speed * 0.15 * wave;

    VSOutput o;
    o.position = mul(mvp, float4(pos, 1.0));
    o.world_normal = input.normal;
    o.color = input.color;
    return o;
}

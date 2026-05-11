[[vk::push_constant]]
cbuffer PC {
    float4x4 mvp;
};

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float3 color    : COLOR;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
    [[vk::location(1)]] float3 color : COLOR;
};

VSOutput main(VSInput input)
{
    VSOutput o;
    o.position     = mul(mvp, float4(input.position, 1.0));
    o.world_normal = input.normal;
    o.color        = input.color;
    return o;
}

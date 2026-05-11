[[vk::push_constant]]
cbuffer PC {
    float4x4 mvp;
    float joint_radius;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4x4> g_joints;

struct VSInput {
    [[vk::location(0)]] uint joint_index : TEXCOORD0;
};

struct VSOutput {
    float4 position : SV_Position;
};

VSOutput main(VSInput input)
{
    float4x4 jw = g_joints[input.joint_index];
    float3 pos = mul(jw, float4(0, 0, 0, 1)).xyz;

    VSOutput o;
    o.position = mul(mvp, float4(pos, 1.0));
    return o;
}

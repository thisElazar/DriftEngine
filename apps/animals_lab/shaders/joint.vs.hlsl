[[vk::push_constant]]
cbuffer PC {
    float4x4 mvp;
    float joint_radius;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4x4> g_joints;

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
};

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 world_normal : NORMAL;
};

VSOutput main(VSInput input, uint instance_id : SV_InstanceID)
{
    float4x4 jw = g_joints[instance_id];
    float3 joint_pos = mul(jw, float4(0, 0, 0, 1)).xyz;
    float3 world_pos = joint_pos + input.position * joint_radius;

    VSOutput o;
    o.position     = mul(mvp, float4(world_pos, 1.0));
    o.world_normal = input.normal;
    return o;
}

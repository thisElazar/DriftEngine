// fullscreen.vs.hlsl — single fullscreen triangle from the vertex index.
// Emits clip position at z = 0 (reverse-Z far plane, matching the depth-clear
// value so a depth-EQUAL sky pass touches only unrendered pixels), plus the
// NDC xy for ray reconstruction and a [0,1] uv for post passes.

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 ndc : TEXCOORD0;
    [[vk::location(1)]] float2 uv  : TEXCOORD1;
};

VSOutput main(uint vid : SV_VertexID)
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOutput o;
    o.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    o.ndc = o.position.xy;
    o.uv = uv;
    return o;
}

// Fullscreen triangle — no vertex buffer, uses SV_VertexID
// Outputs clip-space position and screen UV

struct VSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

VSOutput main(uint vid : SV_VertexID)
{
    // Triangle covering the full screen: vertices at (-1,-1), (3,-1), (-1,3)
    float2 pos = float2((vid << 1) & 2, vid & 2);
    VSOutput o;
    o.position = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(pos.x, 1.0 - pos.y);
    return o;
}

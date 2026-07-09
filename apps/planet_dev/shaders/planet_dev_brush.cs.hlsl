// planet_dev terrain brush — terrain_brush.hlsl on a Texture2DArray layer.
// brush_x/y are TILE-LOCAL grid coords and may lie outside [0, grid): the
// caller dispatches every overlapped tile with the same world-space brush
// expressed in that tile's frame, so edits straddling a seam stay seamless.

[[vk::binding(0, 0)]] RWTexture2DArray<float> terrain;

[[vk::push_constant]]
cbuffer PC {
    float brush_x;
    float brush_y;
    float brush_radius;
    float brush_amount;
    uint grid_w;
    uint grid_h;
    uint layer;
    uint _pad0;
};

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= grid_w || dtid.y >= grid_h) return;
    float2 d = float2(dtid.xy) - float2(brush_x, brush_y);
    float r2 = dot(d, d);
    float falloff = exp(-r2 / (brush_radius * brush_radius));
    if (falloff < 0.001) return;
    uint3 idx = uint3(dtid.xy, layer);
    float h = terrain[idx];
    terrain[idx] = h + brush_amount * falloff;
}

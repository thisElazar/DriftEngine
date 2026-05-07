[[vk::binding(0, 0)]] RWTexture2D<float> terrain;

[[vk::push_constant]]
cbuffer PC {
    float brush_x;
    float brush_y;
    float brush_radius;
    float brush_amount;
    uint grid_w;
    uint grid_h;
    uint _pad0;
    uint _pad1;
};

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= grid_w || dtid.y >= grid_h) return;
    float2 d = float2(dtid.xy) - float2(brush_x, brush_y);
    float r2 = dot(d, d);
    float falloff = exp(-r2 / (brush_radius * brush_radius));
    if (falloff < 0.001) return;
    float h = terrain[dtid.xy];
    terrain[dtid.xy] = h + brush_amount * falloff;
}

struct PSInput {
    float4 position : SV_Position;
};

float4 main(PSInput input) : SV_Target
{
    return float4(0.6, 0.6, 0.6, 1.0);
}

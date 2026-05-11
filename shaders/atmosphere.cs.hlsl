// atmosphere.cs.hlsl — Ported from Drift V2 AtmosphereCompute.usf
// Milestones 3+4+5: Buoyancy, mountain waves, sustained circulation.
// Ping-pong state/wind instead of in-place UAV read-write.

#define THREADGROUP_SIZE 8
#define TWO_PI 6.28318530718
#define PI 3.14159265359

[[vk::binding(0, 0)]] Texture2D<float>   terrain_tex;
[[vk::binding(7, 0)]] SamplerState       terrain_sampler;
[[vk::binding(1, 0)]] Texture2D<float4>   state_read;
[[vk::binding(2, 0)]] RWTexture2D<float4> state_write;
[[vk::binding(3, 0)]] Texture2D<float2>   wind_read;
[[vk::binding(4, 0)]] RWTexture2D<float2> wind_write;
[[vk::binding(5, 0)]] RWTexture2D<float>  precipitation_out;
[[vk::binding(6, 0)]] RWTexture2D<float4> render_out;

[[vk::push_constant]]
cbuffer PC {
    float dt;
    float accumulated_time;
    uint  grid_w;
    uint  grid_h;
    float terrain_scale;
    float orographic_lift_coeff;
    float adiabatic_cooling_rate;
    float rain_shadow_intensity;
    uint  force_init;
    uint  _pad0; uint _pad1; uint _pad2;
};

// ==========================================
// PHYSICS CONSTANTS — verbatim from .usf Milestone 5
// ==========================================

static const float CONDENSATION_RATE = 0.012;
static const float EVAPORATION_RATE = 0.008;
static const float PRECIPITATION_THRESHOLD = 0.65;
static const float CLOUD_DISSIPATION = 0.0008;
static const float EVAPORATION_THRESHOLD = 0.9;

static const float WIND_MOMENTUM = 0.98;
static const float WIND_FORCING = 0.12;
static const float WIND_DIFFUSION = 0.02;
static const float WIND_DAMPING = 0.9995;
static const float VORTICITY_STRENGTH = 0.18;

static const float SOLAR_CONSTANT = 0.5;
static const float RAIN_COOLING_RATE = 0.3;
static const float EVAPORATION_COEFF = 0.003;
static const float LATENT_HEAT_FACTOR = 6.0;
static const float THERMAL_EXPANSION = 0.006;
static const float TURBULENCE_STRENGTH = 0.35;

static const float PREVAILING_WIND_STRENGTH = 2.0;
static const float PRESSURE_GRADIENT_STRENGTH = 0.8;
static const float CONVECTIVE_CLOUD_FORMATION = 0.0018;
static const float MOISTURE_ADVECTION_RATE = 0.15;

static const float LAND_HEATING_FACTOR = 1.3;
static const float ELEVATION_COOLING_RATE = 0.4;
static const float RADIATIVE_COOLING_RATE = 0.08;

static const float BRUNT_VAISALA_FREQ = 0.01;
static const float WAVE_AMPLITUDE_SCALE = 0.2;
static const float WAVE_DECAY_DISTANCE = 3.0;

static const float MIN_WIND_ENERGY = 0.1;
static const float ENERGY_INJECTION_RATE = 0.03;
static const float WIND_NOISE_AMPLITUDE = 0.04;

// ==========================================
// HELPERS
// ==========================================

float ComputeSaturationPressure(float Temperature)
{
    float TempC = Temperature - 273.15;
    return 611.0 * exp((17.27 * TempC) / (TempC + 237.3));
}

float hash(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float noise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + float2(1.0, 0.0));
    float c = hash(i + float2(0.0, 1.0));
    float d = hash(i + float2(1.0, 1.0));

    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// PORT NOTE: wrapping preserved — atmosphere represents a larger air mass
int2 WrapCoord(int2 c)
{
    return ((c % int2(grid_w, grid_h)) + int2(grid_w, grid_h)) % int2(grid_w, grid_h);
}

// ==========================================
// OROGRAPHIC LIFT (Milestone 4)
// ==========================================

float2 CalculateOrographicLift(float2 UV, float2 Wind, out float2 TerrainGradient)
{
    float2 TexelSize = 1.0 / float2(grid_w, grid_h);

    float HeightC = terrain_tex.SampleLevel(terrain_sampler, UV, 0).r;
    float HeightL = terrain_tex.SampleLevel(terrain_sampler, UV + float2(-TexelSize.x, 0), 0).r;
    float HeightR = terrain_tex.SampleLevel(terrain_sampler, UV + float2( TexelSize.x, 0), 0).r;
    float HeightT = terrain_tex.SampleLevel(terrain_sampler, UV + float2(0, -TexelSize.y), 0).r;
    float HeightB = terrain_tex.SampleLevel(terrain_sampler, UV + float2(0,  TexelSize.y), 0).r;

    TerrainGradient = float2(HeightR - HeightL, HeightB - HeightT) / (2.0 * terrain_scale);

    float WindSpeed = length(Wind);
    if (WindSpeed < 0.001)
        return float2(0, HeightC);

    float2 WindDir = normalize(Wind);

    float VerticalVelocity = dot(WindDir, TerrainGradient) * WindSpeed * orographic_lift_coeff;

    float N = BRUNT_VAISALA_FREQ;
    float Nh0_U = (N * HeightC * terrain_scale) / max(WindSpeed, 0.1);

    if (Nh0_U > 0.3 && HeightC > 0.05)
    {
        float waveAmplitude = WindSpeed * Nh0_U * WAVE_AMPLITUDE_SCALE;
        float wavelength = (TWO_PI * WindSpeed) / N;
        wavelength *= terrain_scale;

        float maxHeight = HeightC;
        float2 peakOffset = float2(0, 0);

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                float2 sampleUV = UV + float2(dx, dy) * TexelSize;
                float h = terrain_tex.SampleLevel(terrain_sampler, sampleUV, 0).r;
                if (h > maxHeight)
                {
                    maxHeight = h;
                    peakOffset = float2(dx, dy) * TexelSize;
                }
            }
        }

        float2 toPeak = peakOffset;
        float downwindDist = -dot(toPeak, WindDir) * grid_w * terrain_scale;

        if (downwindDist > 0.0)
        {
            float phase = (downwindDist / wavelength) * TWO_PI;
            float decay = exp(-downwindDist / (wavelength * WAVE_DECAY_DISTANCE));
            float waveContribution = waveAmplitude * sin(phase) * decay;
            VerticalVelocity += waveContribution;
        }
    }

    return float2(VerticalVelocity, HeightC);
}

// ==========================================
// MAIN
// ==========================================

[numthreads(THREADGROUP_SIZE, THREADGROUP_SIZE, 1)]
void main(uint3 DispatchThreadId : SV_DispatchThreadID)
{
    uint2 PixelCoord = DispatchThreadId.xy;

    if (PixelCoord.x >= grid_w || PixelCoord.y >= grid_h)
        return;

    float2 UV = (float2(PixelCoord) + 0.5) / float2(grid_w, grid_h);

    // ---- INIT BRANCH ----
    if (force_init == 1)
    {
        float moistureNoise = sin(UV.x * 4.0) * cos(UV.y * 4.0) * 0.3 + 0.5;

        state_write[PixelCoord] = float4(
            0.0,
            moistureNoise,
            288.15 / 400.0,
            0.0
        );

        float2 initialWind = float2(
            cos(UV.y * PI) * 0.5 + 0.3,
            sin(UV.x * TWO_PI) * 0.2
        );
        wind_write[PixelCoord] = initialWind;
        precipitation_out[PixelCoord] = 0.0;
        render_out[PixelCoord] = float4(0, 0, 0, 0);
        return;
    }

    // ---- READ STATE ----
    // PORT NOTE: reads from state_read / wind_read (ping-pong) instead of in-place
    float4 CurrentState = state_read[PixelCoord];
    float CloudCover = CurrentState.r;
    float Moisture = CurrentState.g;
    float Temperature = CurrentState.b * 400.0;
    float Precipitation = CurrentState.a;

    // ==========================================
    // SOLAR HEATING
    // ==========================================
    float solarAngle = max(0.0, sin((accumulated_time * 0.0001) * TWO_PI));

    float2 TerrainGradient;
    float2 OrographicData = CalculateOrographicLift(UV, wind_read[PixelCoord], TerrainGradient);
    float TerrainHeight = OrographicData.y;
    float isWater = (TerrainHeight < 0.1) ? 1.0 : 0.0;

    float terrainAbsorption = lerp(LAND_HEATING_FACTOR, 1.0, isWater);
    float elevationCooling = saturate(TerrainHeight * ELEVATION_COOLING_RATE);

    float solarHeating = SOLAR_CONSTANT * solarAngle * (1.0 - CloudCover * 0.7);
    solarHeating *= terrainAbsorption * (1.0 - elevationCooling);
    Temperature += solarHeating * dt;

    float nightCooling = (1.0 - solarAngle) * RADIATIVE_COOLING_RATE;
    nightCooling *= (1.0 - Moisture * 0.3) * (1.0 - CloudCover * 0.5);
    Temperature -= nightCooling * dt;

    // ==========================================
    // WIND DYNAMICS
    // ==========================================
    float2 WindCurrent = wind_read[PixelCoord];

    float2 baseWind = float2(
        cos(accumulated_time * 0.00003 + UV.y * PI * 0.5) * PREVAILING_WIND_STRENGTH,
        sin(accumulated_time * 0.00002 + UV.x * PI * 0.3) * PREVAILING_WIND_STRENGTH * 0.5
    );

    float latitudeFactor = cos(UV.y * PI);
    float2 latitudeWind = float2(latitudeFactor * 0.8, 0.0);

    float2 seasonalVariation = float2(
        sin(accumulated_time * 0.00001) * 0.3,
        cos(accumulated_time * 0.000008) * 0.2
    );

    float2 persistentWind = (baseWind + latitudeWind + seasonalVariation) * WIND_FORCING;

    float VerticalVelocity = OrographicData.x;

    // Neighbor wind reads
    int2 PixN = WrapCoord(int2(PixelCoord) + int2(0, -1));
    int2 PixS = WrapCoord(int2(PixelCoord) + int2(0,  1));
    int2 PixE = WrapCoord(int2(PixelCoord) + int2( 1, 0));
    int2 PixW = WrapCoord(int2(PixelCoord) + int2(-1, 0));

    float2 WindN = wind_read[PixN];
    float2 WindS = wind_read[PixS];
    float2 WindE = wind_read[PixE];
    float2 WindW = wind_read[PixW];

    float2 WindAvg = (WindN + WindS + WindE + WindW) * 0.25;

    // Vorticity confinement
    float2 gradX = (WindE - WindW) * 0.5;
    float2 gradY = (WindN - WindS) * 0.5;
    float vorticity = gradX.y - gradY.x;

    float2 vorticityGrad = float2(
        ((WindE.y - WindW.y) - (WindE.x - WindW.x)) * 0.5,
        ((WindN.y - WindS.y) - (WindN.x - WindS.x)) * 0.5
    );
    float vorticityMag = length(vorticityGrad);

    float2 vorticityForce = float2(0, 0);
    if (vorticityMag > 0.0001)
    {
        float2 vorN = normalize(vorticityGrad);
        vorticityForce = VORTICITY_STRENGTH * float2(-vorN.y, vorN.x) * vorticity;
    }

    // Multi-scale turbulence
    float2 turbulence = float2(0, 0);
    float amplitude = TURBULENCE_STRENGTH;
    float frequency = 2.0;
    float2 timeOffset = float2(accumulated_time * 0.01, accumulated_time * 0.0137);

    for (int octave = 0; octave < 3; octave++)
    {
        float2 samplePos = UV * frequency + timeOffset;
        float2 n = float2(
            noise(samplePos) * 2.0 - 1.0,
            noise(samplePos + 100.0) * 2.0 - 1.0
        );
        turbulence += n * amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    // ==========================================
    // BUOYANCY (Milestone 3+5)
    // ==========================================
    float ambientTemp = 288.15;
    float tempAnomaly = Temperature - ambientTemp;
    float2 buoyancyForce = float2(0, 0);

    float4 StateN = state_read[PixN];
    float4 StateS = state_read[PixS];
    float4 StateE = state_read[PixE];
    float4 StateW = state_read[PixW];

    float TempN = StateN.b * 400.0;
    float TempS = StateS.b * 400.0;
    float TempE = StateE.b * 400.0;
    float TempW = StateW.b * 400.0;

    float2 tempGradient = float2(TempE - TempW, TempN - TempS) * 0.5;
    float tempGradMag = length(tempGradient);

    if (tempGradMag > 0.005)
    {
        float2 pressureGradient = -normalize(tempGradient + float2(0.001, 0.001));
        float pressureForceStrength = tempGradMag * PRESSURE_GRADIENT_STRENGTH;
        buoyancyForce += pressureGradient * pressureForceStrength;
    }

    if (abs(tempAnomaly) > 1.5)
    {
        float2 pressureGradient = -normalize(tempGradient + float2(0.001, 0.001));
        float buoyancyStrength = tempAnomaly * THERMAL_EXPANSION;
        buoyancyForce += pressureGradient * buoyancyStrength;

        if (tempAnomaly > 3.0)
        {
            float divergence = 0.015 * (tempAnomaly - 3.0);
            buoyancyForce += normalize(WindCurrent + float2(0.001, 0.001)) * divergence;
        }
    }

    // Integrate wind
    float2 WindNew = WindCurrent * WIND_MOMENTUM;
    WindNew += vorticityForce * WIND_FORCING;
    WindNew += turbulence * WIND_FORCING;
    WindNew += buoyancyForce * WIND_FORCING;
    WindNew += persistentWind;
    WindNew = lerp(WindNew, WindAvg, WIND_DIFFUSION);
    WindNew *= WIND_DAMPING;

    // Energy floor
    float windSpeed = length(WindNew);
    if (windSpeed < MIN_WIND_ENERGY)
    {
        float2 energyBoost = normalize(WindNew + float2(0.001, 0.001)) * ENERGY_INJECTION_RATE;
        WindNew += energyBoost;

        float2 randomPush = float2(
            hash(UV + accumulated_time * 0.001) - 0.5,
            hash(UV.yx + accumulated_time * 0.001) - 0.5
        ) * WIND_NOISE_AMPLITUDE;
        WindNew += randomPush;
    }

    // Soft velocity cap
    windSpeed = length(WindNew);
    if (windSpeed > 2.8)
    {
        WindNew = normalize(WindNew) * (2.8 + (windSpeed - 2.8) * 0.25);
    }

    // ==========================================
    // ADVECTION (semi-Lagrangian, toroidal wrap)
    // ==========================================
    float2 CloudOffset = WindNew * dt * 0.001;
    float2 CloudUV = UV - CloudOffset;
    CloudUV = frac(CloudUV + float2(1.0, 1.0));

    float2 CloudPixel = CloudUV * float2(grid_w, grid_h) - 0.5;
    int2 CloudBase = int2(floor(CloudPixel));
    float2 CloudFrac = frac(CloudPixel);

    int2 P00 = WrapCoord(CloudBase);
    int2 P10 = WrapCoord(CloudBase + int2(1, 0));
    int2 P01 = WrapCoord(CloudBase + int2(0, 1));
    int2 P11 = WrapCoord(CloudBase + int2(1, 1));

    float4 S00 = state_read[P00];
    float4 S10 = state_read[P10];
    float4 S01 = state_read[P01];
    float4 S11 = state_read[P11];

    float4 AdvectedState = lerp(
        lerp(S00, S10, CloudFrac.x),
        lerp(S01, S11, CloudFrac.x),
        CloudFrac.y
    );

    CloudCover = AdvectedState.r;
    Moisture = AdvectedState.g;

    // ==========================================
    // CLOUD PHYSICS (Milestone 5)
    // ==========================================
    float SaturationPressure = ComputeSaturationPressure(Temperature);
    float CurrentVaporPressure = Moisture * SaturationPressure;

    // Condensation
    if (CurrentVaporPressure > SaturationPressure * 0.95)
    {
        float Condensation = (CurrentVaporPressure - SaturationPressure) / SaturationPressure;
        Condensation = min(Condensation * CONDENSATION_RATE, Moisture);
        CloudCover += Condensation;
        Moisture -= Condensation;
        Temperature += Condensation * 2.0 * dt;
    }

    // Orographic condensation
    if (VerticalVelocity > 0.01)
    {
        float OrographicCondensation = VerticalVelocity * Moisture * 0.5;
        CloudCover += OrographicCondensation * dt;
        Moisture = max(Moisture - OrographicCondensation * dt, 0.0);
        Temperature -= OrographicCondensation * adiabatic_cooling_rate * dt;
    }

    // Convective cloud formation
    if (tempAnomaly > 2.5 && Moisture > 0.25)
    {
        float convectiveCondensation = (tempAnomaly - 2.5) * Moisture * CONVECTIVE_CLOUD_FORMATION;
        CloudCover += convectiveCondensation * dt;
        Moisture -= convectiveCondensation * dt * 0.6;
        Temperature += convectiveCondensation * LATENT_HEAT_FACTOR * dt * 0.3;
    }

    // Evaporation over water
    if (isWater > 0.5)
    {
        float evapRate = EVAPORATION_COEFF * (Temperature / 288.15) * (1.0 + windSpeed / 8.0);
        evapRate *= (1.0 - Moisture * 0.7);
        Moisture = min(Moisture + evapRate * dt, 1.0);
        Temperature -= evapRate * LATENT_HEAT_FACTOR * dt;
    }

    // Horizontal moisture advection
    float MoistureN = StateN.g;
    float MoistureS = StateS.g;
    float MoistureE = StateE.g;
    float MoistureW = StateW.g;

    float2 moistureGradient = float2(MoistureE - MoistureW, MoistureN - MoistureS) * 0.5;
    float moistureAdvection = dot(normalize(WindNew + float2(0.001, 0.001)), moistureGradient);
    Moisture += moistureAdvection * windSpeed * MOISTURE_ADVECTION_RATE * dt;

    // Precipitation
    if (CloudCover > PRECIPITATION_THRESHOLD)
    {
        float RainRate = (CloudCover - PRECIPITATION_THRESHOLD) * 0.05;
        Precipitation = min(Precipitation + RainRate, 1.0);
        CloudCover -= RainRate * 0.5;
        Moisture = max(Moisture - RainRate * 0.3, 0.0);
    }
    else
    {
        Precipitation *= 0.95;
    }

    // Rain cooling
    if (Precipitation > 0.01)
    {
        float coolingRate = Precipitation * RAIN_COOLING_RATE;
        Temperature = max(Temperature - coolingRate * dt, 250.0);
    }

    // Cloud dissipation (wind-dependent)
    float dissipationFactor = lerp(0.6, 1.0, saturate(windSpeed / 2.0));
    CloudCover = max(CloudCover - CLOUD_DISSIPATION * dt * dissipationFactor, 0.0);

    // ==========================================
    // SAFETY & OUTPUT
    // ==========================================
    CloudCover = saturate(CloudCover);
    Moisture = saturate(Moisture);
    Temperature = clamp(Temperature, 200.0, 350.0);
    Precipitation = saturate(Precipitation);

    // PORT NOTE: bit-test NaN guard for driver compat
    uint nanTest = asuint(CloudCover + Moisture + Temperature + Precipitation);
    if ((nanTest & 0x7F800000) == 0x7F800000 && (nanTest & 0x007FFFFF) != 0)
    {
        CloudCover = 0.5;
        Moisture = 0.5;
        Temperature = 288.15;
        Precipitation = 0.0;
    }

    state_write[PixelCoord] = float4(CloudCover, Moisture, Temperature / 400.0, Precipitation);
    wind_write[PixelCoord] = WindNew;
    precipitation_out[PixelCoord] = Precipitation;

    // ==========================================
    // VISUAL RENDERING
    // ==========================================
    float3 CloudColor = float3(1.0, 1.0, 1.0);

    float Darkness = 1.0;
    if (Precipitation > 0.01)
        Darkness = lerp(1.0, 0.3, saturate(Precipitation * 2.0));
    else if (CloudCover > 0.5)
        Darkness = lerp(1.0, 0.7, (CloudCover - 0.5) * 2.0);

    CloudColor *= Darkness;

    float3 ColdTint = float3(0.9, 0.95, 1.0);
    float3 WarmTint = float3(1.0, 0.98, 0.9);
    float TempNorm = saturate((Temperature - 273.15) / 30.0);
    CloudColor *= lerp(ColdTint, WarmTint, TempNorm);

    float Shimmer = sin(accumulated_time * 2.0 + UV.x * 10.0) * 0.05 + 1.0;
    CloudColor *= Shimmer;

    float FinalAlpha = saturate(CloudCover * 1.5);

    render_out[PixelCoord] = float4(CloudColor * FinalAlpha, FinalAlpha);
}

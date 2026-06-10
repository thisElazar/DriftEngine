#pragma once

#include "vk_util.h"

#include <cstdint>
#include <vector>

struct StaticGridMesh {
    GpuBuffer vbo{};
    GpuBuffer ibo{};
    uint32_t  index_count = 0;
};

StaticGridMesh build_grid_mesh(VmaAllocator alloc, uint32_t grid_w, uint32_t grid_h);

void cpu_apply_brush(std::vector<float>& hm, uint32_t grid_w, uint32_t grid_h,
                     float bx, float by, float radius, float amount);

float sample_hm_bilinear(const std::vector<float>& hm,
                         uint32_t grid_w, uint32_t grid_h,
                         float gx, float gy);

std::vector<float> build_moisture_grid(const std::vector<float>& water_depth,
                                       uint32_t grid_w, uint32_t grid_h,
                                       float capillary_depth,
                                       int capillary_blur_radius);

std::vector<float> readback_water_depth(VkDevice device, VmaAllocator alloc,
                                        VkQueue queue, uint32_t family,
                                        VkImage state_img,
                                        uint32_t grid_w, uint32_t grid_h,
                                        uint32_t layer = 0);

#pragma once

#include "vk_common.h"
#include <vk_mem_alloc.h>
#include <vector>
#include <cstdint>

struct DepthBuffer {
    VkImage       image;
    VmaAllocation allocation;
    VkImageView   view;
};

struct HeightmapData {
    std::vector<float> values;
    uint32_t width;
    uint32_t height;
};

struct HeightmapGPU {
    VkImage       image;
    VmaAllocation allocation;
    VkImageView   view;
};

struct SweImage {
    VkImage       image;
    VmaAllocation allocation;
    VkImageView   view;
};

DepthBuffer create_depth_buffer(VkDevice device, VmaAllocator allocator, VkExtent2D extent);
void destroy_depth_buffer(VkDevice device, VmaAllocator allocator, DepthBuffer& db);

SweImage create_swe_image(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h);
void destroy_swe_image(VkDevice device, VmaAllocator allocator, SweImage& img);

SweImage create_sediment_image(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h);
SweImage create_wind_image(VkDevice device, VmaAllocator allocator, uint32_t w, uint32_t h);
SweImage create_volume_image(VkDevice device, VmaAllocator allocator,
                             uint32_t w, uint32_t h, uint32_t d, VkFormat format);

HeightmapGPU upload_heightmap(VkDevice device, VmaAllocator allocator,
                              VkQueue queue, uint32_t queue_family,
                              const HeightmapData& hm);

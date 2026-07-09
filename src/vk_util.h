#pragma once

#include "vk_common.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

struct GpuBuffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

GpuBuffer create_host_buffer(VmaAllocator alloc, VkDeviceSize size, VkBufferUsageFlags usage);
GpuBuffer create_readback_buffer(VmaAllocator alloc, VkDeviceSize size);
void      destroy_buffer(VmaAllocator alloc, GpuBuffer& b);

// ---------------------------------------------------------------------------
// SPIR-V / shader helpers
// ---------------------------------------------------------------------------

std::vector<uint32_t> load_spirv(const char* path);
VkShaderModule        make_shader(VkDevice device, const char* path);

// ---------------------------------------------------------------------------
// One-shot command buffer
// ---------------------------------------------------------------------------

struct OneShot {
    VkDevice        device;
    VkQueue         queue;
    VkCommandPool   pool;
    VkCommandBuffer cmd;
};

OneShot oneshot_begin(VkDevice device, VkQueue queue, uint32_t family);
void    oneshot_end(OneShot& s);

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------

void image_barrier(VkCommandBuffer cmd, VkImage img,
                   VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                   VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                   VkImageLayout old_layout, VkImageLayout new_layout,
                   uint32_t base_layer = 0, uint32_t layer_count = 1);

void compute_memory_barrier(VkCommandBuffer cmd);

// ---------------------------------------------------------------------------
// Compute pipeline factory
// ---------------------------------------------------------------------------

struct ComputePipeline {
    VkShaderModule        shader   = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl      = VK_NULL_HANDLE;
    VkPipelineLayout      layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
};

ComputePipeline make_compute_pipeline(VkDevice device, const char* spv,
                                      const std::vector<VkDescriptorType>& bindings,
                                      uint32_t push_size);
void destroy_compute_pipeline(VkDevice device, ComputePipeline& cp);

// ---------------------------------------------------------------------------
// Data format helpers
// ---------------------------------------------------------------------------

float half_to_float(uint16_t h);

// ---------------------------------------------------------------------------
// Image upload
// ---------------------------------------------------------------------------

void update_r32_image(VkDevice device, VmaAllocator alloc,
                      VkQueue queue, uint32_t family,
                      VkImage img, const std::vector<float>& data,
                      uint32_t w, uint32_t h, uint32_t layer = 0);

// Upload a tightly-packed RGBA32F array image (all `layers` of a 2D array,
// data laid out layer-contiguous: layer0 full w*h*4 floats, then layer1, ...).
// `old_layout` is the image's current layout (UNDEFINED for the first upload,
// GENERAL for subsequent re-uploads); it is left in GENERAL on return.
void update_rgba32f_array(VkDevice device, VmaAllocator alloc,
                          VkQueue queue, uint32_t family,
                          VkImage img, const std::vector<float>& data,
                          uint32_t w, uint32_t h, uint32_t layers,
                          VkImageLayout old_layout);

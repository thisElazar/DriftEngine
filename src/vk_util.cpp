#include "vk_util.h"

#include <cstdio>
#include <cstring>
#include <fstream>

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

GpuBuffer create_host_buffer(VmaAllocator alloc, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size  = size;
    ci.usage = usage;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    GpuBuffer b{};
    VK_CHECK(vmaCreateBuffer(alloc, &ci, &ai, &b.buffer, &b.allocation, nullptr));
    return b;
}

GpuBuffer create_readback_buffer(VmaAllocator alloc, VkDeviceSize size)
{
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size  = size;
    ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
             | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    GpuBuffer b{};
    VK_CHECK(vmaCreateBuffer(alloc, &ci, &ai, &b.buffer, &b.allocation, nullptr));
    return b;
}

void destroy_buffer(VmaAllocator alloc, GpuBuffer& b)
{
    if (b.buffer) { vmaDestroyBuffer(alloc, b.buffer, b.allocation); b = {}; }
}

// ---------------------------------------------------------------------------
// SPIR-V / shader helpers
// ---------------------------------------------------------------------------

std::vector<uint32_t> load_spirv(const char* path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "Failed to open SPIR-V file: %s\n", path);
        std::abort();
    }
    auto size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));
    return buffer;
}

VkShaderModule make_shader(VkDevice device, const char* path)
{
    auto spv = load_spirv(path);
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(uint32_t);
    ci.pCode    = spv.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
    return mod;
}

// ---------------------------------------------------------------------------
// One-shot command buffer
// ---------------------------------------------------------------------------

OneShot oneshot_begin(VkDevice device, VkQueue queue, uint32_t family)
{
    OneShot s{device, queue, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = family;
    VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &s.pool));

    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = s.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &cai, &s.cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(s.cmd, &bi));
    return s;
}

void oneshot_end(OneShot& s)
{
    VK_CHECK(vkEndCommandBuffer(s.cmd));
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(s.device, &fci, nullptr, &fence));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s.cmd;
    VK_CHECK(vkQueueSubmit(s.queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(s.device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(s.device, fence, nullptr);
    vkDestroyCommandPool(s.device, s.pool, nullptr);
}

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------

void image_barrier(VkCommandBuffer cmd, VkImage img,
                   VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                   VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                   VkImageLayout old_layout, VkImageLayout new_layout)
{
    VkImageMemoryBarrier2 b{};
    b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask  = src_stage;
    b.srcAccessMask = src_access;
    b.dstStageMask  = dst_stage;
    b.dstAccessMask = dst_access;
    b.oldLayout     = old_layout;
    b.newLayout     = new_layout;
    b.image         = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}

void compute_memory_barrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                     | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                     | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

// ---------------------------------------------------------------------------
// Compute pipeline factory
// ---------------------------------------------------------------------------

ComputePipeline make_compute_pipeline(VkDevice device, const char* spv,
                                      const std::vector<VkDescriptorType>& bindings,
                                      uint32_t push_size)
{
    ComputePipeline cp{};
    cp.shader = make_shader(device, spv);

    std::vector<VkDescriptorSetLayoutBinding> b(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        b[i].binding         = static_cast<uint32_t>(i);
        b[i].descriptorType  = bindings[i];
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = static_cast<uint32_t>(b.size());
    dslci.pBindings    = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &cp.dsl));

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size       = push_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &cp.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &push;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &cp.layout));

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cp.shader;
    stage.pName  = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = cp.layout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &cp.pipeline));

    return cp;
}

void destroy_compute_pipeline(VkDevice device, ComputePipeline& cp)
{
    vkDestroyPipeline(device, cp.pipeline, nullptr);
    vkDestroyPipelineLayout(device, cp.layout, nullptr);
    vkDestroyDescriptorSetLayout(device, cp.dsl, nullptr);
    vkDestroyShaderModule(device, cp.shader, nullptr);
    cp = {};
}

// ---------------------------------------------------------------------------
// Data format helpers
// ---------------------------------------------------------------------------

float half_to_float(uint16_t h)
{
    uint32_t s = (h & 0x8000u) << 16;
    uint32_t e = (h & 0x7C00u) >> 10;
    uint32_t m =  h & 0x03FFu;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            while ((m & 0x0400u) == 0) { m <<= 1; --e; }
            ++e; m &= 0x03FFu;
            f = s | ((e + (127 - 15)) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000u | (m << 13);
    } else {
        f = s | ((e + (127 - 15)) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// ---------------------------------------------------------------------------
// Image upload
// ---------------------------------------------------------------------------

void update_r32_image(VkDevice device, VmaAllocator alloc,
                      VkQueue queue, uint32_t family,
                      VkImage img, const std::vector<float>& data,
                      uint32_t w, uint32_t h)
{
    VkDeviceSize bytes = VkDeviceSize{w} * h * sizeof(float);
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
             | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo staging_info{};
    VK_CHECK(vmaCreateBuffer(alloc, &bci, &ai, &staging, &staging_alloc, &staging_info));
    std::memcpy(staging_info.pMappedData, data.data(), bytes);
    vmaFlushAllocation(alloc, staging_alloc, 0, VK_WHOLE_SIZE);

    OneShot os = oneshot_begin(device, queue, family);
    VkImageMemoryBarrier2 b1{};
    b1.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b1.srcStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                     | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    b1.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b1.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    b1.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b1.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    b1.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b1.image         = img;
    b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d1{};
    d1.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d1.imageMemoryBarrierCount = 1; d1.pImageMemoryBarriers = &b1;
    vkCmdPipelineBarrier2(os.cmd, &d1);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(os.cmd, staging, img,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier2 b2 = b1;
    b2.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    b2.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b2.dstStageMask  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                     | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    b2.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b2.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b2.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    VkDependencyInfo d2{};
    d2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d2.imageMemoryBarrierCount = 1; d2.pImageMemoryBarriers = &b2;
    vkCmdPipelineBarrier2(os.cmd, &d2);
    oneshot_end(os);
    vmaDestroyBuffer(alloc, staging, staging_alloc);
}

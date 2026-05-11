#include "grid_util.h"

#include <algorithm>
#include <cmath>
#include <cstring>

StaticGridMesh build_grid_mesh(VmaAllocator alloc, uint32_t grid_w, uint32_t grid_h)
{
    const uint32_t verts_x = grid_w + 1;
    const uint32_t verts_y = grid_h + 1;

    std::vector<float> verts;
    verts.reserve(verts_x * verts_y * 2);
    for (uint32_t y = 0; y <= grid_h; ++y) {
        for (uint32_t x = 0; x <= grid_w; ++x) {
            verts.push_back(static_cast<float>(x));
            verts.push_back(static_cast<float>(y));
        }
    }

    std::vector<uint32_t> idx;
    idx.reserve(grid_w * grid_h * 6);
    for (uint32_t y = 0; y < grid_h; ++y) {
        for (uint32_t x = 0; x < grid_w; ++x) {
            uint32_t i00 = y * verts_x + x;
            uint32_t i10 = i00 + 1;
            uint32_t i01 = i00 + verts_x;
            uint32_t i11 = i01 + 1;
            idx.push_back(i00); idx.push_back(i10); idx.push_back(i11);
            idx.push_back(i00); idx.push_back(i11); idx.push_back(i01);
        }
    }

    StaticGridMesh m{};
    m.index_count = static_cast<uint32_t>(idx.size());

    VkDeviceSize vbs = verts.size() * sizeof(float);
    VkDeviceSize ibs = idx.size()   * sizeof(uint32_t);
    m.vbo = create_host_buffer(alloc, vbs, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m.ibo = create_host_buffer(alloc, ibs, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    void* mapped = nullptr;
    VK_CHECK(vmaMapMemory(alloc, m.vbo.allocation, &mapped));
    std::memcpy(mapped, verts.data(), vbs);
    vmaUnmapMemory(alloc, m.vbo.allocation);

    VK_CHECK(vmaMapMemory(alloc, m.ibo.allocation, &mapped));
    std::memcpy(mapped, idx.data(), ibs);
    vmaUnmapMemory(alloc, m.ibo.allocation);

    return m;
}

void cpu_apply_brush(std::vector<float>& hm, uint32_t grid_w, uint32_t grid_h,
                     float bx, float by, float radius, float amount)
{
    int cx = static_cast<int>(bx);
    int cy = static_cast<int>(by);
    int half = static_cast<int>(std::ceil(radius * 3.0f));
    int x0 = std::max(0, cx - half);
    int x1 = std::min(static_cast<int>(grid_w) - 1, cx + half);
    int y0 = std::max(0, cy - half);
    int y1 = std::min(static_cast<int>(grid_h) - 1, cy + half);
    float r2 = radius * radius;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            float dx_ = static_cast<float>(x) - bx;
            float dy_ = static_cast<float>(y) - by;
            float d2 = dx_ * dx_ + dy_ * dy_;
            float falloff = std::exp(-d2 / r2);
            if (falloff < 0.001f) continue;
            hm[y * grid_w + x] += amount * falloff;
        }
    }
}

float sample_hm_bilinear(const std::vector<float>& hm,
                         uint32_t grid_w, uint32_t grid_h,
                         float gx, float gy)
{
    int x0 = std::clamp(static_cast<int>(std::floor(gx)), 0, static_cast<int>(grid_w) - 1);
    int y0 = std::clamp(static_cast<int>(std::floor(gy)), 0, static_cast<int>(grid_h) - 1);
    int x1 = std::min(x0 + 1, static_cast<int>(grid_w) - 1);
    int y1 = std::min(y0 + 1, static_cast<int>(grid_h) - 1);
    float fx = gx - x0;
    float fy = gy - y0;
    float h00 = hm[y0 * grid_w + x0];
    float h10 = hm[y0 * grid_w + x1];
    float h01 = hm[y1 * grid_w + x0];
    float h11 = hm[y1 * grid_w + x1];
    return (1 - fx) * (1 - fy) * h00 + fx * (1 - fy) * h10
         + (1 - fx) * fy       * h01 + fx * fy       * h11;
}

std::vector<float> build_moisture_grid(const std::vector<float>& water_depth,
                                       uint32_t grid_w, uint32_t grid_h,
                                       float capillary_depth,
                                       int capillary_blur_radius)
{
    std::vector<float> raw(grid_w * grid_h, 0.0f);
    for (size_t i = 0; i < raw.size(); ++i) {
        float d = std::max(0.0f, water_depth[i]);
        raw[i] = std::clamp(d / capillary_depth, 0.0f, 1.0f);
    }

    if (capillary_blur_radius <= 0) return raw;

    std::vector<float> tmp(grid_w * grid_h, 0.0f);
    int r = capillary_blur_radius;
    int gw = static_cast<int>(grid_w);
    int gh = static_cast<int>(grid_h);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            float sum = 0.0f;
            int count = 0;
            for (int dy = -r; dy <= r; ++dy) {
                int yy = std::clamp(y + dy, 0, gh - 1);
                for (int dx = -r; dx <= r; ++dx) {
                    int xx = std::clamp(x + dx, 0, gw - 1);
                    sum += raw[yy * gw + xx];
                    ++count;
                }
            }
            tmp[y * gw + x] = sum / count;
        }
    }
    for (size_t i = 0; i < tmp.size(); ++i)
        tmp[i] = std::max(raw[i], tmp[i]);
    return tmp;
}

std::vector<float> readback_water_depth(VkDevice device, VmaAllocator alloc,
                                        VkQueue queue, uint32_t family,
                                        VkImage state_img,
                                        uint32_t grid_w, uint32_t grid_h)
{
    VkDeviceSize bytes = VkDeviceSize{grid_w} * grid_h * 8;
    GpuBuffer staging = create_readback_buffer(alloc, bytes);

    OneShot s = oneshot_begin(device, queue, family);

    image_barrier(s.cmd, state_img,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_READ_BIT,
                  VK_IMAGE_LAYOUT_GENERAL,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent      = {grid_w, grid_h, 1};
    vkCmdCopyImageToBuffer(s.cmd, state_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           staging.buffer, 1, &copy);

    image_barrier(s.cmd, state_img,
                  VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_TRANSFER_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                    | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_IMAGE_LAYOUT_GENERAL);

    oneshot_end(s);

    VmaAllocationInfo ai{};
    vmaGetAllocationInfo(alloc, staging.allocation, &ai);
    auto* hp = static_cast<const uint16_t*>(ai.pMappedData);

    std::vector<float> depth(grid_w * grid_h, 0.0f);
    for (size_t i = 0; i < depth.size(); ++i) {
        depth[i] = half_to_float(hp[i * 4 + 0]);
    }

    destroy_buffer(alloc, staging);
    return depth;
}

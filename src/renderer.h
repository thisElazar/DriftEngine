#pragma once

#include "vk_common.h"
#include "resources.h"
#include <vk_mem_alloc.h>
#include <VkBootstrap.h>
#include <array>
#include <vector>

struct GLFWwindow;

constexpr uint32_t FRAMES_IN_FLIGHT = 2;

struct FrameData {
    VkCommandPool   pool;
    VkCommandBuffer cmd;
    VkSemaphore     image_available;
    VkSemaphore     render_finished;
    VkFence         in_flight;
};

struct Renderer {
    GLFWwindow* window = nullptr;

    vkb::Instance vkb_inst;
    vkb::PhysicalDevice vkb_phys;
    vkb::Device vkb_device;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;
    uint32_t gfx_family = 0;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VmaAllocator allocator = VK_NULL_HANDLE;

    vkb::Swapchain vkb_swapchain;
    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_views;
    DepthBuffer depth_buffer{};

    std::array<FrameData, FRAMES_IN_FLIGHT> frames{};
    uint32_t current_frame = 0;
    VkQueryPool query_pool = VK_NULL_HANDLE;

    VkDescriptorPool imgui_pool = VK_NULL_HANDLE;
};

void renderer_init(Renderer& r, int width, int height, const char* title);
void renderer_rebuild_swapchain(Renderer& r);
bool renderer_begin_frame(Renderer& r, FrameData*& frame, uint32_t& image_index, VkExtent2D& extent);
void renderer_end_frame(Renderer& r, FrameData& frame, uint32_t image_index);
void renderer_shutdown(Renderer& r);

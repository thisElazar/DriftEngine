#pragma once

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>

#define VK_CHECK(expr)                                                         \
    do {                                                                        \
        VkResult _r = (expr);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            std::fprintf(stderr, "Vulkan error %d at %s:%d: %s\n",              \
                         static_cast<int>(_r), __FILE__, __LINE__, #expr);      \
            std::abort();                                                        \
        }                                                                       \
    } while (0)

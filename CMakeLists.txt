cmake_minimum_required(VERSION 3.24)
project(drift_engine LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clangd / IDE tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to RelWithDebInfo if no build type set (sane default for engine work)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies (FetchContent)
# ---------------------------------------------------------------------------
include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        v1.4.349
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.2.1
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(glfw vk-bootstrap VulkanMemoryAllocator)

# ---------------------------------------------------------------------------
# Vulkan
# ---------------------------------------------------------------------------
# Honors VULKAN_SDK env var. On macOS this resolves to MoltenVK under the hood.
find_package(Vulkan REQUIRED)

message(STATUS "Vulkan include dir: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan library:     ${Vulkan_LIBRARIES}")

# ---------------------------------------------------------------------------
# Engine target
# ---------------------------------------------------------------------------
add_executable(drift_engine
    src/main.cpp
    src/vma_impl.cpp
)

target_link_libraries(drift_engine PRIVATE
    Vulkan::Vulkan
    glfw
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
)
target_compile_definitions(drift_engine PRIVATE GLFW_INCLUDE_NONE)

# Strict-ish warnings; expand later
if(MSVC)
    target_compile_options(drift_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(drift_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# macOS: ensure we link against the right SDK frameworks if/when we add them
if(APPLE)
    target_compile_definitions(drift_engine PRIVATE VK_USE_PLATFORM_METAL_EXT)
    # GLFW and vk-bootstrap dlopen libvulkan at runtime; the Homebrew loader
    # isn't in the default search path, so embed an rpath.
    get_filename_component(_vulkan_lib_dir "${Vulkan_LIBRARIES}" DIRECTORY)
    target_link_options(drift_engine PRIVATE "LINKER:-rpath,${_vulkan_lib_dir}")
endif()

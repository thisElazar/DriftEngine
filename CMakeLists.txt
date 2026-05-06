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
)

target_link_libraries(drift_engine PRIVATE Vulkan::Vulkan)

# Strict-ish warnings; expand later
if(MSVC)
    target_compile_options(drift_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(drift_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# macOS: ensure we link against the right SDK frameworks if/when we add them
if(APPLE)
    target_compile_definitions(drift_engine PRIVATE VK_USE_PLATFORM_METAL_EXT)
endif()

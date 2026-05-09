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

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.5
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(glfw vk-bootstrap VulkanMemoryAllocator glm imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

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
    src/camera.cpp
    src/pipeline.cpp
    src/renderer.cpp
    src/resources.cpp
    src/ui.cpp
    src/vma_impl.cpp
)

target_link_libraries(drift_engine PRIVATE
    Vulkan::Vulkan
    glfw
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glm::glm
    imgui
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

# ---------------------------------------------------------------------------
# HLSL → SPIR-V shader compilation
# ---------------------------------------------------------------------------
find_program(GLSLC glslc REQUIRED HINTS ENV VULKAN_SDK PATH_SUFFIXES bin)
message(STATUS "glslc: ${GLSLC}")

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

file(GLOB HLSL_SHADERS ${SHADER_DIR}/*.hlsl)

set(SPV_OUTPUTS "")
foreach(HLSL ${HLSL_SHADERS})
    get_filename_component(FNAME ${HLSL} NAME)
    string(REGEX REPLACE "\\.hlsl$" "" STEM ${FNAME})

    # Detect shader stage from filename convention
    if(STEM MATCHES "\\.vs$")
        set(SHADER_STAGE vertex)
    elseif(STEM MATCHES "\\.fs$")
        set(SHADER_STAGE fragment)
    else()
        set(SHADER_STAGE compute)
    endif()

    # Replace dots with underscores for SPV output to avoid CMake .rule conflicts
    string(REPLACE "." "_" SPV_NAME ${STEM})
    set(SPV ${SHADER_OUT_DIR}/${SPV_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} -x hlsl -fshader-stage=${SHADER_STAGE} -fentry-point=main
                --target-env=vulkan1.3 ${HLSL} -o ${SPV}
        DEPENDS ${HLSL}
        COMMENT "HLSL→SPIR-V [${SHADER_STAGE}]: ${FNAME}"
        VERBATIM
    )
    list(APPEND SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})
add_dependencies(drift_engine shaders)

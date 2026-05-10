# World Lab — single-tile sandbox app.
#
# Brings together the existing single-tile shaders (swe_init, swe_step,
# terrain_brush) with the bestiary plant pipeline so we can validate the
# water -> moisture -> plant-density loop on a 256x256 tile before deploying
# it on the planet.

add_executable(world_lab
    main.cpp
    ${CMAKE_SOURCE_DIR}/src/renderer.cpp
    ${CMAKE_SOURCE_DIR}/src/resources.cpp
    ${CMAKE_SOURCE_DIR}/src/vma_impl.cpp
)

target_include_directories(world_lab PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(world_lab PRIVATE
    bestiary
    Vulkan::Vulkan
    glfw
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glm::glm
    imgui
)

target_compile_definitions(world_lab PRIVATE GLFW_INCLUDE_NONE)

if(MSVC)
    target_compile_options(world_lab PRIVATE /W4 /permissive-)
else()
    target_compile_options(world_lab PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(APPLE)
    target_compile_definitions(world_lab PRIVATE VK_USE_PLATFORM_METAL_EXT)
    get_filename_component(_vulkan_lib_dir "${Vulkan_LIBRARIES}" DIRECTORY)
    target_link_options(world_lab PRIVATE "LINKER:-rpath,${_vulkan_lib_dir}")
endif()

# --- HLSL -> SPIR-V for World Lab shaders ---
# Engine shaders (swe_init, swe_step, terrain_brush) are already compiled
# by the top-level shader rule; we only build app-local render shaders here.
set(WL_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB WL_HLSL_SHADERS ${WL_SHADER_DIR}/*.hlsl)

set(WL_SPV_OUTPUTS "")
foreach(HLSL ${WL_HLSL_SHADERS})
    get_filename_component(FNAME ${HLSL} NAME)
    string(REGEX REPLACE "\\.hlsl$" "" STEM ${FNAME})

    if(STEM MATCHES "\\.vs$")
        set(WL_STAGE vertex)
    elseif(STEM MATCHES "\\.fs$")
        set(WL_STAGE fragment)
    else()
        set(WL_STAGE compute)
    endif()

    string(REPLACE "." "_" SPV_NAME ${STEM})
    set(SPV ${CMAKE_BINARY_DIR}/shaders/${SPV_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} -x hlsl -fshader-stage=${WL_STAGE} -fentry-point=main
                --target-env=vulkan1.3 ${HLSL} -o ${SPV}
        DEPENDS ${HLSL}
        COMMENT "HLSL->SPIR-V [${WL_STAGE}]: ${FNAME}"
        VERBATIM
    )
    list(APPEND WL_SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(world_lab_shaders ALL DEPENDS ${WL_SPV_OUTPUTS})
add_dependencies(world_lab world_lab_shaders shaders)

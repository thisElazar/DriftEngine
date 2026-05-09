# Species Lab — standalone executable.
#
# Links engine Vulkan/GLFW/ImGui setup via the engine's renderer/resources/
# vma_impl translation units. When the engine becomes a real library, those
# get pulled in via target_link_libraries instead of by-source.

add_executable(species_lab
    main.cpp
    ${CMAKE_SOURCE_DIR}/src/renderer.cpp
    ${CMAKE_SOURCE_DIR}/src/resources.cpp
    ${CMAKE_SOURCE_DIR}/src/vma_impl.cpp
)

target_include_directories(species_lab PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(species_lab PRIVATE
    bestiary
    Vulkan::Vulkan
    glfw
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glm::glm
    imgui
)

target_compile_definitions(species_lab PRIVATE GLFW_INCLUDE_NONE)

if(MSVC)
    target_compile_options(species_lab PRIVATE /W4 /permissive-)
else()
    target_compile_options(species_lab PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(APPLE)
    target_compile_definitions(species_lab PRIVATE VK_USE_PLATFORM_METAL_EXT)
    get_filename_component(_vulkan_lib_dir "${Vulkan_LIBRARIES}" DIRECTORY)
    target_link_options(species_lab PRIVATE "LINKER:-rpath,${_vulkan_lib_dir}")
endif()

# --- HLSL -> SPIR-V for Species Lab shaders ---
set(SL_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SL_HLSL_SHADERS ${SL_SHADER_DIR}/*.hlsl)

set(SL_SPV_OUTPUTS "")
foreach(HLSL ${SL_HLSL_SHADERS})
    get_filename_component(FNAME ${HLSL} NAME)
    string(REGEX REPLACE "\\.hlsl$" "" STEM ${FNAME})

    if(STEM MATCHES "\\.vs$")
        set(SL_STAGE vertex)
    elseif(STEM MATCHES "\\.fs$")
        set(SL_STAGE fragment)
    else()
        set(SL_STAGE compute)
    endif()

    string(REPLACE "." "_" SPV_NAME ${STEM})
    set(SPV ${CMAKE_BINARY_DIR}/shaders/${SPV_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} -x hlsl -fshader-stage=${SL_STAGE} -fentry-point=main
                --target-env=vulkan1.3 ${HLSL} -o ${SPV}
        DEPENDS ${HLSL}
        COMMENT "HLSL->SPIR-V [${SL_STAGE}]: ${FNAME}"
        VERBATIM
    )
    list(APPEND SL_SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(species_lab_shaders ALL DEPENDS ${SL_SPV_OUTPUTS})
add_dependencies(species_lab species_lab_shaders)

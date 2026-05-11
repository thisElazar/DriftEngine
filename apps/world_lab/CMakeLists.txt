# World Lab — single-tile ecosystem sandbox (terrain + water + plants + creatures).

add_executable(world_lab main.cpp world_lab.cpp)
target_link_libraries(world_lab PRIVATE drift_engine_core bestiary)
target_include_directories(world_lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

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

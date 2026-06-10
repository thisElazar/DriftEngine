# PLANET (DEV) — multi-tile world architecture sandbox (2x2 World-Lab tiles,
# cross-tile water via the planet SWE shader, world-spanning plants).

add_executable(planet_dev main.cpp planet_dev.cpp)
target_link_libraries(planet_dev PRIVATE drift_engine_core bestiary)
target_include_directories(planet_dev PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# --- HLSL -> SPIR-V for app-local render shaders ---
# The SWE compute shader (planet_swe_step) and the clump shaders are built by
# the top-level / world_lab shader rules; only the tiled terrain shaders live
# here.
set(PD_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB PD_HLSL_SHADERS CONFIGURE_DEPENDS ${PD_SHADER_DIR}/*.hlsl)

set(PD_SPV_OUTPUTS "")
foreach(HLSL ${PD_HLSL_SHADERS})
    get_filename_component(FNAME ${HLSL} NAME)
    string(REGEX REPLACE "\\.hlsl$" "" STEM ${FNAME})

    if(STEM MATCHES "\\.vs$")
        set(PD_STAGE vertex)
    elseif(STEM MATCHES "\\.fs$")
        set(PD_STAGE fragment)
    else()
        set(PD_STAGE compute)
    endif()

    string(REPLACE "." "_" SPV_NAME ${STEM})
    set(SPV ${CMAKE_BINARY_DIR}/shaders/${SPV_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} -x hlsl -fshader-stage=${PD_STAGE} -fentry-point=main
                --target-env=vulkan1.3 ${HLSL} -o ${SPV}
        DEPENDS ${HLSL}
        COMMENT "HLSL->SPIR-V [${PD_STAGE}]: ${FNAME}"
        VERBATIM
    )
    list(APPEND PD_SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(planet_dev_shaders ALL DEPENDS ${PD_SPV_OUTPUTS})
add_dependencies(planet_dev planet_dev_shaders world_lab_shaders shaders)

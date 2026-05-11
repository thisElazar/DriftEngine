# simCoupleTesting — simulation coupling test harness (SWE + atmosphere + erosion).

add_executable(sim_couple_test main.cpp)
target_link_libraries(sim_couple_test PRIVATE drift_engine_core)

# --- HLSL -> SPIR-V for local render shaders ---
set(SCT_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SCT_HLSL_SHADERS ${SCT_SHADER_DIR}/*.hlsl)

set(SCT_SPV_OUTPUTS "")
foreach(HLSL ${SCT_HLSL_SHADERS})
    get_filename_component(FNAME ${HLSL} NAME)
    string(REGEX REPLACE "\\.hlsl$" "" STEM ${FNAME})

    if(STEM MATCHES "\\.vs$")
        set(SCT_STAGE vertex)
    elseif(STEM MATCHES "\\.fs$")
        set(SCT_STAGE fragment)
    else()
        set(SCT_STAGE compute)
    endif()

    string(REPLACE "." "_" SPV_NAME ${STEM})
    set(SPV ${CMAKE_BINARY_DIR}/shaders/${SPV_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} -x hlsl -fshader-stage=${SCT_STAGE} -fentry-point=main
                --target-env=vulkan1.3 ${HLSL} -o ${SPV}
        DEPENDS ${HLSL}
        COMMENT "HLSL->SPIR-V [${SCT_STAGE}]: ${FNAME}"
        VERBATIM
    )
    list(APPEND SCT_SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(sim_couple_test_shaders ALL DEPENDS ${SCT_SPV_OUTPUTS})
add_dependencies(sim_couple_test sim_couple_test_shaders shaders)

# Plant Lab — plant morphology viewer.

add_executable(plant_lab main.cpp plant_lab.cpp)
target_include_directories(plant_lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. )
target_link_libraries(plant_lab PRIVATE drift_engine_core bestiary)

# --- HLSL -> SPIR-V for Plant Lab shaders ---
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

add_custom_target(plant_lab_shaders ALL DEPENDS ${SL_SPV_OUTPUTS})
add_dependencies(plant_lab plant_lab_shaders)

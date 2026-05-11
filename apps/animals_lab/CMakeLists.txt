# Animals Lab — procedural animal model viewer + animation testbed.

add_executable(animals_lab main.cpp animals_lab.cpp)
target_include_directories(animals_lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. )
target_link_libraries(animals_lab PRIVATE drift_engine_core bestiary)

# --- HLSL -> SPIR-V for Animals Lab shaders ---
set(AL_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB AL_HLSL_SHADERS ${AL_SHADER_DIR}/*.hlsl)

set(AL_SPV_OUTPUTS "")
foreach(HLSL ${AL_HLSL_SHADERS})
    get_filename_component(FNAME ${HLSL} NAME)
    string(REGEX REPLACE "\\.hlsl$" "" STEM ${FNAME})

    if(STEM MATCHES "\\.vs$")
        set(AL_STAGE vertex)
    elseif(STEM MATCHES "\\.fs$")
        set(AL_STAGE fragment)
    else()
        set(AL_STAGE compute)
    endif()

    string(REPLACE "." "_" SPV_NAME ${STEM})
    set(SPV ${CMAKE_BINARY_DIR}/shaders/${SPV_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${GLSLC} -x hlsl -fshader-stage=${AL_STAGE} -fentry-point=main
                --target-env=vulkan1.3 ${HLSL} -o ${SPV}
        DEPENDS ${HLSL}
        COMMENT "HLSL->SPIR-V [${AL_STAGE}]: ${FNAME}"
        VERBATIM
    )
    list(APPEND AL_SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(animals_lab_shaders ALL DEPENDS ${AL_SPV_OUTPUTS})
add_dependencies(animals_lab animals_lab_shaders)

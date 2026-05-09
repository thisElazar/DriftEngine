# bestiary — shared library for species/morphology/phenotype code.
#
# Links into both apps/species_lab and (eventually) apps/drift_engine when
# the planet engine starts using it. v0.0.1: just the clump morphology stub.

add_library(bestiary STATIC
    morphology/clump.cpp
    morphology/bush.cpp
    morphology/tree.cpp
    species_file.cpp
    environment.cpp
    distribution.cpp
)

target_include_directories(bestiary PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(bestiary PUBLIC tomlplusplus::tomlplusplus glm::glm)

target_compile_features(bestiary PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(bestiary PRIVATE /W4 /permissive-)
else()
    target_compile_options(bestiary PRIVATE -Wall -Wextra -Wpedantic)
endif()

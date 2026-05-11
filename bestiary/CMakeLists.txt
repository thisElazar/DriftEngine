# bestiary — shared library for species/morphology/phenotype code.
#
# Links into both apps/plant_lab and (eventually) apps/drift_engine when
# the planet engine starts using it. v0.0.1: just the clump morphology stub.

add_library(bestiary STATIC
    morphology/clump.cpp
    morphology/bush.cpp
    morphology/tree.cpp
    morphology/lplant.cpp
    morphology/wildflower.cpp
    creature/creature_profile.cpp
    creature/creature_mesh.cpp
    skeleton/skeleton.cpp
    skeleton/animation.cpp
    skeleton/skinning.cpp
    animals/animal_mesh.cpp
    animals/herbivore.cpp
    animals/predator.cpp
    animals/rabbit.cpp
    animals/bird.cpp
    animals/snake.cpp
    animals/walks/herbivore_walk.cpp
    animals/walks/predator_walk.cpp
    animals/walks/rabbit_walk.cpp
    animals/walks/bird_walk.cpp
    animals/walks/snake_walk.cpp
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

# Drift Launcher — unified single-window hub.

add_executable(launcher
    main.cpp
    ../plant_lab/plant_lab.cpp
    ../animals_lab/animals_lab.cpp
    ../world_lab/world_lab.cpp
    ../globe/globe.cpp
)

target_include_directories(launcher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(launcher PRIVATE drift_engine_core bestiary)
add_dependencies(launcher world_lab_shaders shaders)

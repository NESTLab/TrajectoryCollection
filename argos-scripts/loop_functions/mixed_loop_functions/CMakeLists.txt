link_directories(${CMAKE_BINARY_DIR}/controllers/khepera_mixed)
set(mixed_loop_functions_SOURCES mixed_loop_functions.cpp)

if(ARGOS_COMPILE_QTOPENGL)
  set(mixed_loop_functions_SOURCES
    ${mixed_loop_functions_SOURCES}
    foraging_qt_user_functions.cpp)
endif(ARGOS_COMPILE_QTOPENGL)

add_library(mixed_loop_functions MODULE ${mixed_loop_functions_SOURCES})
target_link_libraries(mixed_loop_functions
  khepera_mixed
  argos3core_simulator
  argos3plugin_simulator_dynamics2d
  argos3plugin_simulator_entities
  argos3plugin_simulator_kheperaiv
  argos3plugin_simulator_genericrobot
  argos3plugin_simulator_media)

if(ARGOS_COMPILE_QTOPENGL)
  target_link_libraries(mixed_loop_functions argos3plugin_simulator_qtopengl)
endif(ARGOS_COMPILE_QTOPENGL)

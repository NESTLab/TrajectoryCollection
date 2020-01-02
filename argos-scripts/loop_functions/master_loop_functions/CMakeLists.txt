set(HEADERS
	master_loop_functions.h)

set(SOURCES
	master_loop_functions.cpp)

add_library(argos3plugin_simulator_master_loop_functions SHARED ${HEADERS} ${SOURCES})

target_link_libraries(argos3plugin_simulator_master_loop_functions
  argos3core_simulator)

install(TARGETS argos3plugin_simulator_master_loop_functions DESTINATION lib/argos3)

FILE(RELATIVE_PATH relative_dir ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES ${HEADERS} DESTINATION "include/${relative_dir}")


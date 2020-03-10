include_directories(${CMAKE_SOURCE_DIR})

# Descend into the foraging_loop_functions directory
add_subdirectory(foraging_loop_functions)
add_subdirectory(flocking_loop_functions)
add_subdirectory(mixed_loop_functions)

# # Descend into the custom_distributions_loop_functions directory
# add_subdirectory(custom_distributions_loop_functions)

add_subdirectory(master_loop_functions)
add_subdirectory(trajectory_collection_loop_functions)


# # If Qt+OpenGL dependencies were found, descend into these directories
#   add_subdirectory(id_loop_functions)
#   add_subdirectory(manualcontrol_loop_functions)


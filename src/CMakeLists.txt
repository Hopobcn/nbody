add_subdirectory(gpu)

include_directories( "../include" "/usr/local/cuda/include" ${GLEW_INCLUDE_DIRS} SYSTEM ${INCLUDE_DIRS})
add_library(libglrender render_particles.cpp)
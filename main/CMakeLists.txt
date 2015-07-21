
include_directories(${INCLUDE_DIRS} "../include" ${GLEW_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

cuda_add_executable(nbody main.cpp)
target_link_libraries(nbody libgpunbody libglrender GL glut libglew)
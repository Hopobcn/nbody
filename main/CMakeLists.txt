cuda_add_executable(nbody main.cpp)
target_link_libraries(nbody libgpunbody libglrender ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES})
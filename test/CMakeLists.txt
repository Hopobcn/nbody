#list of tests
#add_subdirectory(nbody)

include_directories( ${GTEST_INCLUDE_DIR} ${GMOCK_INCLUDE_DIR})

add_executable(nbody-test
        nbody/nbody_test.cpp
        util/TestFloatingPoint.cpp)

target_link_libraries(nbody-test
        #    ${CMAKE_THREAD_LIBS_INIT}
        libgpunbody
        libglrender
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${GTEST_LIBRARY}
        ${GMOCK_LIBRARY}
        )

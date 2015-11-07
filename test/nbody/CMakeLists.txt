file(GLOB TEST_NBODY_SRC *.cpp)

include_directories( ${GTEST_INCLUDE_DIR} ${GMOCK_INCLUDE_DIR})
add_executable(nbody-test ${TEST_NBODY_SRC})
add_dependencies(nbody-test gtest gmock glew)
target_link_libraries(nbody-test libgpunbody ${GTEST_LIBRARY} ${GMOCK_LIBRARY} ${GLEW_LIBRARIES})

install(TARGETS nbody-test DESTINATION bin)
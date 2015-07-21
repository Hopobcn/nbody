file(GLOB TEST_NBODY_SRC *.cpp)

include_directories( ${GTEST_INCLUDE_DIRS} ${GMOCK_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} )
add_executable(nbody-test ${TEST_NBODY_SRC})
add_dependencies(nbody-test gtest gmock)
target_link_libraries(nbody-test libgpunbody libgtest libgmock libglew)

install(TARGETS nbody-test DESTINATION bin)
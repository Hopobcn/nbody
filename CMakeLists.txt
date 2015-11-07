cmake_minimum_required(VERSION 3.0)
project(nbody)

# We have to tell CMAKE to include our own FindGTest script
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# specify custom cmake find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )

find_package(Threads REQUIRED)
find_package(OpenMP)

# Specify CUDA_TOOLKIT_ROOT_DIR if cuda isn't installed in a common place
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-7.5")
find_package(CUDA    REQUIRED)
if (CUDA_FOUND)
    add_definitions("-DUSE_CUDA")
    include_directories(${CUDA_INCLUDE_DIRS})

    set(CUDA_HOST_COMPILER "/usr/bin/g++")

    if (CUDA_VERSION_MAJOR LESS "7")
        message(FATAL_ERROR "NVIDIA CUDA Compiler 7.0 or newer is required")
    endif(CUDA_VERSION_MAJOR LESS "7")
else(CUDA_FOUND)
    message(FATAL_ERROR "NVIDIA CUDA Compiler is required")
endif(CUDA_FOUND)

find_package(OpenGL  REQUIRED)
if (OPENGL_FOUND)
    message(STATUS "OpenGL FOUND")
    message(STATUS "OPENGL_INCLUDE_DIR   ${OPENGL_INCLUDE_DIR}")
    message(STATUS "OPENGL_LIBRARIES     ${OPENGL_LIBRARIES}")

    include_directories(${OPENGL_INCLUDE_DIR})
else(OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(OPENGL_FOUND)

find_package(GLUT REQUIRED)
if (GLUT_FOUND)
    message(STATUS "OpenGL Utility Toolkit FOUND")
    message(STATUS "GLUT_INCLUDE_DIR     ${GLUT_INCLUDE_DIR}")
    message(STATUS "GLUT_LIBRARIES       ${GLUT_LIBRARIES}")

    include_directories(${GLUT_INCLUDE_DIR})
else(GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(GLUT_FOUND)

find_package(GLEW REQUIRED)
if (GLEW_FOUND)
    message(STATUS "OpenGL Extension Wrangler Library FOUND")
    message(STATUS "GLEW_INCLUDE_DIRS    ${GLEW_INCLUDE_DIRS}")
    message(STATUS "GLEW_LIBRARY_PATH    ${GLEW_LIBRARY_PATH}")
    message(STATUS "GLEW_LIBRARIES       ${GLEW_LIBRARIES}")

    include_directories(${GLEW_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Google Testing Framework NOT FOUND")
endif()

find_package(GTest REQUIRED)
if (GTEST_FOUND)
    message(STATUS "Google Testing Framework FOUND")
    message(STATUS "GTEST_INCLUDE_DIR    ${GTEST_INCLUDE_DIR}")
    message(STATUS "GTEST_LIBRARY_PATH   ${GTEST_LIBRARY_PATH}")
    message(STATUS "GTEST_LIBRARY        ${GTEST_LIBRARY}")
    message(STATUS "GTEST_MAIN_IBRARY    ${GTEST_MAIN_LIBRARY}")
    message(STATUS "GTEST_BOTH_LIBRARIES ${GTEST_BOTH_LIBRARIES}")
else()
    message(FATAL_ERROR "Google Testing Framework NOT FOUND")
endif()

find_package(GMock REQUIRED)
if (GMOCK_FOUND)
    message(STATUS "Google Mocking Framework FOUND")
    message(STATUS "GMOCK_INCLUDE_DIR    ${GMOCK_INCLUDE_DIR}")
    message(STATUS "GMOCK_LIBRARY_PATH   ${GMOCK_LIBRARY_PATH}")
    message(STATUS "GMOCK_LIBRARY        ${GMOCK_LIBRARY}")
else()
    message(FATAL_ERROR "Google Mocking Framework NOT FOUND")
endif()

include(ExternalProject REQUIRED)

if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
    ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Wformat -Wformat-security")

    include_directories(${CMAKE_SOURCE_DIR}/include)

    if (OPENMP_FOUND)
        add_definitions("-DOPENMP")

        set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
    endif()

    set(CMAKE_CXX_FLAGS_DEBUG   "${COMMON_CXX_FLAGS} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_CXX_FLAGS} -g -O3 -march=native")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG    "${CMAKE_EXE_LINKER_FLAGS_DEBUG}   -g -O0")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE  "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -g -O3")
endif(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
      ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))

add_subdirectory(src)
add_subdirectory(main)
add_subdirectory(test)

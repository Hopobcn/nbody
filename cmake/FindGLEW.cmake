include(ExternalProject)

set_directory_properties(PROPERTIRES EP_PREFIX ${CMAKE_BINARY_DIR}/ext/glew)

ExternalProject_Add(
        openglew
        #--Download step----
        DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
        URL https://sourceforge.net/projects/glew/files/glew/1.12.0/glew-1.12.0.zip
        #URL_MD5 ${glew_md5} TODO
        #--Update/Patch step---
        #--Configure step---
        CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        #--Build step---
        #--Install step--- (DISABLED)
        INSTALL_COMMAND ""
        #--Test step--
        #--Output logging--
        #--Custom targets--
)

# Specify include dir
ExternalProject_Get_Property(openglew SOURCE_DIR)
set(GLEW_INCLUDE_DIRS ${SOURCE_DIR}/include)

# Library
ExternalProject_Get_Property(openglew BINARY_DIR)
set(GLEW_LIBRARY_PATH ${BINARY_DIR})
set(GLEW_LIB ${BINARY_DIR}/lib/libGLEW.a)
set(GLEW_LIBRARIES glew)
add_library(${GLEW_LIBRARIES} STATIC IMPORTED GLOBAL)
link_libraries(${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
set_property(TARGET ${GLEW_LIBRARIES} PROPERTY IMPORTED_LOCATION ${GLEW_LIB})
add_dependencies(${GLEW_LIBRARIES} openglew)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLEW
        FOUND_VAR GLEW_FOUND
        REQUIRED_VARS GLEW_LIBRARIES GLEW_INCLUDE_DIRS
        VERSION_VAR GLEW_VERSION
        FAIL_MESSAGE "GLEW NOT FOUND")

ExternalProject_Add(
    glew
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}
    #--Download step-------------
    DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
    URL https://sourceforge.net/projects/glew/files/glew/1.12.0/glew-1.12.0.zip
    #URL_MD5 ${glew_md5}
    #--Update/Patch step----------
    #--Configure step-------------
    CMAKE_ARGS -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
               -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    #--Build step-----------------
    #--Install step--------------- (DISABLED)
    INSTALL_COMMAND ""
    #--Test step------------------
    #--Output logging-------------
    #--Custom targets-------------
)

# Create a libglew target to be used as a dependency by test programs
add_library(libglew IMPORTED STATIC GLOBAL)
add_dependencies(libglew glew)

# Set gmock properties
ExternalProject_Get_Property(glew SOURCE_DIR BINARY_DIR)
set_target_properties(libglew PROPERTIES
    "IMPORTED_LOCATION"                 "${BINARY_DIR}/lib/libGLEW.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
#    "INTERFACE_INCLUDE_DIRECTORIES"    "${SOURCE_DIR}/include"
)
# I couldn't make it work with INTERFACE_INCLUDE_DIRECTORIES
include_directories("${SOURCE_DIR}/include")

set(GLEW_INCLUDE_DIRS ${SOURCE_DIR}/include PARENT_SCOPE)
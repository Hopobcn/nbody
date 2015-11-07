include(ExternalProject)

set_directory_properties(PROPERTIRES EP_PREFIX ${CMAKE_BINARY_DIR}/ext/gmock)

# Download and install GoogleMock
ExternalProject_Add(
        googlemock
        #--Download step-------------
        URL https://googlemock.googlecode.com/files/gmock-1.7.0.zip
        #URL_HASH SHA1=TODO
        #--Update/Patch step----------
        #--Configure step-------------
        CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DBUILD_SHARED_LIBS=OFF
                   -Dgmock_build_tests=OFF
        #--Build step-----------------
        #--Install step--------------- (DISABLED)
        INSTALL_COMMAND ""
        #--Test step------------------
        #--Output logging-------------
        #--Custom targets-------------
)

# Specify include dir
ExternalProject_Get_Property(googlemock SOURCE_DIR)
set(GMOCK_INCLUDE_DIR ${SOURCE_DIR}/include)

# Library
ExternalProject_Get_Property(googlemock BINARY_DIR)
set(GMOCK_LIBRARY_PATH ${BINARY_DIR})
set(GMOCK_LIB ${BINARY_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gmock.a)
set(GMOCK_LIBRARY gmock)
add_library(${GMOCK_LIBRARY} STATIC IMPORTED GLOBAL)
set_property(TARGET ${GMOCK_LIBRARY} PROPERTY IMPORTED_LOCATION ${GMOCK_LIB})
add_dependencies(${GMOCK_LIBRARY} googlemock)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMock
        FOUND_VAR GMOCK_FOUND
        REQUIRED_VARS GMOCK_LIBRARY GMOCK_INCLUDE_DIR
        VERSION_VAR GMOCK_VERSION
        FAIL_MESSAGE "GMock NOT FOUND")
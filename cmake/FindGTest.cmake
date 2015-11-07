include(ExternalProject)

set_directory_properties(PROPERTIRES EP_PREFIX ${CMAKE_BINARY_DIR}/ext/gtest)

ExternalProject_Add(
        googletest
        #--Download step----
        URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
        URL_HASH SHA1=f85f6d2481e2c6c4a18539e391aa4ea8ab0394af
        #--Update/Patch step---
        #--Configure step---
        CMAKE_ARGS -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -Dgtest_force_shared_crt=OFF
                   -Dgtest_build_samples=OFF
                   -Dgtest_build_tests=OFF
                   -Dgtest_disable_pthreads=OFF
        #--Build step---
        #--Install step--- (DISABLED)
        INSTALL_COMMAND ""
        #--Test step--
        #--Output logging--
        #--Custom targets--
)

# Specify include dir
ExternalProject_Get_Property(googletest SOURCE_DIR)
set(GTEST_INCLUDE_DIR ${SOURCE_DIR}/include)

# Library
ExternalProject_Get_Property(googletest BINARY_DIR)
set(GTEST_LIBRARY_PATH ${BINARY_DIR})
set(GTEST_LIB ${BINARY_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_MAIN_LIB ${BINARY_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main.a)
set(GTEST_LIBRARY gtest)
set(GTEST_MAIN_LIBRARY gtest-main)
set(GTEST_BOTH_LIBRARIES gtest-both)
add_library(${GTEST_LIBRARY} STATIC IMPORTED GLOBAL)
add_library(${GTEST_MAIN_LIBRARY} STATIC IMPORTED GLOBAL)
add_library(${GTEST_BOTH_LIBRARIES} STATIC IMPORTED GLOBAL)
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION ${GTEST_LIB})
set_property(TARGET ${GTEST_MAIN_LIBRARY} PROPERTY IMPORTED_LOCATION ${GTEST_MAIN_LIB})
set_property(TARGET ${GTEST_BOTH_LIBRARIES} PROPERTY IMPORTED_LOCATION ${GTEST_LIB} ${GTEST_MAIN_LIB})
add_dependencies(${GTEST_LIBRARY} googletest)
add_dependencies(${GTEST_MAIN_LIBRARY} googletest)

include(FindPackageHandleStandardArgs)
# Mode 1 of specifiying that GTest is found if ${GTEST_LIBRARY} and ${GTEST_INLCUDE_DIR} are valid
#find_package_handle_standard_args(GTest DEFAULT_MSG GTEST_LIBRARY GTEST_INCLUDE_DIR)
# Mode 2 of sepcifiyin that GTest is found:
find_package_handle_standard_args(GTest
        FOUND_VAR GTEST_FOUND
        REQUIRED_VARS GTEST_LIBRARY GTEST_MAIN_LIBRARY GTEST_BOTH_LIBRARIES GTEST_INCLUDE_DIR
        VERSION_VAR GTEST_VERSION
        FAIL_MESSAGE "GTest NOT FOUND")
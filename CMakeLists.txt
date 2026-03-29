cmake_minimum_required(VERSION 3.18...3.27)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DEEPTIME_VERSION "0.0.0" CACHE STRING "The version without commit offset. Defaults to 0.0.0")
set(DEEPTIME_VERSION_INFO "0.0.0" CACHE STRING "The full version. Defaults to 0.0.0")
set(DEEPTIME_BUILD_CPP_TESTS OFF CACHE BOOL "Whether to build the c++ unit tests")

if(DEFINED PROJECT_NAME)
    set(DEEPTIME_IS_SUBPROJECT ON)
endif()
project(deeptime LANGUAGES C CXX VERSION ${DEEPTIME_VERSION})
set(DEEPTIME_ROOT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")

message(STATUS "Got deeptime version ${DEEPTIME_VERSION}")
message(STATUS "Got deeptime full version ${DEEPTIME_VERSION_INFO}")
message(STATUS "Building tests: ${DEEPTIME_BUILD_CPP_TESTS}")

include(GNUInstallDirs)

find_package(OpenMP 4)
if(OpenMP_FOUND)
    add_definitions(-DUSE_OPENMP)
endif()

if(MSVC)
    add_compile_options(/W3 /EHsc /bigobj /permissive- /std:c++17)
endif()

if(DEEPTIME_BUILD_CPP_TESTS)
    # Tests link pybind11::embed for Python C API symbols used in pybind11 headers
    find_package(Python COMPONENTS Interpreter Development.Module Development.Embed NumPy REQUIRED)
else()
    find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
endif()
find_package(pybind11 CONFIG REQUIRED)

function(register_pybind_module name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "")
    pybind11_add_module(${name} ${ARG_UNPARSED_ARGUMENTS})
    target_link_libraries(${name} PUBLIC deeptime::deeptime)
    if (OpenMP_FOUND)
        target_link_libraries(${name} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/src/deeptime" "${CMAKE_CURRENT_LIST_DIR}")
    install(TARGETS ${name} DESTINATION deeptime/${REL_PATH})
endfunction()

add_subdirectory(src/deeptime/src)
add_subdirectory(src/deeptime/numeric)
add_subdirectory(src/deeptime/data)
add_subdirectory(src/deeptime/covariance/util/covar_c)
add_subdirectory(src/deeptime/clustering)
add_subdirectory(src/deeptime/basis)
add_subdirectory(src/deeptime/markov)
add_subdirectory(src/deeptime/plots)

if(DEEPTIME_BUILD_CPP_TESTS)
    add_subdirectory(tests)
endif()

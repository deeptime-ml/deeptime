cmake_minimum_required(VERSION 3.15...3.19)

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

if(SKBUILD)
    # Scikit-Build does not add your site-packages to the search path
    # automatically, so we need to add it _or_ the pybind11 specific directory
    # here.
    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE _tmp_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
    list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

find_package(OpenMP 4)
if(OpenMP_FOUND)
    add_definitions(-DUSE_OPENMP)
endif()

if(MSVC)
    add_compile_options(/W3 /EHsc /bigobj /permissive- /std:c++17)
endif()

# use cmake 3.12+ FindPython
# find_package(Python COMPONENTS Interpreter Development)  # this doesn't work with conda, picks up the wrong python interpreter on OSX
# now find pybind
find_package(pybind11 REQUIRED CONFIG)

function(register_pybind_module name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "")
    pybind11_add_module(${name} ${ARG_UNPARSED_ARGUMENTS})
    target_link_libraries(${name} PUBLIC deeptime::deeptime)
    if (OpenMP_FOUND)
        target_link_libraries(${name} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/deeptime" "${CMAKE_CURRENT_LIST_DIR}")
    install(TARGETS ${name} DESTINATION ${REL_PATH})
    if(NOT SKBUILD)
        set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    endif()
endfunction()

add_subdirectory(deeptime/src)
add_subdirectory(deeptime/numeric)
add_subdirectory(deeptime/data)
add_subdirectory(deeptime/covariance/util/covar_c)
add_subdirectory(deeptime/clustering)
add_subdirectory(deeptime/basis)
add_subdirectory(deeptime/markov)

if (NOT SKBUILD)
    add_subdirectory(examples/clustering_custom_metric)
endif()

if(DEEPTIME_BUILD_CPP_TESTS)
    add_subdirectory(tests)
endif()

cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)

set(DEEPTIME_BUILD_CPP_TESTS OFF CACHE BOOL "Whether to build the c++ unit tests")

if(DEFINED PROJECT_NAME)
    set(DEEPTIME_IS_SUBPROJECT ON)
endif()
project(deeptime LANGUAGES C CXX VERSION 0.0.0)

find_package(OpenMP)
if(OpenMP_FOUND)
    add_definitions(-DUSE_OPENMP)
endif()

if(MSVC)
    add_compile_options(/W3 /EHsc /std:c++17)
else()
    #add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

find_package(pybind11 REQUIRED)

add_subdirectory(deeptime/src)
set(common_includes "${CMAKE_CURRENT_LIST_DIR}/deeptime/src/include")
set(numeric_includes "${CMAKE_CURRENT_LIST_DIR}/deeptime/numeric/_bindings/include")

add_subdirectory(deeptime/numeric/_bindings)
add_subdirectory(deeptime/data)
add_subdirectory(deeptime/covariance/util/covar_c)
add_subdirectory(deeptime/clustering)
add_subdirectory(deeptime/basis)
add_subdirectory(deeptime/markov/_bindings)
add_subdirectory(deeptime/markov/hmm/_bindings)
add_subdirectory(deeptime/markov/tools/estimation/dense/_bindings)
add_subdirectory(deeptime/markov/tools/estimation/sparse/_bindings)
add_subdirectory(deeptime/markov/tools/estimation/sparse/mle/newton)

if(DEEPTIME_BUILD_CPP_TESTS)
    add_subdirectory(tests)
endif()

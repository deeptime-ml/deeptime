cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED PROJECT_NAME)
    set(DEEPTIME_IS_SUBPROJECT ON)
endif()
project(deeptime LANGUAGES C CXX VERSION 0.0.0)

find_package(OpenMP REQUIRED)
add_definitions(-DUSE_OPENMP)

if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

include("${PYBIND_CMAKE_DIR}/pybind11Config.cmake")

set(common_includes "${CMAKE_CURRENT_LIST_DIR}/deeptime/src/include")

add_subdirectory(deeptime/data)
add_subdirectory(deeptime/covariance/util/covar_c)
add_subdirectory(deeptime/clustering)
add_subdirectory(deeptime/basis)
add_subdirectory(deeptime/markov/_bindings)
add_subdirectory(deeptime/markov/hmm/_bindings)
add_subdirectory(deeptime/markov/tools/kahandot)
add_subdirectory(deeptime/markov/tools/estimation/dense/_bindings)
add_subdirectory(deeptime/markov/tools/estimation/sparse/_bindings)
add_subdirectory(deeptime/markov/tools/estimation/sparse/mle/newton)

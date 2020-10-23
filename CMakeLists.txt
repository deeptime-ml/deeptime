cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED PROJECT_NAME)
    set(DEEPTIME_IS_SUBPROJECT ON)
endif()
project(deeptime LANGUAGES C CXX VERSION 0.0.0)

find_package(OpenMP REQUIRED)
add_definitions(-DUSE_OPENMP)

if (NOT DEEPTIME_IS_SUBPROJECT)
    option(DEEPTIME_BUILD_TESTING "Build deeptime tests" ON)
endif()

if(DEEPTIME_BUILD_TESTING)
    if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/lib/Catch2/CMakeLists.txt)
        #add_subdirectory(lib/Catch2)
        #include(${CMAKE_CURRENT_LIST_DIR}/lib/Catch2/contrib/Catch.cmake)
    elseif (NOT TARGET Catch2::Catch2)
        find_package(Catch2 QUIET)
        if (CATCH2_FOUND)
            include(Catch)
        else()
            # message(SEND_ERROR "Catch2 not found.")
        endif()
    endif ()
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/ext/pybind11)

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

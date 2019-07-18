cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED PROJECT_NAME)
    set(SKTIME_IS_SUBPROJECT ON)
endif()
project(sktime LANGUAGES C CXX VERSION 0.0.0)

if (NOT SKTIME_IS_SUBPROJECT)
    option(SKTIME_BUILD_TESTING "Build sktime tests" ON)
endif()

if(SKTIME_BUILD_TESTING)
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

add_subdirectory(${CMAKE_SOURCE_DIR}/lib/pybind11)

add_subdirectory(sktime/covariance/util/covar_c)
add_subdirectory(sktime/clustering)
add_subdirectory(sktime/markovprocess)

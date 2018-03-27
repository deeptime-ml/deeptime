cmake_minimum_required (VERSION 2.8.12)
project (covartools)


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../../../pybind11
                 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)
pybind11_add_module(_covartools covartools.cpp)

include_directories(.)

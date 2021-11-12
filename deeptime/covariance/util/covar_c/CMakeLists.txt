project(covartools CXX)

set(SRC covartools.hpp covartools.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include ${common_includes})

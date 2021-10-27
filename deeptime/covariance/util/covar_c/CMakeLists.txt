project(covartools CXX)

set(SRC covartools.hpp covartools.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})

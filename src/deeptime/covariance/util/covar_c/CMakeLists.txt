project(_covartools CXX)

set(SRC covartools.hpp covartools.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

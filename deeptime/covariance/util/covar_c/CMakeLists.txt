project(covartools CXX)

set(SRC covartools.hpp covartools.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC deeptime::deeptime)

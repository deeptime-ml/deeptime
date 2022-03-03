project(_basis_bindings CXX)

set(SRC src/basis_bindings.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

project(_numeric_bindings CXX)

set(SRC src/numeric_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

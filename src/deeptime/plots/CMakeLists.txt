project(_plots_bindings CXX)

set(SRC src/plots_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

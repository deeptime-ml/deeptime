project(_mle_bindings CXX)

register_pybind_module(${PROJECT_NAME} src/mle_module.cpp)

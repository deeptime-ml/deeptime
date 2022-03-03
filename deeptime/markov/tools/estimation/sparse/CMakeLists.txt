project(_mle_sparse_bindings CXX)

register_pybind_module(${PROJECT_NAME} src/mle_sparse_module.cpp)

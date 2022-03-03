project(_hmm_bindings CXX)

set(SRC src/hmm_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

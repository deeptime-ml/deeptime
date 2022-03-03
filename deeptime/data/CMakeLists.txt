project(_data_bindings CXX)

set(SRC src/data_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

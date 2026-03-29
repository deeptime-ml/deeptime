project(_tram_bindings)

set(SRC _bindings/src/tram_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

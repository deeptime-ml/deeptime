project(_clustering_bindings CXX)

set(SRC src/clustering_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

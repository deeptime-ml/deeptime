project(objective_sparse_ops CXX)

set(SRC objective_sparse_ops.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC deeptime::common OpenMP::OpenMP_CXX)

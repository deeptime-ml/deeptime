project(_clustering_bindings CXX)

set(SRC src/clustering_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

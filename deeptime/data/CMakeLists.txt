project(deeptime_data CXX)

set(SRC src/data_module.cpp)
pybind11_add_module(_data_bindings ${SRC})
target_link_libraries(_data_bindings PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(_data_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

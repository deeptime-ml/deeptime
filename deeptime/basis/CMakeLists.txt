project(deeptime_basis CXX)

set(SRC src/basis_bindings.cpp)
pybind11_add_module(_basis_bindings ${SRC})
target_link_libraries(_basis_bindings PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(_basis_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

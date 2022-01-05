project(deeptime_msm_mle_sparse CXX)

pybind11_add_module(_mle_sparse_bindings src/mle_sparse_module.cpp)
target_link_libraries(_mle_sparse_bindings PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(_mle_sparse_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

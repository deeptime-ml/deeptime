project(deeptime_msm_mle CXX)

pybind11_add_module(_mle_bindings src/mle_module.cpp)
target_link_libraries(_mle_bindings PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(_mle_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

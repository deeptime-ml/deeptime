project(deeptime_hmm CXX)

set(SRC src/hmm_module.cpp)
pybind11_add_module(_hmm_bindings ${SRC})
target_link_libraries(_hmm_bindings PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(_hmm_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

project(deeptime_markov CXX)

set(SRC src/markov_module.cpp)
pybind11_add_module(_markov_bindings ${SRC})
target_link_libraries(_markov_bindings PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(_markov_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

project(_markov_bindings CXX)

add_subdirectory(hmm)
add_subdirectory(msm/tram)
add_subdirectory(tools/estimation/dense)
add_subdirectory(tools/estimation/sparse)

set(SRC src/markov_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

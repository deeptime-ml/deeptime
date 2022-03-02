project(_markov_bindings CXX)

add_subdirectory(hmm)
add_subdirectory(msm/tram)
add_subdirectory(tools/estimation/dense)
add_subdirectory(tools/estimation/sparse)

set(SRC src/markov_module.cpp)
pybind11_add_module(_markov_bindings ${SRC})
target_link_libraries(_markov_bindings PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(_markov_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()
file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/deeptime" "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS ${PROJECT_NAME} DESTINATION ${REL_PATH})


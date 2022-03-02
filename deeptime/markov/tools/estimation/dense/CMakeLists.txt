project(_mle_bindings CXX)

pybind11_add_module(_mle_bindings src/mle_module.cpp)
target_link_libraries(_mle_bindings PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(_mle_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()
file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/deeptime" "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS ${PROJECT_NAME} DESTINATION ${REL_PATH})

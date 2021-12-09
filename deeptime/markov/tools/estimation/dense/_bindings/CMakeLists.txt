project(deeptime_msm_mle CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
target_link_libraries(${PROJECT_NAME} INTERFACE deeptime::common)
add_library(deeptime::msm_mle ALIAS ${PROJECT_NAME})

pybind11_add_module(_mle_bindings src/mle_module.cpp)
target_link_libraries(_mle_bindings PUBLIC deeptime::msm_mle)
if(OpenMP_FOUND)
    target_link_libraries(_mle_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

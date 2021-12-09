project(deeptime_hmm CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
target_link_libraries(${PROJECT_NAME} INTERFACE deeptime::common)
add_library(deeptime::hmm ALIAS ${PROJECT_NAME})

set(SRC src/hmm_module.cpp)
pybind11_add_module(_hmm_bindings ${SRC})
target_link_libraries(_hmm_bindings PUBLIC deeptime::hmm)
if(OpenMP_FOUND)
    target_link_libraries(_hmm_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

project(deeptime_markov CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
target_link_libraries(${PROJECT_NAME} INTERFACE deeptime::common)
add_library(deeptime::markov ALIAS ${PROJECT_NAME})

set(SRC src/markov_module.cpp)
pybind11_add_module(_markov_bindings ${SRC})
target_link_libraries(_markov_bindings PUBLIC deeptime::markov)
if(OpenMP_FOUND)
    target_link_libraries(_markov_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

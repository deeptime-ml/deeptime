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

add_library(deeptime_tram INTERFACE)
target_include_directories(deeptime_tram INTERFACE include)
target_link_libraries(deeptime_tram INTERFACE deeptime::numeric deeptime::common)
add_library(deeptime::tram ALIAS deeptime_tram)

pybind11_add_module(_tram_bindings src/tram_module.cpp)
target_link_libraries(_tram_bindings PUBLIC deeptime::tram)
if(OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

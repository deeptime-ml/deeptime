project(_markov_bindings CXX)

set(SRC src/markov_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include ${common_includes})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

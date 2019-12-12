project(_bhmm_hidden_bindings CXX)

set(SRC src/bhmm_hidden_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

project(_bhmm_output_models CXX)

set(SRC src/bhmm_output_models_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})

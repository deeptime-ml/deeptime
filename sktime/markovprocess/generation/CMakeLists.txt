project(_markovprocess_generation_bindings CXX)

pybind11_add_module(${PROJECT_NAME} bindings.cpp)

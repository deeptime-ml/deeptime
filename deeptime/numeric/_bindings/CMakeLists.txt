project(_numeric_bindings CXX)

set(SRC src/numeric_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC include ${common_includes})
if (DEEPTIME_USE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

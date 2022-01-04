project(_numeric_bindings CXX)

add_library(deeptime_numeric INTERFACE)
target_include_directories(deeptime_numeric INTERFACE include)
target_link_libraries(deeptime_numeric INTERFACE deeptime::common)
add_library(deeptime::numeric ALIAS deeptime_numeric)

set(SRC src/numeric_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC include deeptime::numeric)
if (DEEPTIME_USE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

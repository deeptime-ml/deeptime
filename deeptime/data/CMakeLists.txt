project(deeptime_data CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
target_link_libraries(${PROJECT_NAME} INTERFACE deeptime::common)
add_library(deeptime::data ALIAS ${PROJECT_NAME})

set(SRC src/data_module.cpp)
pybind11_add_module(_data_bindings ${SRC})
target_link_libraries(_data_bindings PUBLIC deeptime::data)
if(OpenMP_FOUND)
    target_link_libraries(_data_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

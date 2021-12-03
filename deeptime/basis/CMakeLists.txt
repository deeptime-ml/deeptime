project(deeptime_basis CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
target_link_libraries(${PROJECT_NAME} INTERFACE deeptime::common)
add_library(deeptime::basis ALIAS ${PROJECT_NAME})

set(SRC src/basis_bindings.cpp)
pybind11_add_module(_basis_bindings ${SRC})
target_link_libraries(_basis_bindings PUBLIC deeptime::basis)
if(OpenMP_FOUND)
    target_link_libraries(_basis_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()

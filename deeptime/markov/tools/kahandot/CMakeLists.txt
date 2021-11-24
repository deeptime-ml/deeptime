project(kahandot CXX)

set(SRC kahandot_module.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PUBLIC)
target_link_libraries(${PROJECT_NAME} PUBLIC deeptime::common OpenMP::OpenMP_CXX)

project(sktime_data C CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

set(SKTIME_DATA_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/include/sktime-data/sktime-data.h)

target_sources(${PROJECT_NAME} INTERFACE ${SKTIME_DATA_SOURCES})

pybind11_add_module("${PROJECT_NAME}_bindings" ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries("${PROJECT_NAME}_bindings" PUBLIC ${PROJECT_NAME})

if(SKTIME_BUILD_TESTING)
    add_subdirectory(test)
endif()

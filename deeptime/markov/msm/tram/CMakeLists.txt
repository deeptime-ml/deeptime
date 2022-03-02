project(_tram_bindings)

pybind11_add_module(${PROJECT_NAME} _bindings/src/tram_module.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC deeptime::deeptime)
if(OpenMP_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/deeptime" "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS ${PROJECT_NAME} DESTINATION ${REL_PATH})

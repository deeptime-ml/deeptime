project(_covartools CXX)

set(SRC covartools.hpp covartools.cpp)
pybind11_add_module(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC deeptime::deeptime)
file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/deeptime" "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS ${PROJECT_NAME} DESTINATION ${REL_PATH})

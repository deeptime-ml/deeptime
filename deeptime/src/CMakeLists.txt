project(deeptime_common)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
add_library(deeptime::common ALIAS ${PROJECT_NAME})

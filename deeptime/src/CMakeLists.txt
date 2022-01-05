project(deeptime)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include)
add_library(deeptime::deeptime ALIAS ${PROJECT_NAME})

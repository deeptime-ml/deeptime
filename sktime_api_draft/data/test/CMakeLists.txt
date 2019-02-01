add_executable ("${PROJECT_NAME}_test" main.cpp)
target_link_libraries("${PROJECT_NAME}_test" sktime_data Catch2::Catch2)
catch_discover_tests("${PROJECT_NAME}_test")

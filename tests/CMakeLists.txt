project(deeptime_tests)

include(${CMAKE_CURRENT_LIST_DIR}/../devtools/cmake/Conan.cmake)
run_conan()
find_package(Catch2 REQUIRED)

set(TEST_SRC src/test_common_utilities.cpp)
pybind11_add_module(testing_module SHARED ${TEST_SRC})
target_link_libraries(testing_module PRIVATE deeptime::common)

add_executable(tests catch_main.cpp ${TEST_SRC})
target_link_libraries(tests PRIVATE Catch2::Catch2 deeptime::common pybind11::embed)

add_custom_target(
        run_tests
        COMMAND tests --durations=yes -r console
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS tests
)

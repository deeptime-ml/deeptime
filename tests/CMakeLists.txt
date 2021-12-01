project(deeptime_tests)

include(${CMAKE_SOURCE_DIR}/devtools/cmake/Conan.cmake)
run_conan()
include(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

set(TEST_SRC src/test_common_utilities.cpp src/test_swap_arr.cpp src/test_kahan_summation.cpp src/test_TRAMInput.cpp)
add_executable(tests catch_main.cpp ${TEST_SRC})
target_link_libraries(tests PUBLIC ${CONAN_LIBS} CONAN_PKG::catch2 deeptime::tram pybind11::embed)

add_custom_target(
        run_tests
        COMMAND tests --durations=yes -r console
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS tests
)

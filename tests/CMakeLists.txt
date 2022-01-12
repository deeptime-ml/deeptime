project(deeptime_tests)

include(${CMAKE_SOURCE_DIR}/devtools/cmake/Conan.cmake)
run_conan()
include(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

set(TEST_SRC src/test_common_utilities.cpp src/test_swap_arr.cpp
        src/test_kahan_summation.cpp src/test_traminput.cpp src/test_tram.cpp)
add_executable(tests catch_main.cpp ${TEST_SRC})
set_target_properties(tests PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_link_libraries(tests PUBLIC ${CONAN_LIBS} CONAN_PKG::catch2 deeptime::deeptime pybind11::embed)
if(OpenMP_FOUND)
    target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
endif()

add_custom_target(
        run_tests
        COMMAND tests --durations=yes -r console
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS tests
)

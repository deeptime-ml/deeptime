project(deeptime_tests)

Include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(Catch2)

set(TEST_SRC src/test_common_utilities.cpp src/test_swap_arr.cpp
        src/test_kahan_summation.cpp src/test_traminput.cpp src/test_tram.cpp)
add_executable(tests ${TEST_SRC})
set_target_properties(tests PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_link_libraries(tests PUBLIC ${CONAN_LIBS} Catch2::Catch2WithMain deeptime::deeptime pybind11::embed)
if(OpenMP_FOUND)
    target_link_libraries(tests PUBLIC OpenMP::OpenMP_CXX)
endif()

add_custom_target(
        run_tests
        COMMAND tests --durations=yes -r console
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS tests
)

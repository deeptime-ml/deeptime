project(_numeric_bindings CXX)

set(SRC src/numeric_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

# Cython extension: eig_qr (pre-cythonized from eig_qr.pyx)
Python_add_library(eig_qr MODULE eig_qr.c WITH_SOABI)
target_include_directories(eig_qr PRIVATE ${Python_NumPy_INCLUDE_DIRS})
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import scipy; print(scipy.get_include())"
    OUTPUT_VARIABLE SCIPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
target_include_directories(eig_qr PRIVATE ${SCIPY_INCLUDE_DIR})
file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/src/deeptime" "${CMAKE_CURRENT_LIST_DIR}")
install(TARGETS eig_qr DESTINATION deeptime/${REL_PATH})

project(_numeric_bindings CXX)

set(SRC src/numeric_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

# Cython extension: eig_qr (pre-cythonized from eig_qr.pyx)
# All scipy types are inlined in the generated C; no scipy headers needed at compile time.
# To regenerate: cython src/deeptime/numeric/eig_qr.pyx -o src/deeptime/numeric/eig_qr.c
Python_add_library(eig_qr MODULE eig_qr.c WITH_SOABI)
target_include_directories(eig_qr PRIVATE ${Python_NumPy_INCLUDE_DIRS})
file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/src/deeptime" "${CMAKE_CURRENT_LIST_DIR}")
install(TARGETS eig_qr DESTINATION deeptime/${REL_PATH})

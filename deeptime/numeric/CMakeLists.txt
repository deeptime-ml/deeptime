project(_numeric_bindings CXX)

set(SRC src/numeric_module.cpp)
register_pybind_module(${PROJECT_NAME} ${SRC})

find_package(Cython)
if(Cython_FOUND)
    find_package(PythonExtensions REQUIRED)
    add_cython_target(eig_qr eig_qr.pyx)
    add_library(eig_qr MODULE ${eig_qr})
    python_extension_module(eig_qr)
    file(RELATIVE_PATH REL_PATH "${DEEPTIME_ROOT_DIRECTORY}/deeptime" "${CMAKE_CURRENT_LIST_DIR}")
    install(TARGETS eig_qr DESTINATION ${REL_PATH})
endif()

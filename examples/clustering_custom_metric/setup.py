import setuptools
import pybind11
import deeptime

setuptools.setup(
    name='custom_metric',
    packages=setuptools.find_packages(),
    version='0.1',
    ext_modules=[
        setuptools.Extension(
            name='bindings',
            sources=['bindings.cpp'],
            language='c++',
            include_dirs=deeptime.capi_includes(inc_clustering=True) + [pybind11.get_include()],
            extra_compile_args=[
                '-std=c++17',  # c++17 standard
                '-fopenmp'  # OpenMP support, optional
            ],
            extra_link_args=['-lgomp']  # OpenMP support, optional
        )
    ]
)

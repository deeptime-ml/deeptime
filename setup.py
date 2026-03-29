# based on https://github.com/pybind/scikit_build_example/blob/master/setup.py

import os
import sys

from setuptools import find_namespace_packages
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

try:
    from skbuild import setup
except ImportError:
    print("Please update pip, you need pip 10 or greater", file=sys.stderr)
    raise

with open('pyproject.toml', 'rb') as f:
    pyproject = tomllib.load(f)


def load_long_description():
    with open(pyproject["project"]["readme"], mode='r', encoding="utf-8") as f:
        return f.read()


cmake_args = [
    "-DDEEPTIME_VERSION=0.0.0",
    "-DDEEPTIME_VERSION_INFO=0.0.0"
]

excludes = ("tests", "tests.*", "examples", "examples.*", "docs", "docs.*", "devtools", "devtools.*")

metadata = \
    dict(
        long_description=load_long_description(),
        long_description_content_type='text/markdown',
        zip_safe=False,
        packages=find_namespace_packages(where=".", exclude=excludes),
        package_dir={"deeptime": "deeptime"},
        cmake_install_dir="deeptime/",
        cmake_args=cmake_args,
        include_package_data=True,
        ext_modules=[],
    )

if __name__ == '__main__':
    setup(**metadata)

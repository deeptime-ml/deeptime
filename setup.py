# based on https://github.com/pybind/scikit_build_example/blob/master/setup.py

import os
import sys

from setuptools import find_namespace_packages
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

sys.path.insert(0, os.path.dirname(__file__))
import versioneer
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
    f"-DDEEPTIME_VERSION={versioneer.get_version().split('+')[0]}",
    f"-DDEEPTIME_VERSION_INFO={versioneer.get_version()}"
]

excludes = ("tests", "tests.*", "examples", "examples.*", "docs", "docs.*", "devtools", "devtools.*")

metadata = \
    dict(
        long_description=load_long_description(),
        long_description_content_type='text/markdown',
        zip_safe=False,
        packages=find_namespace_packages(where=".", exclude=excludes),
        package_dir={"deeptime": "deeptime", "versioneer": "."},
        cmake_install_dir="deeptime/",
        cmake_args=cmake_args,
        include_package_data=True,
        ext_modules=[],
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass()
    )

if __name__ == '__main__':
    setup(**metadata)

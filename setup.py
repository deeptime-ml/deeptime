# based on https://github.com/pybind/scikit_build_example/blob/master/setup.py

import os
import sys

from setuptools import find_packages
import toml

sys.path.insert(0, os.path.dirname(__file__))
import versioneer
try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

pyproject = toml.load("pyproject.toml")


def load_long_description():
    with open(pyproject["project"]["readme"], mode='r', encoding="utf-8") as f:
        return f.read()


cmake_args = [
    f"-DDEEPTIME_VERSION={versioneer.get_version().split('+')[0]}",
    f"-DDEEPTIME_VERSION_INFO={versioneer.get_version()}"
]

metadata = \
    dict(
        name=pyproject["project"]["name"],
        version=versioneer.get_version(),
        author=pyproject["project"]["authors"][0]["name"],
        author_email=pyproject["project"]["authors"][0]["email"],
        url=pyproject["project"]["urls"]["repository"],
        description=pyproject["project"]["description"],
        long_description=load_long_description(),
        long_description_content_type='text/markdown',
        zip_safe=False,
        install_requires=pyproject["project"]["dependencies"],
        extras_require=pyproject["project"]["optional-dependencies"],
        packages=find_packages(where="."),
        package_dir={"deeptime": "deeptime", "versioneer": "."},
        cmake_install_dir="deeptime/",
        cmake_args=cmake_args,
        include_package_data=True,
        python_requires=pyproject["project"]["requires-python"],
        ext_modules=[],
        cmdclass=versioneer.get_cmdclass()
    )

if __name__ == '__main__':
    setup(**metadata)

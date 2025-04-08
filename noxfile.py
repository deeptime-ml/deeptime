import os
import shutil
import tempfile
from pathlib import Path
from sys import platform

import nox


def setup_environment(session: nox.Session):
    # remove when https://github.com/scikit-build/scikit-build/issues/740 is fixed
    session.env['SETUPTOOLS_ENABLE_FEATURES'] = "legacy-editable"


PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    setup_environment(session)
    if 'cpp' in session.posargs:
        session.install("cmake")
        session.install("scikit-build")

        cmake_module_path = session.run("python", "devtools/cmake/find_cmake_module_path.py", silent=True).strip()
        session.log(f"Found cmake module path {cmake_module_path}")

        site_packages_dir = session.run("python", "devtools/cmake/find_site_packages.py", silent=True).strip()
        session.log(f"Found site-packages {site_packages_dir}, adding to PYTHONPATH")
        session.env['PYTHONPATH'] = site_packages_dir

        pybind11_module_dir = session.run(*"python -m pybind11 --cmakedir".split(" "), silent=True).strip()
        session.log(f"Found pybind11 module dir: {pybind11_module_dir}")
        tmpdir = session.create_tmp()
        session.run("cmake", "-S", ".", "-B", tmpdir, '-DDEEPTIME_BUILD_CPP_TESTS=ON', 
                    '-DCMAKE_MODULE_PATH:PATH={}'.format(cmake_module_path),
                    "-Dpybind11_DIR={}".format(pybind11_module_dir), '-DCMAKE_BUILD_TYPE=Release', silent=True)
        if session.python != "3.13" and platform != "windows":
            session.run("cmake", "--build", tmpdir, "--target", "run_tests")
    else:
        pytest_args = []
        for arg in session.posargs:
            if arg.startswith('numprocesses'):
                n_processes = arg.split('=')[1]
                session.log(f"Running tests with n={n_processes} jobs.")
                pytest_args.append(f'--numprocesses={n_processes}')
        session.install("-e", ".[tests,plotting,units]", '-v', silent=False)

        if 'lldb_torch_setup' in session.posargs:
            session.run("lldb", "--batch", "-o", "run", "-o", "bt", "-o", "c", "--", "python", "-m", "pytest",
                        "tests/base/test_pytorch_setup.py")
            session.run("pytest", "tests/base/test_pytorch_setup.py")
        if 'cov' in session.posargs:
            session.log("Running with coverage")
            xml_results_dest = Path(os.getenv('SYSTEM_DEFAULTWORKINGDIRECTORY', tempfile.gettempdir()))
            assert xml_results_dest.exists() and xml_results_dest.is_dir(), 'no dest dir available'
            cover_pkg = 'deeptime'
            junit_xml = str((xml_results_dest / 'junit.xml').absolute())
            cov_xml = str((xml_results_dest / 'coverage.xml').absolute())

            pytest_args += [f'--cov={cover_pkg}', f"--cov-report=xml:{cov_xml}", f"--junit-xml={junit_xml}"]
                            # "--cov-config=pyproject.toml"]
        else:
            session.log("Running without coverage")

        test_dirs = [str((Path.cwd() / 'tests').absolute())]  # python tests
        if session.python != "3.8" and platform != 'darwin':
            test_dirs += [str((Path.cwd() / 'deeptime').absolute())]  # doctests

        with session.cd("tests"):
            session.run("python", "-m", "pytest", '-vv', '--doctest-modules', '--durations=20', *pytest_args,
                        '--pyargs', *test_dirs)


@nox.session(reuse_venv=True)
def make_docs(session: nox.Session) -> None:
    setup_environment(session)
    if not session.posargs or 'noinstall' not in session.posargs:
        session.install("-e", ".[tests,docs]", '-v', silent=False)
    session.chdir("docs")
    if session.posargs and 'clean' in session.posargs:
        session.log("First run clean")
        shutil.rmtree('source/api/generated')
        shutil.rmtree('source/examples')
        shutil.rmtree('source/datasets')
        session.run("sphinx-build", "-M", "clean", "source", "build")
    session.run("sphinx-build", "-M", "html", "source", "build", "-t", "notebooks")


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    setup_environment(session)
    session.install("build")
    session.log("Building normal files")
    session.run("python", "-m", "build")

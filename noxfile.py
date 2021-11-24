import os
import tempfile
import nox

PYTHON_VERSIONS = ["3.6", "3.7", "3.8", "3.9", "3.10"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("-r", "tests/requirements.txt")

    if session.posargs and 'cpp_tests' in session.posargs:
        session.log("Running C++ unit tests")
        tmpdir = session.create_tmp()
        session.install("cmake")
        session.install("conan")
        pybind11_module_dir = session.run(*"python -m pybind11 --cmakedir".split(" "), silent=True).strip()
        session.log(f"Found pybind11 module dir: {pybind11_module_dir}")
        session.run("cmake", "-S", ".", "-B", tmpdir, '-DDEEPTIME_BUILD_CPP_TESTS=ON',
                    f"-Dpybind11_DIR=\"{pybind11_module_dir}\"")
        session.run("cmake", "--build", tmpdir, "--config=Release", "--target", "run_tests")

    if session.posargs and 'cov' in session.posargs:
        session.log("Running with coverage")
        xml_results_dest = os.getenv('SYSTEM_DEFAULTWORKINGDIRECTORY', tempfile.gettempdir())
        assert os.path.isdir(xml_results_dest), 'no dest dir available'
        cover_pkg = 'deeptime'
        junit_xml = os.path.join(xml_results_dest, 'junit.xml')
        cov_xml = os.path.join(xml_results_dest, 'coverage.xml')

        cov_args = [f'--cov={cover_pkg}', f"--cov-report=xml:{cov_xml}", f"--junit-xml={junit_xml}",
                    "--cov-config=.coveragerc"]
    else:
        session.log("Running without coverage")
        cov_args = []

    test_dirs = ["tests/"]
    try:
        import torch
        # only run doctests if torch is available
        test_dirs.append('deeptime')
    except ImportError:
        pass

    session.install("-e", ".", '-v', silent=False)
    session.run("pytest", '-vv', '--doctest-modules', '--durations=20', *cov_args, '--pyargs', *test_dirs)


@nox.session(python=PYTHON_VERSIONS)
def cpp_tests(session: nox.Session) -> None:
    tmpdir = session.create_tmp()
    session.install("cmake")
    session.install("conan")
    session.install("-r", "tests/requirements.txt")
    session.install("-e", ".", '-v', silent=False)
    session.run("cmake", "-S", ".", "-B", tmpdir, '-DDEEPTIME_BUILD_CPP_TESTS=ON')
    session.run("cmake", "--build", tmpdir, "--config=Release", "--target", "run_tests")


@nox.session(reuse_venv=True)
def make_docs(session: nox.Session) -> None:
    session.install("-r", "tests/requirements.txt")
    session.install("-r", "docs/requirements.txt")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", "source", "build", "-t", "notebooks")


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    session.install("build")
    session.log("Building normal files")
    session.run("python", "-m", "build")

import os
import shutil
import tempfile
from pathlib import Path

import nox


def setup_environment(session: nox.Session):
    session.install("setuptools")

    import setuptools
    if setuptools.__version__ != 'unknown' and int(setuptools.__version__.split('.')[0]) >= 56:
        session.install("scikit-build")
        import skbuild
        skbuild_version = [int(x) for x in skbuild.__version__.split(".")]
        if skbuild_version[0] == 0 and skbuild_version[1] <= 15:
            # https://github.com/scikit-build/scikit-build/issues/740
            session.log("Enabling setuptools legacy features so that CMake may be invoked")
            session.env['SETUPTOOLS_ENABLE_FEATURES'] = "legacy-editable"
        else:
            session.log(f"Got setuptools version {setuptools.__version__} and skbuild version {skbuild_version}, "
                        f"no action required")
    else:
        session.log(f"Got setuptools version {setuptools.__version__}, no action required")


PYTHON_VERSIONS = ["3.8", "3.9", "3.10"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    setup_environment(session)
    if 'cpp' in session.posargs:
        session.install("cmake")
        session.install("conan")

        site_packages_dir = session.run("python", "devtools/cmake/find_site_packages.py", silent=True).strip()
        session.log(f"Found site-packages {site_packages_dir}, adding to PYTHONPATH")
        session.env['PYTHONPATH'] = site_packages_dir

        pybind11_module_dir = session.run(*"python -m pybind11 --cmakedir".split(" "), silent=True).strip()
        session.log(f"Found pybind11 module dir: {pybind11_module_dir}")
        tmpdir = session.create_tmp()
        session.run("cmake", "-S", ".", "-B", tmpdir, '-DDEEPTIME_BUILD_CPP_TESTS=ON',
                    "-Dpybind11_DIR={}".format(pybind11_module_dir), '-DCMAKE_BUILD_TYPE=Release', silent=True)
        session.run("cmake", "--build", tmpdir, "--target", "run_tests")
    else:
        pytest_args = []
        for arg in session.posargs:
            if arg.startswith('numprocesses'):
                n_processes = arg.split('=')[1]
                session.log(f"Running tests with n={n_processes} jobs.")
                pytest_args.append(f'--numprocesses={n_processes}')
        session.install("-e", ".", '-v', silent=False)
        session.install("-r", "tests/requirements.txt", silent=False)
        if 'cov' in session.posargs:
            session.log("Running with coverage")
            xml_results_dest = Path(os.getenv('SYSTEM_DEFAULTWORKINGDIRECTORY', tempfile.gettempdir()))
            assert xml_results_dest.exists() and xml_results_dest.is_dir(), 'no dest dir available'
            cover_pkg = 'deeptime'
            junit_xml = str((xml_results_dest / 'junit.xml').absolute())
            cov_xml = str((xml_results_dest / 'coverage.xml').absolute())

            pytest_args += [f'--cov={cover_pkg}', f"--cov-report=xml:{cov_xml}", f"--junit-xml={junit_xml}",
                            "--cov-config=.coveragerc"]
        else:
            session.log("Running without coverage")

        test_dirs = [str((Path.cwd() / 'tests').absolute())]  # python tests
        test_dirs += [str((Path.cwd() / 'deeptime').absolute())]  # doctests

        with session.cd("tests"):
            session.run("python", "-m", "pytest", '-vv', '--doctest-modules', '--durations=20', *pytest_args,
                        '--pyargs', *test_dirs)


@nox.session(reuse_venv=True)
def make_docs(session: nox.Session) -> None:
    setup_environment(session)
    if not session.posargs or 'noinstall' not in session.posargs:
        session.install("-e", ".", '-v', silent=False)
        session.install("-r", "tests/requirements.txt")
        session.install("-r", "docs/requirements.txt")
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

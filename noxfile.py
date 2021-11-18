import os
import shutil
import tempfile

import nox

PYTHON_VERSIONS = ["3.6", "3.7", "3.8", "3.9", "3.10"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("-r", "tests/requirements.txt")
    session.install("-e", ".", '-v', silent=False)
    if session.posargs and session.posargs[0] == 'cov':
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

    session.run("pytest", '-vv', '--doctest-modules', '--durations=20', *cov_args, '--pyargs', *test_dirs)


@nox.session(reuse_venv=True)
def make_docs(session: nox.Session) -> None:
    session.install("-r", "tests/requirements.txt")
    session.install("-r", "docs/requirements.txt")
    session.install("-e", ".", '-v', silent=False)
    session.chdir("docs")
    if session.posargs and session.posargs[0] == 'clean':
        session.log("First run clean")
        shutil.rmtree('source/api/generated')
        shutil.rmtree('source/examples')
        shutil.rmtree('source/datasets')
        session.run("sphinx-build", "-M", "clean", "source", "build")
    session.run("sphinx-build", "-M", "html", "source", "build", "-t", "notebooks")


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    session.install("build")
    session.log("Building normal files")
    session.run("python", "-m", "build")

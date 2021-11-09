import os
import tempfile

import nox

PYTHON_VERSIONS = ["3.6", "3.7", "3.8", "3.9", "3.10"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("-r", "tests/requirements.txt")
    session.install(".", '--ignore-installed', '--no-cache-dir', '-v', silent=False)
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
        session.run("Running without coverage")
        cov_args = []
    session.run("pytest", '-vv', '--doctest-modules', '--durations=20', *cov_args, '--pyargs', "tests/", 'deeptime')


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    session.install("build")
    session.log("Building normal files")
    session.run("python", "-m", "build")

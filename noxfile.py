import nox

PYTHON_VERSIONS = ["3.6", "3.7", "3.8", "3.9", "3.10"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("-r", "tests/requirements.txt")
    session.install(".")
    session.run("pytest", "tests/")


@nox.session(reuse_venv=True)
def build(session: nox.Session) -> None:
    session.install("build")
    session.log("Building normal files")
    session.run("python", "-m", "build")

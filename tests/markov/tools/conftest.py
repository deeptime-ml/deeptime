import warnings
import pytest


@pytest.fixture(scope='session', autouse=True)
def session_setup():
    warnings.filterwarnings('once', category=DeprecationWarning)
    warnings.filterwarnings('once', category=PendingDeprecationWarning)


@pytest.fixture(params=[False, True], ids=lambda x: f"{'sparse' if x else 'dense'}")
def sparse_mode(request):
    yield request.param

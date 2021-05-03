import warnings
import pytest


@pytest.fixture(scope='session', autouse=True)
def session_setup():
    warnings.filterwarnings('once', category=DeprecationWarning)
    warnings.filterwarnings('once', category=PendingDeprecationWarning)

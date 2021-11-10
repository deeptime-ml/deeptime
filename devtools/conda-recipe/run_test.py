import sys
import pytest

if __name__ == '__main__':
    sys.exit(pytest.main("-vv --doctest-modules --pyargs tests/ deeptime".split(' ')))

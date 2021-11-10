import sys
import pytest

if __name__ == '__main__':
    sys.exit(pytest.main("-vv --pyargs tests/".split(' ')))

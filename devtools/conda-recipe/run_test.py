import os
import sys

import pytest

cover_pkg = 'sktime'

# where to write junit xml
junit_xml = os.path.join(os.getenv('CIRCLE_TEST_REPORTS', os.path.expanduser('~')),
                         'reports', 'junit.xml')
target_dir = os.path.dirname(junit_xml)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
print('junit destination:', junit_xml)

pytest_args = ("-vv "
               "--cov={cover_pkg} "
               "--cov-report=xml:{dest_report} "
               "--doctest-modules "
               "--junit-xml={junit_xml} "
               "--durations=20 "
               "tests/ "
               .format(cover_pkg=cover_pkg,
                       junit_xml=junit_xml,
                       dest_report=os.path.join(os.path.expanduser('~/'), 'coverage.xml'),
                       )
               .split(' '))

print("args:", pytest_args)
print('cwd:', os.getcwd())
print('content:\n', os.listdir(os.getcwd()))

if __name__ == '__main__':
    res = pytest.main(pytest_args)
    sys.exit(res)

import os
import sys

import pytest

test_pkg = 'sktime'
cover_pkg = test_pkg

# where to write junit xml
junit_xml = os.path.join(os.getenv('CIRCLE_TEST_REPORTS', os.path.expanduser('~')),
                         'reports', 'junit.xml')
target_dir = os.path.dirname(junit_xml)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
print('junit destination:', junit_xml)
#njobs_args = '-p no:xdist' if os.getenv('TRAVIS') or os.getenv('CIRCLECI') else '-n2'
njobs_args = ''

pytest_args = ("-v tests/ "
               "--cov={cover_pkg} "
               "--cov-report=xml:{dest_report} "
               "--doctest-modules "
               "{njobs_args} "
               "--junit-xml={junit_xml} "
               #"--durations=20 "
               .format(test_pkg=test_pkg, cover_pkg=cover_pkg,
                       junit_xml=junit_xml,
                       dest_report=os.path.join(os.path.expanduser('~/'), 'coverage.xml'),
                       njobs_args=njobs_args,
                       )
               .split(' '))
print("args:", pytest_args)
res = pytest.main(pytest_args)

sys.exit(res)


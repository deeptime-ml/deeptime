import os
import sys

import pytest

test_pkg = 'pyemma'
cover_pkg = test_pkg

# where to write junit xml
junit_xml = os.path.join(os.getenv('CIRCLE_TEST_REPORTS', os.path.expanduser('~')),
                         'reports', 'junit.xml')

njobs_args = '-p no:xdist' if os.getenv('TRAVIS') else '-n2'

pytest_args = ("-v --pyargs {test_pkg} "
               "--cov={cover_pkg} "
               "--cov-report=xml:{dest_report} "
               "--doctest-modules "
               "{njobs_args} "
               "--junit-xml={junit_xml} "
               "-c {pytest_cfg}"
               .format(test_pkg=test_pkg, cover_pkg=cover_pkg,
                       junit_xml=junit_xml, pytest_cfg='setup.cfg',
                       dest_report=os.path.join(os.path.expanduser('~/'), 'coverage.xml'),
                       njobs_args=njobs_args,
                       )
               .split(' '))
print("args:", pytest_args)
res = pytest.main(pytest_args)

sys.exit(res)


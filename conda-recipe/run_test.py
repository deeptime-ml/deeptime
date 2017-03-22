import os
import sys

import pytest

test_pkg = 'pyemma'
cover_pkg = test_pkg

junit_xml = os.path.join(os.getenv('CIRCLE_TEST_REPORTS', '.'), 'junit.xml')


pytest_args = ("-v --pyargs {test_pkg} "
               "--cov={cover_pkg} "
               "--cov-report=xml:{dest_report} "
               "--doctest-modules "
               #"-n 2 "# -p no:xdist" # disable xdist in favour of coverage plugin
               "--junit-xml={junit_xml} "
               "-c {pytest_cfg}"
               .format(test_pkg=test_pkg, cover_pkg=cover_pkg,
                       junit_xml=junit_xml, pytest_cfg='setup.cfg',
                       dest_report=os.path.join(os.path.expanduser('~/'), 'coverage.xml'),
                       )
               .split(' '))
print("args:", pytest_args)
res = pytest.main(pytest_args)

sys.exit(res)


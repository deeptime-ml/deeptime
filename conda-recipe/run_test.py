import tempfile
import os
import sys
import pytest
import shutil
import pkg_resources

test_pkg = 'pyemma'
cover_pkg = test_pkg

junit_xml = os.path.join(os.getenv('CIRCLE_TEST_REPORTS', '.'), 'junit.xml')
if os.getenv('CONDA_BUILD', False):
    pytest_cfg = pkg_resources.resource_filename(test_pkg, 'setup.cfg')
else:
    pytest_cfg = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../setup.cfg')
assert os.path.exists(pytest_cfg), pytest_cfg
print("Using pytest config file: %s" % pytest_cfg)

# chdir to an path outside of conda-bld, which is known to persist the build phase
run_dir = tempfile.mkdtemp()
os.chdir(run_dir)

# matplotlib headless backend
with open('matplotlibrc', 'w') as fh:
    fh.write('backend: Agg')

pytest_args = ("-v --pyargs {test_pkg} "
               "--cov={cover_pkg} "
               "--cov-report=xml "
               "--doctest-modules "
               #"-n 2 "# -p no:xdist" # disable xdist in favour of coverage plugin
               "--junit-xml={junit_xml} "
               "-c {pytest_cfg} "
               .format(test_pkg=test_pkg, cover_pkg=cover_pkg,
                       junit_xml=junit_xml, pytest_cfg=pytest_cfg)
               .split(' '))
print("args:", pytest_args)
res = pytest.main(pytest_args)

# copy it to home, so we can process it with codecov etc.
shutil.copy('coverage.xml', os.path.expanduser('~/'))

sys.exit(res)


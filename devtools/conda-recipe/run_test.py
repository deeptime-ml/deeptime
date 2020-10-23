import os
import sys

import pytest
import tempfile

cover_pkg = 'deeptime'
xml_results_dest = os.getenv('SYSTEM_DEFAULTWORKINGDIRECTORY', tempfile.gettempdir())
assert os.path.isdir(xml_results_dest), 'no dest dir available'
# where to write junit xml
junit_xml = os.path.join(xml_results_dest, 'junit.xml')
cov_xml = os.path.join(xml_results_dest, 'coverage.xml')
print('junit destination:', junit_xml)
print('coverage dest:', cov_xml)
pytest_args = ("-vv "
               "--cov={cover_pkg} "
               "--cov-report=xml:{dest_report} "
               "--doctest-modules "
               "--junit-xml={junit_xml} "
               "--durations=20 "
               "--cov-config {cov_config} "
               "--pyargs tests/ deeptime"
               .format(cover_pkg=cover_pkg,
                       junit_xml=junit_xml,
                       dest_report=cov_xml,
                       cov_config=".coveragerc"
                       )
               .split(' '))

print("args:", pytest_args)
print('cwd:', os.getcwd())
print('content:\n', os.listdir(os.getcwd()))

if __name__ == '__main__':
    res = pytest.main(pytest_args)
    sys.exit(res)

"""
Cleanup old development builds on Anaconda.org
 
Assumes one has set 4 environment variables:

1. BINSTAR_TOKEN: token to authenticate with anaconda.org
2. DEV_BUILD_N_KEEP: int, how many builds to keep, delete oldest first.
3. ORGNAME: str, anaconda.org organisation/user
4. PACKGENAME: str, name of package to clean up

author: Martin K. Scherer
data: 20.4.16
"""
from __future__ import print_function, absolute_import

import os

from binstar_client.utils import get_server_api
from pkg_resources import parse_version

token = os.environ['BINSTAR_TOKEN']
org = os.environ['ORGNAME']
pkg = os.environ['PACKAGENAME']
n_keep = int(os.getenv('DEV_BUILD_N_KEEP', 10))

b = get_server_api(token=token)
package = b.package(org, pkg)

# sort releases by version number, oldest first
sorted_by_version = sorted(package['releases'],
                           key=lambda rel: parse_version(rel['version']),
                           reverse=True
                          )
to_delete = []
print("Currently have {n} versions online. Going to remove {x}.".
      format(n=len(sorted_by_version), x=len(sorted_by_version) - n_keep))

while len(sorted_by_version) > n_keep:
    to_delete.append(sorted_by_version.pop())


# remove old releases from anaconda.org 
for rel in to_delete:
    version = rel['version']
    print("removing {version}".format(version=version))
    b.remove_release(org, pkg, version)

"""
Cleanup old development builds on Anaconda.org
 
Assumes one has set two environment variables:

1. BINSTAR_TOKEN: token to authenticate with anaconda.org
2. DEV_BUILD_N_KEEP: int, how many builds to keep, delete oldest first.
3. ORGNAME: str, anaconda.org organisation/user
4. PACKGENAME: str, name of package to clean up

author: Martin K. Scherer
data: 20.4.16
"""
from __future__ import print_function, absolute_import
from binstar_client.utils import get_server_api
from pkg_resources import parse_version
from operator import getitem
import os

token = os.getenv['BINSTAR_TOKEN']
org = os.getenv['ORGNAME']
pkg = os.getenv['PACKAGENAME']
n_keep = int(os.getenv('DEV_BUILD_N_KEEP'))

b = get_server_api(token=token)
package = b.package(org, pkg)

# sort releases by version number, oldest first
sorted_by_version = sorted(package['releases'],
                           key=lambda rel: parse_version(rel['version']),
                           reverse=True
                          )
to_delete = []

while len(sorted_by_version) > N_KEEP:
    to_delete.append(sorted_by_version.pop())

# remove old releases from anaconda.org 
for rel in to_delete:
    spec = rel['full_name']
    version = rel['version']
    for dist in rel['distributions']:
        b.remove_dist(org, package_name=pkg, release=version, basename=dist)
        print("removed file %s" % dist)


#!/usr/bin/env python
# author: marscher
# purpose: update version.json file on new software release.
from __future__ import print_function
import json
import sys

from argparse import ArgumentParser

from six.moves.urllib.request import urlopen
from distutils.version import LooseVersion as parse


def hash_(self):
    return hash(self.vstring)
parse.__hash__ = hash_


def make_version_dict(URL, version, url_prefix='v', latest=False):
    return {'version': version,
            'display': version,
            # git tags : vx.y.z
            'url': URL + '/' + url_prefix + version,
            'latest': latest}


def find_latest(versions):
    for v in versions:
        if v['latest'] == True:
            return v


def main(argv=None):
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    parser = ArgumentParser()
    parser.add_argument('-u', '--url', dest='url', required=True, help="base url (has to contain versions json)")
    parser.add_argument('-o', '--output', dest='output')
    parser.add_argument('-a', '--add_version', dest='version')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-l', '--latest-version', dest='latest', action='store_true')

    args = parser.parse_args()

    URL = args.url
    # get dict
    versions = json.load(urlopen(URL + '/versions.json'))
    # add new version
    if args.version:
        versions.append(make_version_dict(URL, args.version))

    # create Version objects to compare them
    version_objs = [parse(s['version']) for s in versions]

    # unify and sort
    version_objs = set(version_objs)
    version_objs = sorted(list(version_objs))

    versions = [make_version_dict(URL, str(v)) for v in version_objs if v != 'devel']

    # last element should be the highest version
    versions[-1]['latest'] = True
    versions.append(make_version_dict(URL, 'devel', '', False))

    if args.verbose:
        print("new versions json:")
        json.dump(versions, sys.stdout, indent=1)
        print()

    if args.latest:
        print(find_latest(versions)['version'])
        return 0

    if args.output:
        with open(args.output, 'w') as v:
            json.dump(versions, v, indent=1)
            v.flush()

if __name__ == '__main__':
    sys.exit(main())

import os

from numpy.testing import assert_

import deeptime as dt


def test_headers_present():
    include_dirs = dt.capi_includes(inc_clustering=True, inc_markov=True, inc_markov_hmm=True, inc_data=True)
    for include_dir in include_dirs:
        headers = [entry.name for entry in os.scandir(include_dir) if entry.is_file()]
        headers = [h for h in headers if h.endswith('.h')]
        assert_(len(headers) > 0)

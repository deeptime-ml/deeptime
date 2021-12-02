import os

from numpy.testing import assert_

import deeptime as dt


def test_headers_present():
    include_dirs = dt.capi_includes(inc_clustering=True, inc_markov=True, inc_markov_hmm=True, inc_data=True)
    assert_(len(include_dirs) > 0)

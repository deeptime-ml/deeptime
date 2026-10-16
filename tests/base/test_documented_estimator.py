"""Exercise the custom-estimator examples from the developer guide."""
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("documented_class", [False, True])
def test_mean_estimator_example(documented_class):
    guide = Path(__file__).resolve().parents[2] / "docs/source/index_dev.rst"
    blocks = [textwrap.dedent(block) for block in re.findall(
        r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)", guide.read_text(encoding="utf-8"))]
    namespace = {}
    for block in blocks[:2]:
        exec(block, namespace)
    if documented_class:
        exec(blocks[5], namespace)

    data = np.array([[1., 3.], [5., 7.]])
    estimator = namespace["MeanEstimator"](axis=-1)
    assert estimator.fetch_model() is None
    np.testing.assert_allclose(estimator.fit(data).fetch_model().mean, [2., 6.])
    np.testing.assert_allclose(estimator.fit_fetch(data).mean, [2., 6.])

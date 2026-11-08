"""Executes the Python code blocks of the top-level README so that they cannot rot.

Every fenced ```python block in README.md is run in its own namespace, in document order. A block is
identified by the closest Markdown heading above it, which is used as the test id.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from deeptime.util.platform import module_available

README = Path(__file__).resolve().parents[1] / "README.md"


def _code_blocks(text):
    heading = "README"
    in_block = False
    lines = []
    blocks = []
    for line in text.splitlines():
        if in_block:
            if line.startswith("```"):
                in_block = False
                blocks.append((heading, "\n".join(lines)))
            else:
                lines.append(line)
        elif line.startswith("```python"):
            in_block = True
            lines = []
        elif line.startswith("#"):
            heading = line.lstrip("#").strip()
    return blocks


def _blocks():
    if not README.exists():
        return []
    blocks = _code_blocks(README.read_text(encoding="utf-8"))
    counts = {}
    result = []
    for heading, code in blocks:
        counts[heading] = counts.get(heading, 0) + 1
        ident = heading if counts[heading] == 1 else f"{heading}-{counts[heading]}"
        result.append(pytest.param(code, id=re.sub(r"\s+", "-", ident.lower())))
    return result


@pytest.mark.skipif(not README.exists(), reason="README.md only available in a source checkout")
@pytest.mark.parametrize("code", _blocks())
def test_readme_code_block(code):
    if "import torch" in code and not module_available("torch"):
        pytest.skip("README block requires PyTorch")
    namespace = {"__name__": "__readme__"}
    exec(compile(code, str(README), "exec"), namespace)

    # The quickstart claims to recover the slowest relaxation timescale of the generating process (16.2 steps).
    if "msm" in namespace:
        np.testing.assert_allclose(namespace["msm"].timescales(k=1), 16.2, rtol=0.1)
    if "hmm" in namespace:
        np.testing.assert_allclose(namespace["hmm"].transition_model.timescales(), 16.2, rtol=0.1)
    # The VAMPnet example claims a VAMP-2 score of about 1.77 where linear VAMP reaches 1.28.
    if "vampnet" in namespace:
        from deeptime.decomposition import VAMP
        score = VAMP(lagtime=1, observable_transform=namespace["vampnet"]).fit_fetch(namespace["traj"]).score(2)
        assert score > 1.6

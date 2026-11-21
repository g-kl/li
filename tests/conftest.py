import sys

import numpy as np
import pytest

OLD = "/mnt/home/gokulg/lineage-inpainting"


@pytest.fixture(scope="session")
def toy():
    from li.toy import toy
    return toy([7.0, 7.5, 8.0, 8.5], n_per_time=2, dim=4)


@pytest.fixture(scope="session")
def old():
    """the publication repo, used as an oracle."""
    if OLD not in sys.path:
        sys.path.insert(0, OLD)
    return OLD


def subtrees(data, b, n=None):
    """(tree, nodes, cell) items for every indexed subtree at bin b."""
    items = [(t, t.subtree(r, b), t.cell) for t in data.trees.values() for r in t.index.get(b, [])]
    return items if n is None else [items[i] for i in np.random.default_rng(0).permutation(len(items))[:n]]

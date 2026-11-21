import numpy as np
import pandas as pd

from li.data.tree import Tree
from li.eval.nodes import summarize
from li.plot.tree import layout


def test_summarize_by_hand():
    coup = pd.DataFrame({"tree": "t", "node": ["a"] * 4 + ["b"] * 4, "time": 8.0, "dist": [1.0, 2.0, 3.0, 4.0] * 2,
                         "sample": [0, 1, 2, 3] * 2, "cell_type": ["x", "x", "x", "y"] + ["x", "y", "z", "z"]})
    s = summarize(coup).set_index("node")
    assert s.loc["a", "modal"] == "x" and s.loc["a", "agreement"] == 0.75 and s.loc["a", "dist"] == 2.5
    assert s.loc["b", "modal"] == "z" and s.loc["b", "agreement"] == 0.5
    assert np.isclose(s.loc["a", "entropy"], -(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
    assert np.isclose(s.loc["b", "entropy"], -(2 * 0.25 * np.log(0.25) + 0.5 * np.log(0.5)))


def test_layout(toy):
    for t in toy.trees.values():
        y = layout(t, 0)
        assert (np.sort(y[t.leaf]) == np.arange(t.leaf.sum())).all()
        assert (np.diff(y[t.leaf]) == 1).all()  # preorder
        for i in np.flatnonzero(~t.leaf):
            assert np.isclose(y[i], y[t.children(i)].mean())
    # a clade below the root
    t = Tree("t", np.array(list("abcdefg"), dtype=object), [-1, 0, 1, 1, 0, 4, 4], [0, 1, 2, 2, 1, 2, 2], [-1] * 7)
    assert layout(t, 4).tolist() == [0.5, 0.0, 1.0]

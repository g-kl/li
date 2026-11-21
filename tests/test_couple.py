import numpy as np
import pandas as pd
import torch

from li.couple import couple
from li.metrics import germ_crossing
from li.net import TreeNet
from li.tree import Tree


def test_couple_toy(toy):
    torch.manual_seed(0)
    net = TreeNet(4, d=16, layers=1, hidden=32, flow=True)
    c = couple(net, toy, n_steps=2)
    bank = toy.bank
    for name, g in c.groupby("tree"):
        t = toy.trees[name]
        pos = pd.Index(t.nodes).get_indexer(g.node)
        assert (pos >= 0).all() and not t.leaf[pos].any() and (t.time[pos] >= 7.0).all()
        # every internal node from E7.0 on whose layer has a root gap earlier gets a cell
        want = ~t.leaf & (t.time >= 7.0)
        near = np.abs(t.time[:, None] - bank.times[None]).argmin(1)
        want &= bank.times[near] - 0.5 >= t.time[0]
        assert set(pos) == set(np.flatnonzero(want))
    # snapped within one observation time either side
    cell_t = pd.Series(bank.time, index=bank.names)[c.cell].values
    node_b = np.abs(c.time.values[:, None] - bank.times[None]).argmin(1)
    cell_b = np.searchsorted(bank.times, cell_t)
    assert (np.abs(node_b - cell_b) <= 1).all() and (c.dist >= 0).all()


def test_germ_crossing_by_hand():
    # 0 -> 1 -> (2, 3); 0 -> 4 -> (5, 6)
    t = Tree("t", np.array(list("abcdefg"), dtype=object), [-1, 0, 1, 1, 0, 4, 4],
             [7.5, 8.0, 8.5, 8.5, 8.0, 8.5, 8.5], [-1] * 7)
    germ = pd.Series({"x": "mesoderm", "y": "mesoderm", "z": "ectoderm", "e": "epiblast"})
    coup = pd.DataFrame({"tree": "t", "node": ["a", "b", "e"], "cell": ["x", "y", "z"]})
    r = germ_crossing(coup, {"t": t}, germ, [7.5, 8.0, 8.5])
    assert r[r.bin == "all"].rate.item() == 0.5 and r[r.bin == "all"].n_edges.item() == 2
    # an unassigned middle node is skipped over: a -> (c via b) when b is unassigned
    coup = pd.DataFrame({"tree": "t", "node": ["a", "c"], "cell": ["x", "z"]})
    assert germ_crossing(coup, {"t": t}, germ, [7.5, 8.0, 8.5]).rate.iloc[0] == 1.0
    # epiblast sources don't count
    coup = pd.DataFrame({"tree": "t", "node": ["a", "b"], "cell": ["e", "z"]})
    r = germ_crossing(coup, {"t": t}, germ, [7.5, 8.0, 8.5])
    assert r[r.bin == "all"].n_edges.item() == 0

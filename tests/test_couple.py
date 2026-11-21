from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from li.data.tree import Tree, scramble
from li.eval import fills
from li.eval.couple import couple, modal_descendant
from li.eval.metrics import germ_crossing
from li.model.net import TreeNet


def net():
    torch.manual_seed(0)
    return TreeNet(4, d=16, layers=1, hidden=32, flow=True)


FILLS = {
    "sampler": lambda: fills.sampler(net(), n_steps=2),
    "posterior_mean": lambda: fills.posterior_mean(net(), k=2, n_steps=2),
    "mean_pool": lambda: fills.mean_pool,
    "random_descendant": lambda: fills.random_descendant,
    "random_cell": lambda: fills.random_cell,
}


@pytest.mark.parametrize("name", FILLS)
def test_couple_toy(toy, name):
    c = couple(FILLS[name](), toy, n_samples=2)
    bank = toy.bank
    assert sorted(c["sample"].unique()) == [0, 1]
    for (tn, s), g in c.groupby(["tree", "sample"]):
        t = toy.trees[tn]
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
    if name == "random_cell":  # already real cells, from the node's own time
        assert (c.dist < 1e-3).all() and (node_b == cell_b).all()


def test_mean_pool_deterministic(toy):
    c = couple(fills.mean_pool, toy, n_samples=2)
    a, b = c[c["sample"] == 0], c[c["sample"] == 1]
    assert (a.cell.values == b.cell.values).all()


def test_modal_descendant(toy):
    lab = pd.Series(np.where(toy.bank.emb[:, 0].numpy() > 0, "pos", "neg"), index=toy.bank.names)
    data = SimpleNamespace(trees=toy.trees, bank=toy.bank, labels=pd.DataFrame({"germ_layer": lab}))
    md = modal_descendant(data)
    for _, r in md.sample(50, random_state=0).iterrows():
        t = toy.trees[r.tree]
        i = list(t.nodes).index(r.node)
        nd = np.arange(i, t.end[i])
        leaves = lab[toy.bank.names[t.cell[nd[t.leaf[nd]]]]]
        assert (leaves == r.label).sum() == leaves.value_counts().max()


def test_scramble(toy):
    rng = np.random.default_rng(0)
    for t in toy.trees.values():
        s = scramble(t, rng)
        assert (s.parent == t.parent).all() and (s.time == t.time).all()
        assert sorted(s.cell[s.leaf]) == sorted(t.cell[t.leaf]) and (s.cell[~s.leaf] == -1).all()


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
    # a label column is used as is
    coup = pd.DataFrame({"tree": "t", "node": ["a", "b", "e"], "label": ["mesoderm", "mesoderm", "ectoderm"]})
    assert germ_crossing(coup, {"t": t}, germ, [7.5, 8.0, 8.5]).rate.iloc[0] == 0.5

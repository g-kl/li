import numpy as np
import pandas as pd
import pytest
import torch

from li.eval import fills
from li.eval.couple import couple
from li.model.net import TreeNet
from li.perturb.project import target_bank
from li.perturb.rollout import rollout

K = 2


@pytest.fixture(scope="module")
def run(toy):
    """a control and a knockout rollout of the toy E8.5 trees under a fake gene."""
    torch.manual_seed(0)
    bank = toy.bank
    net = TreeNet(bank.emb.shape[1], d=16, heads=2, layers=1, hidden=32, flow=True)
    trees = [t for t in toy.trees.values() if t.leaf_time == max(bank.times)]
    from types import SimpleNamespace
    c = couple(fills.sampler(net, n_steps=2), SimpleNamespace(trees={t.name: t for t in trees}, bank=bank), K)
    row = pd.Series(np.arange(len(bank.names)), index=bank.names)
    coup = []
    for k in range(K):
        ck = c[c["sample"] == k]
        d = {}
        for t in trees:
            a = np.full(len(t), -1, np.int64)
            q = ck[ck.tree == t.name]
            pos = pd.Index(t.nodes).get_indexer(q.node)
            a[pos[pos >= 0]] = row.reindex(q.cell).values[pos >= 0]
            d[t.name] = a
        coup.append(d)
    gene = (np.random.default_rng(0).random(len(bank.names)) < 0.5).astype(np.float32)
    off = gene <= 0.0
    kw = dict(gene_col=gene, cutoff=0.0, lo=bank.times[0], hi=bank.times[-1],
              window=1, n_steps=2, n_samples=K)
    ctl = rollout(net, bank, trees, coup, None, **kw)
    ko = rollout(net, bank, trees, coup, target_bank(bank, off, "cpu"), **kw)
    return toy, trees, off, ctl, ko


def test_same_nodes(run):
    """fixed topology: a knockout only moves states, never the tree."""
    _, _, _, ctl, ko = run
    key = ["tree", "node", "sample"]
    assert set(map(tuple, ctl[key].values)) == set(map(tuple, ko[key].values))


def test_control_is_the_observed_tree(run):
    """control changes nothing and reproduces every observed leaf."""
    toy, trees, _, ctl, _ = run
    assert not ctl.changed.any() and (ctl.score == 0).all()
    for t in trees:
        obs = pd.Series(toy.bank.names[t.cell[t.leaf]], index=t.nodes[t.leaf])
        c = ctl[(ctl.tree == t.name) & ctl.node.isin(obs.index)]
        assert len(c) and (c.cell.values == obs.reindex(c.node).values).all()


def test_knockout_lands_on_gene_off_cells(run):
    """every state is a real cell, and under a knockout one that does not express."""
    toy, _, off, _, ko = run
    assert ko.cell.isin(set(toy.bank.names[off])).all()
    assert ko.changed.mean() > 0  # the fake gene is on in half the bank


def test_score_is_a_path_max(run):
    """score never falls from parent to child, so one threshold prunes a whole subtree."""
    _, _, _, _, ko = run
    for _, g in ko.groupby(["tree", "sample"]):
        s = g.set_index("node")["score"]
        p = g.parent.values
        has = pd.notna(p) & pd.Index(p).isin(s.index)
        assert (s.values[has] + 1e-9 >= s.reindex(p[has]).values).all()


def test_off_at_ratio_zero_is_changed_plus_descendants(run):
    """the recorded score reproduces the old off_manifold_descendants rule."""
    _, _, _, _, ko = run
    g = ko[(ko.tree == ko.tree.iloc[0]) & (ko["sample"] == 0)]
    kid = {}
    for n, p in zip(g.node, g.parent):
        kid.setdefault(p, []).append(n)
    exp, stack = set(), list(g.node[g.changed])
    while stack:
        x = stack.pop()
        if x not in exp:
            exp.add(x)
            stack.extend(kid.get(x, []))
    assert set(g.node[g.score > 0]) == exp

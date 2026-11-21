from types import SimpleNamespace

import numpy as np

from li.bank import Bank
from li.tree import Tree


def grow(rng, name, t0, t_end, dim, rate=2.5):
    """birth process from t0 with brownian states; every lineage alive at t_end is a leaf."""
    nodes, parent, time, states = [], [], [], []
    stack = [(-1, t0, np.zeros(dim))]
    while stack:
        p, t, x = stack.pop()
        i = len(nodes)
        nodes.append(f"{name}_{i}")
        parent.append(p)
        time.append(t)
        states.append(x)
        if t >= t_end:
            continue
        for _ in range(2):
            tc = min(t + rng.exponential(1 / rate), t_end)
            stack.append((i, tc, x + rng.normal(size=dim) * np.sqrt(tc - t)))
    return nodes, np.array(parent), np.array(time), np.array(states)


def toy(time_bins, n_per_time=3, dim=8, max_leaves=1024, t0=6.5, seed=0, device="cpu"):
    """a few small trees per leaf time; the bank is every leaf."""
    rng = np.random.default_rng(seed)
    raw, emb, ctime, names = [], [], [], []
    for T in time_bins[1:]:
        for c in range(n_per_time):
            name = f"E{T}-C{c}"
            nodes, parent, time, states = grow(rng, name, t0, T, dim)
            leaf = ~np.isin(np.arange(len(parent)), parent)
            cell = np.full(len(parent), -1)
            cell[leaf] = len(names) + np.arange(leaf.sum())
            names += [n for n, lf in zip(nodes, leaf) if lf]
            emb.append(states[leaf])
            ctime += [T] * leaf.sum()
            raw.append((name, nodes, parent, time, cell))
    bank = Bank(np.concatenate(emb), np.array(ctime), np.array(names), device)
    trees = {}
    for name, nodes, parent, time, cell in raw:
        t = Tree(name, np.array(nodes, dtype=object), parent, time, cell)
        t.build_index(time_bins, max_leaves)
        trees[name] = t
    return SimpleNamespace(trees=trees, bank=bank, ref=bank, labels=None, time_bins=list(time_bins))

import networkx as nx
import numpy as np

from li.data.tree import EPS, Tree, from_nx, prune, to_nx


def test_preorder(toy):
    for t in toy.trees.values():
        assert t.parent[0] == -1 and (t.parent[1:] < np.arange(1, len(t))).all()
        g = to_nx(t)
        for i in np.random.default_rng(0).choice(len(t), 20):
            desc = nx.descendants(g, t.nodes[i]) | {t.nodes[i]}
            assert set(t.nodes[i:t.end[i]]) == desc


def test_times_increase(toy):
    for t in toy.trees.values():
        assert (t.time[1:] >= t.time[t.parent[1:]]).all()


def test_nx_roundtrip(toy):
    t = next(iter(toy.trees.values()))
    t2 = from_nx(to_nx(t), t.name)
    assert (t2.nodes == t.nodes).all() and (t2.parent == t.parent).all() and np.allclose(t2.time, t.time)


def test_prune(toy):
    t = next(iter(toy.trees.values()))
    cell = t.cell.copy()
    leaves = np.flatnonzero(t.leaf)
    cell[leaves[::3]] = -1
    p = prune(Tree(t.name, t.nodes, t.parent, t.time, cell))
    assert (p.cell[p.leaf] >= 0).all()
    assert set(p.nodes[p.leaf]) == set(t.nodes[leaves[cell[leaves] >= 0]])
    # parent links survive by name
    pos = {n: i for i, n in enumerate(t.nodes)}
    for i in range(1, len(p)):
        assert t.nodes[t.parent[pos[p.nodes[i]]]] == p.nodes[p.parent[i]]


def test_subtrees_partition_leaves(toy):
    for t in toy.trees.values():
        for b in toy.time_bins:
            if b < t.time[0]:
                continue
            seen = []
            for r in t.index[b]:
                nd = t.subtree(r, b)
                assert t.time[r] <= b + EPS and (t.time[nd[1:]] > b + EPS).all()
                seen += list(nd[t.leaf[nd]])
            assert len(seen) == len(set(seen)) == t.leaf.sum()


def test_max_leaves(toy):
    t = max(toy.trees.values(), key=len)
    t.build_index(toy.time_bins, 5)
    for b, roots in t.index.items():
        for r in roots:
            nd = t.subtree(r, b)
            assert 1 <= t.leaf[nd].sum() <= 5
    t.build_index(toy.time_bins, 1024)


def test_index_matches_old(toy, old):
    """same subtrees as the old TreeRecord on the same trees."""
    from datasets.tree_record import _extract_subtree, _leaf_counts, _node_depths, _snapshot_at
    for t in toy.trees.values():
        g = to_nx(t)
        sizes, depths = _leaf_counts(g), _node_depths(g)
        for b in toy.time_bins:
            want = set()
            for r in _snapshot_at(g, b):
                st = _extract_subtree(g, r, sizes, depths, t.name, bin_time=b)
                if 1 <= st["size"] <= 1024:
                    want.add((r, frozenset(st["graph"].nodes)))
            got = {(t.nodes[r], frozenset(t.nodes[t.subtree(r, b)])) for r in t.index[b]}
            assert got == want

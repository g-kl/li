import networkx as nx
import numpy as np
import torch

from li.batch import collate, cross_section, subsample
from li.tree import to_nx
from conftest import subtrees


def test_fields(toy):
    items = subtrees(toy, 7.5)
    b = collate(items, toy.bank.emb)
    for k, (t, nd, _) in enumerate(items):
        n = len(nd)
        assert b.mask[k, :n].all() and not b.mask[k, n:].any()
        g = to_nx(t).subgraph(t.nodes[nd])
        par = b.parent[k, :n].numpy()
        assert par[0] == -1
        for i in range(1, n):
            assert g.has_edge(t.nodes[nd[par[i]]], t.nodes[nd[i]])
        assert b.leaf[k, :n].numpy().tolist() == [g.out_degree(v) == 0 for v in t.nodes[nd]]
        root = t.nodes[nd[0]]
        assert b.depth[k, :n].numpy().tolist() == [nx.shortest_path_length(g, root, v) for v in t.nodes[nd]]
        obs = t.cell[nd] >= 0
        assert (b.obs[k, :n].numpy() == obs).all()
        assert torch.equal(b.x[k, :n][obs], toy.bank.emb[t.cell[nd][obs]])


def test_anc_and_mrca(toy):
    items = subtrees(toy, 7.0, n=6)
    b = collate(items, toy.bank.emb)
    for k, (t, nd, _) in enumerate(items):
        g = to_nx(t).subgraph(t.nodes[nd])
        names = t.nodes[nd]
        lca = dict(nx.tree_all_pairs_lowest_common_ancestor(g, root=names[0]))
        for i in range(len(nd)):
            anc = set(nx.ancestors(g, names[i])) | {names[i]}
            assert set(names[b.anc[k, :len(nd), i].numpy()]) == anc
            for j in range(len(nd)):
                m = names[i] if i == j else lca.get((names[i], names[j])) or lca.get((names[j], names[i]))
                assert abs(b.mrca[k, i, j].item() - g.nodes[m]["time"]) < 1e-5


def test_cross_section_marks_each_cut_branch(toy):
    # each lineage crosses the cut on one edge, whose parent is in the cross-section;
    # leaves already reached by the cut are in it themselves
    b = collate(subtrees(toy, 7.0, n=8), toy.bank.emb)
    lo = b.time.masked_fill(~b.mask, 1e9).amin(1)
    hi = b.time.masked_fill(~b.mask, -1e9).amax(1)
    for u in np.linspace(0, 1, 7):
        t = lo + u * (hi - lo)
        cs = cross_section(b, t)
        for k, n in torch.nonzero(b.leaf & b.mask).tolist():
            path = torch.nonzero(b.anc[k, :, n]).flatten().tolist()  # root to leaf
            tk = t[k].item()
            if b.time[k, n] <= tk:
                assert cs[k, n]
                continue
            cuts = [p for p, c in zip(path, path[1:]) if b.time[k, p] <= tk < b.time[k, c]]
            assert len(cuts) == 1 and cs[k, cuts[0]]


def test_subsample(toy):
    t = max(toy.trees.values(), key=len)
    nd = t.subtree(0, t.time[0])
    for cap in [10, 40, 100]:
        np.random.seed(1)
        a = subsample(t, nd, cap)
        np.random.seed(1)
        assert (a == subsample(t, nd, cap)).all()
        deepest = len(t.levels())  # one leaf's path is the floor
        assert len(a) <= max(cap, deepest) and a[0] == nd[0]
        keep = set(a)
        assert all(t.parent[i] in keep for i in a[1:])

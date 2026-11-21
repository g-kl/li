import pickle
import sys

import numpy as np
import pytest
import torch

from li.eval.couple import read_legacy
from li.data.mouse import load, read_trees
from li.eval.metrics import germ_crossing
from li.data.tree import Tree, to_nx

pytestmark = pytest.mark.slow
ROOT = "/mnt/home/gokulg/data/2610_lineage_inpainting_data_avail"
BINS = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]


@pytest.fixture(scope="module")
def mouse():
    return load(ROOT, BINS, 1024, device="cuda")


def test_shapes(mouse):
    assert len(mouse.trees) == 56
    assert mouse.bank.emb.shape == (1834199, 50)
    assert sorted({t.leaf_time for t in mouse.trees.values()}) == [7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    for t in mouse.trees.values():
        assert (t.cell[t.leaf] >= 0).all() and (t.cell[~t.leaf] == -1).all()
        assert (t.time[1:] >= t.time[t.parent[1:]]).all()
        assert all(len(t.index[b]) for b in BINS if b <= t.leaf_time)


def test_index_matches_old_cache(mouse):
    """old cache (max_size=128) vs ours rebuilt at 128 on a small tree."""
    old = pickle.load(open("/mnt/home/gokulg/pool-and-split/cache/final_topology.h5td_E7.5-R1-C1.pkl", "rb"))
    t = mouse.trees["E7.5-R1-C1"]
    t2 = Tree(t.name, t.nodes, t.parent, t.time, t.cell)
    t2.build_index(BINS, 128)
    for b in BINS:
        want = {(st["root"], frozenset(st["graph"].nodes)) for st in old["subtree_index"].get(b, [])}
        got = {(t2.nodes[r], frozenset(t2.nodes[t2.subtree(r, b)])) for r in t2.index[b]}
        assert got == want


def test_germ_matches_old(mouse, old):
    """same germ crossing as the old metric on the paper coupling."""
    from analysis.lib.metrics.germ_layer_eval import assignment_germ_layer_crossing_by_bin
    path = f"{ROOT}/model/assignments_step20000_adj_n8_splitter.pkl"
    gl = mouse.labels["germ_layer"].dropna()
    trees = {name: to_nx(Tree(name, nodes, parent, time, np.full(len(parent), -1)))
             for name, nodes, parent, time in read_trees(f"{ROOT}/embryos/final_topology.h5td")}
    want = assignment_germ_layer_crossing_by_bin(trees, pickle.load(open(path, "rb")), gl.to_dict(), list(mouse.bank.times))
    got = germ_crossing(read_legacy(path), mouse.trees, mouse.labels["germ_layer"], mouse.bank.times)
    assert abs(got[got.bin == "all"].rate.item() - want["all"]) < 1e-12
    for _, r in got[got.bin != "all"].iterrows():
        assert abs(r.rate - want[r.bin]) < 1e-12


def test_old_weights_load(mouse):
    from li.model.net import load_sampler
    net = load_sampler(f"{ROOT}/model")
    assert next(net.parameters()).is_cuda

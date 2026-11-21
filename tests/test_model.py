from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from li.train.align import energy
from li.data.batch import collate
from li.model.inpaint import baseline, impute, sample
from li.model.net import TreeNet
from li.data.tree import Tree
from conftest import subtrees


def seven():
    # 0 -> 1 -> (2, 3); 0 -> 4 -> (5, 6), leaves hold 1, 3, 5, 7
    t = Tree("t", np.array([str(i) for i in range(7)], dtype=object), [-1, 0, 1, 1, 0, 4, 4],
             [0.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0], [-1, -1, 0, 1, -1, 2, 3])
    return t, torch.tensor([[1.0], [3.0], [5.0], [7.0]])


def test_baseline_leaves():
    t, emb = seven()
    b = collate([(t, np.arange(7), t.cell)], emb)
    base = baseline(b, b.x, b.obs)[0, :, 0].tolist()
    assert base == [4.0, 2.0, 1.0, 3.0, 6.0, 5.0, 7.0]


def test_baseline_cut():
    t, emb = seven()
    b = collate([(t, np.arange(7), t.cell)], emb)
    x = torch.zeros_like(b.x)
    x[0, 1], x[0, 5], x[0, 6] = 10.0, 5.0, 7.0
    anchor = torch.zeros_like(b.obs)
    anchor[0, [1, 5, 6]] = True
    base = baseline(b, x, anchor)[0, :, 0].tolist()
    assert base == [8.0, 10.0, 10.0, 10.0, 6.0, 5.0, 7.0]


def test_energy_matches_geomloss():
    from geomloss import SamplesLoss
    torch.manual_seed(0)
    x, y = torch.randn(200, 5), torch.randn(300, 5) + 0.5
    assert torch.allclose(energy(x, y), SamplesLoss("energy")(x, y), atol=1e-5)
    assert energy(x, x).abs() < 1e-5


def old_view(b):
    """the old TreeBatch fields the old modules read."""
    B, N = b.mask.shape
    adj = torch.zeros(B, N, N)
    k, j = torch.nonzero(b.parent >= 0, as_tuple=True)
    adj[k, b.parent[k, j], j] = 1.0
    return SimpleNamespace(mask=b.mask, node_times=b.time, mrca_times=b.mrca, parent_idx=b.parent, depth=b.depth,
                           adj=adj, max_depth=b.max_depth, observed_mask=b.obs, observed_states=b.x)


def test_matches_old_modules(toy, old):
    """old weights load into the new nets and give the same outputs."""
    from splitters.bridge_tree_flow import BridgeDeltaTreeFlow
    from splitters.bridges.vpsde import DiffusionBridge
    from splitters.tree_former import DeltaTreeFormer
    torch.manual_seed(0)
    o_imp, o_smp = DeltaTreeFormer(4, d_internal=16, n_layers=2, hidden=32), BridgeDeltaTreeFlow(4, d_internal=16, n_layers=2, hidden=32, bridge=DiffusionBridge())
    with torch.no_grad():
        for m in (o_imp, o_smp):
            m.attention_bias_scale.normal_()
    imp, smp = TreeNet(4, d=16, layers=2, hidden=32), TreeNet(4, d=16, layers=2, hidden=32, flow=True)
    imp.load_state_dict(o_imp.state_dict())
    smp.load_state_dict(o_smp.state_dict())
    b = collate(subtrees(toy, 7.0, n=8), toy.bank.emb)
    ob = old_view(b)
    assert torch.allclose(baseline(b, b.x, b.obs), o_smp._output_baseline(ob, b.x, b.obs), atol=1e-5)
    torch.manual_seed(1)
    want = o_imp.pool_batch(ob)
    torch.manual_seed(1)
    assert torch.allclose(impute(imp, b), want, atol=1e-4)
    torch.manual_seed(2)
    want = o_smp.split_layer_batch(ob, b.x, b.obs, n_steps=4)
    torch.manual_seed(2)
    assert torch.allclose(sample(smp, b, b.x, b.obs, n_steps=4), want, atol=1e-3)


def test_train_smoke(toy, tmp_path):
    from li.train.loop import step
    cfg = OmegaConf.create(dict(
        model=dict(d=16, heads=2, layers=1, hidden=32),
        train=dict(n_trees=8, max_nodes=64, lookback=[0.0, 0.5], min_snap_time=7.0),
        align=dict(weights=[1.0, 0.25, 0.25], n_ref=64, n_samples=4, min_samples=2)))
    torch.manual_seed(0)
    np.random.seed(0)
    imp, smp = TreeNet(4, **cfg.model), TreeNet(4, flow=True, **cfg.model)
    opt = torch.optim.Adam(list(imp.parameters()) + list(smp.parameters()), 1e-3)
    for i in range(6):
        loss, m = step(i, toy, imp, smp, cfg)
        assert torch.isfinite(loss) and m["split"] > 0
        opt.zero_grad()
        loss.backward()
        opt.step()

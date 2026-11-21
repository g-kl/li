import pickle

import numpy as np
import pandas as pd
import torch

from li.batch import collate
from li.model import sample


def chunks(items, max_b=256, budget=2e8):
    """batches of similar-size subtrees, keeping B * N^2 under budget."""
    items = sorted(items, key=lambda it: len(it[1]))
    out, cur = [], []
    for it in items:
        if cur and (len(cur) >= max_b or (len(cur) + 1) * len(it[1]) ** 2 > budget):
            out.append(cur)
            cur = []
        cur.append(it)
    return out + [cur] if cur else out


@torch.no_grad()
def couple_tree(net, tree, bank, n_steps=8, gap=0.5, min_snap_time=7.0):
    """paper procedure: from the latest observation time back, sample each layer
    from a subtree rooted gap earlier and anchored on the layer below, then snap
    it to the nearest real cell (own time and one either side)."""
    cell = tree.cell.copy()
    dist = np.full(len(tree), np.nan)
    near = np.abs(tree.time[:, None] - bank.times[None]).argmin(1)
    levels = tree.levels()
    for bi in range(len(bank.times) - 1, -1, -1):
        t_root = bank.times[bi] - gap
        if t_root < tree.time[0]:
            continue
        targets = np.flatnonzero(~tree.leaf & (tree.time >= min_snap_time) & (near == bi))
        if not len(targets):
            continue
        # root each target at its latest ancestor alive at t_root
        a = targets.copy()
        while True:
            mv = (tree.time[a] > t_root) & (tree.parent[a] >= 0)
            if not mv.any():
                break
            a[mv] = tree.parent[a[mv]]
        ok = tree.time[a] <= t_root
        is_target = np.zeros(len(tree), dtype=bool)
        is_target[targets[ok]] = True
        # clip below nodes that already have a state
        has = cell >= 0
        cov = np.zeros(len(tree), dtype=bool)
        for lv in levels[1:]:
            p = tree.parent[lv]
            cov[lv] = cov[p] | has[p]
        items = []
        for r in np.unique(a[ok]):
            nd = np.arange(r, tree.end[r])
            items.append((tree, nd[~cov[nd]], cell))
        nodes, states = [], []
        for ch in chunks(items):
            b = collate(ch, bank.emb)
            x = sample(net, b, b.x, b.obs, n_steps)
            for k, (_, nd) in enumerate(b.items):
                pos = np.flatnonzero(is_target[nd])
                nodes.append(nd[pos])
                states.append(x[k, torch.as_tensor(pos, device=x.device)])
        nodes = np.concatenate(nodes)
        rows, d = bank.nn(torch.cat(states), torch.as_tensor(tree.time[nodes], device=bank.emb.device), window=1)
        cell[nodes] = rows.cpu().numpy()
        dist[nodes] = d.cpu().numpy()
    done = ~tree.leaf & (cell >= 0)
    return pd.DataFrame({"tree": tree.name, "node": tree.nodes[done], "time": tree.time[done],
                         "cell": bank.names[cell[done]], "dist": dist[done]})


def couple(net, data, n_steps=8, seed=0):
    torch.manual_seed(seed)
    net.eval()
    return pd.concat([couple_tree(net, t, data.bank, n_steps) for t in data.trees.values()], ignore_index=True)


def read_legacy(path):
    """old {tree: {node: cell}, nn_distances: {tree: {node: squared l2}}} pickle."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    nd = d.pop("nn_distances", {})
    rows = [(t, n, c, np.sqrt(nd.get(t, {}).get(n, np.nan))) for t, a in d.items() for n, c in a.items()]
    return pd.DataFrame(rows, columns=["tree", "node", "cell", "dist"])

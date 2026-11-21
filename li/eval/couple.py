import pickle

import numpy as np
import pandas as pd
import torch

from li.data.batch import collate


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


def layers(tree, bank, gap=0.5, min_snap_time=7.0):
    """observation time (index into bank.times) each node is coupled at, -1 if it isn't."""
    near = np.abs(tree.time[:, None] - bank.times[None]).argmin(1)
    ok = ~tree.leaf & (tree.time >= min_snap_time) & (bank.times[near] - gap >= tree.time[0])
    return np.where(ok, near, -1)


@torch.no_grad()
def couple_tree(fill, tree, bank, gap=0.5, min_snap_time=7.0):
    """paper procedure: from the latest observation time back, fill each layer
    in a subtree rooted gap earlier and anchored on the layer below, then snap
    it to the nearest real cell (own time and one either side)."""
    cell = tree.cell.copy()
    dist = np.full(len(tree), np.nan)
    layer = layers(tree, bank, gap, min_snap_time)
    levels = tree.levels()
    for bi in range(len(bank.times) - 1, -1, -1):
        targets = np.flatnonzero(layer == bi)
        if not len(targets):
            continue
        # root each target at its latest ancestor alive at t_root
        t_root = bank.times[bi] - gap
        a = targets.copy()
        while True:
            mv = (tree.time[a] > t_root) & (tree.parent[a] >= 0)
            if not mv.any():
                break
            a[mv] = tree.parent[a[mv]]
        is_target = np.zeros(len(tree), dtype=bool)
        is_target[targets] = True
        # clip below nodes that already have a state
        has = cell >= 0
        cov = np.zeros(len(tree), dtype=bool)
        for lv in levels[1:]:
            p = tree.parent[lv]
            cov[lv] = cov[p] | has[p]
        items = []
        for r in np.unique(a):
            nd = np.arange(r, tree.end[r])
            items.append((tree, nd[~cov[nd]], cell))
        nodes, states = [], []
        for ch in chunks(items):
            b = collate(ch, bank.emb)
            x = fill(b, bank)
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


def couple(fill, data, n_samples=4, seed=0):
    """long table (tree, node, time, cell, dist, sample), sample k seeded with seed + k."""
    out = []
    for k in range(n_samples):
        torch.manual_seed(seed + k)
        c = pd.concat([couple_tree(fill, t, data.bank) for t in data.trees.values()], ignore_index=True)
        out.append(c.assign(sample=k))
    return pd.concat(out, ignore_index=True)


def modal_descendant(data, col="germ_layer"):
    """label baseline: each node that would be coupled takes the commonest label among its leaves."""
    out = []
    for t in data.trees.values():
        nd = np.flatnonzero(layers(t, data.bank) >= 0)
        codes, uniq = pd.factorize(data.labels[col].reindex(data.bank.names[t.cell[t.leaf]]))
        oh = np.zeros((len(t) + 1, len(uniq)))
        oh[np.flatnonzero(t.leaf)[codes >= 0] + 1, codes[codes >= 0]] = 1
        cs = oh.cumsum(0)
        cnt = cs[t.end[nd]] - cs[nd]
        lab = np.where(cnt.sum(1) > 0, np.asarray(uniq, dtype=object)[cnt.argmax(1)], None)
        out.append(pd.DataFrame({"tree": t.name, "node": t.nodes[nd], "time": t.time[nd], "label": lab}))
    return pd.concat(out, ignore_index=True)


def read_legacy(path):
    """old {tree: {node: cell}, nn_distances: {tree: {node: squared l2}}} pickle."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    nd = d.pop("nn_distances", {})
    rows = [(t, n, c, np.sqrt(nd.get(t, {}).get(n, np.nan))) for t, a in d.items() for n, c in a.items()]
    return pd.DataFrame(rows, columns=["tree", "node", "cell", "dist"])

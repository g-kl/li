# forward rollout of a perturbation: the mirror of li.eval.couple, marching the time
# grid from start to readout instead of backwards from the leaves. topology is the
# observed tree's, so control and knockout share a node set and only states differ.
import numpy as np
import pandas as pd
import torch

from li.data.batch import collate
from li.eval.couple import chunks
from li.model.inpaint import sample
from li.perturb.project import project

EPS = 1e-6


def layers(tree, bank, lo, hi):
    """time bin index per node over [lo, hi], -1 outside. leaves are in: at the readout
    time they are what we score, and in a perturbed lineage they are re-predicted."""
    near = np.abs(tree.time[:, None] - bank.times[None]).argmin(1)
    ok = (bank.times[near] >= lo - EPS) & (bank.times[near] <= hi + EPS)
    return np.where(ok, near, -1)


def bands(tree, roots, stop, levels):
    """(root, nodes) per root: each root plus its descendants down to the first node in
    `stop`, which is included. a root inside another root's band starts its own."""
    n = len(tree)
    is_root = np.zeros(n, bool)
    is_root[roots] = True
    own = np.where(is_root, np.arange(n), -1)
    go = ~stop | is_root
    for lv in levels[1:]:
        p = tree.parent[lv]
        own[lv] = np.where(is_root[lv], lv, np.where(go[p], own[p], -1))
    sel = np.flatnonzero(own >= 0)
    o = own[sel]
    order = np.argsort(o, kind="stable")
    sel, o = sel[order], o[order]
    return [(int(o[g[0]]), sel[g]) for g in
            np.split(np.arange(len(sel)), np.flatnonzero(np.diff(o)) + 1)]


@torch.no_grad()
def rollout_tree(net, tree, bank, coup, target, gene_col, cutoff, lo, hi,
                 window=1, n_steps=16, roots=None, sham=False):
    """one tree, one draw. `coup` is the sampler coupling's bank row per node (-1 where
    absent); leaves use their observed cell. `roots` limits the start layer to those node
    indices, so only their clades roll forward. returns None if nothing starts."""
    n = len(tree)
    lay = layers(tree, bank, lo, hi)
    levels = tree.levels()
    base = np.where(tree.leaf, tree.cell, coup)  # the unperturbed state of each node
    cur = np.full(n, -1, np.int64)               # the node's state, as a bank row
    chg = np.zeros(n, bool)                      # the gene was on here
    pert = np.zeros(n, bool)                     # an ancestor changed: resample this one
    res = np.zeros(n, bool)                      # state came from the sampler, not a copy
    dist = np.full(n, np.nan)
    proj = np.full(n, np.nan)
    loc = np.full(n, np.nan)
    dev = bank.emb.device

    def assign(nodes, states, root):
        new, d, pj, lc, ch = project(states, tree.time[nodes], bank, target,
                                     gene_col, cutoff, window, sham)
        cur[nodes] = new.cpu().numpy()
        dist[nodes] = d.cpu().numpy()
        proj[nodes] = pj.cpu().numpy()
        loc[nodes] = lc.cpu().numpy()
        chg[nodes] = ch.cpu().numpy()
        if root is not None:
            pert[nodes] = pert[root] | chg[root]

    lo_bin = int(np.abs(bank.times - lo).argmin())
    hi_bin = int(np.abs(bank.times - hi).argmin())
    start = np.zeros(n, bool)
    start[np.arange(n) if roots is None else np.asarray(roots, np.int64)] = True
    seed = np.flatnonzero((lay == lo_bin) & (base >= 0) & start)
    if not len(seed):
        return None
    assign(seed, bank.emb[torch.as_tensor(base[seed], device=dev)], None)
    is_seed = np.zeros(n, bool)
    is_seed[seed] = True
    seed_of = np.where(is_seed, np.arange(n), -1)  # each node's clade, fixed for the run
    for lv in levels[1:]:
        seed_of[lv] = np.where(is_seed[lv], lv, seed_of[tree.parent[lv]])

    for bi in range(lo_bin, hi_bin + 1):
        while True:
            tg = np.flatnonzero((lay == bi) & (cur < 0))
            if not len(tg):
                break
            a = tree.parent[tg].copy()               # nearest ancestor holding a state
            while True:
                mv = (a >= 0) & (cur[np.maximum(a, 0)] < 0)
                if not mv.any():
                    break
                a[mv] = tree.parent[a[mv]]
            tg, a = tg[a >= 0], a[a >= 0]
            if not len(tg):
                break
            root = np.full(n, -1, np.int64)
            root[tg] = a
            run = (pert[a] | chg[a]) & (target is not None)  # perturbed -> resample
            done = int((cur >= 0).sum())

            if run.any():
                want = np.zeros(n, bool)
                want[tg[run]] = True
                is_t = np.zeros(n, bool)
                is_t[tg] = True
                # a band is a whole clade down to this layer: everything already placed is
                # an anchor, the layer itself is free, and nothing past it is pulled in
                stop = is_t | (lay > bi) | (lay < 0)
                roots = np.unique(seed_of[tg[run]])
                items = [(tree, nd, cur) for _, nd in bands(tree, roots[roots >= 0], stop, levels)
                         if len(nd) > 1 and want[nd].any()]
                nodes, states = [], []
                for ch_ in chunks(items):
                    b = collate(ch_, bank.emb)
                    x = sample(net, b, b.x, b.obs, n_steps)
                    for k, (_, nd) in enumerate(b.items):
                        pos = np.flatnonzero(want[nd])
                        if len(pos):
                            nodes.append(nd[pos])
                            states.append(x[k, torch.as_tensor(pos, device=dev)])
                if nodes:
                    nd = np.concatenate(nodes)
                    res[nd] = True
                    assign(nd, torch.cat(states), root[nd])

            rest = tg[(~run) & (cur[tg] < 0)]                # copy the observed state
            if len(rest):
                st = np.where(base[rest] >= 0, base[rest], cur[root[rest]])
                assign(rest, bank.emb[torch.as_tensor(st, device=dev)], root[rest])
            if int((cur >= 0).sum()) == done:                # nothing moved, avoid a spin
                break

    # off-manifold score: the path max of proj/local over changed nodes, so thresholding
    # it at any ratio prunes a node together with everything below it.
    own = np.where(chg, proj / np.where(loc > 0, loc, 1e-12), -np.inf)
    score = own.copy()
    for lv in levels[1:]:
        score[lv] = np.maximum(own[lv], score[tree.parent[lv]])

    nd = np.flatnonzero(cur >= 0)
    par = tree.parent[nd]
    return pd.DataFrame({
        "tree": tree.name, "node": tree.nodes[nd], "time": tree.time[nd],
        "parent": np.where(par >= 0, tree.nodes[np.maximum(par, 0)], None),
        "cell": bank.names[cur[nd]], "dist": dist[nd], "proj_dist": proj[nd],
        "local_dist": loc[nd], "changed": chg[nd], "resampled": res[nd],
        "score": np.where(np.isfinite(score[nd]), score[nd], 0.0),
    })


def rollout(net, bank, trees, coup, target, gene_col, cutoff, lo, hi,
            window=1, n_steps=16, n_samples=8, seed=0, roots=None, sham=False):
    """long table (tree, node, time, parent, cell, dist, proj_dist, local_dist, changed,
    score, sample). coup[k][tree] is that draw's coupled bank row per node, roots[tree]
    the clade roots to start from."""
    out = []
    for k in range(n_samples):
        torch.manual_seed(seed + k)
        rows = [rollout_tree(net, t, bank, coup[k][t.name], target, gene_col, cutoff,
                             lo, hi, window, n_steps,
                             None if roots is None else roots[t.name], sham)
                for t in trees]
        rows = [r for r in rows if r is not None]
        out.append(pd.concat(rows, ignore_index=True).assign(sample=k))
    return pd.concat(out, ignore_index=True)

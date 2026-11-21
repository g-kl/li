from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    """B padded subtrees with N node slots each. root is slot 0, parent < child."""
    parent: torch.Tensor  # (B, N) long, -1 for root and padding
    time: torch.Tensor    # (B, N)
    mask: torch.Tensor    # (B, N) real nodes
    leaf: torch.Tensor    # (B, N) leaves of the subtree
    obs: torch.Tensor     # (B, N) nodes with a known state
    x: torch.Tensor       # (B, N, D) known states, 0 elsewhere
    cell: torch.Tensor    # (B, N) bank row of known states, -1 elsewhere
    depth: torch.Tensor   # (B, N) long
    anc: torch.Tensor     # (B, N, N) anc[b, a, n]: a is an ancestor-or-self of n
    mrca: torch.Tensor    # (B, N, N) time of the most recent common ancestor
    max_depth: int
    items: list           # (tree, node indices) per row, to map back


def subsample(tree, nodes, max_nodes):
    """keep a random set of leaves (and their ancestors) so at most max_nodes remain."""
    lp = np.searchsorted(nodes, tree.parent[nodes])
    lp[0] = -1
    n = len(nodes)
    has_kid = np.zeros(n, dtype=bool)
    has_kid[lp[1:]] = True
    leaves = np.flatnonzero(~has_kid)
    k = max(1, int(len(leaves) * max_nodes / n))
    while True:
        keep = np.zeros(n, dtype=bool)
        front = np.random.choice(leaves, size=min(k, len(leaves)), replace=False)
        keep[front] = True
        while len(front):
            front = lp[front]
            front = np.unique(front[front >= 0])
            front = front[~keep[front]]
            keep[front] = True
        if keep.sum() <= max_nodes or k == 1:
            return nodes[keep]
        k = max(1, min(k - 1, int(k * max_nodes / keep.sum())))


def collate(items, emb, max_nodes=None):
    """items are (tree, sorted node indices, cell rows per tree node) triples."""
    if max_nodes:
        items = [(t, subsample(t, nd, max_nodes) if len(nd) > max_nodes else nd, c) for t, nd, c in items]
    B = len(items)
    N = max(len(nd) for _, nd, _ in items)
    parent = np.full((B, N), -1, dtype=np.int64)
    time = np.zeros((B, N), dtype=np.float32)
    cell = np.full((B, N), -1, dtype=np.int64)
    for i, (t, nd, c) in enumerate(items):
        lp = np.searchsorted(nd, t.parent[nd])
        lp[0] = -1
        parent[i, :len(nd)] = lp
        time[i, :len(nd)] = t.time[nd]
        cell[i, :len(nd)] = c[nd]
    dev = emb.device
    parent = torch.from_numpy(parent).to(dev)
    time = torch.from_numpy(time).to(dev)
    cell = torch.from_numpy(cell).to(dev)
    lens = torch.tensor([len(nd) for _, nd, _ in items], device=dev)
    mask = torch.arange(N, device=dev)[None] < lens[:, None]
    obs = cell >= 0
    x = emb[cell.clamp(min=0)] * obs[..., None]

    # leaves: real nodes nobody points to
    has_kid = torch.zeros(B, N + 1, dtype=torch.bool, device=dev)
    has_kid.scatter_(1, torch.where(parent >= 0, parent, N), True)
    leaf = mask & ~has_kid[:, :N]

    # k-hop ancestors by pointer jumping, then root-to-node paths
    me = torch.arange(N, device=dev).expand(B, N)
    up = torch.where(parent >= 0, parent, me)
    hops = [me]
    while True:
        nxt = up.gather(1, hops[-1])
        if (nxt == hops[-1]).all():
            break
        hops.append(nxt)
    hops = torch.stack(hops, -1)                       # (B, N, D)
    depth = (hops[..., 1:] != hops[..., :-1]).sum(-1)  # (B, N)
    D = hops.shape[-1]
    path = hops.gather(2, (depth[..., None] - torch.arange(D, device=dev)).clamp(min=0))  # path[b, n, d] = ancestor at depth d

    # anc[b, a, n] = path[b, n, depth[a]] == a
    da = depth[:, None, :].expand(B, N, N)              # (B, n, a)
    anc = (path.gather(2, da) == me[:, None, :]).transpose(1, 2)
    anc = anc & (depth[:, :, None] <= depth[:, None, :]) & mask[:, :, None] & mask[:, None, :]

    # common ancestors of i and j form a chain from the root, so their count gives the mrca depth
    common = torch.bmm(anc.transpose(1, 2).float(), anc.float()).long()
    m = path.gather(2, (common - 1).clamp(min=0, max=D - 1))
    mrca = time.gather(1, m.view(B, -1)).view(B, N, N)
    return Batch(parent, time, mask, leaf, obs, x, cell, depth, anc, mrca, int(depth.max()), [(t, nd) for t, nd, _ in items])


def max_child_time(b):
    out = torch.full_like(b.time, float("-inf"))
    src = torch.where(b.mask & (b.parent >= 0), b.time, torch.full_like(b.time, float("-inf")))
    return out.scatter_reduce(1, b.parent.clamp(min=0), src, reduce="amax", include_self=True)


def cross_section(b, t, max_child=None):
    """nodes just above each branch the cut at t crosses (leaves count if before t)."""
    if torch.is_tensor(t) and t.dim() == 1:
        t = t[:, None]
    if max_child is None:
        max_child = max_child_time(b)
    return b.mask & (b.time <= t) & (b.leaf | (max_child > t))

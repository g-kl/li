import torch

from li.batch import cross_section, max_child_time

SLICE_EPS = 0.05  # cut just after a bin so nodes sitting on it fall inside


def energy(x, y):
    """energy distance / 2 (v-statistic), same as geomloss SamplesLoss('energy')."""
    def dist(a, b):
        return (a.pow(2).sum(-1)[:, None] + b.pow(2).sum(-1)[None] - 2 * a @ b.T).clamp(min=1e-8).sqrt()
    return dist(x, y).mean() - 0.5 * dist(x, x).mean() - 0.5 * dist(y, y).mean()


def marginal(b, x, bank, n_ref=4096, min_n=8):
    """imputed nodes near each observation time vs real cells observed then."""
    tol = 0.5 * min(b2 - b1 for b1, b2 in zip(bank.times, bank.times[1:]))
    free = b.mask & ~b.obs
    near = bank.bucket(b.time.flatten()).view_as(b.time)
    gap = (b.time - torch.as_tensor(bank.times, device=b.time.device, dtype=b.time.dtype)[near]).abs()
    loss = x.new_zeros(())
    for i in range(len(bank.times)):
        q = x[free & (near == i) & (gap <= tol)]
        ref = bank.emb[bank.sample(i, n_ref)]
        if len(q) >= min_n and len(ref) >= min_n:
            loss = loss + energy(q, ref)
    return loss


def slice_draw(b, cs, start, q):
    """q nodes on cross-section cs inside each start node's subtree, without replacement if possible."""
    B, M, N = *start.shape, b.mask.shape[1]
    ok = b.anc.gather(1, start[..., None].expand(B, M, N)) & cs[:, None]
    count = ok.sum(-1)
    wo = torch.rand(B, M, N, device=ok.device).masked_fill(~ok, float("-inf")).topk(q, -1).indices
    w = torch.where((count == 0)[..., None], torch.ones_like(ok, dtype=torch.float), ok.float())
    wr = torch.multinomial(w.view(B * M, N), q, replacement=True).view(B, M, q)
    return torch.where((count >= q)[..., None], wo, wr), count >= 1


def slice_features(b, x, grid, n_samples=16):
    """path (one lineage through every grid time) and common-ancestor
    (ancestor at t_i, two descendants at t_j) draws, keyed by stratum."""
    comps = {}
    if len(grid) >= 2:
        comps[("path", *grid)] = [(t, 1) for t in grid]
    for i, ti in enumerate(grid):
        for tj in grid[i + 1:]:
            comps[("ca", ti, tj)] = [(ti, 1), (tj, 2)]
    B, D = len(x), x.shape[-1]
    mc = max_child_time(b)
    bi = torch.arange(B, device=x.device).view(B, 1, 1)
    out = {}
    for key, levels in comps.items():
        cur = torch.zeros(B, n_samples, dtype=torch.long, device=x.device)  # root is slot 0
        idx, valid = [], None
        for k, (t, q) in enumerate(levels):
            i, v = slice_draw(b, cross_section(b, t + SLICE_EPS, mc), cur, q)
            idx.append(i)
            valid = v if valid is None else valid & v
            cur = i[..., 0]
        idx = torch.cat(idx, -1)
        f = x[bi.expand_as(idx), idx].reshape(B * n_samples, -1)[valid.flatten()]
        if len(f):
            out[key] = f
    return out


def align_loss(batches, xs, bank, time_bins, weights=(1.0, 0.25, 0.25), n_ref=4096, n_samples=16, min_n=8):
    """marginal against the bank for each record, path + ca between the two records."""
    marg = sum(marginal(b, x, bank, n_ref, min_n) for b, x in zip(batches, xs))
    # grid: bins inside the window both records cover
    t_max = min(b.time[b.mask & b.leaf].max().item() for b in batches)
    t_min = max(b.time[b.mask & (b.parent < 0)].max().item() for b in batches)
    tol = 0.5 * min(b2 - b1 for b1, b2 in zip(time_bins, time_bins[1:]))
    grid = [t for t in time_bins if t_min - tol <= t <= t_max + tol]
    f1, f2 = (slice_features(b, x, grid, n_samples) for b, x in zip(batches, xs))
    zero = xs[0].new_zeros(())
    terms = {"marginal": marg, "path": zero, "ca": zero}
    for key in f1.keys() & f2.keys():
        if len(f1[key]) >= min_n and len(f2[key]) >= min_n:
            terms[key[0]] = terms[key[0]] + energy(f1[key], f2[key])
    total = weights[0] * terms["marginal"] + weights[1] * terms["path"] + weights[2] * terms["ca"]
    return total, {k: v.item() for k, v in terms.items()}

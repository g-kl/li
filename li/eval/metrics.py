import numpy as np
import pandas as pd
import torch

from li.data.batch import collate
from li.model.inpaint import sample


def node_labels(c, t, germ):
    """per tree node label from coupling rows c (one tree, one sample): the label column if
    there is one, else germ of the cell. None for nodes not in c."""
    lab = np.full(len(t), None, dtype=object)
    pos = pd.Index(t.nodes).get_indexer(c["node"])
    v = c["label"].values if "label" in c else germ.reindex(c["cell"]).values
    lab[pos[pos >= 0]] = v[pos >= 0]
    return lab


def edges(t, lab, exclude="epiblast"):
    """condensed edges (nearest labelled strict ancestor i -> labelled node j), sources labelled exclude dropped."""
    has = pd.notna(lab)
    up = np.full(len(t), -1)
    for lv in t.levels()[1:]:
        p = t.parent[lv]
        up[lv] = np.where(has[p], p, up[p])
    j = np.flatnonzero(has & (up >= 0))
    i = up[j]
    keep = lab[i] != exclude
    return i[keep], j[keep]


def germ_crossing(coup, trees, germ, bins, exclude="epiblast"):
    """fraction of condensed edges (assigned node -> nearest assigned descendants)
    whose germ layers differ, epiblast sources excluded. pooled ("all") and per
    source bin, per sample if coup has a sample column."""
    if "sample" not in coup:
        coup = coup.assign(sample=0)
    bins = np.asarray(bins, dtype=float)
    out = []
    for s, cs in coup.groupby("sample"):
        src, crossed = [], []
        for name, c in cs.groupby("tree"):
            t = trees[name]
            gl = node_labels(c, t, germ)
            i, j = edges(t, gl, exclude)
            src.append(np.abs(t.time[i][:, None] - bins[None]).argmin(1))
            crossed.append(gl[i] != gl[j])
        src, crossed = np.concatenate(src), np.concatenate(crossed).astype(float)
        out.append((s, "all", len(crossed), crossed.mean() if len(crossed) else np.nan))
        for k in np.unique(src):
            out.append((s, bins[k], int((src == k).sum()), crossed[src == k].mean()))
    return pd.DataFrame(out, columns=["sample", "bin", "n_edges", "rate"])


# clonal distortion: the pool-and-split metric, same constants
OFFSET, TOL, MIN_LEAVES, N_CLONES, MAX_NODES, SEED = 1.0, 0.5, 9, 400, 256, 212121


def energy(x, y, m):
    """energy distance per padded row, the unbiased u-statistic at full scale
    (2 E|x-y| - E|x-x'| - E|y-y'|). not li.train.align.energy, which is the
    half-scale v-statistic and inflates small clouds."""
    x, y = x.double(), y.double()
    w = (m[:, :, None] & m[:, None, :]).double()
    off = w * (1 - torch.eye(m.shape[1], device=m.device, dtype=w.dtype))
    n = m.sum(1).double()
    pair = (n * (n - 1)).clamp(min=1)
    cross = (torch.cdist(x, y) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)
    return 2 * cross - (torch.cdist(x, x) * off).sum((1, 2)) / pair \
                     - (torch.cdist(y, y) * off).sum((1, 2)) / pair


def ancestor_at(t, back):
    """per node, the ancestor closest to its own time minus back, and that gap."""
    tgt = t.time - back
    cur = np.arange(len(t))
    best, gap = cur.copy(), np.abs(t.time - tgt)
    while True:
        p = t.parent[cur]
        live = p >= 0
        if not live.any():
            return best, gap
        cur = np.where(live, p, cur)
        d = np.abs(t.time[cur] - tgt)
        up = live & (d < gap)
        best, gap = np.where(up, cur, best), np.where(up, d, gap)


def clones(data):
    """sampled (tree, leaf, ancestor): the ancestor sits OFFSET back and carries
    at least MIN_LEAVES descendant leaves."""
    out = []
    for name, t in data.trees.items():
        anc, gap = ancestor_at(t, OFFSET)
        nl = t.n_leaves()
        ok = t.leaf & (t.cell >= 0) & (gap <= TOL) & (anc != np.arange(len(t))) & (nl[anc] >= MIN_LEAVES)
        out += [(name, int(g), int(anc[g])) for g in np.flatnonzero(ok)]
    rng = np.random.default_rng(SEED)
    return [out[int(i)] for i in rng.choice(len(out), min(N_CLONES, len(out)), replace=False)]


@torch.no_grad()
def clonal_distortion(net, data, n_steps=8, batch=32):
    """leaf -> ancestor -> leaf round trip: anchor the observed leaves and predict
    the subtree root, then anchor only that predicted root with every leaf masked
    and regenerate them. energy distance to the true leaf cloud, so the loss
    through the single root is what it measures."""
    net.eval()
    torch.manual_seed(SEED)
    picks, out = clones(data), []
    for s in range(0, len(picks), batch):
        chunk = picks[s:s + batch]
        items = [(data.trees[n], np.arange(a, data.trees[n].end[a]), data.trees[n].cell)
                 for n, _, a in chunk]
        b = collate(items, data.bank.emb, MAX_NODES)
        up = sample(net, b, b.x, b.obs, n_steps)
        x, anc = torch.zeros_like(b.x), torch.zeros_like(b.mask)
        x[:, 0], anc[:, 0] = up[:, 0], True
        dn = sample(net, b, x, anc, n_steps)
        idx = torch.argsort((~b.leaf).int(), dim=1, stable=True)[:, :int(b.leaf.sum(1).max())]
        m = b.leaf.gather(1, idx)
        gi = idx[..., None].expand(-1, -1, b.x.shape[-1])
        ed = energy(dn.gather(1, gi), b.x.gather(1, gi), m).cpu().numpy()
        for (name, g, a), e, nl in zip(chunk, ed, m.sum(1).cpu().numpy()):
            t = data.trees[name]
            out.append((name, t.nodes[g], t.time[g], t.time[a], int(nl), float(e)))
    return pd.DataFrame(out, columns=["tree", "leaf", "leaf_time", "internal_time",
                                      "n_descendants", "energy_distance"])


# perturbation sanity check: does the sham keep the control's composition?
READOUT, READOUT_TOL = 9.5, 0.26


def composition(c, labels, col="subtype"):
    """readout label proportions per draw, for leaves at the readout time."""
    x = c[np.abs(c.time - READOUT) < READOUT_TOL]
    return pd.crosstab(x["sample"].values, labels[col].reindex(x.cell).values, normalize="index")


def sham_drift(control, sham, labels, col="subtype"):
    """total variation between each sham draw's readout composition and the
    control's. the sham moves most leaf identities, so this is the check that it
    still matches the control in aggregate and is a fair baseline for the gene."""
    a, b = composition(control, labels, col).mean(), composition(sham, labels, col)
    cols = a.index.union(b.columns)
    a, b = a.reindex(cols, fill_value=0.0), b.reindex(columns=cols, fill_value=0.0)
    return (b.sub(a, axis=1).abs().sum(1) / 2).rename("tv")

import numpy as np
import pandas as pd


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

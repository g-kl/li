import numpy as np
import pandas as pd


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
            gl = np.full(len(t), None, dtype=object)
            pos = pd.Index(t.nodes).get_indexer(c["node"])
            gl[pos[pos >= 0]] = germ.reindex(c["cell"]).values[pos >= 0]
            assigned = np.array([g is not None and g == g for g in gl])
            # nearest assigned strict ancestor
            up = np.full(len(t), -1)
            for lv in t.levels()[1:]:
                p = t.parent[lv]
                up[lv] = np.where(assigned[p], p, up[p])
            j = np.flatnonzero(assigned & (up >= 0))
            i = up[j]
            keep = gl[i] != exclude
            i, j = i[keep], j[keep]
            src.append(np.abs(t.time[i][:, None] - bins[None]).argmin(1))
            crossed.append(gl[i] != gl[j])
        src, crossed = np.concatenate(src), np.concatenate(crossed).astype(float)
        out.append((s, "all", len(crossed), crossed.mean() if len(crossed) else np.nan))
        for k in np.unique(src):
            out.append((s, bins[k], int((src == k).sum()), crossed[src == k].mean()))
    return pd.DataFrame(out, columns=["sample", "bin", "n_edges", "rate"])

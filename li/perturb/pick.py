# clades worth looking at: a gene-expressing progenitor at the start time, and enough
# leaves of a cell type the gene is known to make.
import numpy as np
import pandas as pd

# tbx5 makes the forelimb and the posterior, venous pole of the heart. the other two
# sets are the neighbouring structures it does not make (the hindlimb is tbx4/pitx1,
# the arterial pole is the anterior second heart field), kept as columns: at E8.0 a
# clade almost never holds both, so the contrast is read across clades, not within one.
TARGETS = {
    "forelimb": ["forelimb field"],
    "heart_post": ["atria", "inflow tract", "atrioventricular canal", "left ventricle"],
}
CONTROLS = {
    "hindlimb": ["hindlimb field"],
    "heart_ant": ["outflow tract", "right ventricle", "second heart field (anterior)"],
}
SETS = {**TARGETS, **CONTROLS}


def pick(data, coup, gene_col, cutoff, start=8.0, n_min=20, n=6):
    """n clades per target type: the progenitor expresses the gene in most draws and the
    clade holds at least n_min observed leaves of that type."""
    bank = data.bank
    row = pd.Series(np.arange(len(bank.names)), index=bank.names)
    sub = data.labels["subtype"]
    bi = int(np.abs(bank.times - start).argmin())
    on = coup.assign(on=gene_col[row.reindex(coup.cell).values] > cutoff)
    rows = []
    for name, c in on.groupby("tree"):
        t = data.trees[name]
        agree = c.groupby("node").on.mean()
        pos = pd.Index(t.nodes).get_indexer(agree.index)
        keep = (pos >= 0) & (agree.values > 0.5)
        pos, agree = pos[keep], agree.values[keep]
        at = np.abs(t.time[pos][:, None] - bank.times[None]).argmin(1) == bi
        pos, agree = pos[at], agree[at]
        if not len(pos):
            continue
        lf = np.flatnonzero(t.leaf)
        s = sub.reindex(bank.names[t.cell[lf]]).values
        cum = {}
        for k, v in SETS.items():
            a = np.zeros(len(t) + 1)
            a[lf + 1] = np.isin(s, v)
            cum[k] = a.cumsum()
        nl = t.n_leaves()
        for r, g in zip(pos, agree):
            rows.append(dict(tree=name, root=int(r), node=t.nodes[r], time=float(t.time[r]),
                             n_leaves=int(nl[r]), on=float(g),
                             **{k: int(cum[k][t.end[r]] - cum[k][r]) for k in SETS}))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    picked = []
    for kind in TARGETS:
        got = 0
        for _, c in df[df[kind] >= n_min].sort_values(kind, ascending=False).iterrows():
            if got == n:
                break
            end = data.trees[c.tree].end  # clades in a tree are nested or disjoint
            if all(p["tree"] != c.tree or end[min(p["root"], c.root)] <= max(p["root"], c.root)
                   for p in picked):
                picked.append(dict(c, kind=kind))
                got += 1
    return pd.DataFrame(picked)

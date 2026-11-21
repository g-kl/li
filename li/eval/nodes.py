import numpy as np
import pandas as pd
import torch

from li.eval.metrics import edges, node_labels


def label(coup, labels):
    """coupling rows with the snapped cell's labels."""
    return coup.join(labels, on="cell")


def summarize(coup, col="cell_type"):
    """per node: time, modal label, agreement (its share of samples), label entropy, mean snap distance."""
    out = coup.groupby(["tree", "node"]).agg(time=("time", "first"), dist=("dist", "mean"), k=("sample", "size"))
    n = coup.groupby(["tree", "node", col]).size().rename("n").reset_index()
    p = n.n / n.groupby(["tree", "node"]).n.transform("sum")
    n["h"] = -p * np.log(p)
    top = n.sort_values("n", ascending=False).drop_duplicates(["tree", "node"]).set_index(["tree", "node"])
    out["modal"] = top[col]
    out["agreement"] = top.n / out.k
    out["entropy"] = n.groupby(["tree", "node"]).h.sum()
    return out.reset_index()


def to_anndata(coup, data):
    """one obs per (tree, node, sample) with the snapped cell's latent, umap and labels."""
    import anndata as ad
    obs = label(coup, data.labels).reset_index(drop=True)
    obs.index = obs.tree + "/" + obs.node.astype(str) + "/" + obs["sample"].astype(str)
    a = ad.AnnData(obs=obs)
    rows = torch.as_tensor(pd.Index(data.bank.names).get_indexer(coup.cell), device=data.bank.emb.device)
    a.obsm["X_scvi"] = data.bank.emb[rows].cpu().numpy()
    a.obsm["X_umap"] = data.umap.reindex(coup.cell).values
    return a


def clade(data, coup, tree, r, col="cell_type"):
    """clade nodes x samples of col: coupled nodes from coup, leaves observed, None elsewhere."""
    nd = np.arange(r, tree.end[r])
    c = coup[coup.tree == tree.name]
    pos = pd.Index(tree.nodes[nd]).get_indexer(c.node)
    k = pos >= 0
    out = np.full((len(nd), coup["sample"].max() + 1), None, dtype=object)
    out[pos[k], c["sample"].values[k]] = c[col].values[k]
    cells = data.bank.names[tree.cell[nd[tree.leaf[nd]]]]
    out[tree.leaf[nd]] = (cells if col == "cell" else data.labels[col].reindex(cells).values)[:, None]
    return out


def crossing(data, coup, tree, r):
    """per clade node, the share of samples whose edge into it crosses germ layers."""
    nd = np.arange(r, tree.end[r])
    out = np.zeros(len(nd))
    for _, c in coup[coup.tree == tree.name].groupby("sample"):
        gl = node_labels(c, tree, data.labels["germ_layer"])
        i, j = edges(tree, gl)
        x = (i >= r) & (j < tree.end[r])
        np.add.at(out, j[x] - r, gl[i[x]] != gl[j[x]])
    return out / (coup["sample"].max() + 1)


def clades(data, coup, n=8, seed=0):
    """non-overlapping clades of 30 to 150 leaves rooted in E7.0 to E8.0: n at random, round robin
    over embryo stages, then the 3 with the most germ crossing."""
    rows = []
    for t in data.trees.values():
        nl = t.n_leaves()
        rs = np.flatnonzero(~t.leaf & (nl >= 30) & (nl <= 150) & (t.time >= 7.0) & (t.time <= 8.0))
        e, x = np.zeros(len(t) + 1), np.zeros(len(t) + 1)  # edges and crossings by source, cumulative
        for _, c in coup[coup.tree == t.name].groupby("sample"):
            gl = node_labels(c, t, data.labels["germ_layer"])
            i, j = edges(t, gl)
            np.add.at(e, i + 1, 1)
            np.add.at(x, i + 1, gl[i] != gl[j])
        e, x = e.cumsum(), x.cumsum()
        for r in rs:
            ne = e[t.end[r]] - e[r]
            rows.append((t.name, t.leaf_time, r, t.nodes[r], nl[r], ne, (x[t.end[r]] - x[r]) / max(ne, 1)))
    cand = pd.DataFrame(rows, columns=["tree", "stage", "root", "node", "n_leaves", "n_edges", "crossing"])
    rng = np.random.default_rng(seed)
    cand["order"] = rng.permutation(len(cand))
    cand["turn"] = cand.sort_values("order").groupby("stage").cumcount()
    picked = []
    for why, df, k in (("random", cand.sort_values(["turn", "order"]), n),
                       ("worst", cand[cand.n_edges >= 20].sort_values("crossing", ascending=False), 3)):
        for _, c in df.iterrows():
            if sum(p["why"] == why for p in picked) == k:
                break
            # clades in a tree are nested or disjoint
            end = data.trees[c.tree].end
            if all(p["tree"] != c.tree or end[min(p["root"], c.root)] <= max(p["root"], c.root) for p in picked):
                picked.append(dict(c.drop(["order", "turn"]), why=why))
    return pd.DataFrame(picked)

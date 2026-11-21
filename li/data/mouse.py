import os
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

from li.data.bank import Bank
from li.data.tree import Tree, from_nx, prune


def read_trees(path, cache="cache"):
    """raw trees as (name, nodes, parent, time) arrays. the h5td read is slow so cache it."""
    key = os.path.join(cache, f"trees_{os.path.basename(path)}_{int(os.path.getmtime(path))}.pkl")
    if os.path.exists(key):
        with open(key, "rb") as f:
            return pickle.load(f)
    import treedata
    td = treedata.read_h5td(path)
    raw = []
    for name, g in td.obst.items():
        t = from_nx(g, name)
        raw.append((name, t.nodes, t.parent, t.time))
    os.makedirs(cache, exist_ok=True)
    with open(key, "wb") as f:
        pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
    return raw


def read_bank(path, device):
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        emb = f["obsm"]["X_scvi"][:]
    return Bank(emb, obs["time"].astype(float).values, obs.index.values, device), obs


def read_labels(subtype_csv, type_csv):
    """per cell: subtype, cell type, lineage, germ layer (as the paper metrics did)."""
    sub = pd.read_csv(subtype_csv, index_col=0)["cell_subtype"]
    ti = pd.read_csv(type_csv, index_col=0)
    type_to_gl = dict(zip(ti["cell_type"], ti["germ_layer"]))  # last row wins
    out = pd.DataFrame({"subtype": sub})
    out["cell_type"] = sub.map(ti["cell_type"])
    out["lineage"] = sub.map(ti["lineage"])
    out["germ_layer"] = out["cell_type"].map(type_to_gl)
    return out


def load(root, time_bins, max_leaves, device="cpu"):
    emb = os.path.join(root, "embryos")
    bank, obs = read_bank(os.path.join(emb, "devmap_latents.h5ad"), device)
    rows = pd.Series(np.arange(len(bank.names)), index=bank.names)
    trees = {}
    for name, nodes, parent, time in read_trees(os.path.join(emb, "final_topology.h5td")):
        cell = rows.reindex(nodes).fillna(-1).values.astype(np.int64)
        t = Tree(name, nodes, parent, time, np.where(np.isin(np.arange(len(parent)), parent), -1, cell))
        t = prune(t)
        t.build_index(time_bins, max_leaves)
        trees[name] = t
    labels = read_labels(os.path.join(emb, "final_subtype_v2_lower.csv"), os.path.join(emb, "final_type_info_v2_lower.csv"))
    return SimpleNamespace(trees=trees, bank=bank, ref=bank, clonal=obs["clone"].notna().values,
                           labels=labels, obs=obs, time_bins=list(time_bins))

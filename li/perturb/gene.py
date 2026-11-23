# gene expression per bank cell. whatever the source, this hands back a counts-shaped
# column, so the gene-off set is always `col <= cutoff`.
import os

import numpy as np
import torch

# the old states h5ad, for source="counts". not part of the zenodo bundle.
COUNTS = "/mnt/home/gokulg/data/embryos/qiu24_integrated_hvgs_and_scvi.h5ad"


def decoder(path, device, cov_name="devmap"):
    """the scvi decoder as a function of the latent alone, rebuilt from the checkpoint.
    scvi.model.SCVI.load would want the full counts adata back."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    covs = list(ck["attr_dict"]["registry_"]["field_registries"]
                ["extra_categorical_covs"]["state_registry"]["mappings"]["dataset"])
    cov = torch.zeros(1, len(covs), device=device)
    cov[0, covs.index(cov_name)] = 1.0
    layers, i = [], 0
    while f"decoder.px_decoder.fc_layers.Layer {i}.0.weight" in sd:
        q = f"decoder.px_decoder.fc_layers.Layer {i}"
        layers.append(tuple(sd[f"{q}.{k}"].to(device) for k in
                            ("0.weight", "0.bias", "1.weight", "1.bias",
                             "1.running_mean", "1.running_var")))
        i += 1
    W = sd["decoder.px_scale_decoder.0.weight"].to(device)
    B = sd["decoder.px_scale_decoder.0.bias"].to(device)

    def run(z):
        h = z
        for w, b, g, bb, m, v in layers:  # linear, then batchnorm (eps 1e-3), then relu
            h = torch.nn.functional.linear(torch.cat([h, cov.expand(len(h), -1)], 1), w, b)
            h = torch.relu((h - m) / (v + 1e-3).sqrt() * g + bb)
        return torch.softmax(h @ W.T + B, -1)

    return run, list(ck["var_names"])


def decoded(bank, gene, root, device, chunk=200_000):
    """scvi px_scale for one gene over every bank cell, per million."""
    run, var = decoder(os.path.join(root, "scvi_model", "model.pt"), device)
    j = var.index(gene)
    out = np.empty(len(bank.names), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(out), chunk):
            z = bank.emb[s:s + chunk].to(device)
            out[s:s + chunk] = run(z)[:, j].float().cpu().numpy()
    return out * 1e6


def counts(bank, gene, path=COUNTS, chunk=50_000):
    """raw counts for one gene, reindexed onto the bank. the csr is 2.2e9 nnz, so pull
    the one column out by scanning the row blocks."""
    import h5py
    import pandas as pd
    with h5py.File(path, "r") as f:
        txt = lambda d: np.array([x.decode() if isinstance(x, bytes) else x for x in d])
        j = list(txt(f["var"][f["var"].attrs["_index"]][:])).index(gene)
        names = txt(f["obs"][f["obs"].attrs["_index"]][:])
        X = f["layers"]["counts"]
        ptr = X["indptr"][:]
        col = np.zeros(len(names), dtype=np.float32)
        for s in range(0, len(names), chunk):
            e = min(s + chunk, len(names))
            lo, hi = int(ptr[s]), int(ptr[e])
            hit = np.flatnonzero(X["indices"][lo:hi] == j)
            if len(hit):
                rows = np.searchsorted(ptr[s:e + 1], hit + lo, side="right") - 1 + s
                col[rows] = X["data"][lo:hi][hit]
    return pd.Series(col, index=names).reindex(bank.names).fillna(0).values.astype(np.float32)


def column(bank, gene, source, root, device="cuda", cache="cache"):
    """gene expression per bank cell, cached. decoder: scvi px_scale per million,
    from the bundle alone. counts: raw counts from the old states h5ad."""
    os.makedirs(cache, exist_ok=True)
    key = os.path.join(cache, f"gene_{source}_{gene}.npy")
    if os.path.exists(key):
        col = np.load(key)
        if len(col) == len(bank.names):
            return col
    col = decoded(bank, gene, root, device) if source == "decoder" else counts(bank, gene)
    np.save(key, col)
    return col


def off_mask(col, bank, cutoff, lo, hi):
    """cells with the gene off. a gene with none at some time in the window has an empty
    projection target there, which is the old genes_with_off_reference filter."""
    off = col <= cutoff
    for i, t in enumerate(bank.times):
        if lo - 1e-6 <= t <= hi + 1e-6:
            n = int(off[bank.rows[i].cpu().numpy()].sum())
            assert n, f"no gene-off cell at t={t}"
    return off

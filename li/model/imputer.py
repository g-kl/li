import torch.nn as nn

from li.model.inpaint import baseline, impute
from li.model.net import TreeNet


def snap(b, x, keep, bank, min_snap_time=7.0):
    """free nodes (not observed, not kept) from min_snap_time on become their nearest real cell."""
    m = b.mask & ~b.obs & ~keep & (b.time >= min_snap_time)
    rows, _ = bank.nn(x[m].detach(), b.time[m])
    out = x.clone()
    out[m] = bank.emb[rows]
    return out


class Learned(TreeNet):
    """one-step transformer on top of mean pool, trained by the alignment loss."""

    def fill(self, b, bank):
        return impute(self, b)


class MeanPool(nn.Module):
    """nonparametric: hierarchical mean of the observed states, snapped to real cells."""

    def __init__(self, dim, min_snap_time=7.0):  # dim unused, every imputer is built from it
        super().__init__()
        self.min_snap_time = min_snap_time

    def fill(self, b, bank):
        return snap(b, baseline(b, b.x, b.obs), b.obs, bank, self.min_snap_time)

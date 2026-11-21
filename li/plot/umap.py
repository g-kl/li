import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from li.plot.tree import GREY


def atlas(ax, umap, n=200_000, seed=0):
    """light grey background of n random cells."""
    u = umap.sample(min(n, len(umap)), random_state=seed).values
    ax.scatter(u[:, 0], u[:, 1], s=0.3, c="#e6e6e6", lw=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])


def path(ax, umap, tree, r, cells, colors, leaf_lab, cmap="viridis"):
    """the clade's leaves in type colour and, per sample, each snapped ancestor joined to its
    nearest snapped ancestor, coloured by node time. zooms to the clade."""
    nd = np.arange(r, tree.end[r])
    t, lf = tree.time[nd], tree.leaf[nd]
    norm = Normalize(t[~lf].min(), t.max())
    K = cells.shape[1]
    pts = []
    for k in range(K):
        xy = umap.reindex(cells[:, k]).values
        has = ~np.isnan(xy[:, 0])
        up = np.full(len(nd), -1)
        for j in range(1, len(nd)):  # preorder, so a parent's up is set first
            p = tree.parent[nd[j]] - r
            up[j] = p if has[p] else up[p]
        j = np.flatnonzero(has & (up >= 0))
        segs = np.stack([xy[up[j]], xy[j]], 1)
        ax.add_collection(LineCollection(segs, colors=colormaps[cmap](norm(t[up[j]])), linewidths=0.6,
                                         alpha=max(0.2, 1 / K)))
        a = has & ~lf
        ax.scatter(xy[a, 0], xy[a, 1], s=6, c=t[a], cmap=cmap, norm=norm, lw=0, alpha=max(0.3, 1 / K), zorder=3)
        pts.append(xy[has])
    xy = umap.reindex(cells[lf, 0]).values
    ax.scatter(xy[:, 0], xy[:, 1], s=10, c=[colors.get(v, GREY) for v in leaf_lab], lw=0.3,
               edgecolors="white", zorder=4)
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.15 * (hi - lo).max() + 0.5
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    return norm

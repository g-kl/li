import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

GREY = "#c8c8c8"


def layout(tree, r):
    """y per node of the clade at r: leaves 0, 1, ... in preorder, parents at their children's mean."""
    nd = np.arange(r, tree.end[r])
    lf = tree.leaf[nd]
    y = np.where(lf, np.cumsum(lf) - 1, 0).astype(float)
    s, c = np.zeros(len(nd)), np.zeros(len(nd))
    for k in range(len(nd) - 1, -1, -1):  # children come after their parent
        if not lf[k]:
            y[k] = s[k] / c[k]
        if k:
            p = tree.parent[nd[k]] - r
            s[p] += y[k]
            c[p] += 1
    return y


def modal(lab):
    """modal label per row (None if no sample has one) and its share of samples."""
    m, share = np.full(len(lab), None, dtype=object), np.zeros(len(lab))
    for i, row in enumerate(lab):
        v = pd.Series(row).dropna().value_counts()
        if len(v):
            m[i], share[i] = v.index[0], v.iloc[0] / len(row)
    return m, share


def card(ax, tree, r, lab, colors, cross=None, lw=(0.4, 2.4)):
    """the clade at r over time. lab is clade nodes x samples (None = no state). edges take the
    child's modal colour, faded by disagreement; each coupled node gets a tick per sample; leaves
    are dots; red x where an edge crosses germ layers (alpha = share of samples)."""
    nd = np.arange(r, tree.end[r])
    y, t, lf = layout(tree, r), tree.time[nd], tree.leaf[nd]
    m, share = modal(lab)
    col = np.array([to_rgba(colors.get(v, GREY), 0.2 + 0.8 * a) if v is not None else to_rgba(GREY)
                    for v, a in zip(m, share)])
    nl = tree.n_leaves()[nd]
    w = lw[0] + (lw[1] - lw[0]) * np.log(nl) / np.log(max(nl.max(), 2))
    par = tree.parent[nd[1:]] - r
    lo, hi = y.copy(), y.copy()
    np.minimum.at(lo, par, y[1:])
    np.maximum.at(hi, par, y[1:])
    segs = [[(t[p], y[k]), (t[k], y[k])] for k, p in zip(range(1, len(nd)), par)]
    segs += [[(t[k], lo[k]), (t[k], hi[k])] for k in np.flatnonzero(~lf)]
    ax.add_collection(LineCollection(segs, colors=np.concatenate([col[1:], col[~lf]]),
                                     linewidths=np.concatenate([w[1:], w[~lf]]), capstyle="round"))
    # one tick per sample, stepping back along the incoming edge
    span = t.max() - t.min()
    k, s = np.nonzero((~lf)[:, None] & pd.notna(lab))
    ax.scatter(t[k] - (s + 1) * 0.01 * span, y[k], s=4, marker="s", lw=0,
               c=[colors.get(v, GREY) for v in lab[k, s]], zorder=3)
    ax.scatter(t[lf], y[lf], s=8, lw=0, c=[colors.get(v, GREY) for v in lab[lf, 0]], zorder=3)
    if cross is not None and (cross > 0).any():
        x = cross > 0
        ax.scatter(t[x], y[x], s=14, marker="x", lw=0.8, c=[(0.85, 0.1, 0.1, a) for a in cross[x]], zorder=4)
    ax.set_xlim(t.min() - 0.06 * span, t.max() + 0.03 * span)
    ax.set_ylim(-1, y.max() + 1)
    ax.set_yticks([])
    ax.set_xlabel("time (E)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def key(ax, labs, colors, **kw):
    """legend of the labels that appear in labs."""
    seen = pd.Series(np.ravel(labs)).dropna().value_counts().index
    ax.legend(handles=[Patch(color=colors.get(v, GREY), label=v) for v in seen], frameon=False, **kw)

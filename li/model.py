import torch

from li import vpsde
from li.batch import cross_section


def baseline(b, x, anchor):
    """felsenstein-style displacement: anchors copied down to descendants,
    hierarchical mean of children above them."""
    anchor = anchor & b.mask
    par = b.parent.clamp(min=0)
    below = x * anchor[..., None]
    has = anchor.clone()
    for d in range(1, b.max_depth + 1):
        fill = (b.depth == d) & b.mask & ~anchor & has.gather(1, par)
        below = torch.where(fill[..., None], below.gather(1, par[..., None].expand_as(x)), below)
        has = has | fill

    above = b.mask & ~has
    up = x * anchor[..., None]
    kid = (b.mask & (b.parent >= 0)).float()
    n_kids = torch.zeros_like(b.time).scatter_add(1, par, kid).clamp(min=1)[..., None]
    for d in range(b.max_depth - 1, -1, -1):
        tgt = (b.depth == d) & above
        sums = torch.zeros_like(up).scatter_add(1, par[..., None].expand_as(up), up * kid[..., None])
        up = torch.where(tgt[..., None], sums / n_kids, up)
    return torch.where(above[..., None], up, below) * b.mask[..., None]


def impute(net, b):
    """one pass: observed nodes clamped, the rest start as noise."""
    anchor = b.obs & b.mask
    x = torch.where(anchor[..., None], b.x, torch.randn_like(b.x))
    pred = net(x, b, anchor) + baseline(b, b.x, anchor)
    return torch.where(anchor[..., None], b.x, pred) * b.mask[..., None]


def sampler_loss(net, b, target, anchor):
    """x1-prediction mse on the non-anchor nodes at a random flow time."""
    anchor = anchor & b.mask
    base = baseline(b, target, anchor)
    t = torch.rand(len(target), device=target.device)
    xt = vpsde.interpolate(torch.randn_like(target), target - base, t)
    pred = net(torch.where(anchor[..., None], target, xt), b, anchor, t) + base
    m = (~anchor & b.mask)[..., None]
    return ((pred - target) ** 2 * m).sum() / (m.sum().clamp(min=1) * target.shape[-1])


@torch.no_grad()
def sample(net, b, x, anchor, n_steps=8):
    """fill in every non-anchor node given anchor states x, by reverse sde."""
    anchor = anchor & b.mask
    a, free = anchor[..., None], (b.mask & ~anchor)[..., None]
    base = baseline(b, x, anchor)
    xt = torch.where(a, x, torch.randn_like(x))
    ss = torch.linspace(1.0, 0.0, n_steps + 1, device=x.device)
    for k in range(n_steps):
        pred = net(xt, b, anchor, (1 - ss[k]).expand(len(x)))
        xt = torch.where(free, vpsde.step(ss[k], ss[k + 1], xt, pred), xt)
    return torch.where(a, x, pred + base) * b.mask[..., None]


def anchor_mask(b):
    """a cross-section at a random cut per tree; past the last node it is the leaf layer."""
    t = b.time.masked_fill(~b.mask, float("inf")).amin(1)
    hi = b.time.masked_fill(~b.mask, float("-inf")).amax(1)
    span = (hi - t).clamp(min=1e-6)
    cut = t + torch.rand(len(t), device=t.device) * span * (1 + 1 / max(b.max_depth, 1))
    return cross_section(b, cut)

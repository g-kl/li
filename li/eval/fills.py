import torch

from li.model.inpaint import baseline, sample


def sampler(net, n_steps=8):
    """the paper procedure: one reverse-sde draw per layer."""
    net.eval()
    return lambda b, bank: sample(net, b, b.x, b.obs, n_steps)


def posterior_mean(net, k=8, n_steps=8):
    """mean of k sampler draws, the like-for-like counterpart to mean pool."""
    net.eval()
    return lambda b, bank: torch.stack([sample(net, b, b.x, b.obs, n_steps) for _ in range(k)]).mean(0)


def mean_pool(b, bank):
    """hierarchical mean of the anchors below. left unsnapped so the coupling records its distance."""
    return baseline(b, b.x, b.obs)


def random_descendant(b, bank):
    """state of a random anchor below each node."""
    below = b.anc & b.obs[:, None, :]
    j = torch.rand(below.shape, device=b.x.device).masked_fill(~below, -1).argmax(-1)
    return b.x.gather(1, j[..., None].expand_as(b.x))


def random_cell(b, bank):
    """a random real cell observed at each node's time (chance floor)."""
    bi = bank.bucket(b.time.flatten())
    rows = torch.empty_like(bi)
    for i in bi.unique().tolist():
        sel = bi == i
        r = bank.rows[i]
        rows[sel] = r[torch.randint(len(r), (int(sel.sum()),), device=r.device)]
    return bank.emb[rows].view_as(b.x)

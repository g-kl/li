# the projection perturbation: snap a state to the nearest real cell, then onto the
# nearest cell that does not express the gene. every state stays a real cell.
import numpy as np
import torch

from li.data.bank import Bank


def target_bank(bank, off, device):
    """bank over the gene-off cells, plus their rows in the full bank."""
    idx = np.flatnonzero(off)
    rows = torch.as_tensor(idx, device=bank.emb.device)
    return Bank(bank.emb[rows], bank.time[idx], bank.names[idx], device), idx


def project(states, times, bank, target, gene_col, cutoff, window=1, sham=False):
    """returns (new bank row, dist, proj_dist, local_dist, changed) per state.

    dist is the raw state's distance to the real manifold, before anything moves.
    proj_dist is how far the snapped cell had to travel to stop expressing the gene,
    local_dist the distance to the nearest other real cell there. off-manifold is
    proj_dist > ratio * local_dist, decided post hoc.
    """
    dev = bank.emb.device
    t = torch.as_tensor(np.asarray(times), device=dev)
    cell, dist = bank.nn(states, t, window=window)
    if target is None:  # control: nothing moves
        z = torch.zeros(len(cell), device=dev)
        return cell, dist, z, z + float("nan"), z.bool()
    tb, idx = target
    snapped = bank.emb[cell]
    changed = torch.as_tensor(gene_col[cell.cpu().numpy()] > cutoff, device=dev)
    _, d2 = bank.nn(snapped, t, window=window, k=2)  # [:, 1] is the nearest other cell
    if sham:  # the same lineages resample, but nothing moves: pure rollout drift
        return cell, dist, torch.zeros_like(dist), d2[:, 1], changed
    rows, proj = tb.nn(snapped, t, window=window)
    new = torch.as_tensor(idx, device=dev)[rows]
    return new, dist, proj, d2[:, 1], changed

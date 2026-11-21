import copy
import csv
import glob
import logging
import os
import time

import numpy as np
import torch
from hydra.utils import instantiate

from li.data.bank import Bank
from li.data.batch import collate
from li.model.inpaint import anchor_mask, impute, sampler_loss
from li.model.net import TreeNet
from li.train.align import align_loss

log = logging.getLogger(__name__)


def pick_records(trees, gap=0.5, tol=0.1):
    """two trees whose leaf times differ by about gap, earlier one first."""
    names = list(trees)
    lt = {n: trees[n].leaf_time for n in names}
    partners = {n: [m for m in names if m != n and abs(lt[m] - lt[n] - gap) <= tol] for n in names}
    first = [n for n in names if partners[n]]
    a = first[np.random.randint(len(first))]
    return trees[a], trees[partners[a][np.random.randint(len(partners[a]))]]


def draw(tree, b, n):
    roots = tree.index[b]
    roots = roots if len(roots) <= n else np.random.choice(roots, n, replace=False)
    return [(tree, tree.subtree(r, b), tree.cell) for r in roots]


def nn_pair(b, x, anchor, bank, min_snap_time):
    """snap imputed nodes (not observed, not anchors) to their nearest real cell."""
    m = b.mask & ~b.obs & ~anchor & (b.time >= min_snap_time)
    rows, _ = bank.nn(x[m].detach(), b.time[m])
    out = x.clone()
    out[m] = bank.emb[rows]
    return out


def nets(dim, cfg):
    return TreeNet(dim, **cfg), TreeNet(dim, flow=True, **cfg)


def step(i, data, imp, smp, cfg):
    tr = cfg.train
    while True:
        recs = pick_records(data.trees)
        lb = tr.lookback[np.random.randint(len(tr.lookback))]
        b = min(data.time_bins, key=lambda t: abs(t - (recs[0].leaf_time - lb)))
        if all(len(r.index.get(b, [])) for r in recs):
            break
    batches = [collate(draw(r, b, tr.n_trees), data.bank.emb, tr.max_nodes) for r in recs]
    xs = [impute(imp, bt) for bt in batches]
    align, terms = align_loss(batches, xs, data.ref, data.time_bins, cfg.align.weights,
                              cfg.align.n_ref, cfg.align.n_samples, cfg.align.min_samples)
    split = 0
    for bt, x in zip(batches, xs):
        anchor = anchor_mask(bt)
        if i % 2 == 1:  # odd steps train the sampler on real cells
            x = nn_pair(bt, x, anchor, data.bank, tr.min_snap_time)
        split = split + sampler_loss(smp, bt, x, anchor)
    return split + align, dict(split=split.item(), align=align.item(), **terms, bin=b)


def fit(cfg, out):
    """train imputer and sampler on cfg.data, checkpoints and metrics.csv go in out."""
    ckpts = sorted(glob.glob(f"{out}/ckpt_*.pt"), key=lambda p: int(p.split("_")[-1][:-3]))
    if ckpts and not cfg.resume:
        raise SystemExit(f"{out} already has checkpoints, pass resume=true")
    os.makedirs(out, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data = instantiate(cfg.data, device=cfg.device, _convert_="all")
    if cfg.align.clonal:  # marginal against lineage-traced cells only
        keep = data.clonal
        data.ref = Bank(data.bank.emb[torch.as_tensor(keep, device=cfg.device)], data.bank.time[keep], data.bank.names[keep], cfg.device)
    dim = data.bank.emb.shape[1]
    imp, smp = (m.to(cfg.device) for m in nets(dim, cfg.model))
    params = list(imp.parameters()) + list(smp.parameters())
    tr, op = cfg.train, cfg.optimizer
    opt = torch.optim.AdamW(params, lr=op.lr, weight_decay=op.weight_decay, eps=op.eps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / op.warmup))
    ema = [copy.deepcopy(imp), copy.deepcopy(smp)]
    ema_params = [p for m in ema for p in m.parameters()]
    start = 0
    if ckpts:
        ck = torch.load(ckpts[-1], map_location=cfg.device)
        imp.load_state_dict(ck["imputer"]), smp.load_state_dict(ck["sampler"])
        ema[0].load_state_dict(ck["ema_imputer"]), ema[1].load_state_dict(ck["ema_sampler"])
        opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start = ck["step"]
        np.random.seed(cfg.seed + start)
        torch.manual_seed(cfg.seed + start)
        log.info(f"resumed from {ckpts[-1]}")

    f = open(f"{out}/metrics.csv", "a", newline="")
    writer = None
    tick = time.time()
    for i in range(start, tr.n_steps):
        loss, m = step(i, data, imp, smp, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        m["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, op.clip).item()
        opt.step()
        sched.step()
        with torch.no_grad():
            decay = min(op.ema, (i + 2) / (i + 11))  # warm-up decay, as before
            torch._foreach_lerp_(ema_params, params, 1 - decay)
        if i % tr.log_every == 0 or i == tr.n_steps - 1:
            m.update(step=i, sec=(time.time() - tick) / (1 if i == start else tr.log_every))
            if torch.cuda.is_available():
                m["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            tick = time.time()
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(m))
                if f.tell() == 0:
                    writer.writeheader()
            writer.writerow(m)
            f.flush()
            log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
        if (i + 1) % tr.save_every == 0 or i == tr.n_steps - 1:
            torch.save(dict(imputer=imp.state_dict(), sampler=smp.state_dict(),
                            ema_imputer=ema[0].state_dict(), ema_sampler=ema[1].state_dict(),
                            opt=opt.state_dict(), sched=sched.state_dict(), step=i + 1,
                            config=dict(model=dict(cfg.model), dim=dim)), f"{out}/ckpt_{i + 1}.pt")

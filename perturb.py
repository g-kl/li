# forward perturbation rollout for a run dir: control or knockout, K samples
import argparse
import logging
import os
import sys
import time
from types import SimpleNamespace

if sys.version_info >= (3, 14):  # hydra trips over argparse's new help check
    argparse.ArgumentParser._check_help = lambda self, action: None

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from li.eval import fills
from li.eval.couple import couple
from li.model.net import load_sampler
from li.perturb import gene as G
from li.perturb.pick import pick
from li.perturb.project import target_bank
from li.perturb.rollout import rollout

log = logging.getLogger(__name__)
CUTOFF = {"decoder": 100.0, "counts": 0.0}  # px_scale per million, and raw counts


def coupled_rows(coup, trees, bank, n_samples):
    """the coupling's bank row per tree node, per sample."""
    row = pd.Series(np.arange(len(bank.names)), index=bank.names)
    out = []
    for k in range(n_samples):
        ck = coup[coup["sample"] == k]
        d = {}
        for t in trees:
            a = np.full(len(t), -1, np.int64)
            c = ck[ck.tree == t.name]
            pos = pd.Index(t.nodes).get_indexer(c.node)
            a[pos[pos >= 0]] = row.reindex(c.cell).values[pos >= 0]
            d[t.name] = a
        out.append(d)
    return out


@hydra.main(config_path="config", config_name="perturb", version_base=None)
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    data = instantiate(cfg.data, device=cfg.device, _convert_="all")
    net = load_sampler(cfg.ckpt.rstrip("/"), cfg.device)
    trees = [t for t in data.trees.values() if abs(t.leaf_time - cfg.target_time) < 1e-6]
    log.info(f"{len(trees)} trees at E{cfg.target_time}: {[t.name for t in trees]}")

    col = G.column(data.bank, cfg.gene, cfg.source, cfg.data.root, cfg.device)
    cutoff = CUTOFF[cfg.source] if cfg.cutoff is None else cfg.cutoff
    off = G.off_mask(col, data.bank, cutoff, cfg.start_time, cfg.target_time)
    log.info(f"{cfg.gene} ({cfg.source}): {off.mean():.1%} of cells off at <= {cutoff:g}")

    # the coupling gives the start layer and picks the clades; it is gene independent
    cp = os.path.join(out, f"coupling_e{cfg.target_time}_k{cfg.n_samples}.parquet")
    if os.path.exists(cp):
        coup = pd.read_parquet(cp)
        log.info(f"reusing {cp}")
    else:
        tick = time.time()
        one = SimpleNamespace(trees={t.name: t for t in trees}, bank=data.bank)
        coup = couple(fills.sampler(net), one, cfg.n_samples)
        coup.to_parquet(cp)
        log.info(f"coupled {len(coup)} rows in {time.time() - tick:.0f}s")

    clades = pick(data, coup, col, cutoff, cfg.start_time)
    if clades.empty:
        raise SystemExit(f"no clade has a {cfg.gene}-expressing progenitor at E{cfg.start_time}")
    src = "" if cfg.source == "decoder" else f"_{cfg.source}"  # clades do not depend on mode
    clades.to_csv(os.path.join(out, f"perturb_clades_{cfg.gene}{src}.csv"), index=False)
    log.info("picked clades\n" + clades.to_string(index=False))

    use = [t for t in trees if t.name in set(clades.tree)]
    roots = {t.name: clades.root[clades.tree == t.name].values for t in use}
    target = None if cfg.mode == "control" else target_bank(data.bank, off, cfg.device)
    tick = time.time()
    res = rollout(net, data.bank, use, coupled_rows(coup, use, data.bank, cfg.n_samples),
                  target, col, cutoff, cfg.start_time, cfg.target_time,
                  cfg.window, cfg.n_steps, cfg.n_samples, roots=roots,
                  sham=cfg.mode == "sham")
    tag = f"{cfg.gene}_{cfg.mode}{src}"
    res.to_parquet(os.path.join(out, f"perturb_{tag}.parquet"))
    log.info(f"{tag}: {len(res)} rows over {len(use)} trees in {time.time() - tick:.0f}s")


if __name__ == "__main__":
    main()

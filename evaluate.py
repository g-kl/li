# coupling samples + germ crossing for a run dir, an old model dir, or an old coupling pickle (cfg.ckpt)
import argparse
import logging
import os
import sys
import time

if sys.version_info >= (3, 14):  # hydra trips over argparse's new help check
    argparse.ArgumentParser._check_help = lambda self, action: None

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from li.data.tree import scramble
from li.eval import fills
from li.eval.couple import couple, modal_descendant, read_legacy
from li.eval.metrics import germ_crossing
from li.eval.nodes import label, summarize
from li.model.net import load_sampler

log = logging.getLogger(__name__)

FILLS = dict(
    sampler=lambda ck, dev: fills.sampler(load_sampler(ck, dev)),
    posterior_mean=lambda ck, dev: fills.posterior_mean(load_sampler(ck, dev)),
    mean_pool=lambda ck, dev: fills.mean_pool,
    random_descendant=lambda ck, dev: fills.random_descendant,
    random_cell=lambda ck, dev: fills.random_cell,
)


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    data = instantiate(cfg.data, device=cfg.device, _convert_="all")
    tag = cfg.fill + ("_scrambled" if cfg.scramble else "")
    if cfg.scramble:
        rng = np.random.default_rng(0)
        data.trees = {n: scramble(t, rng) for n, t in data.trees.items()}
    tick = time.time()
    if cfg.ckpt.endswith(".pkl"):
        coup, tag = read_legacy(cfg.ckpt), "legacy"
    elif cfg.fill == "modal_descendant":
        coup = modal_descendant(data)
    else:
        coup = couple(FILLS[cfg.fill](cfg.ckpt.rstrip("/"), cfg.device), data, cfg.n_samples)
        coup.to_parquet(os.path.join(out, f"coupling_{tag}.parquet"))
        summarize(label(coup, data.labels)).to_parquet(os.path.join(out, f"nodes_{tag}.parquet"))
    log.info(f"{tag}: {len(coup)} rows in {time.time() - tick:.0f}s")
    res = germ_crossing(coup, data.trees, data.labels["germ_layer"], data.bank.times)
    res.to_csv(os.path.join(out, f"germ_{tag}.csv"), index=False)
    log.info("\n" + res.to_string(index=False))
    log.info(f"pooled: {res[res.bin == 'all'].rate.describe()[['mean', 'std', 'min', 'max']].round(4).to_dict()}")


if __name__ == "__main__":
    main()

# germ-layer crossing for a run dir, an old model dir, or an old coupling pickle (cfg.ckpt)
import argparse
import logging
import os
import sys

if sys.version_info >= (3, 14):  # hydra trips over argparse's new help check
    argparse.ArgumentParser._check_help = lambda self, action: None

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from li.eval.couple import couple, read_legacy
from li.eval.metrics import germ_crossing
from li.model.net import load_sampler

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(cfg):
    out = HydraConfig.get().runtime.output_dir
    data = instantiate(cfg.data, device=cfg.device, _convert_="all")
    if cfg.ckpt.endswith(".pkl"):
        coup = read_legacy(cfg.ckpt)
    else:
        net = load_sampler(cfg.ckpt.rstrip("/"), cfg.device)
        coup = []
        for k in range(cfg.n_samples):
            coup.append(couple(net, data, seed=k).assign(sample=k))
            log.info(f"sample {k}: {len(coup[-1])} nodes coupled")
        coup = pd.concat(coup, ignore_index=True)
        coup.to_parquet(os.path.join(out, "coupling.parquet"))
    res = germ_crossing(coup, data.trees, data.labels["germ_layer"], data.bank.times)
    res.to_csv(os.path.join(out, "germ.csv"), index=False)
    log.info("\n" + res.to_string(index=False))
    log.info(f"pooled: {res[res.bin == 'all'].rate.describe()[['mean', 'std', 'min', 'max']].round(4).to_dict()}")


if __name__ == "__main__":
    main()

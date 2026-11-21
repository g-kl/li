import argparse
import sys

if sys.version_info >= (3, 14):  # hydra trips over argparse's new help check
    argparse.ArgumentParser._check_help = lambda self, action: None

import hydra
from hydra.core.hydra_config import HydraConfig

from li.train.loop import fit


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    fit(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()

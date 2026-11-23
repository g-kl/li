import glob
import os

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

from li.train.loop import fit

HERE = os.path.dirname(__file__)
EXPERIMENTS = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{HERE}/../config/experiment/*.yaml"))


def test_configs_compose():
    with initialize(config_path="../config", version_base=None):
        for e in EXPERIMENTS:
            cfg = compose("config", [f"experiment={e}"])
            assert cfg.run_name == e
            OmegaConf.to_container(cfg, resolve=True)
        cfg = compose("eval", ["run_name=x"])
        assert cfg.ckpt == "runs/x"
        cfg = compose("perturb", ["run_name=x"])
        assert cfg.ckpt == "runs/x" and cfg.gene == "Tbx5"


@pytest.mark.parametrize("imputer", ["learned", "mean_pool"])
def test_fit_toy(tmp_path, imputer):
    with initialize(config_path="../config", version_base=None):
        cfg = compose("config", ["experiment=toy", "device=cpu", "train.n_steps=2", "model.d=16", "model.layers=1",
                                 f"imputer={imputer}"])
    fit(cfg, str(tmp_path))
    assert os.path.exists(tmp_path / "ckpt_2.pt") and os.path.exists(tmp_path / "metrics.csv")

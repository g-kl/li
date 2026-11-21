# li

lineage inpainting, minimal. a rewrite of [lineage-inpainting](https://github.com/g-kl/lineage-inpainting) for the revision. see [roadmap.md](roadmap.md) for the plan and what changed.

```
li/
  tree.py     lineage trees as flat arrays in dfs preorder, subtree index per time bin
  batch.py    padded subtree batches: depth, ancestor matrix, mrca times, cross-sections
  bank.py     observed cells by time, exact nearest-neighbour snapping on gpu
  data.py     mouse embryo loader (zenodo bundle layout), labels
  toy.py      small random trees for tests and notebooks
  net.py      tree transformer (imputer, and sampler with a flow-time embedding)
  vpsde.py    vp-sde interpolant and reverse step
  model.py    felsenstein baseline, impute, sampler loss, sampling, anchor cuts
  align.py    energy distance, time marginal, path and common-ancestor features
  train.py    training loop, hydra entry point
  couple.py   node -> cell coupling (the paper procedure), old pickle reader
  metrics.py  germ-layer crossing
configs/      hydra: train.yaml, data/{mouse,toy}.yaml
scripts/      train.sbatch, eval_germ.py
tests/        pytest; `-m slow` runs the real-data and old-code checks
```

## setup

uses the `lt` conda env. `pip install -e . --no-deps`, then from the repo root:

```bash
python -m li.train run_name=main_v1                 # main model, writes runs/main_v1/
python -m li.train data=toy run_name=toy model.d=32  # toy data, seconds
sbatch -J main_v1 scripts/train.sbatch run_name=main_v1
python scripts/eval_germ.py runs/main_v1             # 4 coupling samples + germ crossing
pytest                                              # fast tests
pytest -m slow                                      # real data, old-code oracles
```

data defaults to the zenodo bundle at `/mnt/home/gokulg/data/2610_lineage_inpainting_data_avail` (`embryos/`, `model/`); set `LI_DATA` to point elsewhere. the raw tree read is cached in `cache/`.

`scripts/eval_germ.py` also takes the old model dir (`$LI_DATA/model`) or an old coupling pickle, so old and new numbers come from the same code.

# li

lineage inpainting, minimal. a rewrite of [lineage-inpainting](https://github.com/g-kl/lineage-inpainting) for the revision. see [roadmap.md](roadmap.md) for the plan and what changed.

```
train.py        hydra entry: train a model into runs/<run_name>/
evaluate.py     hydra entry: coupling samples + germ crossing for a run
li/
  data/
    tree.py     lineage trees as flat arrays in dfs preorder, subtree index per time bin
    batch.py    padded subtree batches: depth, ancestor matrix, mrca times, cross-sections
    bank.py     observed cells by time, exact nearest-neighbour snapping on gpu
    mouse.py    mouse embryo loader (zenodo bundle layout), labels
    toy.py      small random trees for tests and notebooks
  model/
    net.py      tree transformer (imputer, and sampler with a flow-time embedding)
    vpsde.py    vp-sde interpolant and reverse step
    inpaint.py  felsenstein baseline, impute, sampler loss, sampling, anchor cuts
  train/
    align.py    energy distance, time marginal, path and common-ancestor features
    loop.py     training step and loop (fit)
  eval/
    couple.py   node -> cell coupling (the paper procedure), old pickle reader
    metrics.py  germ-layer crossing
config/
  config.yaml   root for train.py; eval.yaml is the root for evaluate.py
  data/ model/ optimizer/ train/ align/
  experiment/   named runs (the parity bisect, toy)
  hydra/launcher/  lids, kempner (slurm via submitit), joblib
tests/          pytest; `-m slow` runs the real-data and old-code checks
```

## setup

uses the `lt` conda env. `pip install -e . --no-deps`, then from the repo root:

```bash
python train.py run_name=main                        # main model, writes runs/main/
python train.py experiment=abl_all                   # a named run, writes runs/abl_all/
python train.py experiment=toy device=cpu            # toy data, seconds
python train.py -m hydra/launcher=lids run_name=main # same, on slurm
python train.py -m hydra/launcher=lids experiment=abl_all seed=0,1 'run_name=abl_all_s${seed}'
python evaluate.py run_name=main                     # 4 coupling samples + germ crossing, into runs/main/
python evaluate.py -m hydra/launcher=lids run_name=main,abl_all
python evaluate.py run_name=paper 'ckpt=${data.root}/model'  # the released weights
pytest                                              # fast tests
pytest -m slow                                      # real data, old-code oracles
```

slurm jobs need `-m` (multirun); submitit logs land in `runs/.submitit/<job id>/`. `hydra/launcher/lids.yaml` and `kempner.yaml` carry our partitions and accounts.

data defaults to the zenodo bundle at `/mnt/home/gokulg/data/2610_lineage_inpainting_data_avail` (`embryos/`, `model/`); set `LI_DATA` to point elsewhere. the raw tree read is cached in `cache/`.

`evaluate.py` also takes the old model dir or an old coupling pickle as `ckpt`, so old and new numbers come from the same code.

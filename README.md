# li

lineage inpainting, minimal. a rewrite of [lineage-inpainting](https://github.com/g-kl/lineage-inpainting) for the revision. see [roadmap.md](roadmap.md) for the plan and what changed.

```
train.py        hydra entry: train a model into runs/<run_name>/
evaluate.py     hydra entry: coupling samples + germ crossing for a run, any fill
perturb.py      hydra entry: forward knockout rollout for a run, K samples
li/
  data/
    tree.py     lineage trees as flat arrays in dfs preorder, subtree index per time bin, scramble
    batch.py    padded subtree batches: depth, ancestor matrix, mrca times, cross-sections
    bank.py     observed cells by time, exact nearest-neighbour snapping on gpu
    mouse.py    mouse embryo loader (zenodo bundle layout), labels, umap, type colours
    toy.py      small random trees for tests
  model/
    net.py      tree transformer (imputer, and sampler with a flow-time embedding)
    vpsde.py    vp-sde interpolant and reverse step
    inpaint.py  felsenstein baseline, impute, sampler loss, sampling, anchor cuts
    imputer.py  pluggable imputer: learned (one-step transformer) or mean pool + snap
  train/
    align.py    energy distance, time marginal, path and common-ancestor features
    loop.py     training step and loop (fit)
  eval/
    couple.py   node -> cell coupling through any fill (the paper procedure), label baseline, old pickles
    fills.py    sampler, posterior mean, mean pool, random descendant, random cell
    nodes.py    labels, per-node summaries, anndata export, clade tables and picking
    metrics.py  germ-layer crossing
  perturb/
    gene.py     gene expression per bank cell: scvi decoder (bundle only) or raw counts
    project.py  snap to the nearest real cell, then onto the nearest one with the gene off
    rollout.py  forward rollout over the time grid, with the off-manifold score per node
    pick.py     clades with a gene-expressing progenitor and a target plus its control
  plot/
    tree.py     clade card: dendrogram over time, coloured by predicted type
    umap.py     a clade's ancestors traced on the atlas umap
notebooks/      01_mouse (couple one clone, the output tables), 02_look (clade gallery for a run),
                03_perturb (the Tbx5 knockout)
config/
  config.yaml   root for train.py; eval.yaml is the root for evaluate.py
  perturb.yaml  root for perturb.py
  data/ model/ imputer/ optimizer/ train/ align/
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
python train.py imputer=mean_pool run_name=imp_mp    # nonparametric imputer (mean pool + snap)
python train.py -m hydra/launcher=lids run_name=main # same, on slurm
python train.py -m hydra/launcher=lids experiment=abl_all seed=0,1 'run_name=abl_all_s${seed}'
python evaluate.py run_name=main                     # 4 coupling samples + germ crossing, into runs/main/
python evaluate.py -m hydra/launcher=lids run_name=main,abl_all
python evaluate.py run_name=paper 'ckpt=${data.root}/model'  # the released weights
pytest                                              # fast tests
pytest -m slow                                      # real data, old-code oracles
```

the baseline ladder and the scrambled-tree control go through the same coupling and metric code:

```bash
python evaluate.py -m hydra/launcher=lids run_name=main \
  fill=sampler,posterior_mean,mean_pool,random_descendant,random_cell,modal_descendant
python evaluate.py -m hydra/launcher=lids run_name=main scramble=true fill=sampler,mean_pool
```

each writes `coupling_<fill>.parquet` (tree, node, time, cell, dist, sample), `nodes_<fill>.parquet` (per node: modal type, agreement, entropy, mean dist) and `germ_<fill>.csv` into the run dir (`_scrambled` appended for the control). `li.eval.nodes.to_anndata` turns a coupling into anndata for scanpy.

slurm jobs need `-m` (multirun); submitit logs land in `runs/.submitit/<job id>/`. `hydra/launcher/lids.yaml` and `kempner.yaml` carry our partitions and accounts.

data defaults to the zenodo bundle at `/mnt/home/gokulg/data/2610_lineage_inpainting_data_avail` (`embryos/`, `model/`); set `LI_DATA` to point elsewhere. the raw tree read is cached in `cache/`.

## perturbation

`perturb.py` rolls a knockout forward along the observed topology, from `start_time` to
`target_time`. At each time bin a state is snapped to its nearest real cell and then
projected onto the nearest cell that does not express the gene; a lineage whose gene was
on is resampled from there by the sampler (never the imputer), the rest carry their own
state. Control and knockout share a node set, so only states differ.

```bash
python perturb.py run_name=cand1 mode=control            # the baseline every diff needs
python perturb.py run_name=cand1 mode=knockout           # Tbx5, E8.0 -> E9.5, 8 samples
python perturb.py run_name=cand1 mode=knockout source=counts
```

"expressing" comes from `source`: `decoder` (default) is the bundled scVI model's
`px_scale` for the gene, in parts per million, cutoff 100; `counts` is the raw count,
cutoff 0, read from the old states h5ad, which is **not** in the zenodo bundle. The
decoder path needs nothing but the bundle and avoids counting dropouts as gene-off
(Tbx5 is detected in only 73% of the cells that express it).

Each run writes `perturb_<gene>_<mode>.parquet` (tree, node, time, parent, cell, dist,
proj_dist, local_dist, changed, score, sample) and `perturb_clades_<gene>.csv`. Nothing
about the off-manifold policy is baked in: `score` is the path max of
`proj_dist / local_dist` over nodes whose gene was on, so a node is off-manifold at a
given ratio exactly when `score > ratio`, and prune (drop it) and freeze (fall back to
the last on-manifold ancestor, via `parent`) are both derivable afterwards.

`evaluate.py` also takes the old model dir or an old coupling pickle as `ckpt`, so old and new numbers come from the same code.

# roadmap

li is a minimal rewrite of `../lineage-inpainting` (publication repo) / `../pool-and-split` (dev repo).
data and the released model live in the zenodo bundle layout at `../data/2610_lineage_inpainting_data_avail` (override with `LI_DATA`). anyone with the zenodo download can run li.

## status

| | |
|---|---|
| core (tree, batch, bank, nets, align, train, couple) | done |
| tree / batch tests (toy, plus oracle checks against the old code) | done |
| germ-layer crossing | done, matches the old metric exactly |
| refit of the main model | done: `runs/cand1`, germ crossing 0.049 vs 0.049 for the paper weights (see "parity") |
| phase 1 interface (pluggable imputer for training, fill-based coupling, node tables, notebooks) | done |
| phase 2 ladder + scrambled control on cand1, germ crossing | done: posterior mean 0.0325 < mean pool 0.0362 < one draw 0.0493 (see "ladder on cand1") |
| clade cards and umap paths | done: `notebooks/02_look`, figures in `runs/cand1/look/` |
| phase 4 perturbation code + the Tbx5 knockout, 8 samples | done: `li/perturb/`, `perturb.py`, `notebooks/03_perturb` (see "phase 4") |


## decision log (phase 0)

the default config fixes these, except two old behaviours that turned out to matter and are now explicit settings (marked below). to try the old value of anything else, use a hydra override.

| quirk | li default | old behaviour |
|---|---|---|
| stale cache ignores params | index rebuilt on load from compact arrays; only the raw tree read is cached | pickled index reused regardless of params |
| leaf cap | `data.max_leaves=1024` | 128 (via the stale cache) |
| node cap | `train.max_nodes=256`, seeded subsample; coupling never subsamples | 128 at train, 256 to 1024 at eval |
| root time per step | root bin = earlier leaf time - lookback, lookback drawn from `train.lookback=[0.0, 0.5]` | same distribution, implicitly: nearest bin of leaf - U[0, 0.5]. **kept on purpose**: making every step use 0.5 cost 0.024 in germ crossing (`runs/main_v1` vs `runs/abl_lookback`) |
| marginal reference | lineage-traced cells (`align.clonal=true`); nn pairing snaps to all devmap cells | same. **kept on purpose**: using all devmap cells for the marginal hurt early times. revision text should say "lineage-traced cells", not "atlases" |
| unobserved leaves get baseline 0 | pruned at load (no-op on the real data: every leaf has a state) | kept |
| snapping nodes at t~5 in training | `train.min_snap_time=7.0` | no cutoff |
| seeding, leaf time jitter, double-counted bin edges, faiss, hashed run dirs | fixed / removed | |

kept on purpose (to ablate later): sum over strata, x1-prediction output, `n_trees=1024`, weight decay on all params, alternating id/nn pairing, 8 rollout steps.

## parity (phase 0 bar)

germ-layer crossing, pooled over edges, same coupling procedure as the paper (sampler, 8 steps, snap within one bin either side, nodes >= E7.0).
each refit below is `python train.py experiment=<run>` (`config/experiment/`); `cand1` is the default config.

| | pooled germ crossing |
|---|---|
| paper coupling pickle, old metric | 0.0502 |
| paper coupling pickle, li metric | 0.0502 (exact match per bin) |
| paper weights + li coupling, 4 samples | 0.0487 +- 0.0002 (per bin 0.094 / 0.062 / 0.030 / 0.019 / 0.018 for E7.5 to E9.5) |
| first refit (`runs/main_v1`, all fixes), 4 samples | 0.088 +- 0.001 (per bin **0.272** / **0.090** / 0.035 / 0.022 / 0.019): misses the bar |
| bisect A: old root-time sampler + snapping before E7.0 (`runs/abl_steps`) | 0.0615 +- 0.0005 (per bin 0.158 / 0.069 / 0.034 / 0.021 / 0.018) |
| bisect A1: old root-time sampler only (`runs/abl_lookback`) | 0.0646 +- 0.0005 (per bin 0.181 / 0.066 / 0.034 / 0.023 / 0.019) |
| bisect B: 128-leaf cap + 128-node cap + clonal marginal bank (`runs/abl_data`) | 0.0642 +- 0.0003 (per bin 0.160 / 0.083 / 0.035 / 0.022 / 0.018) |
| all old settings (`runs/abl_all`) | **0.0517 +- 0.0002** (per bin 0.124 / 0.066 / 0.031 / 0.020 / 0.018): parity |
| all old settings, seed 1 (`runs/abl_all_s1`) | 0.0514 +- 0.0003 (per bin 0.123 / 0.063 / 0.032 / 0.020 / 0.018): seed spread is about 0.0003 |
| **li default** (lookback [0, 0.5] + clonal marginal bank, other fixes kept; `runs/cand1`) | **0.0493 +- 0.0002** (per bin 0.106 / 0.059 / 0.033 / 0.021 / 0.018): parity, the new main model |

## phase 1: the model interface

the thing a scientist should touch is a coupling: for each internal node of a tree, a real cell (and its distance), across K samples.

done:
- **pluggable imputer** (`li/model/imputer.py`, config group `imputer/`). `learned` is the paper's one-step transformer on top of mean pool. `mean_pool` is nonparametric: mean pool snapped to real cells. with `mean_pool` the alignment loss has nothing to train and is only logged, and the sampler always learns from real cells. old weights (paper pooler, cand1) load unchanged.
- **coupling through any fill.** `couple(fill, data, n_samples)` takes `fill(batch, bank) -> states`:
  - the sampler, which reproduces the old `couple` exactly
  - the posterior mean
  - a baseline (`li/eval/fills.py`)
  
  the model's couplings always come from the sampler; the imputer is only used in training. `layers()` is the one rule for which nodes get coupled.
- **outputs of `evaluate.py fill=<name>`**, written to the run dir:
  - `coupling_<fill>.parquet` (tree, node, time, cell, dist, sample)
  - `nodes_<fill>.parquet` (modal type, agreement, entropy, mean dist)
  - `germ_<fill>.csv`
- **`li.eval.nodes`**: labels, per-node summaries, `to_anndata` (one obs per node x sample, with `X_scvi`, `X_umap` and labels), clade tables, and seeded clade picking.
- **notebooks** (`notebooks/`):
  - `01_mouse`: couple one clone; tables, anndata, a clade
  - `02_look`: a clade gallery across methods for a run
- a reader for old pickles: `li.eval.couple.read_legacy`.

left:
- snap distance as a confidence score: check it against agreement and crossing (the old coupling qc found it informative from E8.0 on, not at E7.5).

## phase 2: confidence (are the predictions silly?)

- **baseline ladder on every metric.** each runs through the same snap and metric code (`evaluate.py fill=<name>`, done for germ crossing):
  - random time-matched cell (`random_cell`, the chance floor)
  - mean pool + snap (`mean_pool`)
  - the state of a random descendant anchor, snapped (`random_descendant`)
  - modal descendant label (`modal_descendant`, labels only)
  - genot as published (an old pickle via `read_legacy`, if one exists)
  - control: leaf cells scrambled within each tree (`scramble=true`). this keeps topology and composition but removes the lineage signal.
- **like-for-like.**
  - done: the posterior mean (`posterior_mean`: average 8 draws per layer, then snap) vs mean pool
  - to do: a single sample vs a noise-matched mean pool. this needs the spread of the raw states across samples first. reviewers may well ask about it.
- **ladder on cand1**: germ crossing, 4 samples (sd over samples), same coupling and metric code (`runs/cand1/germ_<fill>.csv`, `nodes_<fill>.parquet`):

  | | pooled | E7.5 / E8.0 / E8.5 / E9.0 / E9.5 | scrambled trees | mean snap dist | agreement over 4 samples |
  |---|---|---|---|---|---|
  | random cell (chance) | 0.563 | 0.639 / 0.562 / 0.547 / 0.534 / 0.521 | | 0 (real cells) | 0.31 |
  | sampler, one draw (the paper's procedure) | 0.0493 +- 0.0002 | 0.108 / 0.060 / 0.033 / 0.021 / 0.018 | 0.364 | 4.59 | 0.80 |
  | random descendant | 0.0504 +- 0.0005 | 0.102 / 0.063 / 0.027 / 0.021 / 0.018 | 0.391 | 2.35 | 0.86 |
  | mean pool | 0.0362 | 0.071 / 0.047 / 0.019 / 0.015 / 0.016 | 0.392 | 3.38 | 1 |
  | posterior mean (8 draws per layer) | **0.0325 +- 0.0004** | 0.070 / 0.039 / 0.021 / 0.016 / 0.017 | not run | 3.60 | 0.90 |
  | modal descendant (labels only) | 0.0274 | 0.060 / 0.028 / 0.016 / 0.014 / 0.012 | 0.264 | | |

  - **posterior mean vs mean pool.** like for like, the model's posterior mean beats mean pool (0.0325 vs 0.0362), mostly at E8.0. a single draw does not, and is no better than copying a random descendant's cell. the gap is sampling noise: in the clade cards, single draws flip short internal nodes near the leaves to another type for one sample.
  - **germ crossing rewards smoothness.** the label-only modal-descendant baseline is the best row. germ crossing can't separate right from smooth, so the other metrics and the views have to carry the argument.
  - **the lineage signal is real.** scrambling leaves within each tree costs every method about 0.3 (0.36 to 0.39).
  - **denominators differ.** edges from epiblast-labelled sources are dropped, so each method is scored on a different edge count (0.61M for the sampler, 0.65M for mean pool). worth fixing (score on a shared edge set) before quoting it.
  - **on-manifold does not favour the model.** mean pool's raw states are closer to real cells than single draws at every time (3.38 vs 4.59 overall). the posterior mean is closer than mean pool only at E7.0 to E8.0 (4.22 / 4.40 / 4.10 vs 4.61 / 4.78 / 4.16).
  - **mean pool's ancestors are anachronistic** (quick check, not yet a metric). this is the case against mean pool that germ crossing misses.
    - every mean-pool node at E7.5 snaps to an E8.0 cell (the snap window allows one time step either side). none are epiblast, primitive streak or nascent mesoderm, which are 52% of the lineage-traced cells at E7.5; instead they are nascent somite, spinal cord, and so on.
    - a single model draw snaps 57% of E7.5 nodes to E7.5 cells.
    - total variation between ancestor types and observed lineage-traced cells at E7.5 / E8.0 / E8.5 / E9.0 / E9.5:
      - one draw: 0.38 / 0.44 / 0.25 / 0.25 / 0.09
      - posterior mean: 0.37 / 0.47 / 0.27 / 0.25 / 0.09
      - mean pool: **0.92 / 0.74** / 0.33 / 0.29 / 0.10
    - to do: make it a metric (it is the label version of marginal distortion), and add an own-time-only snap as a coupling variant.
- noise bands from K samples x seeds on every number.
- held-out clones as the default eval split (`supp_holdout` held out 11 of 56 clones).
- port the other five metrics into `li.eval.metrics` with tidy outputs:
  - oscillation
  - linkage (fix the normalisation that blows up at early cuts)
  - hox (only 11 pairs; consider more genes)
  - marginal and clonal distortion (make the model vs baseline comparisons symmetric)
- label-free tests with real ground truth:
  - sister-clade prediction vs a copy-the-sister baseline
  - held-out-leaf inpainting (needs random leaf masking in training, so it is an ablation)
- things to look at, ranked by confidence gained per hour:
  - done: **clade card** (`li.plot.tree.card`). a dendrogram over time, one row per leaf:
    - edges in the child's modal predicted type, faded by disagreement
    - a tick per sample
    - a red x on germ crossings
  - done: **umap path** (`li.plot.umap.path`): each sample's ancestors joined parent to child on the atlas umap.
  - `notebooks/02_look` shows both for sampler / posterior mean / mean pool / random descendant / scrambled, on 8 random clades plus the 3 worst (`runs/<run>/clades.csv`, figures in `runs/<run>/look/`).
  - on-manifold check: raw-state `dist` per method and time bin, against a real-cell floor (each real cell's distance to its nearest other cell). it needs no labels and costs almost nothing. it is an open question, not a win to expect: see "ladder on cand1" above.
  - backward fate ladder: take maximal clades that are >= 90% one lineage, and show the predicted ancestor types per time bin for each method. read it against textbook progenitor order.
  - spot-check gallery: random nodes stratified by time, showing descendant composition vs the K snapped types.
  - composition per time bin: observed lineage-traced cells vs predicted ancestors.
  - does uncertainty behave? entropy over K vs time and clade size. do the high-entropy / high-dist nodes carry the crossings?
  - marker-gene cards: decode latents with the bundled scVI model (scvi-tools 1.4.2 is in `lt`), then plot markers along root-to-leaf paths of lineage-pure clades.
  - blinded pairwise review page for biologists (model vs mean pool, anonymised).
  - cell-type transition map with per-edge sample support.

## phase 3: ablations and the grid on the new core

- each old quirk as a one-off run in `config/experiment/` (max_leaves 128, max_nodes 128, old root-time sampler, clonal marginal bank)
- root lookback of 1.0 so path features become real multi-time paths
- weight-decay exclusions (norms, embeddings, bias scale)
- id vs nn vs alternating pairing; cfm vs vp-sde; delta vs plain
- nonparametric imputer (`imputer=mean_pool`): does the learned imputer matter? first run, `runs/imp_meanpool`: yes.
  - a single draw gets germ crossing 0.0335 +- 0.0003, vs 0.0493 for cand1 (per bin 0.066 / 0.038 / 0.021 / 0.017 / 0.017).
  - but it inherits mean pool's anachronism: total variation at E7.5 / E8.0 is 0.92 / 0.73, vs 0.38 / 0.44 for cand1.
  - the logged alignment of its targets is about 10x worse (marginal 5.3 vs 0.5).
  - so the learned imputer plus alignment is what keeps ancestors time-appropriate, and germ crossing alone would pick the wrong model.
- port the synthetic benchmarks (crossing, context, landscape). they already build confidence and should run on li unchanged.

## phase 4: perturbation

done: the code (`li/perturb/`, `perturb.py`, `config/perturb.yaml`) and a first Tbx5 run.

- **the rollout** mirrors `couple` forwards: from the start layer, per time bin, snap to the
  nearest real cell, project onto the nearest non-expressing one, and resample every lineage
  whose gene was on, with the clade's whole placed history as anchors (not the old
  parent-only star graph). topology is the observed tree's, so all modes share a node set.
- **three modes.** `control` copies every state and makes no model call. `knockout` projects.
  `sham` resamples exactly the lineages the knockout resamples but moves nothing, so
  control -> sham is rollout drift and sham -> knockout is the gene. the sham is new; the old
  pipeline's only control made no model call, so it could not separate the two.
- **nothing about the off-manifold policy is baked in.** `score` is the path max of
  `proj_dist / local_dist` over nodes whose gene was on, so off-manifold at a ratio is
  `score > ratio`, and prune and freeze are both derivable from the one table.
- **gene-off two ways** (`source=`). `decoder` is the bundled scVI model's `px_scale`, cutoff
  100 per million, and needs only the zenodo bundle. `counts` is the raw count from the old
  states h5ad, which is not in the bundle.

### the Tbx5 knockout on cand1 (E8.0 -> E9.5, 8 samples, 12 clades over 6 E9.5 trees)

clades are picked where the E8.0 progenitor expresses Tbx5 in most draws and the clade holds
at least 20 leaves of a Tbx5-dependent type. at E8.0 no clade holds both a target and its
control (a forelimb clade makes no hindlimb), so the contrast is read across clades.

readout leaves per draw, mean +- sd over 8 draws:

| | control | sham | knockout |
|---|---|---|---|
| forelimb | 398 | 63 +- 40 | **3 +- 2** |
| hindlimb | 1.8 | 1.8 +- 1.5 | **43 +- 26** |
| posterior heart | 224 | 177 +- 40 | **13 +- 9** |
| anterior heart | 19 | 12 +- 14 | **118 +- 74** |

- **a single draw is nearly uninformative.** the sham alone moves 82.5% of leaf identities off
  control, and the modal subtype wins only 0.37 of the 8 draws (0.36 for the knockout). the
  backward coupling's agreement is 0.80. the forward rollout compounds noise over three steps,
  and the published screen took one draw.
- **but against the matched sham the gene effect is real and correctly directed.** forelimb
  leaves: hindlimb field 0.002 -> 0.081. posterior-heart leaves: inflow tract 0.226 -> 0.001,
  atria 0.180 -> 0.000, second heart field (posterior) 0.071 -> 0.000, while right ventricle
  0.016 -> 0.215 and outflow tract 0.005 -> 0.069. that is the textbook phenotype: the venous
  pole is lost and the arterial pole takes over.
- **without the sham, most of the "effect" is drift.** 82.5 of the 91.3 points of
  knockout-vs-control change are already there with no gene move.
- **geometry is sharper than the model.** projecting an observed forelimb leaf straight onto its
  nearest non-Tbx5 cell sends 83% to hindlimb field; the rollout sends 8% there and smears the
  rest over somatic and splanchnic mesoderm. the model lands where geometry would only 10% of
  the time. so the crisp part of the prediction is in the manifold, not the trajectory.
- **the decoder matters.** at E9.5 counts call only 73% of forelimb and 70% of inflow tract
  Tbx5-on, against 99.8% and 99.4% for the decoder, so a counts-based gene-off set is ~27%
  dropouts of the target cells and the projection onto one is a no-op.

left:
1. the noise is the headline problem. report K-sample bands on every perturbation number, and
   check whether a posterior mean over draws (as it did for coupling, 0.0325 vs 0.0493)
   recovers the sharpness.
2. **sweeps, not picks.** sweep ratio and freeze/prune off the recorded `score`.
3. shift time and per cell type log fold change as tidy outputs, against the sham not the control.
4. **screen every hvg the same way.** filters become post-hoc columns, not pre-screen choices.
5. **speculative: interpretation as a claude skill** that reads only those tables and writes a per-gene report card:
   - prediction summary
   - robustness
   - literature / mgi / gnomad support, with sources

## phase 5: analysis and figures

- `li.eval` functions return tidy frames. each analysis writes `results/<run>/<analysis>.parquet`.
- a `runs.yaml` registry (friendly name to run dir), so analysis code never sees hashes or paths.
- figure dirs only read results and plot with polyplot. read polyplot's conventions at the start of this phase.
- a migration table from each old figure dir (`analysis/figures/*`, `synthetic_evals/figure*`) to its li replacement.
- a paper-number checklist: every number in the main text, the script that makes it, and the old vs new value.

## phase 6: release

- updated zenodo bundle (new weights and K-sample couplings as parquet), README, notebooks
- revision-text fixes: subtrees per record, leaf cap and node cap, bank composition vs the "atlases" wording, and any number that moved

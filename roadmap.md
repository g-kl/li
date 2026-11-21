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

- `couple()` already returns a long table (tree, node, time, cell, dist, sample). make it the documented artifact, saved as parquet next to the run.
- summaries over samples: per-node label distribution, label entropy, and modal cell type. snap distance works as a confidence score (the old coupling qc found it informative from E8.0 on, not at E7.5).
- export inferred ancestors as anndata (nodes x samples, `X_scvi`, labels) so people can use scanpy.
- notebooks:
  - `00_quickstart`: toy trees, train in the notebook, sample, couple.
  - `01_mouse`: load the released model, couple one clone, look at it.
- a reader for old pickles exists already (`li.eval.couple.read_legacy`).

## phase 2: confidence (are the predictions silly?)

- **baseline ladder on every metric.** each runs through the same snap and metric code:
  - random time-matched coupling (chance floor)
  - mean pool + snap
  - nn of a random descendant leaf at the earlier time
  - modal descendant label
  - genot as published
- **like-for-like.** compare the model's posterior-mean coupling (average K samples, then snap) against mean pool, and a single model sample against a noise-matched mean pool. this is the fair version of the mean-pool comparison. reviewers may well ask about it.
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
- things to look at:
  - per-clone tree viewer: nodes coloured by modal type, with spread across samples
  - spot-check gallery: random nodes with their K snapped cells
  - marker-gene trajectory cards
  - cell-type transition map with per-edge sample support

## phase 3: ablations and the grid on the new core

- each old quirk as a one-off run in `config/experiment/` (max_leaves 128, max_nodes 128, old root-time sampler, clonal marginal bank)
- root lookback of 1.0 so path features become real multi-time paths
- weight-decay exclusions (norms, embeddings, bias scale)
- id vs nn vs alternating pairing; cfm vs vp-sde; delta vs plain
- port the synthetic benchmarks (crossing, context, landscape). they already build confidence and should run on li unchanged.

## phase 4: perturbation

1. **a declarative spec.**
   - gene
   - start time and readout time
   - projection rule (nearest non-expressing cell, as now)
   - off-manifold policy (freeze vs prune, and the ratio)
   - K samples
2. **standard outputs**, all as tables with uncertainty over samples and seeds:
   - per gene x cell type log fold change
   - shift time
   - off-manifold fraction
3. **sweeps, not picks.** sweep the heuristic choices (freeze vs prune, ratio 1.1 vs others) and report sensitivity, instead of choosing one.
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

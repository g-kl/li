# germ-layer crossing for a run dir, an old model dir, or an old coupling pickle
# usage: python scripts/eval_germ.py <path>
import os
import sys

import pandas as pd

from li.couple import couple, read_legacy
from li.data import load_mouse
from li.metrics import germ_crossing
from li.net import load_sampler

ROOT = os.environ.get("LI_DATA", "/mnt/home/gokulg/data/2610_lineage_inpainting_data_avail")
BINS = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
K = 4  # coupling samples

path = sys.argv[1].rstrip("/")
data = load_mouse(ROOT, BINS, 1024, device="cuda")
if path.endswith(".pkl"):
    coup, out = read_legacy(path), "runs/legacy/germ_pickle.csv"
else:
    out = os.path.join(path if path.startswith("runs/") else "runs/legacy", "germ.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    net = load_sampler(path)
    coup = []
    for k in range(K):
        coup.append(couple(net, data, seed=k).assign(sample=k))
        print(f"sample {k}: {len(coup[-1])} nodes coupled", flush=True)
    coup = pd.concat(coup, ignore_index=True)
    coup.to_parquet(out.replace("germ.csv", "coupling.parquet"))
os.makedirs(os.path.dirname(out), exist_ok=True)
res = germ_crossing(coup, data.trees, data.labels["germ_layer"], data.bank.times)
res.to_csv(out, index=False)
print(res.to_string(index=False))
print("pooled:", res[res.bin == "all"].rate.describe()[["mean", "std", "min", "max"]].round(4).to_dict())

import numpy as np
import torch


class Bank:
    """observed cells: states, times and names, bucketed by observation time."""

    def __init__(self, emb, time, names, device="cpu"):
        self.emb = torch.as_tensor(emb, dtype=torch.float32, device=device)
        self.time = np.asarray(time, dtype=np.float64)
        self.names = np.asarray(names)
        self.times = np.unique(self.time)
        self.rows = [torch.as_tensor(np.flatnonzero(self.time == t), device=device) for t in self.times]
        self.sq = self.emb.pow(2).sum(-1)

    def bucket(self, t):
        """nearest observation time (index into self.times) for node times t."""
        t = torch.as_tensor(t, device=self.emb.device)
        return (t[:, None] - torch.as_tensor(self.times, device=t.device, dtype=t.dtype)).abs().argmin(1)

    def sample(self, i, n):
        r = self.rows[i]
        return r if len(r) <= n else r[torch.randperm(len(r), device=r.device)[:n]]

    @torch.no_grad()
    def nn(self, q, t, window=1, chunk=512, k=1):
        """k nearest cells to each query among its own time bucket and `window` buckets
        either side. returns bank rows and l2 distances, shaped (n,) for k=1 else (n, k)."""
        q = q.float()
        bi = self.bucket(t)
        shape = (len(q),) if k == 1 else (len(q), k)
        rows = torch.empty(shape, dtype=torch.long, device=q.device)
        dist = torch.empty(shape, device=q.device)
        for i in bi.unique().tolist():
            sel = torch.nonzero(bi == i).squeeze(1)
            cand = torch.cat(self.rows[max(0, i - window):i + window + 1])
            e, esq = self.emb[cand], self.sq[cand]
            for s in range(0, len(sel), chunk):
                qi = q[sel[s:s + chunk]]
                d2 = qi.pow(2).sum(-1, keepdim=True) + esq[None] - 2 * qi @ e.T
                if k == 1:
                    j = d2.argmin(1)
                    dist[sel[s:s + chunk]] = (qi - e[j]).norm(dim=-1)
                else:
                    j = d2.topk(min(k, len(cand)), -1, largest=False).indices
                    if j.shape[1] < k:  # bucket smaller than k, repeat the farthest
                        j = torch.cat([j, j[:, -1:].expand(-1, k - j.shape[1])], 1)
                    dist[sel[s:s + chunk]] = (qi[:, None] - e[j]).norm(dim=-1)
                rows[sel[s:s + chunk]] = cand[j]
        return rows, dist

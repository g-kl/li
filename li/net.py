import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_sampler(path, device="cuda"):
    """ema sampler from a run dir (latest ckpt) or an old model dir (split_ema_step20000.pt)."""
    old = os.path.join(path, "split_ema_step20000.pt")
    if os.path.exists(old):
        net = TreeNet(50, flow=True)
        net.load_state_dict(torch.load(old, map_location="cpu"))
    else:
        ck = sorted(glob.glob(f"{path}/ckpt_*.pt"), key=lambda p: int(p.split("_")[-1][:-3]))[-1]
        ck = torch.load(ck, map_location="cpu")
        net = TreeNet(ck["config"]["dim"], flow=True, **ck["config"]["model"])
        net.load_state_dict(ck["ema_sampler"])
    return net.to(device)


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, c))


class Attn(nn.Module):
    # same parameter names as nn.MultiheadAttention so old checkpoints load
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, bias):
        B, N, d = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).view(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out_proj(o.transpose(1, 2).reshape(B, N, d))


class Layer(nn.Module):
    def __init__(self, d, heads, hidden):
        super().__init__()
        self.attn = Attn(d, heads)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = mlp(d, hidden, d)

    def forward(self, x, bias):
        x = x + self.attn(self.norm1(x), bias)
        return x + self.ff(self.norm2(x))


class TreeNet(nn.Module):
    """transformer over tree nodes, attention biased by patristic distance.
    flow=True adds a flow-time embedding (the sampler); the imputer has none."""

    def __init__(self, dim, d=128, heads=4, layers=4, hidden=512, flow=False):
        super().__init__()
        self.heads = heads
        self.input_proj = nn.Linear(dim, d)
        self.out_proj = mlp(d, hidden, dim)
        self.node_time_embed = mlp(1, d, d)
        if flow:
            self.flow_time_embed = mlp(1, d, d)
        self.encoder_layers = nn.ModuleList([Layer(d, heads, hidden) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d)
        self.attention_bias_scale = nn.Parameter(torch.zeros(layers, heads))
        self.role_embed = nn.Embedding(2, d)  # 1 = anchor

    def forward(self, x, b, anchor, t=None):
        B, N = b.mask.shape
        h = self.input_proj(x) + self.node_time_embed(b.time[..., None])
        if t is not None:
            h = h + self.flow_time_embed(t.view(B, 1, 1))
        h = h + self.role_embed(anchor.long())
        dist = (b.time[:, :, None] + b.time[:, None, :] - 2 * b.mrca)[:, None]
        pad = torch.zeros(B, 1, 1, N, device=x.device, dtype=h.dtype).masked_fill(~b.mask[:, None, None], float("-inf"))
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h, self.attention_bias_scale[i].view(1, -1, 1, 1) * dist + pad)
        return self.out_proj(self.final_norm(h)) * b.mask[..., None]

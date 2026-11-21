import torch

# vp-sde on its own clock s (s=0 data, s=1 noise). flow time is t = 1 - s.
B0, B1 = 0.1, 20.0


def alpha_bar(s):
    return torch.exp(-(B0 * s + 0.5 * (B1 - B0) * s ** 2))


def interpolate(x0, x1, t):
    """noise x0 -> data x1 at flow time t (per tree)."""
    a = alpha_bar(1 - t).view(-1, 1, 1)
    return a.sqrt() * x1 + (1 - a).sqrt() * x0


def step(s, s_next, x, x1):
    """one euler-maruyama reverse step given a data prediction x1. last step adds no noise."""
    a = alpha_bar(s)
    beta = B0 + s * (B1 - B0)
    score = -(x - a.sqrt() * x1) / (1 - a)
    dt = s - s_next
    mean = x + (0.5 * beta * x + beta * score) * dt
    if float(s_next) == 0.0:
        return mean
    return mean + (beta * dt).sqrt() * torch.randn_like(x)

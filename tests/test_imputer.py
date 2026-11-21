import torch

from li.data.batch import collate
from li.model.inpaint import impute
from li.model.imputer import Learned, MeanPool
from conftest import subtrees


def test_mean_pool(toy):
    imp = MeanPool(4)
    assert not list(imp.parameters())
    b = collate(subtrees(toy, 7.0, n=8), toy.bank.emb)
    x = imp.fill(b, toy.bank)
    assert torch.equal(x[b.obs], b.x[b.obs])
    # every free node from E7.0 on is a real cell
    m = b.mask & ~b.obs & (b.time >= 7.0)
    assert m.any()
    assert (x[m][:, None] == toy.bank.emb[None]).all(-1).any(1).all()


def test_learned_is_impute(toy):
    torch.manual_seed(0)
    imp = Learned(4, d=16, layers=1, hidden=32)
    b = collate(subtrees(toy, 7.0, n=8), toy.bank.emb)
    torch.manual_seed(1)
    want = impute(imp, b)
    torch.manual_seed(1)
    assert torch.equal(imp.fill(b, toy.bank), want)

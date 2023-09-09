### Test cases for pytest ###

import torch
from model import KGEModel

def test_norm_inv_1():
    m = KGEModel("TransE",1,1,1,1)
    w1 = torch.tensor([1.0])
    w2 = m._norm_inv(w1, 1.0)
    assert(torch.equal(w1, w2))

def test_norm_inv_2():
    m = KGEModel("TransE",1,1,1,1)
    w1 = torch.tensor([0.5, 0.5])
    w2 = m._norm_inv(w1, 1.0)
    assert(torch.equal(w1, w2))

def test_norm_wsum_1():
    m = KGEModel("TransE",1,1,1,1)
    w1 = torch.tensor([1.0])
    w2 = torch.tensor([0.0])
    w3 = m._norm_wsum(w1, w2, 0.0)
    assert(torch.equal(w3, w2))

def test_norm_wsum_1():
    m = KGEModel("TransE",1,1,1,1)
    w1 = torch.tensor([1.0])
    w2 = torch.tensor([0.0])
    w3 = m._norm_wsum(w1, w2, 1.0)
    assert(torch.equal(w3, w1))

import pytest
import torch

from torchaddons import distributions


def run_single(device: torch.device):
    d = distributions.Categorical(0.1 * torch.ones(10, device=device))
    x1 = d.sample(13, 17, 19)

    assert x1.device == device
    assert (x1 < 10).all()
    assert (x1 >= 0).all()
    assert x1.dtype == torch.long
    assert x1.shape == (13, 17, 19)

    mask = torch.ones(10, device=device, dtype=torch.bool)
    mask[torch.randperm(10)[:3]] = False
    constraint = distributions.constraints.CategoricalMask(mask)
    d = constraint.apply_to(d)
    
    x1 = d.sample(13, 17, 19)
    assert mask[x1].all()

def test_single_cpu():
    run_single(torch.device("cpu"))

def test_single_cuda():
    if not torch.cuda.is_available():
        pytest.skip()
    run_single(torch.device("cuda", index=0))


def run_batch(device: torch.device):
    d = distributions.Categorical(0.1 * torch.ones(3, 5, 7, 10, device=device))
    x1 = d.sample(13, 17, 19)

    assert x1.device == device
    assert (x1 < 10).all()
    assert (x1 >= 0).all()
    assert x1.dtype == torch.long
    assert x1.shape == (13, 17, 19, 3, 5, 7)

    mask = torch.ones(3, 5, 7, 10, device=device, dtype=torch.bool)
    mask[:, :, :, torch.randperm(10)[:5]] = False
    constraint = distributions.constraints.CategoricalMask(mask)
    d = constraint.apply_to(d)
    
    x1 = d.sample(13, 17, 19)
    for i in range(3):
        for j in range(5):
            for k in range(7):
                assert mask[i, j, k, x1[:, :, :, i, j, k].view(-1)].all()

def test_batch_cpu():
    run_batch(torch.device("cpu"))

def test_batch_cuda():
    if not torch.cuda.is_available():
        pytest.skip()
    run_batch(torch.device("cuda", index=0))

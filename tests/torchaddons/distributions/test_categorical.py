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


def test_single_cpu():
    run_single(torch.device("cpu"))

def test_single_cuda():
    if not torch.cuda.is_available():
        pytest.skip()
    run_single(torch.device("cuda", index=0))

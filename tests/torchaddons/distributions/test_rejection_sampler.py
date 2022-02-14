import pytest
import torch

from torchaddons import distributions


def run_single(device: torch.device):
    
    d = distributions.Normal(
        torch.randn(2, device=device),
        torch.eye(2, device=device) + 0.5 * torch.ones(2, 2, device=device)
    )

    constraint = distributions.constraints.LowerBound(
        torch.tensor([-0.5, -0.5], device=device)
    )

    d = constraint.apply_to(d)

    assert isinstance(d, distributions.RejectionSampler)

    samples = d.sample(1000)
    assert (samples >= constraint.bound).all()

def test_single_cpu():
    run_single(torch.device("cpu"))

def test_single_cuda():
    if not torch.cuda.is_available():
        pytest.skip()
    run_single(torch.device("cuda", index=0))

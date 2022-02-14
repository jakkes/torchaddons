import pytest
import torch

from torchaddons import distributions


def run_single(device: torch.device):
    d = distributions.Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25], device=device))

    constraint = distributions.constraints.CategoricalMask(
        torch.tensor([True, True, False, False], device=device)
    )

    d = constraint.apply_to(d)

    sample = d.sample(1000)
    assert (sample < 2).all()

def test_single_cpu():
    run_single(torch.device("cpu"))

def test_single_cuda():
    if not torch.cuda.is_available():
        pytest.skip()
    run_single(torch.device("cuda", index=0))


def run_batch(device: torch.device):
    d = distributions.Categorical(
        torch.tensor([
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.4, 0.05, 0.05]
        ], device=device)
    )

    constraint = distributions.constraints.CategoricalMask(
        torch.tensor([
            [True, True, False, False],
            [False, False, True, False]
        ], device=device)
    )

    d = constraint.apply_to(d)

    samples = d.sample(1000)
    assert (samples[:, 0] < 2).all()
    assert (samples[:, 1] == 2).all()

def test_batch_cpu():
    run_batch(torch.device("cpu"))

def test_batch_cuda():
    if not torch.cuda.is_available():
        pytest.skip()
    run_batch(torch.device("cuda", index=0))

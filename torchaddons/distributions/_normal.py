import torch

from torchaddons import distributions


class Normal(distributions.Base):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor) -> None:
        super().__init__()
        self._mean = mean
        self._cov = cov

        

import torch
from torchaddons import distributions
from torchaddons.distributions import constraints


class DiscreteMask(constraints.Base):
    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self._mask = mask

    def check(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mask(self) -> torch.Tensor:
        """Boolean mask"""
        return self._mask

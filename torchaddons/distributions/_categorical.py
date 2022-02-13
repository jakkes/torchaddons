from typing import Tuple

import torch

import torchaddons
from torchaddons import distributions


class Categorical(distributions.Base):
    """Categorical distribution."""

    def __init__(self, probs: torch.Tensor) -> None:
        super().__init__()
        self._probs = probs

    def sample(self, shape: Tuple[int, ...] = ()) -> torch.Tensor:
        return torchaddons.random.choice(self._probs.expand(shape + self._probs.shape))

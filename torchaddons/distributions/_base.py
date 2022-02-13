import abc
from typing import Tuple

import torch

from torchaddons import distributions
from . import constraints


class Base(abc.ABC):
    """Base class for distributions."""

    @abc.abstractmethod
    def sample(self, shape: Tuple[int, ...] = ()) -> torch.Tensor:
        """Samples the distribution.

        Args:
            shape (Tuple[int, ...]): Batch dimensions to prepend to the output. Defaults
                to none. If the distribution was parametrized using batch data, then
                this argument prepends _further_ batches.

        Returns:
            torch.Tensor: Random sample.
        """
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Shape of the output, if called with no extra batch dimensions."""
        pass

    def apply_constraint(self, constraint: constraints.Base) -> "distributions.Base":
        """Applies a constraint to the distribution.

        Args:
            constraint (constraints.Base): constraint to apply.

        Returns:
            distributions.Base: New distribution object respecting the given constraint.
        """
        return distributions.RejectionSampler(self, constraint)

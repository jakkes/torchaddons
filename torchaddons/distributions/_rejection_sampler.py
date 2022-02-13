from typing import Tuple
import functools

import torch

from torchaddons import distributions


class RejectionSampler(distributions.Base):
    """Distribution with constraints that are evaluated using rejection sampling."""

    def __init__(
        self,
        distribution: distributions.Base,
        *constraints: distributions.constraints.Base
    ) -> None:
        """Creates a rejection sampler.

        Args:
            distribution (distributions.Base): Underlying distribution.
            constraints (distributions.constraints.Base): Constraints to apply.
        """
        super().__init__()
        self._distribution = distribution
        self._constraints = list(constraints)

    @property
    def is_continuous(self) -> bool:
        return self._distribution.is_continuous

    @property
    def is_discrete(self) -> bool:
        return self._distribution.is_discrete

    def add_constraint(self, constraint: distributions.constraints.Base):
        """Adds a constraint to the rejection sampling.

        Args:
            constraint (distributions.constraints.Base): Constraint to add.
        """
        self._constraints.append(constraint)

    def sample(self, shape: Tuple[int, ...] = ()) -> torch.Tensor:
        def get_reducer(value: torch.Tensor):
            def reducer(
                mask: torch.Tensor, constraint: distributions.constraints.Base
            ) -> torch.Tensor:
                return mask & constraint.check()

            return reducer

        value = None
        mask = torch.zeros(shape + self.shape, dtype=torch.bool, device=value.device)
        while not mask.all():
            value = super().sample(shape)
            mask = functools.reduce(
                get_reducer(value),
                self._constraints,
                torch.zeros(shape + self.shape, dtype=torch.bool, device=value.device),
            )
        return value

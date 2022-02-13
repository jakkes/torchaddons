import torch

from torchaddons.distributions import constraints


class LowerBound(constraints.Base):
    """Lower bound constraint."""
    def __init__(self, bound: torch.Tensor, allow_equal: bool = True) -> None:
        """
        Args:
            bound (torch.Tensor): Lower bound, all elements must be greater (or equal)
                than this value.
            allow_equal (bool, optional): If True, equal values are allowed. Defaults
                to True.
        """
        super().__init__()
        self._bound = bound
        self._scalar = len(bound.shape) == 0
        self._fn = torch.ge if allow_equal else torch.gt

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if self._scalar:
            return self._fn(value, self._bound)
        return self._fn(value, self._bound).all(-1)

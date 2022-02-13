import abc
import torch


class Base(abc.ABC):
    """Base class for constraints."""
    
    @abc.abstractmethod
    def check(self, value: torch.Tensor) -> torch.Tensor:
        pass

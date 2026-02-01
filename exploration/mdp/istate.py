from abc import ABC, abstractmethod
from torch import Tensor
import torch
import numpy as np


class IState(ABC):
    """
    An interface used to represent State
    """

    def __hash__(self):
        return hash(self.np_encoding)

    def __eq__(self, other: 'IState'):
        return np.all(self.np_encoding == other.np_encoding) and torch.all(self.torch_encoding == other.torch_encoding)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def np_encoding(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def torch_encoding(self) -> Tensor:
        raise NotImplementedError

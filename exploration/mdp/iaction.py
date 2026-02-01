from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class IAction(ABC):
    """
    An interface used to represent Action
    """
    def __hash__(self):
        return self.encoding

    def __eq__(self, other: 'IAction'):
        return np.all(self.encoding == other.encoding)

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def encoding(self) -> Any:
        raise NotImplementedError

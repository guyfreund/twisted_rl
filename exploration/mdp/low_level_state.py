import numpy as np
import torch
from torch import Tensor

from exploration.mdp.istate import IState


class LowLevelState(IState):
    def __init__(self, configuration: np.ndarray):
        assert len(configuration) == 93
        self._configuration = configuration

    def __hash__(self):
        return hash(str(list(self._configuration)))

    def __str__(self):
        return str(self._configuration)

    def __repr__(self):
        return self.__str__()

    @property
    def configuration(self) -> np.ndarray:
        return self._configuration

    @property
    def np_encoding(self) -> np.ndarray:
        return self._configuration

    @property
    def torch_encoding(self) -> Tensor:
        return torch.from_numpy(self._configuration.copy())

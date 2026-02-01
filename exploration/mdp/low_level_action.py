from dataclasses import dataclass
import torch
import numpy as np
from typing import Union

from exploration.mdp.iaction import IAction


@dataclass
class LowLevelAction(IAction):
    """
    An interface used to represent Low Level Action
    """
    link: int
    z: float
    x: float
    y: float
    order_dict = {
        'link': 0,
        'z': 1,
        'x': 2,
        'y': 3
    }

    def __hash__(self):
        return self.encoding

    def __eq__(self, other: 'IAction'):
        return torch.all(self.encoding == other.encoding)

    def __str__(self):
        return f'link={self.link}, z={self.z}, x={self.x}, y={self.y}'

    @classmethod
    def arg_to_idx(cls, arg: str) -> dict[str, int]:
        return cls.order_dict[arg]

    @classmethod
    def idx_to_arg(cls, idx: int) -> dict[int, str]:
        return {v: k for k, v in cls.order_dict.items()}[idx]

    @property
    def encoding(self) -> torch.Tensor:
        encoding = self.torch_encoding
        self._verify_encoding(encoding=encoding)
        return encoding

    @property
    def torch_encoding(self) -> torch.Tensor:
        encoding = torch.tensor([self.link, self.z, self.x, self.y])
        self._verify_encoding(encoding=encoding)
        return encoding

    @property
    def np_encoding(self) -> np.ndarray:
        encoding = np.array([self.link, self.z, self.x, self.y])
        self._verify_encoding(encoding=encoding)
        return encoding

    def link_onehot_encoding(self, num_links: int) -> np.ndarray:
        link_one_hot = np.zeros(num_links)
        link_one_hot[self.link] = 1
        encoding = np.concatenate((link_one_hot, np.array([self.z, self.x, self.y])))
        return encoding

    def _verify_encoding(self, encoding: Union[np.ndarray, torch.Tensor]):
        if isinstance(encoding, torch.Tensor):
            item = encoding.item()
        else:
            item = encoding
        assert self.link == item[0]
        assert self.z == item[1]
        assert self.x == item[2]
        assert self.y == item[3]



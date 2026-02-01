from dataclasses import dataclass
from typing import Dict
import numpy as np

from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.mdp.iaction import IAction


@dataclass
class HighLevelAction(IAction):
    """
    An interface used to represent High Level Action
    """
    src: HighLevelAbstractState
    dst: HighLevelAbstractState
    data: Dict

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self.data) + str(hash(self.src)) + str(hash(self.dst)))

    def __eq__(self, other: 'HighLevelAction'):
        return np.all(self.encoding == other.encoding) and self.src == other.src and self.dst == other.dst and self.data == other.data

    def __str__(self):
        move = f'{self.data["move"]}'
        move = move.replace('cross', 'C')
        args = ','.join([str(v) for k, v in self.data.items() if k != 'move'])
        return f'{move}({args})'

    @property
    def encoding(self):
        return None

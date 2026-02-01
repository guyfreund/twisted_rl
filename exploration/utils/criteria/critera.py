from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from exploration.mdp.istate import IState
from exploration.mdp.state_mapping import StateVisitation
from exploration.utils.criteria.criteria_utils import calculate_state_visitation_entropy


class BaseCriteria(ABC):
    def __str__(self):
        return self.__name__

    def compare(self, state1: IState, state2: IState, state_visitation: StateVisitation[IState, int]) -> \
            Tuple[IState, float]:
        c1 = self.calculate(state=state1, state_visitation=state_visitation)
        c2 = self.calculate(state=state2, state_visitation=state_visitation)

        if c1 > c2:
            return state1, c1
        elif c2 > c1:
            return state2, c2
        else:
            idx = np.random.choice(2)
            if idx == 0:
                return state1, c1
            else:
                return state2, c2

    @abstractmethod
    def calculate(self, state: IState, state_visitation: StateVisitation[IState, int]) -> float:
        raise NotImplementedError


class MaxEntropyCriteria(BaseCriteria):
    def __init__(self, scale: float):
        self._scale = scale

    def calculate(self, state: IState, state_visitation: StateVisitation[IState, int]) -> float:
        e = calculate_state_visitation_entropy(state_to_increase=state, state_visitation=state_visitation)
        return e


class RandomCriteria(BaseCriteria):
    def __init__(self, const: float = 1.0):
        self.const = const

    def calculate(self, state: IState, state_visitation: StateVisitation[IState, int]) -> float:
        return self.const  # random criteria is const for each state

from collections.abc import MutableMapping
import numpy as np
from typing import Any, Dict
from abc import ABC, abstractmethod


class StateMapping(MutableMapping, ABC):
    def __init__(self):
        self._mapping = dict()

    def __getitem__(self, state) -> Any:
        return self._mapping[state]

    def __setitem__(self, state, value: Any):
        assert self.verify_type(value=value)
        self._mapping[state] = value

    def __delitem__(self, state):
        del self._mapping[state]

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    @abstractmethod
    def verify_type(self, value) -> bool:
        raise NotImplementedError

    @classmethod
    def init(cls) -> 'StateMapping':
        return cls()

    def sample(self, size: int) -> 'StateMapping':
        states = list(self._mapping.keys())
        num_states = len(states)
        nominal_size = min(int(num_states / 2), size)
        idxs = np.random.choice(np.arange(num_states), size=nominal_size)
        state_mapping = self.init()
        for idx in idxs:
            state = states[idx]
            state_mapping[state] = self[state]
        return state_mapping


class StateVisitation(StateMapping):
    @property
    def total(self) -> int:
        return sum(self._mapping.values())

    def __getitem__(self, state):
        return self._mapping[state]

    def verify_type(self, value) -> bool:
        return isinstance(value, int)

    def visit(self, state, value: int = 1):
        self[state] = self.get(state, 0) + value

    def unify(self, other: 'StateVisitation'):
        for state, visits in other.items():
            for i in range(visits):
                self.visit(state)

    def per_crossing_number(self) -> Dict[int, int]:
        per_crossing_number_container = {}
        for state, visits in self.items():
            crossing_number = state.crossing_number
            if crossing_number not in per_crossing_number_container:
                per_crossing_number_container[crossing_number] = visits
            else:
                per_crossing_number_container[crossing_number] += visits
        return per_crossing_number_container

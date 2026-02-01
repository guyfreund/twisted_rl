from collections import defaultdict
from typing import List, Optional
from logging import Logger

from exploration.mdp.high_level_action import HighLevelAction
from exploration.utils.mixins import LoggableMixin, PickleableMixin
from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.mdp.low_level_state import LowLevelState
from exploration.reachable_configurations.reachable_configurations import ReachableConfigurations


class InitialStateSelector(LoggableMixin, PickleableMixin):
    def __init__(self, logger: Optional[Logger] = None):
        LoggableMixin.__init__(self=self, logger=logger)

    @property
    def filename(self) -> str:
        return 'InitialStateSelector'

    def select(self, goal_actions: List[HighLevelAction], reachable_configurations: ReachableConfigurations) -> (List[LowLevelState], List[HighLevelAbstractState]):
        high_level_states = [goal_action.src for goal_action in goal_actions]

        high_level_states_idxs = defaultdict(list)
        for idx, high_level_state in enumerate(high_level_states):
            high_level_states_idxs[high_level_state].append(idx)

        low_level_states = [None] * len(high_level_states)
        for high_level_state, idxs in high_level_states_idxs.items():
            low_level_states_for_state = reachable_configurations.get_topology_nodes(topology=high_level_state, n=len(idxs))
            for idx, low_level_state in zip(idxs, low_level_states_for_state):
                low_level_states[idx] = low_level_state.low_level_state

        assert all([state is not None for state in low_level_states])
        return low_level_states, high_level_states

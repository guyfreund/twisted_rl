from typing import Dict, List, Optional
from logging import Logger
import numpy as np

from exploration.mdp.graph.high_level_graph import HighLevelGraph
from exploration.mdp.high_level_action import HighLevelAction
from exploration.reachable_configurations.reachable_configurations import ReachableConfigurations
from exploration.utils.mixins import LoggableMixin, PickleableMixin


class GoalSelector(LoggableMixin, PickleableMixin):
    def __init__(self, min_crosses: int, max_crosses: int, depth: int, high_level_actions: List[str], logger: Optional[Logger] = None):
        LoggableMixin.__init__(self=self, logger=logger)
        self.max_crosses = max_crosses
        self.min_crosses = min_crosses
        self.depth = depth
        self.high_level_actions = high_level_actions
        self.high_level_graph = HighLevelGraph.load_full_graph()
        self.potential_edges = self.get_potential_edges()

    def get_potential_edges(self) -> List[HighLevelAction]:
        potential_edges = []

        for edge in self.high_level_graph.edges:
            move = edge.data['move']
            if move not in self.high_level_actions:
                continue
            if edge.dst.crossing_number > self.max_crosses:
                continue
            if self.depth == 1:
                if edge.src.crossing_number != self.min_crosses:
                    continue
            else:
                if edge.src.crossing_number < self.min_crosses:
                    continue
            potential_edges.append(edge)

        return potential_edges

    def filename(self) -> str:
        raise f'GoalSelector_min_crosses_{self.min_crosses}_max_crosses_{self.max_crosses}_depth_{self.depth}_high_level_actions_{self.high_level_actions}'

    def select(self, reachable_configurations: ReachableConfigurations, total: int):
        potential_edges = [edge for edge in self.potential_edges if reachable_configurations.is_topology_reachable(edge.src)]
        edges_idxs = np.random.choice(len(potential_edges), total)
        edges = [potential_edges[i] for i in edges_idxs]
        return edges


if __name__ == '__main__':
    goal_selector = GoalSelector(
        min_crosses=1,
        max_crosses=2,
        depth=1,
        high_level_actions=['R1']
    )
    class FakeRC:
        def is_topology_reachable(self, state):
            return True
    reachable_configurations = FakeRC()
    selected_edges = goal_selector.select(reachable_configurations, len(goal_selector.potential_edges) * 2)
    print()
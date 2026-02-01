import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import os
from tqdm import tqdm
import pickle
from abc import ABC, abstractmethod

from exploration.mdp.istate import IState
from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.mdp.low_level_action import LowLevelAction
from exploration.mdp.low_level_state import LowLevelState
from exploration.mdp.graph.directed_state_action_graph import DirectedStateActionGraph
from exploration.rl.replay_buffer.in_mem_efficient_per_buffer import InMemEfficientPrioritizedReplayBuffer
from exploration.utils.futures_pool import FuturesMultiprocessingPool, perform_tasks
from mujoco_infra.mujoco_utils.mujoco import get_link_segments


@dataclass
class ReachableConfigurationsState(IState):
    high_level_state: HighLevelAbstractState
    low_level_state: LowLevelState
    low_level_pos: np.ndarray
    link_segments: list
    intersections: list

    def __hash__(self):
        return hash(f'{hash(self.high_level_state)}{hash(self.low_level_state)}')

    def __eq__(self, other: 'ReachableConfigurationsState'):
        return self.high_level_state == other.high_level_state and self.low_level_state == other.low_level_state

    def __str__(self):
        return ''

    @property
    def np_encoding(self):
        return None

    @property
    def torch_encoding(self):
        return None


class IReachableConfigurations(ABC):
    @property
    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_topology_nodes(self, topology: HighLevelAbstractState, n: int = None) -> List[ReachableConfigurationsState]:
        raise NotImplementedError

    @abstractmethod
    def add_node(self, state: ReachableConfigurationsState):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, src: ReachableConfigurationsState, dst: ReachableConfigurationsState, action: LowLevelAction):
        raise NotImplementedError

    @classmethod
    def from_replay_buffer_files(cls, replay_buffer_files_path: str, num_cpus: int, sample_size: int = None, min_crosses: int = None, max_crosses: int = None, raise_exception: bool = True) -> 'IReachableConfigurations':
        raise NotImplementedError


class ReachableConfigurations(DirectedStateActionGraph, IReachableConfigurations):
    def __init__(self):
        super().__init__()
        self._topology_to_i: Dict[HighLevelAbstractState, List[int]] = {}

    def is_topology_reachable(self, state: HighLevelAbstractState) -> bool:
        return state in self._topology_to_i

    def add_node(self, state: ReachableConfigurationsState):
        super().add_node(state)
        if (high_level_state := state.high_level_state) not in self._topology_to_i:
            self._topology_to_i[high_level_state] = [self._i]
        else:
            self._topology_to_i[high_level_state].append(self._i)

    def get_topology_nodes(self, topology: HighLevelAbstractState, n: int = None) -> List[ReachableConfigurationsState]:
        indices = self._topology_to_i.get(topology, [])
        chosen_indices = np.random.choice(indices, n) if n is not None else indices
        assert len(chosen_indices) == n if n is not None else True, f'{len(chosen_indices)=} != {n=}'
        states = [self._graph.nodes[i]['state'] for i in chosen_indices]
        return states

    @classmethod
    def from_replay_buffer_files(cls, replay_buffer_files_path: str, num_cpus: int, sample_size: int = None, min_crosses: int = None, max_crosses: int = None,
                                 raise_exception: bool = True) -> 'ReachableConfigurations':
        return reachable_configurations_from_replay_buffer_files(cls=cls, replay_buffer_files_path=replay_buffer_files_path, num_cpus=num_cpus, sample_size=sample_size, min_crosses=min_crosses, max_crosses=max_crosses, raise_exception=raise_exception)


def create_rc_state_from_experience(experience: Any, location: str) -> ReachableConfigurationsState:
    assert location in ['start', 'end']

    def experience_getter(attribute: str):
        return getattr(experience, f'{location}_{attribute}')

    def experience_setter(attribute: str, value: Any):
        setattr(experience, f'{location}_{attribute}', value)

    # Fix link_segments and intersections
    link_segments, intersections = get_link_segments(np.array(experience_getter('low_level_pos')))
    experience_setter('link_segments', link_segments)
    experience_setter('intersections', intersections)

    rc_state = ReachableConfigurationsState(
        low_level_state=experience_getter('low_level_state'),
        high_level_state=experience_getter('high_level_state'),
        low_level_pos=experience_getter('low_level_pos'),
        link_segments=experience_getter('link_segments'),
        intersections=experience_getter('intersections')
    )
    return rc_state


def get_trajectory_rc_states(trajectory_file_path: str, min_crosses: int, max_crosses: int, raise_exception: bool) -> (Optional[List[ReachableConfigurationsState]], str):
    states = []
    traceback_str = ''

    with open(trajectory_file_path, 'rb') as f:
        trajectory = pickle.load(f)

    try:
        trajectory_length = len(trajectory['infos'])

        for i, info in enumerate(trajectory['infos']):
            experience = info['experience']

            if min_crosses <= experience.start_high_level_state.crossing_number <= max_crosses:
                start_rc_state = create_rc_state_from_experience(experience=experience, location='start')
                states.append(start_rc_state)

            if i == trajectory_length - 1:
                if min_crosses <= experience.end_high_level_state.crossing_number <= max_crosses:
                    end_rc_state = create_rc_state_from_experience(experience=experience, location='end')
                    states.append(end_rc_state)

    except Exception as e:
        if raise_exception:
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e
        else:
            traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
            return None, traceback_str

    return states, traceback_str


def reachable_configurations_from_replay_buffer_files(cls, replay_buffer_files_path: str, num_cpus: int, sample_size: int = None,
                                                      min_crosses: int = None, max_crosses: int = None, raise_exception: bool = True):
    # create reachable_configurations from replay buffer files in parallel
    if cls == SumTreeReachableConfigurations:
        reachable_configurations = cls(crossing_number=min_crosses)
    else:
        reachable_configurations = cls()
    arguments = []
    for path in replay_buffer_files_path:
        arguments += [{
            'trajectory_file_path': os.path.join(path, f),
            'min_crosses': min_crosses or -np.inf,
            'max_crosses': max_crosses or np.inf,
            'raise_exception': raise_exception,
        } for f in os.listdir(path)]
    if sample_size is not None:
        num_arguments = len(arguments)
        indices = list(range(num_arguments))
        effective_sample_size = min(sample_size, num_arguments)
        chosen_indices = np.random.choice(indices, effective_sample_size, replace=False)
        arguments = [arguments[i] for i in chosen_indices]
    parallel = num_cpus > 0
    pool = FuturesMultiprocessingPool(processes=num_cpus, clear_cache_activate=False, catch_exceptions=True, verbose=True) if parallel else None
    results = perform_tasks(
        func=get_trajectory_rc_states, kwargs_list=arguments, total=len(arguments), pool=pool,
        parallel=parallel, init_pool=True, close_pool=True,
    )
    trajectory_rc_states_list = [result[0] for result in results]
    traceback_strs = [result[1] for result in results]
    total = len(trajectory_rc_states_list)
    failed = []
    for i, trajectory_rc_states in tqdm(enumerate(trajectory_rc_states_list), desc=f'{cls.__name__} Creating reachable configurations', total=total):
        if trajectory_rc_states is None:
            arguments[i]['traceback_str'] = traceback_strs[i]
            failed.append(arguments[i])
            continue
        for rc_state in trajectory_rc_states:
            reachable_configurations.add_node(rc_state)
    if len(failed) > 0:
        for f in failed:
            print(f'Failed to create reachable configurations for trajectory file: {f["trajectory_file_path"]} with the following traceback:')
            print('\n'.join(f['traceback_str']))
    return reachable_configurations


class SumTreeReachableConfigurations(ReachableConfigurations):
    def __init__(self, crossing_number: int):
        super().__init__()
        self.get_priority = get_priority.get(crossing_number, no_priority)
        self.crossing_number = crossing_number
        self._topology_to_sum_tree: Dict[HighLevelAbstractState, InMemEfficientPrioritizedReplayBuffer] = {}
        self._num_nodes = 0

    @property
    def num_nodes(self):
        return self._num_nodes

    def is_topology_reachable(self, state: HighLevelAbstractState) -> bool:
        return super().is_topology_reachable(state) or state in self._topology_to_sum_tree

    def add_node(self, state: ReachableConfigurationsState):
        if state.high_level_state.crossing_number == self.crossing_number:
            if (high_level_state := state.high_level_state) not in self._topology_to_sum_tree:
                rb = InMemEfficientPrioritizedReplayBuffer(
                    batch_size=None,  # to be defined later,
                    max_buffer_size=50000,  # TODO: in G3 it will explode due to a lot of initial states
                    output_dir=None,
                    store_preprocessed_trajectory=False,
                    save_at_end=False,
                    to_process=False
                )
                rb.get_priority = self.get_priority
                self._topology_to_sum_tree[high_level_state] = rb

            self._topology_to_sum_tree[high_level_state].add(state)
        else:
            super().add_node(state)

        self._num_nodes += 1

    def add_edge(self, src: ReachableConfigurationsState, dst: ReachableConfigurationsState, action: LowLevelAction):
        pass

    def get_topology_nodes(self, topology: HighLevelAbstractState, n: int = None) -> List[ReachableConfigurationsState]:
        if topology.crossing_number == self.crossing_number:
            rb = self._topology_to_sum_tree[topology]
            rb.batch_size = n or rb.real_size
            states = rb.sample()
        else:
            states = super().get_topology_nodes(topology=topology, n=n)
        return states

    @classmethod
    def from_replay_buffer_files(cls, replay_buffer_files_path: str, num_cpus: int, sample_size: int = None, min_crosses: int = None, max_crosses: int = None,
                                 raise_exception: bool = True) -> 'SumTreeReachableConfigurations':
        return reachable_configurations_from_replay_buffer_files(cls=cls, replay_buffer_files_path=replay_buffer_files_path, num_cpus=num_cpus, sample_size=sample_size, min_crosses=min_crosses, max_crosses=max_crosses, raise_exception=raise_exception)


def get_1_cross_priority(state: ReachableConfigurationsState, calc) -> float:
    assert state.high_level_state.crossing_number == 1
    link_segments, intersections = state.link_segments, state.intersections
    num_segments = len(link_segments)
    num_links = state.low_level_pos.shape[0] - 1
    if num_segments == 3:
        loop_num_links = link_segments[1][1] - link_segments[1][0]
        loop_ratio = loop_num_links / num_links
        dist = loop_ratio - 0.5
        priority = np.exp(-dist ** 2 / 0.02)
    else:
        priority = 0.
    return priority


def no_priority(state: ReachableConfigurationsState, calc) -> float:
    return 1.


get_priority = {
    1: get_1_cross_priority,
}

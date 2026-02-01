import os.path
from collections import OrderedDict
from typing import List, Dict, Iterator, Tuple, Optional, Any
import networkx as nx
from datetime import datetime
from logging import Logger
import ray
from tabulate import tabulate
from tqdm import tqdm

from exploration.utils.futures_pool import FuturesMultiprocessingPool, perform_tasks
from mujoco_infra.mujoco_utils.topology.BFS import generate_next
from exploration.utils.mixins import PickleableMixin, LoggableMixin
from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.mdp.high_level_action import HighLevelAction
from exploration.mdp.graph.directed_state_action_graph import DirectedStateActionGraph
from exploration.utils.constants import HIGH_LEVEL_GRAPHS_PATH, PROBLEMS_PATH


class HighLevelGraph(DirectedStateActionGraph, PickleableMixin, LoggableMixin):
    def __init__(self, initial_max_depth: int, max_crosses: Optional[int] = None, min_crosses: Optional[int] = None,
                 initial_state_index: Optional[int] = None, initial_state: Optional[HighLevelAbstractState] = None,
                 high_level_actions: Optional[List[str]] = None, logger: Optional[Logger] = None,
                 build_parallel: bool = False):
        DirectedStateActionGraph.__init__(self=self)
        LoggableMixin.__init__(self=self, logger=logger)
        self.depth = initial_max_depth
        self.max_crosses = max_crosses
        self.min_crosses = min_crosses
        self.initial_state_index = initial_state_index
        self.high_level_actions = high_level_actions
        self.initial_states: List[HighLevelAbstractState] = [initial_state] if initial_state is not None else []
        self.build_parallel = build_parallel
        if high_level_actions is not None:
            allowed = ['R1', 'R2', 'cross']
            assert all([a in allowed for a in high_level_actions]), f"{high_level_actions=}. {allowed=}"
        if initial_state is None:
            if build_parallel:
                assert self.min_crosses == 0, "Initial state must have 0 crossings"
                self.bfs_build_parallel()
            else:
                self.build()
        else:
            self.bfs_build(root_node=initial_state)

    @classmethod
    def base_path(cls) -> str:
        return HIGH_LEVEL_GRAPHS_PATH

    @property
    def filename(self) -> str:
        return self._get_filename(depth=self.depth, max_crosses=self.max_crosses, min_crosses=self.min_crosses,
                                  initial_state_index=self.initial_state_index, high_level_actions=self.high_level_actions)

    @classmethod
    def load_full_graph(cls, max_crosses: int = 4) -> 'HighLevelGraph':
        return cls.load_kwargs(
            path=cls.base_path(),
            depth=max_crosses,
            max_crosses=max_crosses,
            min_crosses=0,
            high_level_actions=['R1', 'R2', 'cross'],
            build_parallel=True
        )

    @classmethod
    def load_kwargs(cls, path: str, depth: int, max_crosses: Optional[int] = None, min_crosses: Optional[int] = None,
                    initial_state_index: Optional[int] = None, high_level_actions: List[str] = None,
                    logger: Optional[Logger] = None, build: bool = True, cls_class: Optional['HighLevelGraph'] = None,
                    build_parallel: bool = False, **cls_kwargs) -> Optional['HighLevelGraph']:
        cls_class = cls_class or cls
        os.makedirs(path, exist_ok=True)
        filename = cls._get_filename(depth=depth, max_crosses=max_crosses, min_crosses=min_crosses,
                                     initial_state_index=initial_state_index, high_level_actions=high_level_actions)
        full_path = os.path.join(path, f'{filename}.pkl')
        if os.path.exists(full_path):
            obj = cls_class.load(path=full_path)
            obj.logger = logger
            return obj
        else:
            st = datetime.now()
            print(f'{st} Could not find HighLevelGraph with {depth=} {max_crosses=} {min_crosses=} {high_level_actions=} at path {full_path}')
            if build:
                print(f'{st} Building HighLevelGraph with {depth=} {max_crosses=} {min_crosses=} {high_level_actions=}')
                obj = cls_class(initial_max_depth=depth, max_crosses=max_crosses, min_crosses=min_crosses,
                                initial_state_index=initial_state_index, high_level_actions=high_level_actions,
                                build_parallel=build_parallel, **cls_kwargs)
                obj.dump(path=path)
                obj.logger = logger
                return obj
            else:
                return None

    @staticmethod
    def _get_filename(depth: int, max_crosses: Optional[int], min_crosses: Optional[int],
                      initial_state_index: Optional[int], high_level_actions: Optional[List[str]]) -> str:
        base_str = f'HighLevelGraph_depth_{depth}_max_crosses_{max_crosses}_min_crosses_{min_crosses}'
        if initial_state_index is not None:
            base_str += f'_state_index_{initial_state_index}'
        if high_level_actions is not None:
            base_str += f'_{"_".join(high_level_actions)}'
        return base_str

    def all_simple_paths(self, dst: HighLevelAbstractState, src: HighLevelAbstractState = HighLevelAbstractState()) -> List[Tuple[List[HighLevelAbstractState], List[HighLevelAction]]]:
        src_i = self._state_to_i.get(src)
        assert src_i is not None, "all_simple_paths always starting from a reachable node"
        dst_i = self._state_to_i.get(dst)
        assert dst_i is not None, "all_simple_paths always ending in a reachable node"
        raw_paths = [p for p in nx.all_simple_paths(self._graph, source=src_i, target=dst_i)]
        paths = []
        for raw_path in raw_paths:
            actions_path = []
            states_path = [self._graph.nodes[n]['state'] for n in raw_path]
            for u, v in zip(raw_path[:-1], raw_path[1:]):
                action = self._graph.get_edge_data(u=u, v=v)['action']
                actions_path.append(action)
            path = (states_path, actions_path)
            paths.append(path)
        return paths

    def get_nodes_by_p_data(self, p_datas: List[str]) -> List[Optional[HighLevelAbstractState]]:
        all_p_datas = [s.p_data for s in self.states]
        intersection = set(p_datas).intersection(set(all_p_datas))
        raw_indices = [all_p_datas.index(p_data) for p_data in intersection]
        raw_nodes = [self.states[i] for i in raw_indices]
        indices = [p_datas.index(p_data) for p_data in intersection]
        nodes = [None] * len(p_datas)
        for i, node in enumerate(raw_nodes):
            nodes[indices[i]] = node
        return nodes

    def all_simple_paths_string(self, dst: HighLevelAbstractState, src: HighLevelAbstractState = HighLevelAbstractState()):
        paths = self.all_simple_paths(dst=dst, src=src)
        string_paths = []
        for path in paths:
            string_path = ''
            for action in path:
                string_path += f' => {action}'
        string_paths = '\n'.join(string_paths)
        return string_paths

    def get_parents(self, state: HighLevelAbstractState) -> List[HighLevelAbstractState]:
        i = self._state_to_i.get(state)
        assert i is not None, "get_parents always starting from an existing node"

        parents = self._graph.predecessors(i)
        all_parent_states = [self._graph.nodes[n]['state'] for n in parents]
        parent_states = []

        for parent_state in all_parent_states:
            if self.depth == 1 and parent_state.crossing_number != self.min_crosses:  # we do this only for bipartite graphs
                continue

            if self.has_path_with_depth(src=parent_state, dst=state, depth=1):
                parent_states.append(parent_state)

        return parent_states

    def has_path_with_depth(self, src: HighLevelAbstractState, dst: HighLevelAbstractState, depth: int) -> bool:
        src_i = self._state_to_i.get(src)
        assert src_i is not None, "has_path_with_depth always starting from a reachable node"
        dst_i = self._state_to_i.get(dst)
        assert dst_i is not None, "has_path_with_depth always ending in a reachable node"

        has_path = nx.has_path(self._graph, source=src_i, target=dst_i)
        path_depth = nx.shortest_path_length(self._graph, src_i, dst_i)
        path_meets_depth = path_depth == depth

        return has_path and path_meets_depth

    def get_children(self, state: HighLevelAbstractState, extend_children: bool = False) -> List[HighLevelAbstractState]:
        i = self._state_to_i.get(state)
        assert i is not None, "get_children always starting from a reachable node"

        children = self._graph.successors(i)
        children_states = [self._graph.nodes[n]['state'] for n in children]

        if len(children_states) == 0 and extend_children:
            self.extend_node(state=state)  # no children - need to extend node
            children = self._graph.successors(i)
            children_states = [self._graph.nodes[n]['state'] for n in children]

        return children_states

    def has_edge(self, src: HighLevelAbstractState, dst: HighLevelAbstractState, num_extensions: int = 1) -> bool:
        src_i = self._state_to_i.get(src)
        dst_i = self._state_to_i.get(dst)

        if src_i is None or dst_i is None:
            result = False
        else:
            result = self._graph.has_edge(src_i, dst_i)

        if not result and num_extensions > 0:
            # there might be an option that dst is a new state not encountered yet (src on the frontier)
            self.extend_node(state=src)
            result = self.has_edge(src=src, dst=dst, num_extensions=num_extensions - 1)

        return result

    def generate_next(self, state: HighLevelAbstractState) -> Iterator[Tuple[HighLevelAbstractState, Dict]]:
        for next_state, action in generate_next(state=state):
            if self.max_crosses is not None and next_state.crossing_number > self.max_crosses:
                continue

            if self.min_crosses is not None and next_state.crossing_number < self.min_crosses:
                continue

            if self.high_level_actions is not None and action['move'] not in self.high_level_actions:
                continue

            if not self.has_node(state=next_state):
                self.add_node(state=next_state)
            high_level_action = HighLevelAction(src=state, dst=next_state, data=action)
            self.add_edge(src=state, dst=next_state, action=high_level_action)

            yield next_state, action

        return StopIteration

    def extend_frontier(self):
        st = datetime.now()
        self.log_debug(msg='Extending frontier')
        for state in self.get_frontier():
            self.extend_node(state=state)
        et = datetime.now()
        self.log_debug(msg=f'Done extending frontier - total time: {et - st}')

    def extend_node(self, state: HighLevelAbstractState, verbose: bool = False):
        st = datetime.now()
        for _, _ in self.generate_next(state=state):
            pass
        et = datetime.now()
        if verbose:
            self.log_debug(msg=f'Done extending node - total time: {et - st}')

    def get_all_edges(self, move: str):
        return [edge for edge in self.edges if edge.data['move'] == move]

    def get_all_edge_variations(self, src: HighLevelAbstractState, dst: HighLevelAbstractState, from_graph: bool) -> List[HighLevelAction]:
        high_level_actions = []
        for s, a in generate_next(src):
            if s == dst:
                if from_graph:
                    if not (self.has_edge(src=src, dst=dst, num_extensions=0) and a['move'] in self.high_level_actions):
                        continue
                high_level_action = HighLevelAction(src=src, dst=dst, data=a)
                high_level_actions.append(high_level_action)
        return high_level_actions

    def unify(self, other: 'HighLevelGraph'):
        for state in other.states:
            if not self.has_node(state=state):
                self.add_node(state=state)
            for child in other.get_children(state=state, extend_children=False):
                if not self.has_node(state=child):
                    self.add_node(state=child)
                action = other.get_edge(src=state, dst=child)
                if not self.has_edge(src=state, dst=child, num_extensions=0):
                    self.add_edge(src=state, dst=child, action=action)

    def get_initial_states(self) -> List[HighLevelAbstractState]:
        st = datetime.now()

        if self.min_crosses == 0:
            initial_states = [HighLevelAbstractState()]
        else:
            initial_states = set()
            queue = [(HighLevelAbstractState(), 0)]
            while queue:
                state, depth = queue.pop(0)
                if self.depth is not None and depth >= self.depth:
                    break
                for next_state, action in generate_next(state=state):
                    if next_state.crossing_number <= self.min_crosses:
                        queue.append((next_state, 0))  # we want to start only from states with min_crosses
                    else:
                        queue.append((next_state, depth + 1))
                    if next_state.crossing_number == self.min_crosses:
                        initial_states.add(next_state)
            initial_states = list(initial_states)

        et = datetime.now()
        self.log_debug(f'{st} Getting initial states: {initial_states} took {et - st}')
        return initial_states

    @staticmethod
    def build_single_graph(**kwargs) -> Optional['HighLevelGraph']:
        if kwargs['initial_state_index'] is not None and kwargs['index'] != kwargs['initial_state_index']:
            return None
        graph = HighLevelGraph(**kwargs)
        return graph

    def build(self):
        self.initial_states = self.get_initial_states()
        graphs = []

        kwargs_list = [{
            'initial_max_depth': self.depth,
            'max_crosses': self.max_crosses,
            'min_crosses': self.min_crosses,
            'initial_state_index': None,
            'initial_state': initial_state,
            'high_level_actions': self.high_level_actions,
            'logger': None,
        } for initial_state in self.initial_states]

        for kwargs in tqdm(kwargs_list, total=len(kwargs_list), desc='Building graphs'):
            graph = HighLevelGraph.build_single_graph(**kwargs)
            if graph is not None:
                graphs.append(graph)

        for graph in tqdm(graphs, total=len(graphs), desc='Unifying graphs'):
            self.unify(other=graph)

    def bfs_build_parallel(self, num_workers: int = 88, start_depth: int = 2):
        root_node = HighLevelAbstractState()
        self.initial_states = [root_node]
        st = datetime.now()
        self.log_debug(msg=f'{st} Building High Level Graph of depth {self.depth} in parallel with {num_workers} workers')
        if self.min_crosses is not None and root_node.crossing_number >= self.min_crosses:
            self.add_node(state=root_node)
        queue = [(root_node, 0)]

        while queue:
            state, depth = queue.pop(0)

            if self.depth is not None and depth >= self.depth:  # first occurrence of max depth
                break

            if depth == start_depth:  # first occurrence of start_depth:
                queue = [(state, depth)] + queue
                self.log_debug(f'{datetime.now()} Start Depth reached. Queue size: {len(queue)}')
                break

            next_depth = depth + 1
            for next_state, _ in self.generate_next(state=state):
                queue.append((next_state, next_depth))

        if len(queue) > 0:
            kwargs_list = [{
                'initial_max_depth': self.depth - start_depth,
                'max_crosses': self.max_crosses,
                'min_crosses': initial_state.crossing_number,
                'initial_state_index': None,
                'initial_state': initial_state,
                'high_level_actions': self.high_level_actions,
                'logger': None,
            } for initial_state, _ in queue]

            processes = min(min(num_workers, len(queue)), os.cpu_count() - 2)
            self.log_debug(f'{datetime.now()} Starting parallel workers to build {len(queue)} graphs with {processes} processes')
            pool = FuturesMultiprocessingPool(processes=processes, catch_exceptions=True, verbose=True)
            graphs = perform_tasks(
                func=HighLevelGraph.build_single_graph, kwargs_list=kwargs_list, total=len(kwargs_list), pool=pool,
                parallel=True, init_pool=True, close_pool=True,
            )

            for graph in tqdm(graphs, total=len(graphs), desc=f'{datetime.now()} Unifying graphs'):
                if graph is not None:
                    self.unify(other=graph)

            et = datetime.now()
            self.log_debug(msg=f'{et} Done Building High Level Graph of depth {self.depth} - total time: {et - st}')

    def bfs_build(self, root_node: HighLevelAbstractState, log_freq: int = 500):
        """
        Builds the graph in BFS style
        """
        assert root_node is not None, "root_node must be provided"
        st = datetime.now()
        self.log_debug(msg=f'{st} Building High Level Graph of depth {self.depth}')
        if self.min_crosses is not None and root_node.crossing_number >= self.min_crosses:
            self.add_node(state=root_node)
        queue = [(self._i, root_node, 0)]

        while queue:
            node, state, depth = queue.pop(0)

            if self.depth is not None and depth >= self.depth:
                break

            next_depth = depth + 1
            for next_state, _ in self.generate_next(state=state):
                queue.append((self._i, next_state, next_depth))

            if len(queue) % log_freq == 0:
                et = datetime.now()
                self.log_debug(msg=f'{st} Queue size: {len(queue)} - total time so far: {et - st}')

        et = datetime.now()
        self.log_debug(msg=f'{et} Done Building High Level Graph of depth {self.depth} - total time: {et - st}')


@ray.remote
def get_all_simple_paths(high_level_graph: HighLevelGraph, dst: HighLevelAbstractState, src: HighLevelAbstractState = HighLevelAbstractState()):
    return high_level_graph.all_simple_paths(dst=dst, src=src)


class Problem(HighLevelGraph):
    def __init__(self, name: str, description: str, initial_max_depth: int, high_level_actions: List[str],
                 max_crosses: Optional[int] = None, min_crosses: Optional[int] = None,
                 initial_state_index: Optional[int] = None, initial_state: Optional[HighLevelAbstractState] = None,
                 logger: Optional[Logger] = None, build_parallel: bool = False):
        assert high_level_actions is not None, "high_level_actions must be provided"
        super().__init__(
            initial_max_depth=initial_max_depth,
            max_crosses=max_crosses,
            min_crosses=min_crosses,
            initial_state_index=initial_state_index,
            initial_state=initial_state,
            high_level_actions=high_level_actions,
            logger=logger,
            build_parallel=build_parallel,
        )
        self.name = name
        self.description = description

    def __repr__(self):
        return f'{self.name}: {self.description}'

    @classmethod
    def load_kwargs(cls, path: str, depth: int, max_crosses: Optional[int] = None, min_crosses: Optional[int] = None,
                    initial_state_index: Optional[int] = None, high_level_actions: List[str] = None,
                    logger: Optional[Logger] = None, build: bool = True, build_parallel: bool = False,
                    **cls_kwargs) -> Optional['Problem']:
        crossing_number_increment = [2 if a == 'R2' else 1 for a in high_level_actions]
        max_crosses = max_crosses if max_crosses is not None else max(crossing_number_increment) + min_crosses
        return super().load_kwargs(
            path=path, depth=depth, max_crosses=max_crosses, min_crosses=min_crosses,
            initial_state_index=initial_state_index, high_level_actions=high_level_actions,
            logger=logger, build=build, build_parallel=build_parallel, cls_class=cls,
            **cls_kwargs,
        )

    @classmethod
    def base_path(cls) -> str:
        return PROBLEMS_PATH

    def get_problem_states(self, value: Any = None) -> List[OrderedDict[int, OrderedDict[HighLevelAbstractState, int]]]:
        s_c_src = OrderedDict({self.min_crosses: OrderedDict({initial_state: value if value is not None else idx for idx, initial_state in enumerate(self.initial_states)})})
        s_dst = OrderedDict()
        s_dst_len = 0
        for initial_state, idx in s_c_src[self.min_crosses].items():
            children = self.get_children(state=initial_state)
            for child in children:
                child_crossing_number = child.crossing_number
                if child_crossing_number not in s_dst:
                    s_dst[child_crossing_number] = OrderedDict()
                s_dst[child_crossing_number][child] = value if value is not None else s_dst_len
                s_dst_len += 1
        return [s_c_src, s_dst]

    def get_problem_edges(self, value: Any = None) -> OrderedDict[str, OrderedDict[HighLevelAction, int]]:
        edges_len = 0
        edges = OrderedDict({move: OrderedDict() for move in self.high_level_actions})
        for move, move_edges in edges.items():
            for idx, edge in enumerate(self.get_all_edges(move)):
                item = value if value is not None else edges_len
                move_edges[edge] = item
                edges_len += 1
        return edges

    def validate(self, other: 'Problem'):
        missing_in_other = {}
        s_c_src, _ = self.get_problem_states()
        for state, idx in s_c_src[self.min_crosses].items():
            if not other.has_node(state):
                missing_in_other[idx] = state
        validate_str = f'{other.name} C={other.min_crosses} -> {self.name} C={self.min_crosses}'
        error_str = f'Found {len(missing_in_other)} {self.name} s_c_src missing nodes in {other.name} s_dst'
        missing_str = f'Missing nodes in {other.name}: {sorted(missing_in_other.keys())}'
        if len(missing_in_other) > 0:
            print(f'\n{validate_str}\n{error_str}\n{missing_str}')

    def _gather_states_by_crossing_number(self) -> Dict[int, List[HighLevelAbstractState]]:
        s_c_src, s_dst = self.get_problem_states()
        s_c_src.update(s_dst)  # TODO: note that indexes are 0,0...,N - fix this
        return s_c_src

    def print_problem(self, states: bool, edges: bool, crossing_number_states: Optional[Dict[int, List[HighLevelAbstractState]]] = None):
        print(f'\n{self.name}: {self.description}')
        states_container = crossing_number_states or self._gather_states_by_crossing_number()
        columns = ['name', 'crossing_number', 'num_states']
        data = [[self.name, crossing_number, len(states)] for crossing_number, states in states_container.items()]
        data += [[self.name, 'total', sum(item[2] for item in data)]]
        table = tabulate(data, headers=columns, tablefmt='pretty')
        print(table)
        if states:
            self.print_states()
        if edges:
            self.print_edges()

    def print_edges(self):
        columns = ['move', 'num_edges']
        edges = self.get_problem_edges()
        data = [[move, len(move_edges)] for move, move_edges in edges.items()]
        data += [['total', sum(item[1] for item in data)]]
        table = tabulate(data, headers=columns, tablefmt='pretty')
        print(table)

    def print_states(self):
        columns = ['name', 'initial_state_idx'] + [f'num_children_{a}' for a in self.high_level_actions]
        data = []
        s_c_src, _ = self.get_problem_states()
        for state, idx in s_c_src[self.min_crosses].items():
            counts = {a: 0 for a in self.high_level_actions}
            for edge in self.get_edges(src=state):
                counts[edge.data['move']] += 1
            data.append([self.name, idx] + [counts[a] for a in self.high_level_actions])
        table = tabulate(data, headers=columns, tablefmt='pretty')
        print(table)

import networkx as nx
from typing import Optional, List, Tuple
import copy

from exploration.mdp.istate import IState
from exploration.mdp.iaction import IAction


class DirectedStateActionGraph:
    def __init__(self):
        self._i = None
        self._state_to_i = {}
        self._graph = nx.DiGraph()

    @property
    def num_nodes(self) -> int:
        return self._i + 1

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def states(self) -> List[IState]:
        return [self._graph.nodes[n]['state'] for n in self._graph.nodes]

    @property
    def edges(self) -> List[IAction]:
        return [self._graph.edges[e]['action'] for e in self._graph.edges]

    def get_frontier(self) -> List[IState]:
        frontier = [self._graph.nodes[n]['state'] for n in self._graph.nodes if self._graph.out_degree(n) == 0]
        return frontier

    def add_node(self, state: IState):
        if self._i is None:
            self._i = 0
        else:
            self._i += 1
        self._graph.add_node(self._i, state=state)
        self._state_to_i[state] = copy.deepcopy(self._i)

    def get_node(self, state: IState) -> Optional[IState]:
        i = self._state_to_i.get(state)
        if i is not None:
            return self._graph.nodes[i]['state']
        return None

    def get_node_by_index(self, index: int) -> Optional[IState]:
        if index in self._graph.nodes:
            return self._graph.nodes[index]['state']
        return None

    def has_node(self, state: IState) -> bool:
        i = self._state_to_i.get(state)
        if i is None:
            return False
        return True

    def get_node_index(self, state: IState) -> Optional[int]:
        return self._state_to_i.get(state)

    def get_edge_index(self, action: IAction) -> Optional[Tuple[int, int]]:
        src_i = self._state_to_i.get(action.src)
        dst_i = self._state_to_i.get(action.dst)
        if src_i is None or dst_i is None:
            return None
        return src_i, dst_i

    def add_edge(self, src: IState, dst: IState, action: IAction):
        src_i = self._state_to_i.get(src)
        dst_i = self._state_to_i.get(dst)

        if src_i is None:
            self.add_node(state=src)
            src_i = self._state_to_i[src]
        if dst_i is None:
            self.add_node(state=dst)
            dst_i = self._state_to_i[dst]

        self._graph.add_edge(src_i, dst_i, src=src, dst=dst, action=action)

    def get_edge(self, src: IState, dst: IState) -> Optional[IAction]:
        src_i = self._state_to_i.get(src)
        assert src_i is not None, "has_edge always starting from a reachable node"
        dst_i = self._state_to_i.get(dst)
        if dst_i is None:
            return None
        else:
            if self.has_edge(src=src, dst=dst):
                edge = self._graph.edges[src_i, dst_i]['action']
                return edge
            else:
                return None

    def get_edges(self, src: IState) -> List[IAction]:
        src_i = self._state_to_i.get(src)
        assert src_i is not None, "get_edges always starting from a reachable node"
        edges = [self._graph.edges[src_i, dst_i]['action'] for dst_i in self._graph.successors(src_i)]
        return edges

    def has_edge(self, src: IState, dst: IState, **kwargs) -> bool:
        src_i = self._state_to_i.get(src)
        dst_i = self._state_to_i.get(dst)
        if src_i is None:
            result = False
        elif dst_i is None:
            result = False
        else:
            result = self._graph.has_edge(src_i, dst_i)
        return result

    def get_children(self, state: IState, **kwargs) -> List[IState]:
        i = self._state_to_i.get(state)
        assert i is not None, "get_children always starting from a reachable node"

        children = self._graph.successors(i)
        children_states = [self._graph.nodes[n]['state'] for n in children]

        return children_states

    def unify(self, other: 'DirectedStateActionGraph'):
        for state in other.states:
            if not self.has_node(state=state):
                self.add_node(state=state)
            for child in other.get_children(state=state):
                if not self.has_node(state=child):
                    self.add_node(state=child)
                action = other.get_edge(src=state, dst=child)
                if not self.has_edge(src=state, dst=child):
                    self.add_edge(src=state, dst=child, action=action)

from collections import defaultdict
from pyvis.network import Network
import networkx as nx
import numpy as np

from exploration.mdp.graph.high_level_graph import HighLevelGraph
from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.utils.server_utils import SERVER_TO_PREFIX
from system_flow.metrics.states import COMPLEXITY_STATES
from exploration.rl.test_scripts.random_evaluation_analysis import get_edge_success_rate_from_actions_table
from system_flow.metrics.h_values import H_VALUES_3_CROSSES

# ==================================== variables ========================================

server = 'server_new'
white_edge_color = True
only_success_goals = True
# complexity = 'medium'
complexity = None
lower, upper = 0, 150
action_paths = [
    'exploration/outputs/evaluation/ExplorationSAC/06-04-2025_09-39/G1_R1/best_model_collection_7/stochastic/no_inference_her',
    'exploration/outputs/evaluation/ExplorationSAC/06-04-2025_09-39/G1_Cross/best_model_collection/stochastic/no_inference_her',
    'exploration/outputs/evaluation/ExplorationSAC/11-04-2025_11-19/G2_R1/best_model_collection/stochastic/no_inference_her',
    'exploration/outputs/evaluation/ExplorationSAC/11-04-2025_11-19/G2_Cross/best_model_collection_7/stochastic/no_inference_her',
]

# ==================================== variables ========================================


high_level_graph = HighLevelGraph.load_full_graph()

prefix = SERVER_TO_PREFIX[server]
if complexity is not None:
    states = [high_level_graph.get_node_by_index(k) for k in COMPLEXITY_STATES[complexity][3]]
else:
    h_values_new = {k: v for k, v in H_VALUES_3_CROSSES.items() if lower <= v <= upper}

    states = [high_level_graph.get_node_by_index(k) for k, _ in h_values_new.items()]


edge_success_rate = defaultdict(lambda: 0)

for path in action_paths:
    path_edge_success_rate = get_edge_success_rate_from_actions_table(path)
    for k, v in path_edge_success_rate.items():
        if k in edge_success_rate:
            edge_success_rate[k] = max(edge_success_rate[k], v)
        else:
            edge_success_rate[k] = v


paths_per_state = defaultdict(list)
edges_move = defaultdict(str)
for raw_state in states:
    state = HighLevelAbstractState.from_abstract_state(raw_state)

    if only_success_goals:
        state_idx = high_level_graph.get_node_index(state)
        is_success = False
        for (u_idx, v_idx), success_rate in edge_success_rate.items():
            if v_idx == state_idx and success_rate > 0:
                is_success = True
                break
        if not is_success:
            continue

    paths = high_level_graph.all_simple_paths(src=HighLevelAbstractState(), dst=state)
    for path in paths:
        paths_per_state[state].append([(high_level_graph.get_node_index(s), s.crossing_number) for s in path[0]])
        for a in path[1]:
            a_idx = high_level_graph.get_edge_index(a)
            edges_move[a_idx] = a.data['move'] if a.data['move'] != 'cross' else 'C'

G = nx.DiGraph()
node_depths = {}
node_colors = {}

for goal_state, paths in paths_per_state.items():
    for path in paths:
        for node in path:
            index, crossing_number = node
            G.add_node(index)
            if index not in node_depths or crossing_number < node_depths[index]:
                node_depths[index] = crossing_number

            if crossing_number == 0:
                node_colors[index] = 'red'
            elif crossing_number == 3:
                node_colors[index] = 'green'
            else:
                node_colors.setdefault(index, 'gray')

        for u, v in zip(path[:-1], path[1:]):
            G.add_edge(u[0], v[0])

layers = defaultdict(list)
for node, depth in node_depths.items():
    layers[depth].append(node)

pos = {}
x_gap = 300
y_gap = 100


min_y, max_y = np.inf, -np.inf

for depth, nodes in sorted(layers.items()):
    sorted_nodes = sorted(nodes)
    num_nodes = len(sorted_nodes)
    y_offset = (num_nodes - 1) * y_gap / 2.0
    for i, node in enumerate(sorted_nodes):
        x = depth * x_gap
        y = -i * y_gap + y_offset
        pos[node] = (x, y)

net = Network(height="800px", width="100%", directed=True, notebook=True)
net.toggle_physics(False)

for node in G.nodes():
    x, y = pos[node]
    color = node_colors[node]
    net.add_node(
        node,
        label=str(node),
        color=color,
        borderWidth=5,
        borderWidthSelected=5,
        x=x,
        y=y,
        shape='circle',
        font={'color': 'black', 'size': 20},
    )

for u, v in G.edges():
    rate = edge_success_rate.get((u, v), None)
    move = edges_move[(u, v)]
    label = f"{move} {rate:.1f}%" if rate is not None else move
    label_color = 'green' if rate is not None and rate > 0 else 'black'
    edge_color = 'gray' if rate is not None and rate > 0 or not white_edge_color else 'white'
    curve_type = 'curvedCW' if pos[v][1] < 0 else 'curvedCCW'
    is_not_straight = pos[u][1] * pos[v][1] > 0
    net.add_edge(
        u,
        v,
        label=label if edge_color != 'white' else '',
        title=f'{u}-->{v} {label}',
        color=edge_color,
        arrows='to',
        smooth={
            'enabled': True,
            'type': curve_type,
            'roundness': 0.25 * is_not_straight
        },
        font={
            'align': 'horizontal',
            'size': 8,
            'vadjust': -10,
            'multi': 'html',
            'color': label_color,
        },
        labelHighlightBold=True
    )

net.show_buttons(filter_=['physics'])
net.show("interactive_graph.html")

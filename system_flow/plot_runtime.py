import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
from collections import defaultdict
from system_flow.ablations import AVAILABLE_ABLATIONS
from exploration.mdp.graph.high_level_graph import HighLevelGraph
from system_flow.evaluation_automation import StateEvaluationResult, MultiRLStateEvaluationResult

twisted_base = [63.6, 63.6, 154.5, 354.5, 418.2, 581.8, 909.1, 1236.4, 1254.5, 1481.8]

# Load the high level graph
high_level_graph = HighLevelGraph.load_full_graph()


def get_ablation_data(ablation, state_type='3-Easy'):
    """
    Extract runtime data for all states and seeds for a given ablation and state type.
    Returns a dictionary: {state_idx: [runtime1, runtime2, ...]} for all seeds
    """
    path = ablation.get_path_for_state_type(state_type)
    if not path:
        return {}

    # Get all seed directories
    base_path = pathlib.Path(path)
    seed_dirs = [f for f in base_path.iterdir() if f.is_dir() and f.name.isdigit()]

    state_runtimes = defaultdict(list)

    for seed_dir in seed_dirs:
        # For 3-Easy, we look in the '3' subdirectory
        eval_path = seed_dir / '3' if state_type != '4-Eval' else seed_dir / '4'
        if not eval_path.exists():
            continue

        # Load all StateEvaluationResult files
        state_files = [f for f in eval_path.iterdir() if 'StateEvaluationResult' in f.name]
        if len(state_files) == 0:
            continue

        file = state_files[0]
        if 'MultiRLStateEvaluationResult' in file.name:
            cls_method = MultiRLStateEvaluationResult
        else:
            cls_method = StateEvaluationResult

        for state_file in state_files:
            try:
                state_result = cls_method.load(state_file)
                if state_result.success:
                    state_idx = high_level_graph.get_node_index(state_result.state)
                    runtime = state_result.running_time.total_seconds()
                    state_runtimes[state_idx].append(runtime)
            except Exception as e:
                print(f"Error loading {state_file}: {e}")
                continue

    return dict(state_runtimes)

def get_data():
    ablation_names = [
        'TWISTED-RL-G5',
        'TWISTED-RL-A5',
        'TWISTED-RL-C5',
        'TWISTED-RL-AC5'
    ]
    ablations = selected_ablations = [abl for abl in AVAILABLE_ABLATIONS if abl.name in ablation_names]
    TWISTED_RL_C = next(abl for abl in AVAILABLE_ABLATIONS if abl.name == 'TWISTED-RL-C5')
    easy_data = {ablation.name.replace('5', ''): get_ablation_data(ablation, '3-Easy') for ablation in ablations}
    twisted_data = {state_idx: None for state_idx, _ in easy_data['TWISTED-RL-C'].items()}
    idx = 0
    for state_idx, _ in twisted_data.items():
        twisted_data[state_idx] = [twisted_base[idx]] * 5
        idx += 1
    easy_data['TWISTED'] = twisted_data
    test_sets_data = {test_set: get_ablation_data(TWISTED_RL_C, test_set) for test_set in [
        '3-Easy',
        '3-Medium-1h',
        '3-Hard',
        '3-Eval',
        '4-Eval'
    ]}
    all_data = {
        '3-Easy': easy_data,
        'C-across-sets': test_sets_data
    }
    return all_data


def compute_continuous_success_rate(runs, time_cap, testset_name=None):
    num_problems = 30 if testset_name == '3-Eval' else 10
    num_seeds = 5
    total_runs = num_problems * num_seeds

    all_times = []
    for state_idx, run in runs.items():
        all_times.extend(run)

    sorted_times = sorted(all_times)
    success_counts = np.arange(1, len(sorted_times) + 1)
    success_percentages = (success_counts / total_runs) * 100

    plot_times = np.concatenate(([0], sorted_times))
    plot_pcts = np.concatenate(([0], success_percentages))

    if plot_times[-1] < time_cap:
        plot_times = np.concatenate((plot_times, [time_cap]))
        plot_pcts = np.concatenate((plot_pcts, [plot_pcts[-1]]))

    max_time = plot_times[-1]
    ratio = time_cap / max_time if max_time > 0 else 1.0
    plot_times = plot_times * ratio

    return plot_times, plot_pcts


def create_combined_plot(all_data, save_path='combined_ablation_plot.png'):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5),
                                   gridspec_kw={'width_ratios': [1, 2]})

    colors = sns.color_palette('rocket', n_colors=5)

    # Define markers to cycle through
    markers = ['o', 's', '^', 'X', 'v']

    # --- LEFT PLOT: All Variants on 3-Easy ---
    easy_data = all_data['3-Easy']
    time_cap_left = 1800

    for i, (variant, runs) in enumerate(easy_data.items()):
        time_points, pcts = compute_continuous_success_rate(runs, time_cap_left)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax1.step(time_points, pcts, linewidth=2.5, label=variant, color=color, alpha=0.9,
                 marker=marker, markevery=0.1, markersize=8)

    ax1.set_xlabel('Time (seconds)', fontsize=14)
    ax1.set_ylabel('Cumulative Success Rate (%)', fontsize=14)
    ax1.grid(True, alpha=0.5, linestyle='--')
    ax1.set_xticks(np.arange(0, 1880, 300))
    ax1.set_xlim(0, 1880)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(loc='lower right', fontsize=12)
    ax1.set_title('(a) Performance on the 3-Easy task', fontsize=16, pad=20)

    # --- RIGHT PLOT: TWISTED-RL-C Across Test Sets ---
    testsets_data = all_data['C-across-sets']
    test_set_caps = {'3-Easy': 1800, '3-Medium-1h': 3600, '3-Hard': 1800, '3-Eval': 1800, '4-Eval': 1800}

    for i, (testset, runs) in enumerate(testsets_data.items()):
        specific_cap = test_set_caps.get(testset, 1800)
        time_points, pcts = compute_continuous_success_rate(runs, specific_cap, testset_name=testset)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax2.step(time_points, pcts, linewidth=2.5, label=testset, color=color, alpha=0.9,
                 marker=marker, markevery=0.1, markersize=8)

    ax2.axvline(x=1800, color='black', linestyle='--', alpha=0.4, linewidth=2)
    ax2.text(1850, 5, '30-min Time cap', rotation=90, alpha=0.6, fontsize=10)
    ax2.set_xlabel('Time (seconds)', fontsize=14)
    # ax2.set_ylabel('Cumulative Success Rate (%)', fontsize=14)
    ax2.grid(True, alpha=0.5, linestyle='--')
    ax2.set_xticks(np.arange(0, 3680, 300))
    ax2.set_xlim(0, 3680)
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.set_title('(b) TWISTED-RL-C performance across tasks', fontsize=16, pad=20)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    all_data = get_data()
    create_combined_plot(all_data, save_path='anytime_performance_analysis.png')
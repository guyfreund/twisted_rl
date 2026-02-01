import pathlib
import sys
from collections import defaultdict
import numpy as np
from tabulate import tabulate

sys.path.append('.')

from system_flow.ablations import AVAILABLE_ABLATIONS
from system_flow.metrics.h_values import H_VALUES_3_CROSSES, H_VALUES_4_CROSSES
from system_flow.evaluation_automation import StateEvaluationResult, MultiRLStateEvaluationResult
from exploration.mdp.graph.high_level_graph import HighLevelGraph

high_level_graph = HighLevelGraph.load_full_graph()


def get_runtime_stats(runtimes):
    if len(runtimes) == 0:
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
    runtimes = np.array(runtimes)
    mean = int(np.mean(runtimes))
    median = int(np.median(runtimes))
    min_ = int(np.min(runtimes))
    max_ = int(np.max(runtimes))
    std = int(np.std(runtimes))
    return mean, median, min_, max_, std


def print_stats(base_success_idxs, ablation_success_idxs_dict, base_total, ablation_totals,
                ablation_shared_success_runtimes_dict, base_all_success_runtimes, base_name, ablation_names,
                base_seed_rates, ablation_seed_rates):
    columns = ['metric', base_name] + ablation_names
    rows = []

    base_success_rate = len(base_success_idxs) / base_total * 100 if base_total > 0 else 0
    ablation_success_rates = {}
    for ablation_name in ablation_names:
        ablation_success_rate = len(ablation_success_idxs_dict[ablation_name]) / ablation_totals[ablation_name] * 100 if ablation_totals[ablation_name] > 0 else 0
        ablation_success_rates[ablation_name] = ablation_success_rate

    rows.append(['success rate (%)'] + [np.round(base_success_rate, 1)] + [np.round(ablation_success_rates[name], 1) for name in ablation_names])

    # --- New Section: Standard Deviation ---
    base_std = np.std(base_seed_rates) if base_seed_rates else 0
    ablation_stds = {}
    for ablation_name in ablation_names:
        rates = ablation_seed_rates.get(ablation_name, [])
        ablation_stds[ablation_name] = np.std(rates) if rates else 0

    rows.append(['success rate std'] + [np.round(base_std, 1)] + [np.round(ablation_stds[name], 1) for name in ablation_names])
    # ---------------------------------------

    rows.append(['total success'] + [len(base_success_idxs)] + [len(ablation_success_idxs_dict[name]) for name in ablation_names])
    rows.append(['total'] + [base_total] + [ablation_totals[name] for name in ablation_names])

    ablation_runtime_stats_dict = {ablation_name: get_runtime_stats(ablation_shared_success_runtimes_dict[ablation_name]) for ablation_name in ablation_names}
    runtime_metrics = ['mean success', 'median success', 'min success', 'max success', 'std success']
    for i, metric in enumerate(runtime_metrics):
        row = [f'{metric} relative runtime [%]']
        row.append('100')
        for ablation_name in ablation_names:
            row.append(ablation_runtime_stats_dict[ablation_name][i])
        rows.append(row)

    base_all_success_runtimes_ = [x for x in base_all_success_runtimes if x != 'N/A']
    rows.append(['mean runtime [s]', int(np.mean(base_all_success_runtimes_)) if base_all_success_runtimes_ else 'N/A'] + ['N/A' for _ in ablation_names])

    table = tabulate(rows, headers=columns, tablefmt='pretty')
    print(table)


def get_data(state_result, st):
    if state_result is not None:
        success = state_result.success
        running_time = int(state_result.running_time.total_seconds())
        state = state_result.state
        state_idx = high_level_graph.get_node_index(state_result.state)
        crossing_number = state.crossing_number
        count = H_VALUES_3_CROSSES[state.p_data] if '3' in st else H_VALUES_4_CROSSES[state.p_data]
    else:
        success = 'N/A'
        running_time = 'N/A'
        state = None
        state_idx = None
        crossing_number = None
        count = None
    return success, running_time, state, state_idx, crossing_number, count


def analyze(base_eval_path, ablation_eval_paths, st, ablation_names, seed, print_full_table, print_success_only, base_name):
    base_state_paths = [f for f in pathlib.Path(base_eval_path).iterdir() if 'StateEvaluationResult' in f.name]
    base_states = [StateEvaluationResult.load(base_state_path) for base_state_path in base_state_paths]
    base_data = {high_level_graph.get_node_index(s.state): s for s in base_states}

    ablation_data_dict = {}
    for ablation_path in ablation_eval_paths:
        ablation_state_paths = [f for f in pathlib.Path(ablation_path).iterdir() if 'StateEvaluationResult' in f.name]
        ablation_states = [StateEvaluationResult.load(ablation_state_path) for ablation_state_path in ablation_state_paths]
        ablation_data = {high_level_graph.get_node_index(s.state): s for s in ablation_states}
        ablation_data_dict[ablation_path] = ablation_data

    shared_data = defaultdict(dict)
    for idx, state in base_data.items():
        shared_data[idx]['base'] = state

    for i, ablation_path in enumerate(ablation_eval_paths):
        for idx, state in ablation_data_dict[ablation_path].items():
            shared_data[idx][ablation_names[i]] = state

    columns = ['number', 'state_idx', 'crossing_number', 'h_value (count)', f'{base_name} success', f'{base_name} runtime [s]']
    for name in ablation_names:
        columns.append(f'{name} success')
        columns.append(f'{name} runtime [s]')

    rows = []
    base_success_idxs = []
    base_runtimes = []
    base_total = 0
    ablation_success_idxs_dict = defaultdict(list)
    ablation_runtimes_dict = defaultdict(list)
    ablation_totals = defaultdict(int)
    for state_idx, state_data in shared_data.items():
        base_state_result = state_data.get('base', None)
        is_state_success = False

        base_success, base_running_time, base_state, base_state_idx, base_crossing_number, base_count = get_data(base_state_result, st)
        if base_success != 'N/A':
            base_total += 1
        if base_success is True:
            base_success_idxs.append(state_idx)
            base_runtimes.append(base_running_time)
            is_state_success = True

        row = [state_idx, None, None, base_success, base_running_time]

        crossing_number = None
        count = None
        if base_state_result is not None:
            crossing_number = base_crossing_number
            count = base_count

        for name in ablation_names:
            ablation_state_result = state_data.get(name, None)
            if ablation_state_result is not None:
                ablation_success, ablation_running_time, _, _, ablation_crossing_number, ablation_count = get_data(ablation_state_result, st)
                row.extend([ablation_success, ablation_running_time])
                if crossing_number is None:
                    crossing_number = ablation_crossing_number
                    count = ablation_count
                if ablation_success != 'N/A':
                    ablation_totals[name] += 1
                if ablation_success is True:
                    ablation_success_idxs_dict[name].append(state_idx)
                    ablation_runtimes_dict[name].append(ablation_running_time)
                    is_state_success = True
            else:
                row.extend(['N/A', 'N/A'])

        row[1] = crossing_number
        row[2] = count

        if print_success_only and not is_state_success:
            continue
        rows.append(row)

    rows = [[i + 1] + row for i, row in enumerate(rows)]
    rows = sorted(rows, key=lambda x: x[1])

    # add a summary row
    base_success_rate = len(base_success_idxs) / base_total * 100 if base_total > 0 else 0
    base_mean, base_median, base_min, base_max, base_std = get_runtime_stats(base_runtimes)
    base_success_rate = np.round(base_success_rate, 1)
    base_mean = np.round(base_mean, 1) if base_mean != 'N/A' else 'N/A'
    inner_list = []
    for ablation_name in ablation_names:
        ablation_success_rate = len(ablation_success_idxs_dict[ablation_name]) / ablation_totals[ablation_name] * 100 if ablation_totals[ablation_name] > 0 else 0
        ablation_mean, ablation_median, ablation_min, ablation_max, ablation_std = get_runtime_stats(ablation_runtimes_dict[ablation_name])
        ablation_success_rate = np.round(ablation_success_rate, 1)
        ablation_mean = np.round(ablation_mean, 1) if ablation_mean != 'N/A' else 'N/A'
        inner_list.extend([ablation_success_rate, ablation_mean])
    rows.append(['all', 'all', 'all', 'all', base_success_rate, base_mean] + inner_list)

    if print_full_table:
        table = tabulate(rows, headers=columns, tablefmt='pretty')
        print(f'================================================================== seed {seed} ==================================================================')
        print(table)
        print(f'================================================================== seed {seed} ==================================================================')

    ablation_shared_success_runtimes_dict = {}

    for i, name in enumerate(ablation_names):
        shared_idxs = set(base_success_idxs).intersection(set(ablation_success_idxs_dict[name]))
        base_shared_runtimes = [base_runtimes[i] for i, idx in enumerate(base_success_idxs) if idx in shared_idxs]
        ablation_shared_runtimes = [ablation_runtimes_dict[name][i] for i, idx in enumerate(ablation_success_idxs_dict[name]) if idx in shared_idxs]
        ablation_shared_runtimes = np.round(np.array(ablation_shared_runtimes) / np.array(base_shared_runtimes) * 100).astype(int)
        ablation_shared_success_runtimes_dict[name] = ablation_shared_runtimes

    # Pass specific seed totals back to main for rate tracking
    return base_success_idxs, ablation_success_idxs_dict, base_total, ablation_totals, ablation_shared_success_runtimes_dict, base_runtimes


def main(baseline_ablation, comparison_ablations, st_list, print_full_table, print_success_only, mask):
    # Convert ablation objects to AblationPaths objects
    baseline_paths = baseline_ablation
    comparison_paths = comparison_ablations

    # Get ablation names (excluding the baseline)
    ablation_names = [abl.name for abl in comparison_paths]
    base_name = baseline_paths.name

    for st, enable in zip(st_list, mask):
        if not enable:
            continue
        print(
            f'======================================================== {st} Evaluation ========================================================')

        # Get the appropriate paths for this state type
        base_path = baseline_paths.get_path_for_state_type(st)
        if not base_path:
            print(f"Skipping {st} for baseline {base_name} - path not provided")
            continue

        ablation_paths = []
        valid_ablation_names = []
        for abl_path in comparison_paths:
            path = abl_path.get_path_for_state_type(st)
            if path:
                ablation_paths.append(path)
                valid_ablation_names.append(abl_path.name)
            else:
                print(f"Skipping {abl_path.name} for {st} - path not provided")

        if not ablation_paths:
            print(f"No valid ablation paths for {st}, skipping")
            continue

        base_success_idxs_list = []
        ablation_success_idxs_dict = {name: [] for name in valid_ablation_names}
        base_total_list = []
        ablation_totals_dict = {name: [] for name in valid_ablation_names}
        ablation_shared_success_runtimes_dict = {name: [] for name in valid_ablation_names}
        base_all_success_runtime_list = []

        # --- New Lists for Per-Seed Rate Tracking ---
        base_seed_rates = []
        ablation_seed_rates_dict = defaultdict(list)
        # --------------------------------------------

        base_seeds = [f.name for f in pathlib.Path(base_path).iterdir() if f.is_dir() and f.name.isdigit()]

        ablation_seeds = []
        for ablation_path in ablation_paths:
            ablation_seeds.extend(
                [f.name for f in pathlib.Path(ablation_path).iterdir() if f.is_dir() and f.name.isdigit()])

        all_seeds = sorted(set(base_seeds).union(set(ablation_seeds)) - {'342'})

        for seed in all_seeds:
            base_seed_path = pathlib.Path(base_path) / seed
            base_eval_path = base_seed_path / '3' if '3' in st else base_seed_path / '4'
            base_eval_path.mkdir(parents=True, exist_ok=True)

            ablation_eval_paths = []
            for ablation_path in ablation_paths:
                ablation_seed_path = pathlib.Path(ablation_path) / seed
                ablation_eval_path = ablation_seed_path / base_eval_path.name
                ablation_eval_path.mkdir(parents=True, exist_ok=True)
                ablation_eval_paths.append(ablation_eval_path)

            base_success_idxs, seed_ablation_success_idxs_dict, base_total, seed_ablation_totals, seed_ablation_shared_success_runtimes_dict, base_all_success_runtime = analyze(
                base_eval_path, ablation_eval_paths, st, valid_ablation_names, seed, print_full_table,
                print_success_only, base_name)

            # --- Capture Success Rate for this Seed ---
            seed_base_rate = len(base_success_idxs) / base_total * 100 if base_total > 0 else 0
            base_seed_rates.append(seed_base_rate)

            for name in valid_ablation_names:
                seed_total = seed_ablation_totals[name]
                seed_success_count = len(seed_ablation_success_idxs_dict[name])
                seed_rate = seed_success_count / seed_total * 100 if seed_total > 0 else 0
                ablation_seed_rates_dict[name].append(seed_rate)
            # ------------------------------------------

            base_success_idxs_list.extend(base_success_idxs)
            for name in valid_ablation_names:
                ablation_success_idxs_dict[name].extend(seed_ablation_success_idxs_dict[name])

            base_total_list.append(base_total)
            for name in valid_ablation_names:
                ablation_totals_dict[name].append(seed_ablation_totals[name])

            for name in valid_ablation_names:
                ablation_shared_success_runtimes_dict[name].extend(seed_ablation_shared_success_runtimes_dict[name])

            base_all_success_runtime_list.extend(base_all_success_runtime)

        print_stats(
            base_success_idxs_list,
            ablation_success_idxs_dict,
            sum(base_total_list),
            {name: sum(ablation_totals_dict[name]) for name in valid_ablation_names},
            ablation_shared_success_runtimes_dict,
            base_all_success_runtime_list,
            base_name,
            valid_ablation_names,
            base_seed_rates,          # Passed new argument
            ablation_seed_rates_dict  # Passed new argument
        )
        print(
            f'======================================================== {st} Evaluation ========================================================')
        print()
        print()


if __name__ == '__main__':
    # =============================== CONFIGURATION ===============================
    # baseline_ablation_name = 'TWISTED-RL-G'  # Default baseline
    baseline_ablation_name = 'TWISTED-RL-G5'  # Default baseline

    mask = [
        True,  # 3-Easy
        True,  # 3-Medium
        True,  # 3-Medium-1h
        True,  # 3-Hard
        True,  # 4-Easy-Eval
        True,  # 3-Eval
    ]

    ablations_mask = [
        'TWISTED',
        # 'TWISTED-RL-G',
        # 'TWISTED-RL-A',
        # 'TWISTED-RL-C',
        # 'TWISTED-RL-AC',
        # 'TWISTED-RL-H',
        # 'TWISTED-RL-ALL',
        # 'TWISTED-RL-ALL-G',
        # 'TWISTED-RL-ALL-A',
        # 'TWISTED-RL-ALL-C',
        # 'TWISTED-RL-ALL-AC',
        # 'TWISTED-RL-C5S',
        'TWISTED-RL-G5',
        'TWISTED-RL-C5',
        'TWISTED-RL-A5',
        'TWISTED-RL-AC5',
        # 'TWISTED5',
        # 'TWISTED-RL-H5',
    ]

    print_success_only = True
    print_full_table = False
    # =============================== CONFIGURATION ===============================

    assert len(mask) == 6, "Mask should have exactly 6 elements corresponding to the state types."
    assert any(abl_name == baseline_ablation_name for abl_name in ablations_mask), "Baseline ablation should be in the ablations mask."
    baseline_ablation = next(abl for abl in AVAILABLE_ABLATIONS if abl.name == baseline_ablation_name)
    comparison_ablations = [abl for abl in AVAILABLE_ABLATIONS if abl.name != baseline_ablation.name and abl.name in ablations_mask]

    states_types = [
        '3-Easy',
        '3-Medium',
        '3-Medium-1h',
        '3-Hard',
        '4-Easy-Eval',
        '3-Eval',
    ]

    main(baseline_ablation, comparison_ablations, states_types, print_full_table, print_success_only, mask)
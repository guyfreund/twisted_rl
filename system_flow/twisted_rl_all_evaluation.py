import pathlib
import sys
import traceback
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from tabulate import tabulate

sys.path.append('.')

from system_flow.evaluation_automation import MultiRLStateEvaluationResult
from system_flow.ablations import EvaluationSet


@dataclass
class StateAblationData:
    data: Dict[str, Dict[str, bool]]
    success: bool


def print_twisted_rl_all_breakdown(evaluation_set: EvaluationSet, state_types: List[str], mask: List[bool]):
    for state_type, enable in zip(state_types, mask):
        if not enable:
            continue

        print(
            f'======================================================== {state_type} TWISTED-RL-ALL Breakdown ========================================================')

        eval_path = evaluation_set.get_path_for_state_type(state_type)
        if not eval_path:
            print(f"No path found for state type {state_type}")
            continue

        seeds = [f.name for f in pathlib.Path(eval_path).iterdir() if f.is_dir() and f.name.isdigit()]

        if not seeds:
            print(f"No seed directories found in {eval_path}")
            continue

        all_results = []

        for seed in seeds:
            seed_path = pathlib.Path(eval_path) / seed
            sub_dir = '3' if '3' in state_type else '4'
            eval_dir = seed_path / sub_dir

            if not eval_dir.exists():
                continue

            result_files = [f for f in eval_dir.iterdir() if 'StateEvaluationResult' in f.name]

            for result_file in result_files:
                try:
                    result = MultiRLStateEvaluationResult.load(result_file)
                    all_results.append(result)
                except Exception as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    continue

        if not all_results:
            print(f"No valid StateEvaluationResult files with ablation data found for {state_type}")
            continue

        state_ablations_data = []
        for result in all_results:
            data = {}
            plan_length = None
            for ablation_name, ablation in result.ablations.items():
                plan_length = len(ablation)
                if len(ablation) > 0:
                    all_success = all(ablation)
                    final_success = ablation[-1]
                    success = all_success or final_success
                    data[ablation_name] = {'all_success': all_success, 'final_success': final_success, 'success': success}
            assert plan_length is not None, f"Plan length is None for {result}"
            for idx in range(plan_length):
                success_ablations = []
                for ablation_name, ablation in result.ablations.items():
                    if len(ablation) > 0:
                        if ablation[idx]:
                            success_ablations.append(ablation_name)
                for ablation_name, ablation_data in data.items():
                    if len(success_ablations) == 1 and success_ablations[0] == ablation_name:
                        ablation_data['is_critical'] = True
                    else:
                        ablation_data['is_critical'] = False
            state_ablation_data = StateAblationData(data=data, success=result.success)
            state_ablations_data.append(state_ablation_data)

        columns = ["ablation", "SR (%)", "all SR (%)", "final SR (%)", "is critical rate (%)", "all success", "final success", "is critical", "success", "total success", "total trials"]
        rows = []

        total = len(state_ablations_data)
        total_success = len([s for s in state_ablations_data if s.success])
        ablation_names = [list(s.data.keys()) for s in state_ablations_data]
        ablation_names = [item for sublist in ablation_names for item in sublist]
        ablation_names = set(ablation_names)
        ablation_names = sorted(list(ablation_names), key=lambda x: {'G': 0, 'A': 1, 'C': 2, 'AC': 3}.get(x, 4))

        for ablation_name in ablation_names:
            ablation_all_success, ablation_final_success, ablation_success, ablation_critical = 0, 0, 0, 0
            for state_ablation_data in state_ablations_data:
                if ablation_name not in state_ablation_data.data:
                    continue
                data = state_ablation_data.data[ablation_name]
                ablation_all_success += data['all_success']
                ablation_final_success += data['final_success']
                ablation_success += data['success']
                ablation_critical += data['is_critical']
            all_success_rate = np.round(ablation_all_success / total_success * 100, 1) if total_success > 0 else 0
            final_success_rate = np.round(ablation_final_success / total_success * 100, 1) if total_success > 0 else 0
            is_critical = np.round(ablation_critical / total_success * 100, 1) if total_success > 0 else 0
            success_rate = np.round(ablation_success / total * 100, 1) if total > 0 else 0
            rows.append([ablation_name, success_rate, all_success_rate, final_success_rate, is_critical, ablation_all_success, ablation_final_success, ablation_critical, ablation_success, total_success, total])

        table = tabulate(rows, headers=columns, tablefmt='pretty')
        print(table)
        caption = """\nMetrics Legend:
• ablation: Name of the ablation component being analyzed
• SR (%): Percentage of trials where the ablation contributed to success (success / total)
• all SR (%): Percentage of successful trials where this ablation achieved success at every step (all success / total success)
• final SR (%): Percentage of successful trials where this ablation  succeeded at the final step (final success / total success)
• is critical rate (%): Percentage of successful trials where this ablation was the only one to succeed at some step, making it critical for success (critical / total success)
\nStatistic Legend:
• all success: Raw count of trials where this ablation succeeded at every step
• final success: Raw count of trials where this ablation succeeded at the final step
• is critical: Raw count of trials where this ablation was critical for success
• success: Raw count of trials where this ablation contributed to success (either all steps or final step)
• total success: Total number of successful trials across all ablations
• total trials: Total number of trials conducted""".strip()

        # print(caption)
        print()
        print(f'TWISTED-RL-ALL Success Rate: {np.round(total_success / total * 100, 1) if total > 0 else 0}%')
        print(f'======================================================== {state_type} TWISTED-RL-ALL Breakdown ========================================================')
        print()


def main():
    # Define TWISTED-RL-ALL evaluation set
    TWISTED_RL_ALL = EvaluationSet(
        name="TWISTED-RL-ALL",
        easy_path="exploration/outputs/evaluation/twisted_evaluation/19-05-2025_13-55/",
        medium_path="exploration/outputs/evaluation/twisted_evaluation/19-05-2025_13-56/",
        medium_1h_path="exploration/outputs/evaluation/twisted_evaluation/20-05-2025_19-41/",
        hard_path="exploration/outputs/evaluation/twisted_evaluation/19-05-2025_17-46/",
        four_easy_eval_path="exploration/outputs/evaluation/twisted_evaluation/22-05-2025_10-21/",
        three_eval_path="exploration/outputs/evaluation/twisted_evaluation/20-05-2025_19-42/",
    )

    state_types = [
        '3-Easy',
        '3-Medium',
        '3-Medium-1h',
        '3-Hard',
        '4-Easy-Eval',
        '3-Eval',
    ]

    # Configure which state types to evaluate
    mask = [
        True,  # 3-Easy
        True,  # 3-Medium
        True,  # 3-Medium-1h
        True,  # 3-Hard
        True,  # 4-Easy-Eval
        True,  # 3-Eval
    ]

    print_twisted_rl_all_breakdown(TWISTED_RL_ALL, state_types, mask)


if __name__ == '__main__':
    main()
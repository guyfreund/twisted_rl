import glob
import os
from copy import deepcopy
from typing import List, Any, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from tabulate import tabulate
from exploration.mdp.graph.high_level_graph import HighLevelGraph
from exploration.mdp.high_level_state import HighLevelAbstractState


def get_edge_success_rate_from_actions_table(actions_table_path: str) -> dict:
    df = pd.read_csv(os.path.join(actions_table_path, 'goal_actions.csv') if not actions_table_path.endswith('.csv') else actions_table_path)
    move_list = df.move.values.tolist()
    success_rate_list = df.success_rate.values.tolist()
    data = {}
    for move, success_rate in zip(move_list, success_rate_list):
        if not move.startswith('('):
            continue
        uv_str = move.split(')')[0]
        uv_str = uv_str.replace('(', '').replace(')', '')
        u, v = uv_str.split(', ')
        u = int(u)
        v = int(v)
        data[(u, v)] = np.round(success_rate * 100, 1)
    return data



def print_table(rows: List[Any], columns: List[str], print_success_only: bool):
    copy_rows = deepcopy(rows)
    if print_success_only:
        copy_rows = [row for row in copy_rows if row[-3] > 0.02]
    for row in copy_rows:
        row[-3] = f'{str(np.round(row[-3] * 100, 1))}%'  # move success rate to percentage
    table = tabulate(copy_rows, headers=columns, tablefmt='pretty')
    print(table)


def plot_table(rows: List[Any], columns: List[str], her: bool, plot: bool, save_table: bool, save_plot: bool,
               show: bool, print_success_only: bool, deterministic_str: str, algo_name: str, problem_name: str,
               model_name: str, plot_name: str, to_print: bool = True, save_path: Optional[str] = None, log_to_wandb: bool = False,
               wandb_run=None):
    if to_print:
        print(f'{algo_name} {deterministic_str} {model_name} - {problem_name}{" - HER Inference" if her else ""}')
        all_rows = [row for row in rows if row[0] == 'all']
        indices_rows = [row for row in rows if row[0] != 'all']
        # sort indices_rows by indices
        indices_rows = sorted(indices_rows, key=lambda x: x[0])
        rows = all_rows + indices_rows
        print_table(rows=rows, columns=columns, print_success_only=print_success_only)

    # Save to disk
    if save_path is not None and save_table:
        df = pd.DataFrame(rows, columns=columns)
        csv_save_path = os.path.join(save_path, f'{plot_name}.csv')
        df.to_csv(csv_save_path, index=False)
        if to_print:
            print(f'Saved table to {csv_save_path}')

    # Handle plotting
    if not plot:
        return

    # Handle unique rows
    unique_rows = [row for row in rows if row[0] == 'all']
    [rows.remove(row) for row in unique_rows]
    unique_df = pd.DataFrame(unique_rows, columns=columns)
    unique_column = columns[1]
    df = pd.DataFrame(rows, columns=columns)

    # Handle naming
    words = [word.capitalize() for word in plot_name.split('_')]
    name = ' '.join(words)

    # Convert success rate to percentage format
    df.success_rate = df.success_rate.astype(float) * 100
    df.success_rate = df.success_rate.round(1)
    unique_df.success_rate = unique_df.success_rate.astype(float) * 100
    unique_df.success_rate = unique_df.success_rate.round(1)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Create figure and axis
    plt.figure(figsize=(12, 6))

    # Create bar plot
    ax = sns.barplot(x=df.idx, y=df.success_rate, color="royalblue", alpha=0.7)

    # Ensure x-axis labels are correctly set as strings
    ax.set_xticks(df.idx.values.tolist())  # Set correct number of ticks
    ax.set_xticklabels(df[unique_column], rotation=60, ha='right')
    ax.set_yticks(range(0, 101, 10))

    for i, (success, total) in enumerate(zip(df.total_success, df.total)):
        ax.text(i, df.success_rate.iloc[i] + 1, f"$\\frac{{{success}}}{{{total}}}$", ha='center', fontsize=12)

    # Add overall success rate lines
    color_palette = ['black', 'green', 'red']
    for i in range(len(unique_rows)):
        total = unique_df.total.iloc[i]
        success = unique_df.total_success.iloc[i]
        label = f"{unique_df[unique_column].iloc[i]} {unique_column} Success Rate {unique_df.success_rate.iloc[i]}% ({success}/{total})"
        plt.axhline(y=unique_df.success_rate.iloc[i], color=color_palette[i], linestyle='--', label=label)

    # Labels and title
    plt.xlabel(name)
    plt.ylabel(f"{name} Success Rate (%)")
    her_str = ' with Inference HER' if her else ''
    plt.title(f"{algo_name} {deterministic_str} {model_name} - {problem_name} - {name} Success Rate by {unique_column}{her_str}")
    plt.legend(loc='upper center')
    plt.tight_layout()

    # Save to disk
    if save_path is not None and save_plot:
        png_save_path = os.path.join(save_path, f'{plot_name}.png')
        plt.savefig(png_save_path, dpi=500, bbox_inches='tight')
        if to_print:
            print(f"Saved plot to {png_save_path}")

    # Create wandb image before closing the plot
    wandb_image = None
    if log_to_wandb:
        if wandb_run is not None:
            wandb_image = wandb_run.Image(plt)
        else:
            wandb_image = wandb.Image(plt)

    # Show the plot
    if show:
        plt.show()

    # Handle the end of plotting
    plt.close()

    # Return wandb image
    return wandb_image


def plot_table_with_data(container_success, container_total, to_print, unique_column, plot_name, save_path,
                         her, plot, save_table, save_plot, show, deterministic_str, algo_name, problem_name,
                         model_name, log_to_wandb, print_success_only, wandb_run=None):
    total_success = sum(sum(container_success[unique_column].values()) for unique_column in container_success)
    total = sum(sum(container_total[unique_column].values()) for unique_column in container_total)
    success_rate = np.round(total_success / total, 3) if total > 0 else 0
    if to_print:
        print(f'{plot_name} all success rate: {success_rate} ({total_success}/{total})')
    columns = ['idx', unique_column, 'success_rate', 'total_success', 'total']
    rows = [['all', 'all', success_rate, total_success, total]]
    unique_values = sorted(list(container_total.keys()))
    all_rows = [[] for _ in range(len(unique_values))]
    high_level_graph = HighLevelGraph.load_full_graph()
    current_index = 0
    for i, unique_value in enumerate(unique_values):
        total_success = sum(container_success[unique_value].values())
        total = sum(container_total[unique_value].values())
        success_rate = np.round(total_success / total, 3) if total > 0 else 0
        all_rows[i].append(['all', unique_value, success_rate, total_success, total])
        if to_print:
            print(f'{plot_name} with {unique_column}={unique_value} success rate: {success_rate} ({total_success}/{total})')
        for obj, total_success in container_success[unique_value].items():
            if isinstance(obj, HighLevelAbstractState):
                idx = high_level_graph.get_node_index(obj)
                str_idx = idx
            else:
                idx = current_index
                str_idx = high_level_graph.get_edge_index(obj)
            total = container_total[unique_value][obj]
            success_rate = np.round(total_success / total, 3) if total > 0 else 0
            all_rows[i].append([idx, f'{str_idx} {str(obj)}', success_rate, total_success, total])
            current_index += 1
    for unique_rows in all_rows:
        rows.append(unique_rows[0])  # add only the first row of each move - the "all" row
    for unique_rows in all_rows:
        rows.extend(unique_rows[1:])
    image = plot_table(rows=rows, columns=columns, save_path=save_path, her=her, plot=plot, save_table=save_table,
                       save_plot=save_plot, show=show, print_success_only=print_success_only, deterministic_str=deterministic_str,
                       algo_name=algo_name, problem_name=problem_name, model_name=model_name, to_print=to_print,
                       log_to_wandb=log_to_wandb, plot_name=plot_name, wandb_run=wandb_run)
    return image


def run_analysis(episodes, episode_times, is_her_list, model_evaluation_output_dir, her, plot,
                 save_table, save_plot, show, deterministic_str, algo_name, problem, model_name, to_print, log_to_wandb,
                 wandb_run=None):
    _, children_success = problem.get_problem_states(value=0)
    children_total = deepcopy(children_success)
    edges_success = problem.get_problem_edges(value=0)
    edges_total = deepcopy(edges_success)

    for episode, episode_time, is_her in zip(episodes, episode_times, is_her_list):
        states, actions, raw_actions, rewards, done_flags, truncateds, infos = episode
        success = infos[-1]['experience'].info.goal_reached
        goal_state = infos[-1]['experience'].goal_state
        goal_crossing_number = goal_state.crossing_number
        children_total[goal_crossing_number][goal_state] += 1
        goal_action = infos[-1]['experience'].goal_action
        goal_action_move = goal_action.data['move']
        edges_total[goal_action_move][goal_action] += 1
        if success:
            children_success[goal_crossing_number][goal_state] += 1
            edges_success[goal_action_move][goal_action] += 1

    shared_kwargs = {
        'save_path': model_evaluation_output_dir,
        'her': her,
        'plot': plot,
        'save_table': save_table,
        'save_plot': save_plot,
        'show': show,
        'to_print': to_print,
        'deterministic_str': deterministic_str,
        'algo_name': algo_name,
        'problem_name': problem.name,
        'model_name': model_name,
        'log_to_wandb': log_to_wandb,
        'print_success_only': False,
        'wandb_run': wandb_run,
    }
    states_image = plot_table_with_data(
        container_success=children_success,
        container_total=children_total,
        unique_column='crossing_number',
        plot_name='goal_states',
        **shared_kwargs
    )
    actions_image = plot_table_with_data(
        container_success=edges_success,
        container_total=edges_total,
        unique_column='move',
        plot_name='goal_actions',
        **shared_kwargs
    )
    images = [states_image, actions_image]

    return images


def view_results(model_dir, plot, save_table, save_plot, show, print_success_only, algo_name, problem_name):
    if not os.path.exists(model_dir):
        print(f'Evaluation output directory {model_dir} does not exist')
        return

    for model in sorted(os.listdir(model_dir)):
        model_path = os.path.join(model_dir, model)
        model_name = os.path.basename(model_path)
        summary_paths = glob.glob(f'{model_path}/**/summary.txt', recursive=True)
        states_paths = glob.glob(f'{model_path}/**/goal_states.csv', recursive=True)
        actions_paths = glob.glob(f'{model_path}/**/goal_actions.csv', recursive=True)

        for summary_path, states_path, actions_path in zip(summary_paths, states_paths, actions_paths):
            save_path = os.path.dirname(summary_path)
            her_str = os.path.basename(save_path)
            her = 'no' not in her_str
            deterministic_str = os.path.basename(os.path.dirname(save_path))

            print(f'====================== {model} {deterministic_str} {her_str} ======================')
            with open(summary_path, 'r') as f:
                summary_rows = [row.strip() for row in f.readlines()]
            for row in summary_rows:
                print(row)

            states_df = pd.read_csv(states_path)
            states_columns = states_df.columns
            states_rows = [[v for _, v in row.to_dict().items()] for _, row in states_df.iterrows()]
            plot_table(rows=states_rows, columns=states_columns, print_success_only=print_success_only,
                       her=her, plot=plot, save_table=save_table, save_plot=save_plot, show=show,
                       save_path=save_path, deterministic_str=deterministic_str,
                       algo_name=algo_name, problem_name=problem_name, model_name=model_name,
                       to_print=True, log_to_wandb=False, plot_name='goal_states')

            actions_df = pd.read_csv(actions_path)
            actions_columns = actions_df.columns
            actions_rows = [[v for _, v in row.to_dict().items()] for _, row in actions_df.iterrows()]
            plot_table(rows=actions_rows, columns=actions_columns, print_success_only=print_success_only,
                       her=her, plot=plot, save_table=save_table, save_plot=save_plot, show=show,
                       save_path=save_path, deterministic_str=deterministic_str,
                       algo_name=algo_name, problem_name=problem_name, model_name=model_name,
                       to_print=True, log_to_wandb=False, plot_name='goal_actions')
            print(f'====================== {model} {deterministic_str} {her_str} ======================')

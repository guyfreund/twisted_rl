import sys
import traceback
import torch
import yaml
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta
from pytorch_lightning import seed_everything
from typing import List, Dict, Union, Optional
import os
from easydict import EasyDict as edict
from tabulate import tabulate
from tqdm import tqdm
import pytz
import numpy as np
from functools import partial
from uuid import uuid4

sys.path.append('.')

from system_flow.high_level_class.high_level import HighLevelPlanner
from exploration.mdp.graph.high_level_graph import HighLevelGraph
from exploration.rl.environment.exploration_gym_envs import ExplorationGymEnvs
from exploration.utils.config_utils import dump_config, load_config, load_env_config, set_agents_by_ablation
from exploration.utils.mixins import PickleableMixin
from mujoco_infra.mujoco_utils.general_utils import argparse_create, edict2dict, run_with_limited_time
from mujoco_infra.mujoco_utils.topology.representation import AbstractState
from system_flow.metrics.states import COMPLEXITY_STATES, EVAL_3_RANDOM_BINS_STATES, EVAL_4_EASY_800

HIGH_LEVEL_CATALOG = {
    "guided_by_high_level": HighLevelPlanner
}


def update_config_init(cfg):
    cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["config_length"] = (cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["num_of_links"]-1)*2+7
    #config + hot vectore action + x,y,height + position num_of_links*3
    cfg['LOW_LEVEL']['STATE2STATE_PARMS']['input_size'] = cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["config_length"] +\
         cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["num_of_links"] + 3
    cfg['LOW_LEVEL']['STATE2STATE_PARMS']['output_size'] = cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["config_length"]
    if cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["return_with_init_position"]:
        cfg['LOW_LEVEL']['STATE2STATE_PARMS']['input_size'] += 3*(cfg['LOW_LEVEL']["STATE2STATE_PARMS"]["num_of_links"]+1)
    return cfg


@dataclass
class StateEvaluationResult(PickleableMixin):
    state: AbstractState
    num_crosses: int
    running_time: timedelta
    success: bool
    seed: int
    idx: int
    checkpoint_path: str
    checkpoint_paths: Optional[List[str]]

    @property
    def filename(self) -> str:
        return f'{self.__class__.__name__}_{self.seed}_{self.idx}_{self.num_crosses}_{self.success}_{str(self.running_time.total_seconds()).replace(".", "_")}_{uuid4().hex}'

    def print(self):
        print(f'State {self.idx} {self.state} with {self.num_crosses} crosses, {self.success=}, {self.seed=}, {self.running_time=}')


@dataclass
class MultiRLStateEvaluationResult(StateEvaluationResult):
    ablations: Dict[str, List[bool]]


@dataclass
class EvaluationResult(PickleableMixin):
    results: Dict[int, Dict[int, List[StateEvaluationResult]]]
    checkpoint_path: str
    _results_per_crossing_number = None
    algo = 'TWISTED_BASELINE'

    @property
    def filename(self) -> str:
        model_path = ''.join(self.checkpoint_path.replace('/', '_').split('.')[:-1])
        return f'{self.__class__.__name__}_{self.algo}_{model_path}'

    @property
    def results_per_crossing_number(self):
        if self._results_per_crossing_number is None:
            seeds = list(self.results.keys())
            crossing_numbers = list(self.results[seeds[0]].keys())
            results_per_crossing_number = {crossing_number: [] for crossing_number in crossing_numbers}
            for crossing_number in crossing_numbers:
                for seed, seed_results in self.results.items():
                    results_per_crossing_number[crossing_number].extend(seed_results[crossing_number])
            self._results_per_crossing_number = results_per_crossing_number
        return self._results_per_crossing_number

    def print_statistics(self, top_print: bool = True):
        rows = []
        statistics = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'min': np.min,
            'max': np.max,
        }
        columns = {
            'success_rate': 'Success Rate (%)',
            'total': 'Total',
            'total_success': 'Total Success',
            'crossing_number': 'Crosses',
        }
        columns.update({f'running_time_{statistic_name}': f'Runtime {statistic_name.capitalize()} (m)' for statistic_name in statistics})

        for crossing_number, results in sorted(self.results_per_crossing_number.items(), key=lambda x: x[0]):
            row = {}

            total = len(results)
            total_success = sum([result.success for result in results])
            success_rate = total_success / total if total > 0 else 0.
            row['crossing_number'] = crossing_number
            row['success_rate'] = np.round(success_rate * 100, 1)
            row['total'] = total
            row['total_success'] = total_success

            running_times = [result.running_time.total_seconds() for result in results if result.success]
            for statistic_name, statistic_func in statistics.items():
                if len(running_times) > 0:
                    row[f'running_time_{statistic_name}'] = np.round(statistic_func(running_times) / 60, 1)
                else:
                    row[f'running_time_{statistic_name}'] = 0

            rows.append(row)

        table = tabulate(rows, headers=columns, tablefmt='pretty')
        if top_print:
            if self.checkpoint_path:
                print(f'Evaluation results for TWISTED BASELINE with checkpoint: {self.checkpoint_path}')
            print(table)
        return table


@dataclass
class RLEvaluationResult(EvaluationResult):
    checkpoint_paths: Union[List[str], Dict[str, str]]
    algo = 'TWISTED_RL'

    @staticmethod
    def get_name_from_path(path):
        parts = path.split('/')
        level, params, date, best_model = parts[-4:]
        name = f'{level}_{date}_{best_model}'
        return name

    @property
    def filename(self) -> str:
        names = []
        if isinstance(self.checkpoint_paths, list):
            for checkpoint_path in self.checkpoint_paths:
                if checkpoint_path is None:
                    continue
                name = self.get_name_from_path(checkpoint_path)
                names.append(name)
        elif isinstance(self.checkpoint_paths, dict):
            for problem_name, checkpoint_path in self.checkpoint_paths.items():
                if checkpoint_path is not None:
                    name = self.get_name_from_path(checkpoint_path)
                    checkpoint_name = f'{problem_name}_{name}'
                    names.append(checkpoint_name)
        else:
            raise ValueError(f'Unknown type of checkpoint_paths: {type(self.checkpoint_paths)}')
        models = '_'.join(names)
        filename = f'{self.__class__.__name__}_{self.algo}_{models}'
        filename = f'{self.__class__.__name__}_{self.algo}_{uuid4().hex}'
        return filename

    @classmethod
    def load_from_results_path(cls, results_path: str, checkpoint_paths: List[str], checkpoint_path: str = ''):
        results: Dict[int, Dict[int, List[StateEvaluationResult]]] = {}

        for seed in os.listdir(results_path):
            if int(seed) not in results:
                results[int(seed)] = {}
            for crossing_number in os.listdir(os.path.join(results_path, seed)):
                if int(crossing_number) not in results[int(seed)]:
                    results[int(seed)][int(crossing_number)] = []
                for state in os.listdir(os.path.join(results_path, seed, crossing_number)):
                    state_path = os.path.join(results_path, seed, crossing_number, state)
                    state_result = StateEvaluationResult.load(state_path)
                    results[int(seed)][int(crossing_number)].append(state_result)

        obj = cls(results=results, checkpoint_paths=checkpoint_paths, checkpoint_path=checkpoint_path)
        obj.dump(results_path)
        return obj

    def print_statistics(self, top_print: bool = True):
        table = super().print_statistics(top_print=False)
        if top_print:
            print(f'Evaluation results for {self.algo} with checkpoints:')
            for problem_name, checkpoint_path in self.checkpoint_paths.items():
                print(f'{problem_name}: {checkpoint_path}')
            print(table)
        return table


def main(cfg, args):
    if not torch.cuda.is_available():
        raise

    to_raise = False
    resume_dir = args.resume

    if resume_dir is None:
        israel_tz = pytz.timezone('Israel')
        now_in_israel = datetime.now(israel_tz).strftime("%d-%m-%Y_%H-%M")
    else:
        now_in_israel = os.path.basename(os.path.normpath(args.resume))

    high_level_graph = HighLevelGraph.load_full_graph()

    evaluation_cfg = cfg.EVALUATION
    states_cfg = evaluation_cfg.states
    if evaluation_cfg.states_type in ['easy', 'medium', 'hard']:
        p_datas = COMPLEXITY_STATES[evaluation_cfg.states_type]
    elif evaluation_cfg.states_type == '3-eval':
        p_datas = EVAL_3_RANDOM_BINS_STATES
    elif evaluation_cfg.states_type == '4-eval':
        p_datas = EVAL_4_EASY_800
    elif evaluation_cfg.states_type == 'p-data':
        p_datas = [p_data.replace('\\n', '\n') for p_data in evaluation_cfg.p_datas]
    else:
        raise ValueError(f'Unknown states type: {evaluation_cfg.states_type}')

    nodes = high_level_graph.get_nodes_by_p_data(p_datas)
    states = defaultdict(list)
    for node in nodes:
        states[node.crossing_number].append(node)

    number_of_seeds = evaluation_cfg.number_of_seeds
    seeds = [seed for seed in range(number_of_seeds)] if len(evaluation_cfg.seeds) == 0 else evaluation_cfg.seeds
    results = {seed: {int(num_crosses): [] for num_crosses in list(states_cfg.keys())} for seed in seeds}
    failed_runs = []

    if cfg.LOW_LEVEL.NAME == "S2APlanner":
        checkpoint_path = cfg.LOW_LEVEL.STATE2ACTION_PARMS.path
        evaluation_results_cls = partial(EvaluationResult, checkpoint_path=checkpoint_path)
        state_evaluation_result_cls = partial(StateEvaluationResult, checkpoint_path=checkpoint_path, checkpoint_paths=None)
    elif cfg.LOW_LEVEL.NAME == "RLPlanner":
        checkpoint_paths = cfg.LOW_LEVEL.RL.agents
        evaluation_results_cls = partial(RLEvaluationResult, checkpoint_paths=checkpoint_paths, checkpoint_path='')
        state_evaluation_result_cls = partial(StateEvaluationResult, checkpoint_paths=checkpoint_paths, checkpoint_path='')
    elif cfg.LOW_LEVEL.NAME == "MultiRLPlanner":
        checkpoint_paths = cfg.LOW_LEVEL.RL.agents
        evaluation_results_cls = partial(RLEvaluationResult, checkpoint_paths=checkpoint_paths, checkpoint_path='')
        state_evaluation_result_cls = partial(MultiRLStateEvaluationResult, checkpoint_paths=checkpoint_paths, checkpoint_path='')
    else:
        raise ValueError(f'Unknown low level planner: {cfg.LOW_LEVEL.NAME}')

    save_path = f'exploration/outputs/evaluation/twisted_evaluation/{now_in_israel}'
    os.makedirs(save_path, exist_ok=True)
    exceptions_dir = os.path.join(save_path, 'exceptions')
    os.makedirs(exceptions_dir, exist_ok=True)
    print(f'Evaluation results will be saved to {save_path}')

    # dump config
    config_path = os.path.join(save_path, 'twisted_evaluation.yml')
    dump_config(cfg=cfg, path=config_path)
    print(f'Saved config to {config_path}')

    def get_config(path: str, effective_batch_size: int) -> edict:
        config = load_env_config(path)
        config.num_cpus = effective_batch_size
        config.env.reachable_configurations.replay_buffer_files_path = None
        config.env.min_crosses = 0
        config.env.max_crosses = 4
        config.env.depth = 4
        config.env.high_level_actions = ['R1', 'R2', 'cross']
        config.file_system.output_dir = save_path
        config.file_system.exceptions_dir = exceptions_dir
        return config

    env = None
    if cfg.LOW_LEVEL.NAME != "S2APlanner":
        env_path = 'exploration/rl/environment/exploration_env.yaml'
        st = datetime.now()
        print(f"{st} TWISTED Evaluation: Loading environment from {env_path}")
        effective_batch_size = int(1.5 * cfg.LOW_LEVEL.batch_size)
        env_cfg = get_config(env_path, effective_batch_size)
        env = ExplorationGymEnvs.from_cfg(cfg=env_cfg, init_pool_context=True)
        assert env is not None
        et = datetime.now()
        print(f"{et} TWISTED Evaluation: Loading environment took {et - st}")

    episode_length = {a: {i: [] for i in range(3)} for a in ['R1', 'R2', 'cross']}
    try:
        for iteration in range(evaluation_cfg.iterations):
            for seed in seeds:
                seed_everything(seed=seed)
                for num_crosses, num_crosses_cfg in states_cfg.items():
                    state_save_path = os.path.join(save_path, str(seed), str(num_crosses))
                    num_crosses = int(num_crosses)
                    os.makedirs(state_save_path, exist_ok=True)
                    running_timedelta = timedelta(**num_crosses_cfg.time)
                    if num_crosses not in states:
                        continue
                    idxs = range(len(states[num_crosses]))

                    for idx in tqdm(idxs, total=len(idxs), desc=f'{seed=}, {num_crosses=}'):
                        state = states[num_crosses][idx]
                        state_idx = high_level_graph.get_node_index(state)

                        if resume_dir is not None:
                            files = [f for f in os.listdir(state_save_path) if f.endswith('.pkl') and f'{seed}_{state_idx}_{num_crosses}' in f]
                            if len(files) > 0:
                                files_str = '\n'.join(files)
                                if len(files) > 1:
                                    print(f'Found more than one file: {files_str}')
                                    raise ValueError(f'Found more than one file: {files_str}')
                                file = files[0]
                                print(f'Resume is On. Found file {file} for {seed=}, {num_crosses=} and {idx=} and high level graph idx {state_idx=}. Skipping')
                                continue

                        print(f'Running evaluation for {seed=}, {num_crosses=} and {idx=} and high level graph idx {state_idx=}')
                        cfg.GENERAL_PARAMS.output_path = os.path.join(state_save_path, str(state_idx))
                        system = HIGH_LEVEL_CATALOG[cfg["HIGH_LEVEL"]["type"]](args, cfg)
                        system.set_new_goal(state)
                        try:
                            if cfg.LOW_LEVEL.NAME == "S2APlanner":
                                success, running_time = run_with_limited_time(system.tie_knot, (), {'seed': seed}, int(running_timedelta.total_seconds()))
                            else:
                                success, running_time = system.tie_knot(seed=seed, running_time=running_timedelta, env=env)
                                for a, data in episode_length.items():
                                    for crossing_number, _ in data.items():
                                        episode_length[a][crossing_number].extend(system.low_planner.episode_length[a][crossing_number])
                            kwargs = {
                                'running_time': running_time,
                                'success': success,
                                'seed': seed,
                                'state': state,
                                'num_crosses': num_crosses,
                                'idx': state_idx,
                            }
                            if cfg.LOW_LEVEL.NAME == "MultiRLPlanner":
                                kwargs['ablations'] = system.low_planner.current_plan_success
                            state_evaluation_result = state_evaluation_result_cls(**kwargs)
                            state_evaluation_result.print()
                            state_evaluation_result.dump(state_save_path)
                            print(f'Saved state {idx} with {num_crosses} crosses and seed {seed} to {state_save_path}')
                            results[seed][num_crosses].append(state_evaluation_result)

                        except Exception as e:
                            traceback.print_exception(type(e), e, e.__traceback__)
                            if to_raise:
                                raise e
                            else:
                                failed_runs.append((seed, num_crosses, idx, state, traceback.format_exception(type(e), e, e.__traceback__)))
                                continue
                        finally:
                            try:
                                if hasattr(system, 'low_planner'):
                                    if hasattr(system.low_planner, 'env'):
                                        system.low_planner.env = None
                                        system.low_planner.close()
                                    del system.low_planner
                                    print('deleted low_planner')
                                del system
                                print('deleted system')
                                import gc
                                # torch.cuda.set_device(1)
                                with torch.no_grad():
                                    torch.cuda.empty_cache()
                                    print('emptied cuda cache')
                                gc.collect()
                                print('collected garbage')
                            except Exception as e:
                                traceback.print_exception(type(e), e, e.__traceback__)
                                raise e

        for a, data in episode_length.items():
            for crossing_number, episode_lengths in data.items():
                print(f"{a=} {crossing_number=} average episode length: {np.mean(episode_lengths)}, total: {len(episode_lengths)}")
        evaluation_result = evaluation_results_cls(results=results)
        evaluation_result.dump(path=save_path)
        print(f'Saved evaluation results to {save_path}')

        for failed_run in failed_runs:
            seed, num_crosses, idx, state, trackback_str = failed_run
            print(f'Failed run: {seed=}, {num_crosses=}, {idx=}, {state=}')
            print(''.join(trackback_str))

        if len(failed_runs) > 0:
            raise Exception

    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    args = argparse_create()
    args.cfg = "system_flow/config/twisted_evaluation.yml"
    cfg = load_config(path=args.cfg)
    cfg["GENERAL_PARAMS"]["config"] = args.cfg
    args.env_path = cfg['HIGH_LEVEL'].env_path
    low_level_planner = args.low_level_planner
    if low_level_planner is not None:
        cfg["LOW_LEVEL"]["NAME"] = low_level_planner
        print(f'Using low level planner from args: {low_level_planner}')
    states_type = args.states_type
    if states_type is not None:
        cfg["EVALUATION"]["states_type"] = states_type
        print(f'Using states type from args: {states_type}')
    seeds = args.seeds
    if seeds is not None:
        cfg["EVALUATION"]["seeds"] = seeds
        print(f'Using seeds from args: {seeds}')
    timecap = args.timecap
    if timecap is not None:
        for _, num_crosses_cfg in cfg["EVALUATION"].states.items():
            num_crosses_cfg.time.minutes = timecap
            num_crosses_cfg.time.seconds = 0
    search_factor = args.search_factor
    if search_factor is not None:
        print(f'before: {cfg["LOW_LEVEL"]["batch_size"]=}, {cfg["LOW_LEVEL"]["Select_Samples"]["num_samples"]=}')
        cfg["LOW_LEVEL"]["batch_size"] = int(cfg["LOW_LEVEL"]["batch_size"] * search_factor)
        cfg["LOW_LEVEL"]["Select_Samples"]["num_samples"] = int(cfg["LOW_LEVEL"]["Select_Samples"]["num_samples"] * search_factor)
        print(f'Using search factor from args: {search_factor}')
        print(f'after: {cfg["LOW_LEVEL"]["batch_size"]=}, {cfg["LOW_LEVEL"]["Select_Samples"]["num_samples"]=}')
    ablation = args.ablation
    if ablation is not None:
        set_agents_by_ablation(cfg=cfg.LOW_LEVEL, ablation=ablation)
        if ablation == 'ALL':
            all_ablations = ['G', 'A', 'C', 'AC']
            remove_all_ablation = args.remove_all_ablation
            if remove_all_ablation is not None:
                all_ablations.remove(remove_all_ablation)
            cfg.LOW_LEVEL.RL.ablations = all_ablations
        elif low_level_planner == 'MultiRLPlanner':
            cfg.LOW_LEVEL.RL.ablations = [ablation] * 5

    cfg = update_config_init(cfg)
    print(yaml.dump(edict2dict(cfg), indent=4, default_flow_style=False, sort_keys=True))

    main(cfg, args)

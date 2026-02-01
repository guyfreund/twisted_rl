import pickle
import traceback
from datetime import datetime
import pytz
import os
import time
from collections import deque, defaultdict
from functools import partial
import shutil
import numpy as np
import wandb
from pytorch_lightning import seed_everything
from tqdm import tqdm
from absl import logging
from itertools import accumulate
from typing import Iterable

logging.set_verbosity(logging.FATAL)

from exploration.rl.replay_buffer.in_mem_efficient_length_per_buffer import InMemEfficientLengthPrioritizedReplayBuffer
from exploration.mdp.graph.problem_set import ProblemSet
from exploration.rl.test_scripts.random_evaluation_analysis import run_analysis
from exploration.utils.schedule.schedule_factory import ScheduleFactory
from exploration.rl.environment.exploration_gym_envs import ExplorationGymEnvs, plot_collection_summary
from exploration.utils.wandb_utils import wandb_log_fn
from exploration.utils.config_utils import dump_config, load_env_config
from mujoco_infra.mujoco_utils.general_utils import edict2dict
from exploration.rl.environment.pool_episode_runner import get_subprocess_count, get_memory_usage_run, \
    get_memory_usage_machine
from exploration.rl.replay_buffer.in_mem_efficient_per_buffer import InMemEfficientPrioritizedReplayBuffer, \
    OldAndNewInMemEfficientPrioritizedReplayBuffers


def calculate_returns(rewards: Iterable[float], gamma: float):
    reversed_rewards = rewards[::-1]
    acc = list(accumulate(reversed_rewards, lambda x, y: x * gamma + y))
    return acc[::-1]


def _extract_test_metrics(episode_results, env, gamma):
    states, actions, raw_actions, rewards, dones, truncateds, infos = episode_results
    returns_ = calculate_returns(rewards, gamma)
    success_ = episode_results[-1][-1]['experience'].info.goal_reached
    return success_, returns_[0], states[0], raw_actions[0]  # success, full episode return, first state, first action


class Trainer:
    def __init__(self, config, algorithm_init_fn, algorithm_name, name=None, problem=None,
                 agent_load_path=None, replay_buffer_files_path=None, hindsight_sharing=None, seed=None):
        self.env, self.config, self.problem, self.save_path, self.wandb_run, self.her_replay_buffer_files, self.problem_set = self.setup(
            config=config,
            algorithm_name=algorithm_name,
            problem_name=problem,
            name=name,
            agent_load_path=agent_load_path,
            replay_buffer_files_path=replay_buffer_files_path,
            hindsight_sharing=hindsight_sharing,
            seed=seed,
        )
        self.algorithm_init_fn = algorithm_init_fn
        self.buffer = None
        assert self.problem is not None, 'Problem not found'

        if self.config.train.hindsight_sharing:
            level_high_level_actions = ['R1', 'R2'] if self.env.min_crosses == 0 and self.env.depth == 1 else ['R1', 'R2', 'cross']
            self.adj_high_level_actions = list(set(level_high_level_actions) - set(self.env.high_level_actions))
            self.adj_problems = {}
            for hla in self.adj_high_level_actions:
                if self.env.depth == 1:
                    max_crosses = self.env.min_crosses + 1 if hla != 'R2' else 2
                else:
                    max_crosses = self.env.max_crosses
                kwargs = {
                    'min_crosses': self.env.min_crosses,
                    'max_crosses': max_crosses,
                    'depth': self.env.depth,
                    'high_level_actions': [hla],
                }
                problem = self.problem_set.get_problem_by_kwargs(**kwargs)
                self.adj_problems[hla] = problem
                os.makedirs(os.path.join(self.her_replay_buffer_files, hla), exist_ok=True)
            self.last_loaded_her_files = {hla: -1 for hla in self.adj_high_level_actions}
            self.last_saved_her_files = {hla: -1 for hla in self.adj_high_level_actions}
        else:
            self.adj_high_level_actions = []
            self.adj_problems = {}
            self.last_loaded_her_files = {}
            self.last_saved_her_files = {}

        # training stats
        self.start_time: float = None
        self.env_steps: int = 0  # counts env steps
        self.global_steps: int = 0  # counts model updates
        self.episodes: int = 0  # counts collected episodes
        self.best_success_rate: float = -1.
        self.validation_round = 0 # counts validation rounds
        self.collection_success_rate = -1.
        self.best_collection_success_rate = -1.

        # the algorithm itself
        self.agent = self.algorithm_init_fn(config.algorithm, self.env)

        # init the epsilon greedy scheduler
        self.exploration_schedule = ScheduleFactory.create_from_cfg(cfg=config.train.schedule)

        # init the replay buffer scheduler
        if 'replay_buffer_schedule' in config.train:
            self.replay_buffer_schedule = ScheduleFactory.create_from_cfg(cfg=config.train.replay_buffer_schedule)
        else:
            self.replay_buffer_schedule = None

    @staticmethod
    def setup(config, algorithm_name, problem_name, name, agent_load_path, replay_buffer_files_path, hindsight_sharing, seed):
        if agent_load_path is not None:
            config.algorithm.agent_load_path = agent_load_path
            print(f'Using agent load path from args {agent_load_path}')
        if hindsight_sharing is not None:
            config.train.hindsight_sharing = hindsight_sharing
            print(f'Using hindsight sharing from args {hindsight_sharing}')

        env_cfg_path = config.env.env_cfg_path
        env_config = load_env_config(env_cfg_path)

        if seed is not None:
            env_config.seed = seed
            print(f'Using seed from args {seed}')

        if replay_buffer_files_path is not None:
            env_config.env.reachable_configurations.replay_buffer_files_path = replay_buffer_files_path
            print(f'Using replay buffer files path from args {replay_buffer_files_path}')

        problem_set = ProblemSet()
        if problem_name is not None:
            problem = problem_set.get_problem_by_name(problem_name)
            env_config.env.min_crosses = problem.min_crosses
            env_config.env.max_crosses = problem.max_crosses
            env_config.env.depth = problem.depth
            env_config.env.high_level_actions = problem.high_level_actions
            print(f'Using problem {problem_name} from args')
        else:
            kwargs = {
                'min_crosses': env_config.env.min_crosses,
                'max_crosses': env_config.env.max_crosses,
                'depth': env_config.env.depth,
                'high_level_actions': env_config.env.high_level_actions,
            }
            problem = problem_set.get_problem_by_kwargs(**kwargs)

        config.env.update(env_config)
        group = problem.name

        # handle file system
        if name is None:
            israel_tz = pytz.timezone('Israel')
            now_in_israel = datetime.now(israel_tz)
            name = now_in_israel.strftime("%d-%m-%Y_%H-%M")

        output_dir = os.path.join('exploration/outputs/training', algorithm_name, name, group)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        exceptions_dir = os.path.join(output_dir, 'exceptions')
        if os.path.exists(exceptions_dir):
            shutil.rmtree(exceptions_dir)
        os.makedirs(exceptions_dir)

        her_replay_buffer_files = os.path.join(output_dir, 'her_replay_buffer_files')
        if os.path.exists(her_replay_buffer_files):
            shutil.rmtree(her_replay_buffer_files)
        os.makedirs(her_replay_buffer_files)

        # update env cfg
        config.env.file_system.output_dir = output_dir
        config.env.file_system.exceptions_dir = exceptions_dir

        # seed and init ray
        seed_everything(config.env.seed)

        # load env
        env = ExplorationGymEnvs.from_cfg(config.env)

        # get config as type dict
        config_dict = edict2dict(config)

        # init wandb
        # run = wandb.init(project=algorithm_name, group=group, name=f'{group}_{name}', config=config_dict, dir=output_dir)
        run = wandb.init(project='ExplorationSAC', group=group, name=f'{group}_{name}', config=config_dict, dir=output_dir)
        run.define_metric("stats/global_steps")
        run.define_metric("stats/env_steps")
        run.define_metric("stats/val_steps")
        run.define_metric("stats/test_steps")
        run.define_metric("stats/ongoing_test_steps")
        run.define_metric("train/*", step_metric="stats/global_steps")
        run.define_metric("collection/*", step_metric="stats/env_steps")
        run.define_metric("validation/*", step_metric="stats/val_steps")
        run.define_metric("test/*", step_metric="stats/test_steps")
        run.define_metric("ongoing_test/*", step_metric="stats/ongoing_test_steps")

        # dump config
        dump_config(cfg=config, path=os.path.join(output_dir, 'config.yml'))

        return env, config, problem, output_dir, run, her_replay_buffer_files, problem_set

    @property
    def epsilon(self) -> float:
        return self.exploration_schedule.value(step=self.env_steps)

    @property
    def old_replay_buffer_sample_rate(self) -> float:
        if self.replay_buffer_schedule is None:
            return 0.
        else:
            return self.replay_buffer_schedule.value(step=self.env_steps)

    def load_her_replay_buffer_files(self):
        if not self.config.train.hindsight_sharing:
            return

        for hla in self.adj_high_level_actions:
            num_loaded = 0
            adj_problem = self.adj_problems[hla]
            adj_problem_save_path = os.path.join(os.path.dirname(self.save_path), adj_problem.name)

            if os.path.exists(adj_problem_save_path):  # if the adj problem is training
                her_files_path = os.path.join(self.her_replay_buffer_files, hla)
                if not os.path.exists(her_files_path):
                    continue

                her_files = sorted([f for f in os.listdir(her_files_path)])
                if len(her_files) == 0:  # check if her_files_path is empty
                    continue

                for i in range(self.last_loaded_her_files[hla] + 1, len(her_files)):
                    her_file = os.path.join(her_files_path, her_files[i])
                    with open(her_file, 'rb') as f:
                        trajectory = pickle.load(f)
                    self.buffer.add(trajectory)
                    num_loaded += 1

                self.last_loaded_her_files[hla] = len(her_files) - 1
                print(f'{self.problem.name} Problem {self.problem.name} loaded {num_loaded} HER files from {adj_problem.name} her replay buffer files')

    def handle_trajectory(self, trajectory):
        if trajectory['is_her']:
            hla = trajectory['infos'][0]['experience'].goal_action.data['move']
            if hla in self.env.high_level_actions:
                self.buffer.add(trajectory)

            elif self.config.train.hindsight_sharing:
                adj_problem = self.adj_problems.get(hla, None)
                if adj_problem is not None:
                    adj_problem_save_path = os.path.join(os.path.dirname(self.save_path), adj_problem.name)
                    if os.path.exists(adj_problem_save_path):  # if the adj problem is training
                        her_files_path = os.path.join(adj_problem_save_path, 'her_replay_buffer_files', self.problem.high_level_actions[0])
                        os.makedirs(her_files_path, exist_ok=True)
                        her_file = f'{self.last_saved_her_files[hla] + 1}.pkl'
                        her_file_path = os.path.join(her_files_path, her_file)
                        with open(her_file_path, 'wb') as f:
                            pickle.dump(trajectory, f)
                        self.last_saved_her_files[hla] += 1
        else:
            self.buffer.add(trajectory)


    def _get_stats(self):
        return {'env_steps': self.env_steps, 'episodes': self.episodes, 'global_steps': self.global_steps,
                'epsilon_greedy_schedule': self.epsilon, 'best_success_rate': self.best_success_rate,
                'val_steps': self.validation_round, 'test_steps': 0}

    def _test_queries(self, deterministic, stage, step):
        self.agent.eval()
        episodes, times, is_her = self.env.play_test_queries(
            get_actions=partial(self.agent.predict_action, deterministic=deterministic),
            iterations=self.config.train.validation_iterations,
            apply_her=False,
        )
        deterministic_str = 'deterministic' if deterministic else 'stochastic'
        model_name = f'{deterministic} {stage} {step}'
        images = run_analysis(
            episodes=episodes,
            episode_times=times,
            is_her_list=is_her,
            model_evaluation_output_dir=None,
            model_name=model_name,
            deterministic_str=deterministic_str,
            her=True,
            algo_name='TWISTED_RL',
            problem=self.problem,
            plot=True,
            save_table=False,
            save_plot=False,
            show=False,
            to_print=False,
            log_to_wandb=True,
            wandb_run=self.wandb_run,
        ) if self.config.train.run_analysis else None
        per_episode_rewards, per_episode_success = [], []
        first_states, first_actions = [], []
        for episode in episodes:
            states, actions, raw_actions, rewards, done_flags, truncateds, infos = episode
            episode_length = len(actions)
            if episode_length == 0:
                continue  # can be if experienced an exception on the first step in the episode
            episode_result = _extract_test_metrics(episode, self.env, self.config.algorithm.gamma)
            success, episode_return, first_state, first_action = episode_result
            per_episode_rewards.append(episode_return)
            per_episode_success.append(success)
            first_states.append(first_state)
            first_actions.append(first_action)
        return np.mean(per_episode_success), np.mean(per_episode_rewards), first_states, first_actions, per_episode_rewards, images

    def _do_validation(self):
        self.validation_round += 1
        validation_results = self._test_queries(self.config.train.deterministic_test, 'validation', self.validation_round)
        success_rate, mean_rewards, first_states, first_actions, returns, images = validation_results
        q_estimation_bias = np.mean(self.agent.get_q_values(first_states, first_actions).flatten() - np.array(returns))
        print(f'{self.problem.name} validation - env_steps: {self.env_steps}: success_rate: {success_rate}, rewards: {mean_rewards}, q_estimation_bias: {q_estimation_bias}')
        wandb_log_fn({
            "validation": {
                'success_rate': success_rate, 'rewards': mean_rewards, 'q_estimation_bias': q_estimation_bias,
                f'goal_states': images[0] if images is not None else -1,
                f'goal_actions': images[1] if images is not None else -1,
            },
            "stats": {
                'val_steps': self.validation_round,
                'best_success_rate': self.best_success_rate,
            },
            "machine_metrics": {
                'process_count_requested': self.env.num_workers + self.config.train.replay_buffer_workers * (self.config.train.replay_buffer_type != 0),
                'process_count_actual': get_subprocess_count() if self.config.stats.plot_process_count_actual else 0,
                'run_memory_usage': get_memory_usage_run() if self.config.stats.plot_run_memory_usage else 0,
                'machine_memory_usage': get_memory_usage_machine() if self.config.stats.plot_machine_memory_usage else 0
            }
        }, wandb_run=self.wandb_run)
        self._save_if_better(success_rate)

    def _save_if_better(self, validation_success_rate):
        if validation_success_rate > self.best_success_rate:
            self.best_success_rate = validation_success_rate
            self.agent.save(os.path.join(self.save_path, 'best_model'))
            self.agent.save(os.path.join(self.save_path, f'best_model_{self.env_steps // self.config.train.test_playback_resolution}'))

        if self.collection_success_rate > self.best_collection_success_rate:
            self.best_collection_success_rate = self.collection_success_rate
            self.agent.save(os.path.join(self.save_path, 'best_model_collection'))
            self.agent.save(os.path.join(self.save_path, f'best_model_collection_{self.env_steps // self.config.train.test_playback_resolution}'))

    def train(self, close_buffer: bool = True):
        self.start_time = time.time()  # compute the total time of the script

        # replay buffer
        save_path = os.path.join(self.save_path, 'replay_buffer_files')
        replay_buffer_kwargs = {
            'batch_size': self.config.algorithm.batch_size,
            'max_buffer_size': self.config.train.buffer_size,
            'output_dir': save_path,
            'store_preprocessed_trajectory': self.config.train.store_preprocessed_trajectory,
            'save_at_end': True,
            'use_per_weights': self.config.algorithm.per_weights,
            'use_per_sampling': self.config.algorithm.per_sampling,
        }
        if self.config.train.replay_buffer_type == 3:
            self.buffer = InMemEfficientPrioritizedReplayBuffer(
                per_minus=self.config.algorithm.per_minus,
                alpha=self.config.algorithm.per_alpha,
                beta=self.config.algorithm.per_beta,
                **replay_buffer_kwargs
            )
        elif self.config.train.replay_buffer_type == 4:
            self.buffer = OldAndNewInMemEfficientPrioritizedReplayBuffers(
                per_minus=self.config.algorithm.per_minus,
                alpha=self.config.algorithm.per_alpha,
                beta=self.config.algorithm.per_beta,
                **replay_buffer_kwargs
            )
        elif self.config.train.replay_buffer_type == 5:
            self.buffer = InMemEfficientLengthPrioritizedReplayBuffer(
                agent=self.agent,
                batch_size_factor=self.config.algorithm.batch_size_factor,
                **replay_buffer_kwargs
            )
        else:
            raise ValueError(f'Unknown replay buffer type: {self.config.train.replay_buffer_type}')

        # load replay buffer if needed
        if self.config.train.replay_buffer_load_path is not None:
            num_trajectories_to_load = self.config.train.replay_buffer_num_trajectories_to_load if 'replay_buffer_num_trajectories_to_load' in self.config.train else None
            keep_stale_trajectories_ratio = self.config.train.replay_buffer_keep_stale_trajectories_ratio if 'replay_buffer_keep_stale_trajectories_ratio' in self.config.train else 1.
            print(f'{self.problem.name} Loading replay buffer from {self.config.train.replay_buffer_load_path}')
            self.buffer.load(
                path=self.config.train.replay_buffer_load_path,
                num_trajectories_to_load=num_trajectories_to_load,
                num_cpus=self.env.pool_context.workers,
                keep_stale_trajectories_ratio=keep_stale_trajectories_ratio,
            )
            print(f'{self.problem.name} Done loading replay buffer from {self.config.train.replay_buffer_load_path}')

        if self.config.train.collection_starts > 0:
            raise NotImplementedError('collection_starts > 0 is not supported yet')
            # if self.config.train.replay_buffer_load_path is not None:
            #     path = self.config.train.replay_buffer_load_path
            # else:
            #     path = self.config.env.env.reachable_configurations.replay_buffer_files_path
            # self.buffer.load(path=path, num_steps_to_load=self.config.train.replay_buffer_num_steps_to_load)

        # Load agent if needed, e.g. when training G2_R1, load G1_R1
        if 'agent_load_path' in self.config.algorithm and self.config.algorithm.agent_load_path is not None:
            print(f'{self.problem.name} Loading agent from {self.config.algorithm.agent_load_path}')
            self.agent.load(self.config.algorithm.agent_load_path)
            print(f'{self.problem.name} Done loading agent from {self.config.algorithm.agent_load_path}')

        # save initial model
        self.agent.save(os.path.join(self.save_path, 'best_model'))
        self.agent.save(os.path.join(self.save_path, 'best_model_0'))
        self.agent.save(os.path.join(self.save_path, 'best_model_collection_0'))
        # initial validation
        self._do_validation()

        dataloader_iterator = None
        last_validation = 0

        rolling_rewards = deque(maxlen=1000)
        rolling_success = deque(maxlen=1000)
        rolling_lengths = deque(maxlen=1000)
        rolling_moved = deque(maxlen=1000)
        rolling_moved_to_lower_goal_crossing_number = deque(maxlen=1000)
        rolling_moved_to_higher_goal_crossing_number = deque(maxlen=1000)
        rolling_stayed_in_the_same_crossing_number = deque(maxlen=1000)
        rolling_diff_to_goal_crossing_number = deque(maxlen=1000)
        rolling_max_crosses_passed = deque(maxlen=1000)
        rolling_max_steps_reached = deque(maxlen=1000)
        rolling_goal_selection_time = deque(maxlen=1000)
        rolling_initial_state_selection_time = deque(maxlen=1000)
        rolling_episode_postprocess_time = deque(maxlen=1000)
        rolling_episode_time = deque(maxlen=1000)
        rolling_play_episodes_time = deque(maxlen=1000)
        rolling_her_ratio = deque(maxlen=1000)
        rolling_high_level_action_exists = deque(maxlen=1000)
        rolling_R1_success_rate = deque(maxlen=1000)
        rolling_R2_success_rate = deque(maxlen=1000)
        rolling_cross_success_rate = deque(maxlen=1000)
        rolling_actor_stddev_link = deque(maxlen=1000)
        rolling_actor_stddev_z = deque(maxlen=1000)
        rolling_actor_stddev_x = deque(maxlen=1000)
        rolling_actor_stddev_y = deque(maxlen=1000)
        fake_data_collected = 0
        collection_stats = {}

        while self.env_steps < self.config.train.total_timesteps:
            if self.config.train.is_random_validation:
                best_success_rate = self.best_collection_success_rate
            else:
                best_success_rate = self.best_success_rate

            if best_success_rate >= self.config.train.training_end_success_rate:
                print(f'{self.problem.name} model solved the validation environment, cannot improve further')

            # collect data
            all_action_inference_times = []
            total_sample_trajectory_time = 0
            num_exceptions = 0
            collection_started = fake_data_collected >= self.config.train.collection_starts

            if not collection_started:
                data_collected = self.config.train.update_freq + 100
                fake_data_collected += data_collected
            else:
                data_collected = 0

                while data_collected < self.config.train.update_freq and self.env_steps < self.config.train.total_timesteps:
                    self.agent.train()
                    start_time = datetime.now()
                    if self.env_steps < self.config.algorithm.learning_starts:
                        get_actions = None  # random actions
                    else:
                        get_actions = partial(self.agent.predict_action, deterministic=False, epsilon=self.epsilon)
                    episodes, episode_times, is_her_list = self.env.play_episodes(get_actions=get_actions, apply_her=self.env.her)

                    for episode, episode_time, is_her in zip(episodes, episode_times, is_her_list):
                        states, actions, raw_actions, rewards, done_flags, truncateds, infos = episode
                        episode_length = len(actions)
                        if episode_length == 0:
                            num_exceptions += 1
                            continue  # can be if experienced an exception on the first step in the episode
                        data_collected += episode_length
                        last_experience = infos[-1]['experience']
                        exception_occurred = last_experience.exception_occurred
                        num_exceptions += exception_occurred
                        first_info = infos[0]
                        goal_selection_time = first_info['goal_selection_time']
                        initial_state_selection_time = first_info['initial_state_selection_time']
                        episode_postprocess_time = first_info['episode_postprocess_time']
                        rolling_goal_selection_time.append(goal_selection_time)
                        rolling_initial_state_selection_time.append(initial_state_selection_time)
                        rolling_episode_postprocess_time.append(episode_postprocess_time)

                        if not exception_occurred:
                            self.episodes += 1
                            self.env_steps += episode_length

                            max_crosses_passed = last_experience.info.max_crosses_passed
                            high_level_action_exists = last_experience.high_level_action is not None
                            moved_to_lower_goal_crossing_number = last_experience.moved_to_lower_goal_crossing_number
                            moved_to_higher_goal_crossing_number = last_experience.moved_to_higher_goal_crossing_number

                            if not is_her:
                                action_inference_times = [info['experience'].info.step_time for info in infos]  # TODO: this is not correct I think
                                all_action_inference_times.extend(action_inference_times)
                                rolling_lengths.append(len(actions))
                                success = last_experience.info.goal_reached
                                rolling_success.append(success)
                                goal_action = last_experience.goal_action
                                if goal_action.data['move'] == 'R1':
                                    rolling_R1_success_rate.append(success)
                                elif goal_action.data['move'] == 'R2':
                                    rolling_R2_success_rate.append(success)
                                elif goal_action.data['move'] == 'cross':
                                    rolling_cross_success_rate.append(success)
                                rolling_moved.append(last_experience.info.moved_high_level_state)
                                if moved_to_lower_goal_crossing_number is not None:
                                    rolling_moved_to_lower_goal_crossing_number.append(moved_to_lower_goal_crossing_number)
                                if moved_to_higher_goal_crossing_number is not None:
                                    rolling_moved_to_higher_goal_crossing_number.append(moved_to_higher_goal_crossing_number)
                                if last_experience.stayed_in_the_same_crossing_number is not None:
                                    rolling_stayed_in_the_same_crossing_number.append(last_experience.stayed_in_the_same_crossing_number)
                                if last_experience.diff_to_goal_crossing_number is not None:
                                    rolling_diff_to_goal_crossing_number.append(last_experience.diff_to_goal_crossing_number)
                                rolling_high_level_action_exists.append(high_level_action_exists)
                                rolling_max_crosses_passed.append(max_crosses_passed)
                                rolling_max_steps_reached.append(last_experience.info.max_steps_reached)
                                rolling_rewards.append(sum(rewards))
                                rolling_episode_time.append(episode_time)
                                for info in infos:
                                    experience = info['experience']
                                    if experience.stddev_link is not None:
                                        rolling_actor_stddev_link.append(experience.stddev_link)
                                        rolling_actor_stddev_z.append(experience.stddev_z)
                                        rolling_actor_stddev_x.append(experience.stddev_x)
                                        rolling_actor_stddev_y.append(experience.stddev_y)

                            trajectory = {
                                'states': states,
                                'actions': actions,
                                'raw_actions': raw_actions,
                                'rewards': rewards,
                                'done_flags': done_flags,
                                'infos': infos,
                                'is_her': is_her,
                                'start_env_steps': self.env_steps - episode_length,
                                'end_env_steps': self.env_steps
                            }
                            self.handle_trajectory(trajectory)

                    end_time = datetime.now()
                    learning_started = self.env_steps >= self.config.algorithm.learning_starts
                    total_time = (end_time - start_time).total_seconds()
                    if not learning_started:
                        plot_collection_summary(episodes, episode_times, is_her_list, start_time=start_time, end_time=end_time)
                        print(f'{self.problem.name} Collected {len(episodes)} episodes in {total_time} seconds. Learning not started: {learning_started} - {self.env_steps}/{self.config.algorithm.learning_starts}')
                    else:
                        print(f'{self.problem.name} Collected {len(episodes)} episodes in {total_time} seconds. Learning started: {learning_started} - {self.env_steps}/{self.config.train.total_timesteps}')
                    rolling_play_episodes_time.append(total_time)
                    total_sample_trajectory_time += total_time
                    rolling_her_ratio.append(sum(is_her_list) / len(is_her_list))

                self.collection_success_rate = np.mean(rolling_success)

                collection_stats = {
                    'rolling_episode_length': np.mean(rolling_lengths),
                    'rolling_episode_success': self.collection_success_rate,
                    'rolling_episode_rewards': np.mean(rolling_rewards),
                    'rolling_episode_moved': np.mean(rolling_moved),
                    'rolling_episode_max_crosses_passed': np.mean(rolling_max_crosses_passed),
                    'rolling_episode_max_steps_reached': np.mean(rolling_max_steps_reached),
                    'rolling_her_ratio': np.mean(rolling_her_ratio),
                    'data_collected': data_collected,
                    'exception_rate': num_exceptions / data_collected if data_collected > 0 else -1.,
                    'time/sample_trajectory_time': total_sample_trajectory_time,
                    'time/rolling_play_episodes_time': np.mean(rolling_play_episodes_time),
                    'time/simulation_step_time': np.mean(all_action_inference_times),
                    'time/rolling_episode_time': np.mean(rolling_episode_time),
                    'time/rolling_goal_selection_time': np.mean(rolling_goal_selection_time),
                    'time/rolling_initial_state_selection_time': np.mean(rolling_initial_state_selection_time),
                    'time/rolling_episode_postprocess_time': np.mean(rolling_episode_postprocess_time),
                    'num_initial_states': self.env.reachable_configurations.num_nodes,
                    'rolling_high_level_action_exists': np.mean(rolling_high_level_action_exists),
                    'buffer/real_size': self.buffer.real_size,
                    'buffer/success_trajectories_num': self.buffer.num_success_trajectories,
                    'buffer/fail_trajectories_num': self.buffer.num_fail_trajectories,
                    'buffer/stale_trajectories_num': self.buffer.num_stale_trajectories,
                    'buffer/fail_no_stale_trajectories_num': self.buffer.num_fail_no_stale_trajectories,
                    'buffer/success_trajectories_ratio': self.buffer.num_success_trajectories / self.buffer.real_size,
                    'buffer/fail_trajectories_ratio': self.buffer.num_fail_trajectories / self.buffer.real_size,
                    'buffer/stale_trajectories_ratio': self.buffer.num_stale_trajectories / self.buffer.real_size,
                    'buffer/fail_no_stale_trajectories_ratio': self.buffer.num_fail_no_stale_trajectories / self.buffer.real_size,
                    'rolling_R1_success_rate': np.mean(rolling_R1_success_rate) if len(rolling_R1_success_rate) > 0 else -1,
                    'rolling_R2_success_rate': np.mean(rolling_R2_success_rate) if len(rolling_R2_success_rate) > 0 else -1,
                    'rolling_cross_success_rate': np.mean(rolling_cross_success_rate) if len(rolling_cross_success_rate) > 0 else -1,
                }
                if len(rolling_diff_to_goal_crossing_number) > 0:
                    collection_stats['rolling_episode_diff_to_goal_crossing_number'] = np.mean(rolling_diff_to_goal_crossing_number)
                if len(rolling_moved_to_lower_goal_crossing_number) > 0:
                    collection_stats['rolling_episode_moved_to_lower_goal_crossing_number'] = np.mean(rolling_moved_to_lower_goal_crossing_number)
                if len(rolling_moved_to_higher_goal_crossing_number) > 0:
                    collection_stats['rolling_episode_moved_to_higher_goal_crossing_number'] = np.mean(rolling_moved_to_higher_goal_crossing_number)
                if len(rolling_stayed_in_the_same_crossing_number) > 0:
                    collection_stats['rolling_episode_stayed_in_the_same_crossing_number'] = np.mean(rolling_stayed_in_the_same_crossing_number)

                if len(rolling_actor_stddev_link) > 0:
                    collection_stats['rolling_actor/stddev/link'] = np.mean(rolling_actor_stddev_link)
                    collection_stats['rolling_actor/stddev/z'] = np.mean(rolling_actor_stddev_z)
                    collection_stats['rolling_actor/stddev/x'] = np.mean(rolling_actor_stddev_x)
                    collection_stats['rolling_actor/stddev/y'] = np.mean(rolling_actor_stddev_y)

                if self.env_steps < self.config.algorithm.learning_starts:
                    continue  # start learning only after learning_starts steps

            # set the old replay buffer sample rate according to the schedule
            if isinstance(self.buffer, OldAndNewInMemEfficientPrioritizedReplayBuffers):
                self.buffer.set_old_replay_buffer_sample_rate(sample_rate=self.old_replay_buffer_sample_rate)

            if dataloader_iterator is None:
                dataloader_iterator = iter(self.buffer.get_dataloader())
            self.load_her_replay_buffer_files()

            # do updates
            updates = int(data_collected * self.config.train.updates_per_env_step)
            all_metrics = defaultdict(list)
            model_update_times, batch_wait_times = [], []
            start_time = time.time()

            # update freeze schedule
            actor_frozen, critic_frozen = self.agent.update_freeze_schedule(env_steps=self.env_steps)

            for idx, batch_id in tqdm(enumerate(range(updates)), total=updates, desc='Updating model'):
                batch_wait_time = time.time()
                try:
                    batch = next(dataloader_iterator)
                except Exception as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    print(f'{self.problem.name} Exception in batch {idx}')
                    raise e
                batch_wait_time = time.time() - batch_wait_time
                batch_wait_times.append(batch_wait_time)

                model_update_time = time.time()
                metrics, td_errors, tree_idxs = self.agent.update(batch, self.global_steps)
                if all([
                    isinstance(self.buffer, InMemEfficientPrioritizedReplayBuffer),
                    not isinstance(self.buffer, InMemEfficientLengthPrioritizedReplayBuffer),
                ]):
                    if self.config.algorithm.per_bellman:
                        self.buffer.update_priorities(tree_idxs, td_errors.cpu().numpy())
                    if self.config.algorithm.per_beta_linear:
                        start_beta = self.config.algorithm.per_beta
                        self.buffer.beta = start_beta + (1 - start_beta) * self.env_steps / self.config.train.total_timesteps
                self.global_steps += 1
                model_update_time = time.time() - model_update_time
                model_update_times.append(model_update_time)

                metrics.update(self.buffer.get_batch_profiling())
                metrics['time/batch_wait_time'] = batch_wait_time
                metrics['time/model_update_time'] = model_update_time

                for k, v in metrics.items():
                    if isinstance(v, list):
                        all_metrics[k].extend(v)
                    else:
                        all_metrics[k].append(metrics[k])

            model_update_times = np.mean(model_update_times)
            batch_wait_times = np.mean(batch_wait_times)
            print(f'{self.problem.name} model update time: {model_update_times}, batch wait time: {batch_wait_times}, env steps {self.env_steps} of {self.config.train.total_timesteps}')
            end_time = time.time()
            mean_metrics = {k: np.mean(all_metrics[k]) for k in all_metrics}
            total_time = end_time - start_time
            mean_metrics['time/total_time'] = total_time
            if isinstance(self.buffer, OldAndNewInMemEfficientPrioritizedReplayBuffers):
                mean_metrics['update_stats/old_replay_buffer_sample_rate'] = self.old_replay_buffer_sample_rate
            if self.agent.freeze_schedule is not None:
                mean_metrics['losses/actor_frozen'] = int(actor_frozen)
                mean_metrics['losses/critic_frozen'] = int(critic_frozen)
            wandb_log_fn({
                "train": mean_metrics,
                "stats": {
                    'env_steps': self.env_steps,
                    'episodes': self.episodes,
                    'global_steps': self.global_steps,
                    'epsilon_greedy_schedule': self.epsilon,
                },
                "collection": collection_stats
            }, wandb_run=self.wandb_run)

            if last_validation + self.config.train.eval_freq < self.env_steps:
                last_validation = self.env_steps
                self._do_validation()

        if last_validation != self.env_steps:
            self._do_validation()

        if close_buffer:
            self.buffer.close()

    def _test_ongoing_best_model(
            self, final_env_steps: int, prefix: str, test_playback_start_override: int = None, log_wandb: bool = True,
    ):
        results = {}
        test_playback_start = test_playback_start_override or self.config.train.test_playback_start
        # test the model for every config.test_playback_resolution env steps to get a clear picture of the performance
        last_success_rate, last_mean_rewards, images = None, None, []
        model_path = None
        total = int(np.ceil(final_env_steps / self.config.train.test_playback_resolution))
        for i in range(total):
            # if a new model exists, mark it as the model and clear stats
            current_model_path = os.path.join(self.save_path, f'{prefix}_{i}')
            if os.path.exists(current_model_path):
                model_path = current_model_path
                last_success_rate, last_mean_rewards, images = None, None, []

            # after passing config.test_playback_start, we record the results
            if i * self.config.train.test_playback_resolution >= test_playback_start:
                # if the stats are not updated, update them
                if last_success_rate is None:
                    self.agent.load(model_path)
                    test_results = self._test_queries(self.config.train.deterministic_test, prefix, i)
                    success_rate, mean_rewards, _, _, _, images = test_results
                steps = self.config.train.test_playback_resolution * i
                results[steps] = last_success_rate, last_mean_rewards
                if log_wandb:
                    wandb_log_fn({
                        "ongoing_test": {
                            f'{prefix}_success_rate': last_success_rate,
                            f'{prefix}_rewards': last_mean_rewards,
                            f'{prefix}_steps': steps,
                            f'{prefix}_goal_states': images[0] if images is not None else -1,
                            f'{prefix}_goal_actions': images[1] if images is not None else -1,
                        },
                        "stats": {
                            "ongoing_test_steps": i,
                        }
                    }, wandb_run=self.wandb_run)
        return results

    def test(self):
        for prefix in ['best_model', 'best_model_collection']:
            print(f'{self.problem.name} ============================== {prefix} ==============================')
            best_success_rate = self.best_collection_success_rate if prefix == 'best_model_collection' else self.best_success_rate
            print(f'{self.problem.name} loading {prefix} with validation success rate {best_success_rate}, running tests...')
            self.agent.load(os.path.join(self.save_path, prefix))
            # get test results on all_queries (validation was a cheap proxy for this)
            test_results = self._test_queries(self.config.train.deterministic_test, prefix, 0)
            test_success_rate, test_mean_rewards, _, _, _, images = test_results
            # get test results on queries used during training
            # train_queries = self.env.get_training_queries() if self.config.train.test_train > 0 else None  # TODO
            train_queries = None  # TODO
            if train_queries is not None:
                train_success_rate, train_mean_rewards = self._test_queries(self.config.train.deterministic_test)[:2]
            else:
                train_success_rate, train_mean_rewards = None, None

            print(f'{self.problem.name} end of run tests: {test_success_rate}, rewards: {test_mean_rewards}')
            stats = self._get_stats()
            stats['total_time'] = (time.time() - self.start_time) / 3600
            print(f'{self.problem.name} model took {stats["total_time"]} hours')
            final_metrics = {
                "test": {
                    f'{prefix}_success_rate': test_success_rate, f'{prefix}_rewards': test_mean_rewards,
                    f'{prefix}_goal_states': images[0] if images is not None else -1,
                    f'{prefix}_goal_actions': images[1] if images is not None else -1,
                },
                "stats": {
                    'test_steps': 0,
                }
            }
            if train_queries is not None:
                final_metrics["train"] = {f'{prefix}_success_rate': train_success_rate, f'{prefix}_rewards': train_mean_rewards}
            wandb_log_fn(final_metrics, wandb_run=self.wandb_run)

            self._test_ongoing_best_model(
                final_env_steps=self.env_steps,
                prefix=prefix,
                test_playback_start_override=None,
                log_wandb=True
            )


def train_agent(config, algorithm, algorithm_name, problem, name, agent_load_path, replay_buffer_files_path, hindsight_sharing, seed):
    print(f'Training {algorithm_name} with {name=} and {problem=}')
    trainer = Trainer(
        config=config,
        algorithm_init_fn=algorithm,
        problem=problem,
        algorithm_name=algorithm_name,
        name=name,
        agent_load_path=agent_load_path,
        replay_buffer_files_path=replay_buffer_files_path,
        hindsight_sharing=hindsight_sharing,
        seed=seed,
    )
    try:
        trainer.train(close_buffer=False)
        trainer.test()

    except Exception as e:
        traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
        exception_save_path = os.path.join(trainer.save_path, 'exception.pkl')
        with open(exception_save_path, 'wb') as f:
            pickle.dump(traceback_str, f)
        print(f'{trainer.problem.name} Exception occurred, saved to {exception_save_path}:')
        print('\n'.join(traceback_str))
        raise e

    finally:
        print(f'{trainer.problem.name} closing replay buffer')
        trainer.buffer.close()
        print(f'{trainer.problem.name} closing env')
        trainer.env.close()

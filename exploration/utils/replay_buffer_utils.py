import os
from enum import IntEnum

import numpy as np
import pickle
from tqdm import tqdm
import torch

from exploration.utils.futures_pool import FuturesMultiprocessingPool, perform_tasks
from exploration.utils.network_utils import get_device, process_batch, to_torch


def collate_fn(batch, transition_length):
    items = {i: [] for i in range(transition_length)}
    for i in range(transition_length):
        for item in batch[0]:
            item_i = item[i]
            if isinstance(item_i, torch.Tensor):
                items[i].append(item_i)
            elif isinstance(item_i, dict):
                items[i].append({k: torch.tensor(item_i[k]) for k in item_i.keys()})
            else:
                items[i].append(torch.tensor(item_i))
    new_batch = []
    for v in items.values():
        if isinstance(v[0], torch.Tensor):
            new_batch.append(torch.stack(torch.atleast_1d(v)))
        elif isinstance(v[0], dict):
            new_batch.append({k: torch.stack([item[k] for item in v]) for k in v[0].keys()})
        else:
            new_batch.append(v)
    return tuple(new_batch)


def is_trajectory_stale(trajectory):
    is_stale_experiences = []
    for i, info in enumerate(trajectory['infos']):
        experience = info['experience']
        is_stale = experience.start_high_level_state.crossing_number == experience.end_high_level_state.crossing_number
        is_stale_experiences.append(is_stale)
    is_stale_trajectory = all(is_stale_experiences)
    return is_stale_trajectory


def load_trajectory(replay_buffer_files_path: str, trajectory_id: str, raise_exception: bool = True):
    try:
        with open(os.path.join(replay_buffer_files_path, trajectory_id), 'rb') as f:
            trajectory = pickle.load(f)
        trajectory = {k: v if not isinstance(v, torch.Tensor) else v.cpu().numpy() for k, v in trajectory.items()}

        if is_trajectory_stale(trajectory):
            return None, trajectory
        else:
            return trajectory, None

    except Exception as e:
        if raise_exception:
            raise e
        return None, None


def load_replay_buffer_parallel(replay_buffer, replay_buffer_files_path: str, num_cpus: int, num_trajectories_to_load: int = None,
                                raise_exception: bool = True, keep_stale_trajectories_ratio: float = 1.):
    arguments = [{
        'replay_buffer_files_path': replay_buffer_files_path,
        'trajectory_id': trajectory_id,
        'raise_exception': raise_exception,
    } for trajectory_id in sorted(os.listdir(replay_buffer_files_path))]

    if num_trajectories_to_load is not None:
        num_arguments = len(arguments)
        indices = list(range(num_arguments))
        effective_sample_size = min(num_trajectories_to_load, num_arguments)
        chosen_indices = np.random.choice(indices, effective_sample_size, replace=False)
        arguments = [arguments[i] for i in chosen_indices]

    parallel = num_cpus > 0
    pool = FuturesMultiprocessingPool(processes=num_cpus, clear_cache_activate=False, catch_exceptions=True, verbose=True) if parallel else None
    results = perform_tasks(
        func=load_trajectory, kwargs_list=arguments, total=len(arguments), pool=pool,
        parallel=parallel, init_pool=True, close_pool=True,
    )

    move_trajectories = [{'idx': i, 'trajectory': result[0]} for i, result in enumerate(results) if result[0] is not None]
    all_stale_trajectories = [{'idx': i, 'trajectory': result[1]} for i, result in enumerate(results) if result[1] is not None]
    stale_sample_size = int(np.floor(len(all_stale_trajectories) * keep_stale_trajectories_ratio))
    stale_indices = np.random.choice(len(all_stale_trajectories), stale_sample_size, replace=False)
    stale_trajectories = [all_stale_trajectories[i] for i in stale_indices]
    trajectories = sorted(move_trajectories + stale_trajectories, key=lambda x: x['idx'])
    trajectories = [x['trajectory'] for x in trajectories]

    for trajectory in tqdm(trajectories, total=len(trajectories), desc='Adding trajectories to replay buffer'):
        if trajectory is not None:
            replay_buffer.add(trajectory)


def get_per_priority(item, per_minus=False):
    last_reward = item['rewards'][-1]
    if isinstance(last_reward, torch.Tensor):
        last_reward = last_reward.item()
    episode_length = len(item['actions'])
    length_factor = 1 / (episode_length + 1e-5)
    # length_factor = 1.
    # we favor failed and short episodes, thus -last_reward
    per_minus = -1 if per_minus else 1
    priority = (2 ** (per_minus * last_reward + 1)) * length_factor  # TODO: this assumes reward is -1,0,1 for forbidden, in-pleace, goal
    return priority


class TransitionIndexing(IntEnum):
    state = 0
    action = 1
    raw_action = 2
    next_state = 3
    reward = 4
    done = 5
    info = 6
    start_env_steps = 7
    end_env_steps = 8
    weight = 9
    priority = 10
    tree_idx = 11
    episode_success = 12


class KnotTyingTrajectoryDatasetProcessUtils:
    @staticmethod
    def select_transition(item, weight: float = 1.0, priority: float = 1.0, tree_idx: int = -1, episode_success: bool = False):
        current_state_index = np.random.randint(0, len(item['actions']))
        current_state = item['states'][current_state_index]
        action = item['actions'][current_state_index]
        raw_action = item['raw_actions'][current_state_index]
        next_state = item['states'][current_state_index + 1]
        reward = item['rewards'][current_state_index]
        done = item['done_flags'][current_state_index]
        info = item['infos'][current_state_index]
        start_env_steps = item['start_env_steps']
        end_env_steps = item['end_env_steps']
        transition = {
            TransitionIndexing.state: current_state,
            TransitionIndexing.action: action,
            TransitionIndexing.raw_action: raw_action,
            TransitionIndexing.next_state: next_state,
            TransitionIndexing.reward: reward,
            TransitionIndexing.done: done,
            TransitionIndexing.info: info,
            TransitionIndexing.start_env_steps: start_env_steps,
            TransitionIndexing.end_env_steps: end_env_steps,
            TransitionIndexing.weight: weight,
            TransitionIndexing.priority: priority,
            TransitionIndexing.tree_idx: tree_idx,
            TransitionIndexing.episode_success: episode_success,
        }
        res = [v for _, v in sorted(transition.items(), key=lambda x: x[0].value)]
        return tuple(res)

    @staticmethod
    def process_item(item, trajectory_hindsight_logic=None):
        if trajectory_hindsight_logic is not None:
            res = trajectory_hindsight_logic.process_item(item)
        else:
            item = KnotTyingTrajectoryDatasetProcessUtils.process_trajectory(item)
            res = KnotTyingTrajectoryDatasetProcessUtils.select_transition(item)
        return res

    @staticmethod
    def process_trajectory(item, trajectory_hindsight_logic=None):
        device = get_device(to_print=False)
        states = process_batch(item['states'], device=device)
        actions = to_torch(item['actions'], device=device)
        raw_actions = to_torch(item['raw_actions'], device=device)
        rewards = to_torch(item['rewards'], device=device)
        done_flags = to_torch(item['done_flags'], device=device)
        start_env_steps = item['start_env_steps']
        end_env_steps = item['end_env_steps']
        infos = []
        experiences = []
        for info in item['infos']:
            experience = info['experience']
            experiences.append(experience)
            new_info = {
                'goal_reached': experience.info.goal_reached,
                'moved_high_level_state': experience.info.moved_high_level_state,
                'max_crosses_passed': experience.info.max_crosses_passed,
                'moved_to_higher_goal_crossing_number': experience.moved_to_higher_goal_crossing_number,
                'moved_to_lower_goal_crossing_number': experience.moved_to_lower_goal_crossing_number,
                'stayed_in_the_same_crossing_number': experience.stayed_in_the_same_crossing_number,
            }
            new_info = {k: v if v is not None else -1 for k, v in new_info.items()}
            infos.append(new_info)
        processed_trajectory = {
            'states': states,
            'actions': actions,
            'raw_actions': raw_actions,
            'rewards': rewards,
            'done_flags': done_flags,
            'infos': infos,
            'start_env_steps': start_env_steps,
            'end_env_steps': end_env_steps,
            'experiences': experiences
        }
        return processed_trajectory

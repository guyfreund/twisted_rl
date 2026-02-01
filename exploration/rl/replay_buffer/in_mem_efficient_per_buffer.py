# taken from https://github.com/Howuhh/prioritized_experience_replay/tree/main
from functools import partial

import torch
import random
import os
import pickle
from copy import deepcopy
from typing import Optional

from exploration.utils.replay_buffer_utils import load_replay_buffer_parallel, collate_fn, get_per_priority, \
    TransitionIndexing, KnotTyingTrajectoryDatasetProcessUtils
from exploration.rl.replay_buffer.sum_tree import SumTree
from torch.utils.data import IterableDataset, DataLoader


def _get_filepath(dir, item_id):
    return os.path.join(dir, f'{item_id}.pkl')


class InMemEfficientPrioritizedReplayBuffer(IterableDataset):
    def __init__(self, batch_size: Optional[int], max_buffer_size: int, output_dir: str,
                 store_preprocessed_trajectory: bool, save_at_end: bool = True, to_process: bool = True,
                 eps=1e-2, alpha=0.6, beta=0.1, use_per_weights: bool = False, use_per_sampling: bool = False,
                 per_minus: bool = False, per_bellman: bool = False):
        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        self.data = {}
        self.raw_data = {}
        self.count = 0
        self.real_size = 0
        self.size = max_buffer_size
        self.batch_size = batch_size
        self.tree = SumTree(size=max_buffer_size)
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        self.store_preprocessed_trajectory = store_preprocessed_trajectory
        self.save_at_end = save_at_end
        self.to_process = to_process
        self._dataloader: DataLoader = None
        self.use_per_weights = use_per_weights
        self.use_per_sampling = use_per_sampling
        self.per_minus = per_minus
        self.per_bellman = per_bellman
        self.trajectory_status = {}

    @property
    def num_success_trajectories(self):
        return sum(status == 'success' for _, status in self.trajectory_status.items())

    @property
    def num_fail_trajectories(self):
        return sum(status == 'stale' or status == 'fail' for _, status in self.trajectory_status.items())

    @property
    def num_stale_trajectories(self):
        return sum(status == 'stale' for _, status in self.trajectory_status.items())

    @property
    def num_fail_no_stale_trajectories(self):
        return sum(status == 'fail' for _, status in self.trajectory_status.items())

    def __getitem__(self, item):
        raise NotImplementedError('Do not use __getitem__, this is an IterableDataset')

    def __iter__(self):
        while True:
            yield self.sample()

    def get_dataloader(self):
        if self._dataloader is None:
            # batch_size=1 because sample() yields batches
            self._dataloader = DataLoader(self, batch_size=1, num_workers=0, collate_fn=partial(collate_fn, transition_length=len(TransitionIndexing)))
        return self._dataloader

    def get_priority(self, item, calc_priority):
        if calc_priority:
            if self.per_bellman:
                return self.max_priority
            else:
                return get_per_priority(item, self.per_minus)
        else:
            return 1.

    def update_priorities(self, tree_idxs, priorities):
        for tree_idx, priority in zip(tree_idxs.squeeze(), priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(tree_idx.item(), priority)
            self.max_priority = max(self.max_priority, priority)

    def add(self, item):
        priority = self.get_priority(item, self.use_per_sampling)
        self.tree.add(priority, self.count)
        if isinstance(item, dict):
            last_experience = item['infos'][-1]['experience']
            is_success = last_experience.info.goal_reached
            is_stale = last_experience.stayed_in_the_same_crossing_number
            status = 'success' if is_success else 'stale' if is_stale else 'fail'
            self.trajectory_status[self.count] = status

        if self.store_preprocessed_trajectory:
            processed_item = KnotTyingTrajectoryDatasetProcessUtils.process_trajectory(item=item)  # save processed item on GPU
            experiences = processed_item.pop('experiences')
            self.raw_data[self.count] = experiences  # save raw data on CPU
        else:
            processed_item = item
        self.data[self.count] = processed_item
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample_tree(self, batch_size: int):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        item_ids, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, item_id = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            item_ids.append(item_id)

        return item_ids, tree_idxs, priorities

    def sample(self):
        item_ids, tree_idxs, priorities = self.sample_tree(self.batch_size)

        if self.use_per_weights and not self.use_per_sampling:
            items = [self.data[item_id] for item_id in item_ids]
            priorities = np.array([self.get_priority(item, True) for item in items])

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        batch = [self.get(
            item_id=item_id,
            sample_transition=True,
            to_process=self.to_process,
            weight=weight if self.use_per_weights else 1.,
            priority=priority,
            tree_idx=tree_idx,
        ) for item_id, weight, priority, tree_idx in zip(item_ids, weights, priorities, tree_idxs)]
        return batch

    def get(self, item_id: int, sample_transition: bool, weight: float = 1.0, to_process: bool = None, priority: float = 1.,
            tree_idx: int = -1):
        to_process = to_process if to_process is not None else self.to_process
        item = self.data[item_id]

        if self.store_preprocessed_trajectory:
            experiences = self.raw_data[item_id]
            if not sample_transition:
                for i, info in enumerate(item['infos']):
                    info['experience'] = experiences[i]
        else:
            if to_process:
                item = KnotTyingTrajectoryDatasetProcessUtils.process_trajectory(item=item)

        if isinstance(item, dict) and sample_transition:
            experiences = self.raw_data[item_id]
            last_experience = experiences[-1]
            episode_success = last_experience.info.goal_reached
            item = KnotTyingTrajectoryDatasetProcessUtils.select_transition(item=item, weight=weight, priority=priority,
                                                                            tree_idx=tree_idx, episode_success=episode_success)

        return item

    def get_all_data(self):
        return deepcopy([self.get(item_id=item_id, sample_transition=False, to_process=False) for item_id in self.data.keys()])

    def close(self):
        if self.save_at_end:
            self.save()

    def save(self):
        for item_id in self.data.keys():
            trajectory = self.get(item_id=item_id, sample_transition=False, to_process=False)
            trajectory = {k: v if not isinstance(v, torch.Tensor) else v.cpu().numpy() for k, v in trajectory.items()}
            with open(_get_filepath(self.output_dir, item_id), 'wb') as f:
                pickle.dump(trajectory, f)

    def get_batch_profiling(self):
        return {}

    def load(self, path: str, num_trajectories_to_load: int = None, num_cpus: int = 0, keep_stale_trajectories_ratio: float = 1.):
        load_replay_buffer_parallel(
            replay_buffer=self,
            replay_buffer_files_path=path,
            num_trajectories_to_load=num_trajectories_to_load,
            raise_exception=True,
            num_cpus=num_cpus,
            keep_stale_trajectories_ratio=keep_stale_trajectories_ratio,
        )


class OldAndNewInMemEfficientPrioritizedReplayBuffers(IterableDataset):
    def __init__(self, batch_size: int, max_buffer_size: int, output_dir: str,
                 store_preprocessed_trajectory: bool, save_at_end: bool = True, to_process: bool = True,
                 eps=1e-2, alpha=0.1, beta=0.1, use_per_weights: bool = False, use_per_sampling: bool = False,
                 per_minus: bool = False, per_bellman: bool = False):
        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        self._old_replay_buffer_sample_rate = None
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        self.store_preprocessed_trajectory = store_preprocessed_trajectory
        self.save_at_end = save_at_end
        self.to_process = to_process
        self._dataloader: DataLoader = None
        self.use_per_weights = use_per_weights
        self.use_per_sampling = use_per_sampling
        self.per_minus = per_minus
        self.per_bellman = per_bellman
        self.buffers = {age: InMemEfficientPrioritizedReplayBuffer(
            batch_size=None, max_buffer_size=self.max_buffer_size, output_dir=self.output_dir,
            store_preprocessed_trajectory=self.store_preprocessed_trajectory, save_at_end=False,
            to_process=self.to_process, eps=self.eps, alpha=self.alpha, beta=self.beta,
            use_per_weights=self.use_per_weights, use_per_sampling=self.use_per_sampling, per_minus=self.per_minus,
            per_bellman=self.per_bellman,
        ) for age in ['old', 'new']}

    @property
    def num_success_trajectories(self):
        return sum(b.num_success_trajectories for _, b in self.buffers.items())

    @property
    def num_fail_trajectories(self):
        return sum(b.num_fail_trajectories for _, b in self.buffers.items())

    @property
    def num_stale_trajectories(self):
        return sum(b.num_stale_trajectories for _, b in self.buffers.items())

    @property
    def num_fail_no_stale_trajectories(self):
        return sum(b.num_fail_no_stale_trajectories for _, b in self.buffers.items())

    def __iter__(self):
        while True:
            yield self.sample()

    def set_old_replay_buffer_sample_rate(self, sample_rate: float):
        assert 0. <= sample_rate <= 1., "sample_rate should be in [0, 1]"
        old_batch_size = int(self.batch_size * sample_rate)
        new_batch_size = self.batch_size - old_batch_size
        self.buffers['old'].batch_size = old_batch_size
        self.buffers['new'].batch_size = new_batch_size

    def get_dataloader(self):
        if self._dataloader is None:
            self._dataloader = DataLoader(
                self,
                batch_size=1,  # batch_size=1 because we sample yields batches
                num_workers=0,  # in memory buffer, no need for multiprocessing
                collate_fn=self.buffers['new'].collate_fn,  # we can also take old, it does not matter
            )
        return self._dataloader

    def add(self, item: dict, buffer: str = 'new'):
        self.buffers[buffer].add(item)

    def sample(self):
        batch = []
        for _, buffer in self.buffers.items():
            batch.extend(buffer.sample())
        return batch

    def get_all_data(self):
        return self.buffers['new'].get_all_data()

    def close(self):
        if self.save_at_end:
            self.save()

    def save(self):
        index = 0
        buffer = self.buffers['new']
        for item_id in buffer.data.keys():
            trajectory = buffer.get(item_id=item_id, sample_transition=False, to_process=False)
            trajectory = {k: v if not isinstance(v, torch.Tensor) else v.cpu().numpy() for k, v in trajectory.items()}
            with open(_get_filepath(self.output_dir, index), 'wb') as f:
                pickle.dump(trajectory, f)
            index += 1

    def get_batch_profiling(self):
        return {}

    def load(self, path: str, num_trajectories_to_load: int = None, num_cpus: int = 0, keep_stale_trajectories_ratio: float = 1.):
        load_replay_buffer_parallel(
            replay_buffer=self.buffers['old'],
            replay_buffer_files_path=path,
            num_trajectories_to_load=num_trajectories_to_load,
            raise_exception=True,
            num_cpus=num_cpus,
            keep_stale_trajectories_ratio=keep_stale_trajectories_ratio,
        )


def get_trajectory(trajectory_file_path: str):
    with open(trajectory_file_path, 'rb') as f:
        trajectory = pickle.load(f)
    return trajectory

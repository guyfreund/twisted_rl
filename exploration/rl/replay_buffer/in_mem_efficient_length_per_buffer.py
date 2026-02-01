import torch

from exploration.rl.replay_buffer.in_mem_efficient_per_buffer import InMemEfficientPrioritizedReplayBuffer
from exploration.utils.replay_buffer_utils import TransitionIndexing


class InMemEfficientLengthPrioritizedReplayBuffer(InMemEfficientPrioritizedReplayBuffer):
    def __init__(self, agent, batch_size: int, max_buffer_size: int, output_dir: str,
                 store_preprocessed_trajectory: bool, save_at_end: bool = True, to_process: bool = True,
                 use_per_weights: bool = False, use_per_sampling: bool = False, batch_size_factor: int = 10):
        super().__init__(
            batch_size=batch_size,
            max_buffer_size=max_buffer_size,
            output_dir=output_dir,
            store_preprocessed_trajectory=store_preprocessed_trajectory,
            save_at_end=save_at_end,
            to_process=to_process,
            use_per_weights=use_per_weights,
            use_per_sampling=use_per_sampling,
        )
        self.agent = agent
        self.batch_size_factor = batch_size_factor
        self.full_batch_size = self.batch_size * self.batch_size_factor

    def get_priority(self, item):
        trajectory_length = len(item['actions'])
        return trajectory_length

    def update_priorities(self, tree_idxs, priorities):
        raise NotImplementedError("This method is not implemented for this class")

    def sample(self):
        item_ids, tree_idxs, priorities = self.sample_tree(self.full_batch_size)
        full_batch = [self.get(item_id=item_id, sample_transition=True, to_process=self.to_process) for item_id in item_ids]

        states = torch.stack([item[TransitionIndexing.state.value] for item in full_batch]).to(self.agent.device)
        raw_actions = torch.stack([item[TransitionIndexing.raw_action.value] for item in full_batch]).to(self.agent.device)
        next_states = torch.stack([item[TransitionIndexing.next_state.value] for item in full_batch]).to(self.agent.device)
        rewards = torch.stack([item[TransitionIndexing.reward.value] for item in full_batch]).to(self.agent.device)
        done_flags = torch.stack([item[TransitionIndexing.done.value] for item in full_batch]).to(self.agent.device)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _, _ = self.agent.actor.get_action(next_states)
            min_qf_next_target = self.agent.qf_target.get_min(next_states, next_state_actions) - self.agent.alpha * next_state_log_pi
            next_q_value = rewards.flatten() + (1 - done_flags.flatten()) * self.agent.config.gamma * (min_qf_next_target).view(-1)
            td_errors, _, _ = self.agent.qf.get_td_errors(states, raw_actions, next_q_value)
            _, idxs = torch.topk(td_errors, self.batch_size)  # choose batch_size with the highest td error
            td_errors = td_errors.cpu().numpy()

        batch = []
        for idx in idxs.cpu().numpy().tolist():
            item = list(full_batch[idx])
            item[TransitionIndexing.weight.value] = 1.  # equal weight to all transitions with top-k highest td error
            item[TransitionIndexing.priority.value] = td_errors[idx]  # priorities are not lengths but td errors !!!
            item[TransitionIndexing.tree_idx.value] = tree_idxs[idx]
            batch.append(tuple(item))

        return batch

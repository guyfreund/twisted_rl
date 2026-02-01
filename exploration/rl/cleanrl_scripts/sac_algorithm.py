# based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
import argparse
import time
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import Optional

import sys

sys.path.append('.')

from absl import logging
logging.set_verbosity(logging.FATAL)

from exploration.mdp.low_level_action import LowLevelAction
from exploration.utils.schedule.schedule_factory import ScheduleFactory
from exploration.rl.cleanrl_scripts.common import DoubleQNetworks
from exploration.rl.cleanrl_scripts.train_agent import train_agent
from exploration.utils.config_utils import load_config
from exploration.utils.network_utils import get_layers, process_batch, clip_gradients_if_required, to_torch, \
    generic_freeze, generic_unfreeze, get_device

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, state_goal_size, action_size, action_high: np.ndarray, action_low: np.ndarray, sizes=(256, 256),
                 freeze_layers: int = 0):
        super().__init__()
        self.net, self.fc_mean, self.fc_logstd = self.generic_create_module(state_goal_size, action_size, sizes)

        # handle freezing
        self.frozen = False
        self.freeze_layers = freeze_layers
        assert self.freeze_layers <= len(sizes), f'{freeze_layers=} must be less than or equal to the length of {sizes=}'

        # action rescaling
        self.register_buffer("action_scale", torch.FloatTensor((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", torch.FloatTensor((action_high + action_low) / 2.0))
        # print number of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Actor number of parameters: {num_params}")

    def forward(self, state):
        return self.generic_forward(state, self.net, self.fc_mean, self.fc_logstd)

    def freeze(self):
        self.frozen = generic_freeze(net=self.net, frozen=self.frozen, freeze_layers=self.freeze_layers, net_name='Actor')
        return self.frozen

    def unfreeze(self):
        self.frozen = generic_unfreeze(net=self.net, frozen=self.frozen, net_name='Actor')
        return self.frozen

    def get_action(self, states):
        mean, log_std = self(states)
        action, log_prob, mean, dist = self.generic_get_action(mean, log_std, self.action_scale, self.action_bias)
        stddev = self.get_stddev(dist)
        return action, log_prob, mean, stddev

    @staticmethod
    def generic_create_module(state_goal_size, output_size, sizes):
        net = get_layers([state_goal_size] + list(sizes))
        fc_mean = nn.Linear(sizes[-1], output_size)
        fc_logstd = nn.Linear(sizes[-1], output_size)
        return net, fc_mean, fc_logstd

    @staticmethod
    def generic_forward(state, net, mean_head, logstd_head):
        x = F.relu(net(state))
        mean = mean_head(x)
        log_std = logstd_head(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    @staticmethod
    def generic_get_action(mean, log_std, action_scale, action_bias):
        normal = Actor.get_normal(mean, log_std)
        x_t = normal.rsample()  # Re-Parametrization Trick
        y_t = torch.tanh(x_t)
        action = y_t * action_scale + action_bias
        log_prob = Actor.get_log_prob(action_scale, normal, x_t, y_t)
        mean = torch.tanh(mean) * action_scale + action_bias
        return action, log_prob, mean, normal

    @staticmethod
    def get_normal(mean, log_std):
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        return normal

    @staticmethod
    def get_log_prob(action_scale, normal, x_t, y_t):
        log_prob = normal.log_prob(x_t)  # Get log prob from the Gaussian distribution
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + 1e-6)  # Adjust log prob for tanh transformation
        log_prob = log_prob.sum(1, keepdim=True)  # Sum over action dimensions
        return log_prob

    def log_prob(self, states, actions):
        """Compute the log probability of given actions under the current policy distribution."""
        mean, log_std = self(states)
        normal = Actor.get_normal(mean, log_std)
        # Reverse the action transformation: convert from action space back to raw Gaussian sample space
        y_t = (actions - self.action_bias) / self.action_scale  # Undo scaling and shifting
        x_t = torch.atanh(y_t)  # Inverse of tanh with numerical stability
        log_prob = Actor.get_log_prob(self.action_scale, normal, x_t, y_t)
        return log_prob

    def get_stddev(self, dist):
        stddev = dist.stddev.detach().cpu().numpy()
        return stddev


class AutoregressiveActor(nn.Module):
    def __init__(self, state_goal_size, action_size, action_high: np.ndarray, action_low: np.ndarray, sizes=(256, 256),
                 freeze_layers: int = 0):
        super().__init__()
        assert action_size == 4 == len(action_high) == len(action_low), 'Only 4 actions are supported for AutoRegressiveActor'
        # create modules
        self.link_net, self.link_fc_mean, self.link_fc_logstd = Actor.generic_create_module(state_goal_size, 1, sizes)
        self.xy_net, self.xy_fc_mean, self.xy_fc_logstd = Actor.generic_create_module(state_goal_size + 1, 2, sizes)
        self.z_net, self.z_fc_mean, self.z_fc_logstd = Actor.generic_create_module(state_goal_size + 3, 1, sizes)

        # handle freezing
        self.frozen = False
        self.freeze_layers = freeze_layers
        assert self.freeze_layers <= len(sizes), f'{freeze_layers=} must be less than or equal to the length of {sizes=}'

        # action rescaling
        self.register_buffer("action_scale", torch.FloatTensor((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", torch.FloatTensor((action_high + action_low) / 2.0))
        # print number of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"AutoRegressiveActor number of parameters: {num_params}")

    def forward(self, state):
        link_mean, link_log_std = Actor.generic_forward(state, self.link_net, self.link_fc_mean, self.link_fc_logstd)
        link, link_log_prob, link_mean, link_dist = Actor.generic_get_action(link_mean, link_log_std, self.action_scale[0], self.action_bias[0])
        state_with_link = torch.cat([state, link], dim=1)

        xy_mean, xy_log_std = Actor.generic_forward(state_with_link, self.xy_net, self.xy_fc_mean, self.xy_fc_logstd)
        xy, xy_log_prob, xy_mean, xy_dist = Actor.generic_get_action(xy_mean, xy_log_std, self.action_scale[2:4], self.action_bias[2:4])
        state_with_link_xy = torch.cat([state_with_link, xy], dim=1)

        z_mean, z_log_std = Actor.generic_forward(state_with_link_xy, self.z_net, self.z_fc_mean, self.z_fc_logstd)
        z, z_log_prob, z_mean, z_dist = Actor.generic_get_action(z_mean, z_log_std, self.action_scale[1], self.action_bias[1])

        action = torch.cat([link, z, xy], dim=1)
        mean = torch.cat([link_mean, z_mean, xy_mean], dim=1)
        # p(link,x,y,z)=p(z|link,x,y)*p(x,y|link)*p(link)
        # log(p(link,x,y,z)) = log(p(z|link,x,y)*p(x,y|link)*p(link)) = log(p(z|link,x,y)) + log(p(x,y|link)) + log(p(link))
        log_prob = link_log_prob + z_log_prob + xy_log_prob
        stddev = self.get_stddev((link_dist, z_dist, xy_dist))
        return action, log_prob, mean, stddev

    def freeze(self):
        link_frozen = generic_freeze(net=self.link_net, frozen=self.frozen, freeze_layers=self.freeze_layers, net_name='Link')
        xy_frozen = generic_freeze(net=self.xy_net, frozen=self.frozen, freeze_layers=self.freeze_layers, net_name='XY')
        z_frozen = generic_freeze(net=self.z_net, frozen=self.frozen, freeze_layers=self.freeze_layers, net_name='Z')
        assert link_frozen == xy_frozen == z_frozen, f'All submodules must be frozen or unfrozen together {link_frozen=}, {xy_frozen=}, {z_frozen=}'
        self.frozen = link_frozen
        return self.frozen

    def unfreeze(self):
        link_frozen = generic_unfreeze(net=self.link_net, frozen=self.frozen, net_name='Link')
        xy_frozen = generic_unfreeze(net=self.xy_net, frozen=self.frozen, net_name='XY')
        z_frozen = generic_unfreeze(net=self.z_net, frozen=self.frozen, net_name='Z')
        assert link_frozen == xy_frozen == z_frozen, f'All submodules must be frozen or unfrozen together {link_frozen=}, {xy_frozen=}, {z_frozen=}'
        self.frozen = link_frozen
        return self.frozen

    def get_action(self, state):
        action, log_prob, mean, stddev = self(state)
        return action, log_prob, mean, stddev

    def get_stddev(self, dist):
        link_dist, z_dist, xy_dist = dist
        link_stddev = link_dist.stddev.detach().cpu().numpy()
        num_samples = link_stddev.shape[0]
        z_stddev = z_dist.stddev.detach().cpu().numpy()
        xy_stddev = xy_dist.stddev.detach().cpu().numpy()
        x_stddev, y_stddev = np.atleast_2d(xy_stddev[:, 0]).T, np.atleast_2d(xy_stddev[:, 1]).T
        stddev = np.stack([link_stddev, z_stddev, x_stddev, y_stddev], axis=1).squeeze().reshape((num_samples, 4))
        return stddev


class SACAlgorithm(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self.config = config
        self.env = env
        self.device = get_device(to_print=True)

        # Freeze Schedule
        self.actor_freeze_layers = config.actor_freeze_layers if 'actor_freeze_layers' in config else 0
        self.critic_freeze_layers = config.critic_freeze_layers if 'critic_freeze_layers' in config else 0
        assert self.actor_freeze_layers == self.critic_freeze_layers, f'Actor and Critic must have the same number of frozen layers. {self.actor_freeze_layers=}, {self.critic_freeze_layers=}'
        if self.actor_freeze_layers > 0 and self.critic_freeze_layers > 0:
            assert 'freeze_schedule' in config, f'freeze_schedule must be provided if freeze_layers > 0. {self.actor_freeze_layers=}, {self.critic_freeze_layers=}'
            self.freeze_schedule = ScheduleFactory.create_from_cfg(cfg=config.freeze_schedule)
        else:
            self.freeze_schedule = None

        state_goal_size = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
        action_size = env.action_space.shape[0]

        actor_hidden_layers = [config.actor_hidden_layer_size] * config.actor_hidden_layer_count
        actor_kwargs = {
            'state_goal_size': state_goal_size,
            'action_size': action_size,
            'action_high': env.action_space.high,
            'action_low': env.action_space.low,
            'sizes': actor_hidden_layers,
            'freeze_layers': self.actor_freeze_layers,
        }
        if 'actor_type' not in config or config.actor_type == 'default':
            self.actor = Actor(**actor_kwargs).to(self.device)
        elif config.actor_type == 'autoregressive':
            self.actor = AutoregressiveActor(**actor_kwargs).to(self.device)
        else:
            raise ValueError(f'Unknown actor type: {config.actor_type}')

        critic_hidden_layers = [config.critic_hidden_layer_size] * config.critic_hidden_layer_count
        critic_kwargs = {
            'config': config,
            'state_goal_size': state_goal_size,
            'action_size': action_size,
            'sizes': critic_hidden_layers,
            'dropout_prob': config.critic_dropout,
            'layer_norm': config.critic_layer_norm,
        }
        self.qf = DoubleQNetworks(**{**critic_kwargs, 'freeze_layers': self.critic_freeze_layers}).to(self.device)
        self.qf_target = DoubleQNetworks(**critic_kwargs).to(self.device)
        self.qf_target.q1.load_state_dict(self.qf.q1.state_dict())
        self.qf_target.q2.load_state_dict(self.qf.q2.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.qf.parameters()), lr=config.q_lr, weight_decay=config.q_weight_decay
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=config.policy_lr, weight_decay=config.policy_weight_decay
        )

        # Automatic entropy tuning
        if config.autotune:
            self.target_entropy = -torch.Tensor([action_size]).to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=config.q_lr)
        else:
            self.target_entropy, self.log_alpha, self.a_optimizer = None, None, None
            self.alpha = config.alpha

        # print number of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SAC number of parameters: {num_params}")

    def actor_need_to_freeze(self, env_steps: int) -> int:
        if self.freeze_schedule is None:
            return 1
        else:
            return self.freeze_schedule.value(step=env_steps)

    def critic_need_to_freeze(self, env_steps: int) -> int:
        if self.freeze_schedule is None:
            return 1
        else:
            return self.freeze_schedule.value(step=env_steps)

    def update_freeze_schedule(self, env_steps: int):
        if self.actor_need_to_freeze(env_steps) == 1:
            actor_frozen = self.actor.freeze()
        else:
            actor_frozen = self.actor.unfreeze()
        if self.critic_need_to_freeze(env_steps) == 1:
            critic_frozen = self.qf.freeze()
        else:
            critic_frozen = self.qf.unfreeze()
        return actor_frozen, critic_frozen

    def update(self, batch, global_step):
        (states, actions, raw_actions, next_states, rewards, done_flags, infos, start_env_steps, end_env_steps, weights,
         priorities, tree_idxs, episode_successes) = batch

        metrics = {}
        metrics['update_stats/mean_rewards'] = np.mean(rewards.cpu().numpy())
        metrics['update_stats/done_ratio'] = np.mean(done_flags.cpu().numpy())
        metrics['update_stats/goal_reached_ratio'] = np.mean(infos['goal_reached'].cpu().numpy())
        metrics['update_stats/fail_ratio'] = 1 - metrics['update_stats/goal_reached_ratio']
        metrics['update_stats/stale_ratio'] = np.mean(infos['stayed_in_the_same_crossing_number'].cpu().numpy())
        metrics['update_stats/fail_no_stale_ratio'] = 1 - metrics['update_stats/goal_reached_ratio'] - metrics['update_stats/stale_ratio']
        metrics['update_stats/moved_high_level_state_ratio'] = np.mean(infos['moved_high_level_state'].cpu().numpy())
        metrics['update_stats/max_crosses_passed_ratio'] = np.mean(infos['max_crosses_passed'].cpu().numpy())
        metrics['update_stats/mean_weights'] = np.mean(weights.cpu().numpy())
        metrics['update_stats/max_weights'] = np.max(weights.cpu().numpy())
        metrics['update_stats/min_weights'] = np.min(weights.cpu().numpy())
        metrics['update_stats/mean_priorities'] = np.mean(priorities.cpu().numpy())

        to_cuda_time = time.time()
        states = states.to(device=self.device)
        raw_actions = raw_actions.to(device=self.device)
        next_states = next_states.to(device=self.device)
        rewards = rewards.to(device=self.device)
        done_flags = done_flags.to(device=self.device)
        weights = weights.to(device=self.device)
        episode_successes = episode_successes.to(device=self.device)
        to_cuda_time = time.time() - to_cuda_time
        metrics['time/to_cuda_time'] = to_cuda_time

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _, _ = self.actor.get_action(next_states)
            min_qf_next_target = self.qf_target.get_min(next_states, next_state_actions) - self.alpha * next_state_log_pi
            metrics['losses/approximated_actor_entropy'] = -next_state_log_pi.mean().cpu().numpy().item()
            next_q_value = rewards.flatten() + (1 - done_flags.flatten()) * self.config.gamma * (min_qf_next_target).view(-1)

        qf_loss, td_errors = self.qf.get_losses(states, raw_actions, next_q_value, weights, metrics)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        q_gradients = clip_gradients_if_required(self.qf.parameters(), self.config.q_grad_clip)
        metrics['losses/q_gradients'] = q_gradients.cpu().numpy()
        self.q_optimizer.step()

        if global_step % self.config.policy_frequency == 0:  # TD 3 Delayed update support
            # update actor
            for _ in range(self.config.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _, stddev = self.actor.get_action(states)
                min_qf_pi = self.qf.get_min(states, pi)
                # Policy objective is maximization of (alpha * logp - Q) with priority weights.
                actor_loss = ((self.alpha * log_pi) - min_qf_pi) * weights
                metrics["losses/actor_loss"] = actor_loss.mean().item()

                if self.config.bc:  # Behavioral Cloning support
                    log_prob_actions = self.actor.log_prob(states, raw_actions)
                    bc_loss = episode_successes * -log_prob_actions
                    metrics["losses/bc_loss"] = bc_loss.mean().item()
                    all_actor_loss = actor_loss + self.config.bc_lambda * bc_loss
                else:
                    all_actor_loss = actor_loss

                all_actor_loss = all_actor_loss.mean()
                metrics["losses/all_actor_loss"] = all_actor_loss.item()
                self.actor_optimizer.zero_grad()
                all_actor_loss.backward()
                actor_gradients = clip_gradients_if_required(self.actor.parameters(), self.config.policy_grad_clip)
                metrics['losses/actor_gradients'] = actor_gradients.cpu().numpy()
                self.actor_optimizer.step()

                for arg in ['link', 'z', 'x', 'y']:
                    metrics[f'actor/stddev/{arg}'] = stddev[:, LowLevelAction.arg_to_idx(arg)].tolist()

                metrics["losses/alpha"] = self.alpha

                if self.config.autotune:
                    with torch.no_grad():
                        _, log_pi, _, _ = self.actor.get_action(states)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

                    metrics["losses/alpha_loss"] = alpha_loss.item()

        # update the target networks
        if global_step % self.config.target_network_frequency == 0:
            for param, target_param in zip(self.qf.q1.parameters(), self.qf_target.q1.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            for param, target_param in zip(self.qf.q2.parameters(), self.qf_target.q2.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return metrics, td_errors, tree_idxs

    def predict_action(self, states, deterministic: bool, epsilon: Optional[float] = None):
        safe_states = [state.copy() for state in states]
        start_goal = process_batch(safe_states, device=self.device)
        random_actions, _ = self.env.get_random_actions(states)

        if epsilon is not None:
            mask = np.random.rand(len(states)) < epsilon
        else:
            mask = np.full(len(states), False)

        with torch.no_grad():
            action_preds, _, action_means, action_dist = self.actor.get_action(start_goal)
            actions = action_means if deterministic else action_preds
            actions = actions.cpu().numpy()

        actions[mask] = random_actions[mask]
        return actions, action_dist

    def get_q_values(self, states, actions):
        first_states = process_batch(states, device=self.device)
        first_actions = to_torch(actions, device=self.device)
        with torch.no_grad():
            q_values = self.qf_target.get_min(first_states, first_actions)
            q_values = q_values.cpu().numpy()
        return q_values

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=self.device), strict=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exploration/rl/config/sac.yml", help="config file to use")
    parser.add_argument("-p", "--problem", type=str, default=None, help="Problem to use")
    parser.add_argument("-alp", "--agent_load_path", type=str, default=None, help="Path to load the agent from")
    parser.add_argument("-rbp", "--replay_buffer_files_path", type=str, default=None, help="Path to load the initial states from the replay buffers")
    parser.add_argument("-n", "--name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("-hs", "--hindsight_sharing", type=int, default=None, help="use hindsight sharing")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed for the experiment")
    args = parser.parse_args()

    config = load_config(args.config)
    train_agent(
        config=config,
        algorithm=SACAlgorithm,
        algorithm_name='TWISTED_RL',
        problem=args.problem,
        agent_load_path=args.agent_load_path,
        replay_buffer_files_path=args.replay_buffer_files_path,
        name=args.name,
        hindsight_sharing=args.hindsight_sharing,
        seed=args.seed,
    )

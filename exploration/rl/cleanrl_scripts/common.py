import torch
import torch.nn as nn
import torch.nn.functional as F

from exploration.utils.network_utils import get_layers, generic_freeze, generic_unfreeze


class QNetwork(nn.Module):
    def __init__(self, state_goal_size, action_size, sizes=(256, 256), dropout_prob: float = 0.,
                 layer_norm: bool = False, freeze_layers: int = 0):
        super().__init__()
        sizes = [state_goal_size + action_size] + list(sizes) + [1]
        layers = get_layers(sizes, dropout_prob=dropout_prob, layer_norm=layer_norm, return_packed=False)

        self.net = nn.Sequential(*layers)

        # handle freezing
        self.frozen = False
        self.freeze_layers = freeze_layers
        assert self.freeze_layers <= len(sizes), f'{self.freeze_layers=} must be less than or equal to the length of {sizes=}'

        # print number of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"QNetwork number of parameters: {num_params}")

    def forward(self, state_goal, action):
        x = torch.cat([state_goal, action], 1)
        return self.net(x)

    def freeze(self, net_name):
        self.frozen = generic_freeze(net=self.net, frozen=self.frozen, freeze_layers=self.freeze_layers, net_name=net_name)
        return self.frozen

    def unfreeze(self, net_name):
        self.frozen = generic_unfreeze(net=self.net, frozen=self.frozen, net_name=net_name)
        return self.frozen


class DoubleQNetworks(nn.Module):
    def __init__(self, config, state_goal_size, action_size, sizes=(256, 256), dropout_prob: float = 0., layer_norm: bool = False,
                 freeze_layers: int = 0):
        super().__init__()
        self.config = config

        # handle freezing
        self.frozen = False
        assert freeze_layers <= len(sizes), f'{freeze_layers=} must be less than or equal to the length of {sizes=}'

        q_kwargs = {
            'state_goal_size': state_goal_size,
            'action_size': action_size,
            'sizes': sizes,
            'dropout_prob': dropout_prob,
            'layer_norm': layer_norm,
            'freeze_layers': freeze_layers,
        }
        self.q1 = QNetwork(**q_kwargs)
        self.q2 = QNetwork(**q_kwargs)

        # print number of parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DoubleQNetworks number of parameters: {num_params}")

    def forward(self, state_goal, action):
        return self.q1(state_goal, action), self.q2(state_goal, action)

    def freeze(self):
        q1_frozen = self.q1.freeze(net_name='Q_function_1')
        q2_frozen = self.q2.freeze(net_name='Q_function_2')
        assert q1_frozen == q2_frozen, f'Q1 and Q2 must be frozen or unfrozen together {q1_frozen=} {q2_frozen=}'
        self.frozen = q1_frozen
        return self.frozen

    def unfreeze(self):
        q1_frozen = self.q1.unfreeze(net_name='Q_function_1')
        q2_frozen = self.q2.unfreeze(net_name='Q_function_2')
        assert q1_frozen == q2_frozen, f'Q1 and Q2 must be frozen or unfrozen together {q1_frozen=} {q2_frozen=}'
        self.frozen = q1_frozen
        return self.frozen

    def get_min(self, state_goal, action):
        q1_values, q2_values = self(state_goal, action)
        return torch.min(q1_values, q2_values)

    def get_td_errors(self, states, actions, bellman_target, qf1_a_values=None, qf2_a_values=None):
        if qf1_a_values is None or qf2_a_values is None:
            qf1_a_values, qf2_a_values = self(states, actions)
        td_error_q1 = torch.abs(qf1_a_values.flatten() - bellman_target).detach()
        td_error_q2 = torch.abs(qf2_a_values.flatten() - bellman_target).detach()
        if self.config.td_error_q_reduction is None:
            td_errors = td_error_q1
        elif self.config.td_error_q_reduction == 'mean':
            td_errors = torch.mean(torch.vstack([td_error_q1, td_error_q2]), dim=0)
        elif self.config.td_error_q_reduction == 'max':
            td_errors = torch.max(td_error_q1, td_error_q2)
        else:
            raise NotImplementedError(f'{self.config.td_error_q_reduction=}')
        return td_errors, td_error_q1, td_error_q2

    def get_losses(self, states, actions, bellman_target, weights, metrics):
        qf1_a_values, qf2_a_values = self(states, actions)
        qf1_bellman_loss = F.mse_loss(qf1_a_values.flatten(), bellman_target, reduction='none') * weights.squeeze()
        qf2_bellman_loss = F.mse_loss(qf2_a_values.flatten(), bellman_target, reduction='none') * weights.squeeze()
        qf_bellman_loss = qf1_bellman_loss + qf2_bellman_loss
        qf_bellman_loss = qf_bellman_loss.mean()

        td_errors, td_error_q1, td_error_q2 = self.get_td_errors(states, actions, bellman_target, qf1_a_values, qf2_a_values)

        metrics["losses/qf1_values"] = qf1_a_values.mean().item()
        metrics["losses/qf1_values_max"] = qf1_a_values.max().item()
        metrics["losses/qf1_values_min"] = qf1_a_values.min().item()
        metrics["losses/qf2_values"] = qf2_a_values.mean().item()
        metrics["losses/qf2_values_max"] = qf2_a_values.max().item()
        metrics["losses/qf2_values_min"] = qf2_a_values.min().item()
        metrics["losses/qf1_bellman_loss"] = qf1_bellman_loss.mean().item()
        metrics["losses/qf2_bellman_loss"] = qf2_bellman_loss.mean().item()
        metrics["losses/qf_bellman_loss"] = qf_bellman_loss.item()
        metrics["losses/td_errors_q1"] = td_error_q1.mean().item()
        metrics["losses/td_errors_q2"] = td_error_q2.mean().item()
        metrics["losses/td_errors"] = td_errors.mean().item()
        metrics["losses/td_errors_max"] = td_errors.max().item()
        metrics["losses/td_errors_min"] = td_errors.min().item()

        return qf_bellman_loss, td_errors

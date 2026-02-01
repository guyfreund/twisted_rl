import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal, Beta

from mujoco_infra.mujoco_utils.mujoco import set_physics_state, get_position_from_physics, \
    convert_topology_state_to_input_vector
from system_flow.low_level_class.base_low_level import LowLevelPlanner


class stochastic_s2a_netowrk(nn.Module):
    def __init__(self, input_size, output_size, output_ranges, minimal_std=0.01, dropout=0.0, device="cuda",
                 only_R1=False, num_of_links=21, use_beta: bool = False):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.mixture_count = 4 #mixture_count3
        self.minimal_std = minimal_std
        self.action_loss = nn.CrossEntropyLoss()
        self.only_R1 = only_R1
        self.num_of_links = num_of_links
        self.use_beta = use_beta

        #defining ranges
        self.output_range_height = output_ranges["height"]
        self.output_range_x = output_ranges["x"]
        self.output_range_y = output_ranges["y"]
        self.output_height_min, self.output_height_max = self.output_range_height
        self.output_x_min, self.output_x_max = self.output_range_x
        self.output_y_min, self.output_y_max = self.output_range_y

        #moving ranges to cuda
        self.output_height_min = torch.tensor(self.output_height_min).to(self.device)
        self.output_height_max = torch.tensor(self.output_height_max).to(self.device)
        self.output_x_min = torch.tensor(self.output_x_min).to(self.device)
        self.output_x_max = torch.tensor(self.output_x_max).to(self.device)
        self.output_y_min = torch.tensor(self.output_y_min).to(self.device)
        self.output_y_max = torch.tensor(self.output_y_max).to(self.device)

        #self.network_output_size = self.mixture_count * (1 + 2 * self.output_size)
        tanh_slack = 1.1

        #defining the distribution
        #self._mu_linear_coefficient = tanh_slack * 0.5 * (self.output_max - self.output_min).view(1, self.output_size)
        #self._mu_bias = tanh_slack * 0.5 * (self.output_max + self.output_min).view(1, self.output_size)

        #network architacture
        self.layer1 = nn.Linear(self.input_size, 512)
        self.layer2 = nn.Linear(512, 1024)
        self.layer3 = nn.Linear(1024, 2048)
        self.layer4 = nn.Linear(2048, 2048)
        self.layer5 = nn.Linear(2048, 2048)
        self.layer6 = nn.Linear(2048, self.output_size)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout_value = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()

    def forward(self, input):
        x = input
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        x = self.activation(x)
        if self.dropout_value > 0:
            x = self.dropout(x)
        x = self.layer5(x)
        x = self.activation(x)
        x = self.layer6(x)
        action_part = self.output_size - 3*2
        action = x[:,:action_part]
        pos = x[:, action_part:]
        if not self.only_R1:
            action = self.softmax(action)
        if self.use_beta:
            # perform soft plus and add 1 to have alpha,beta > 1
            pos = self.softplus(pos) + 1
            if self.only_R1:
                action = self.softplus(action) + 1
        return action, pos

    def compute_loss(self, states, targets, take_mean=True):
        actions, params = self(states)
        action_dis, height_dis, x_dis, y_dis = self._output_to_dist(action=actions, params=params)
        if not isinstance(targets, Tensor):
            print("gt is not tensor")
        #action_loss = self.action_loss(action_dis, targets[:,:21])
        if not self.only_R1:
            action_part = 21
            action_loss = -action_dis.log_prob(torch.argmax(targets[:,:action_part], dim=1)) #(action_dis, targets[:,:21])
        else:
            action_part = 1
            if self.use_beta:
                action_loss = -action_dis.log_prob(self.transform_range(x=targets[:, action_part - 1], from_=(0, 1), to=(0, 1)))
            else:
                action_loss = -action_dis.log_prob(targets[:, action_part - 1])
        if not self.use_beta:
            height_loss = -height_dis.log_prob(targets[:,action_part])
            x_loss = -x_dis.log_prob(targets[:,action_part + 1])
            y_loss = -y_dis.log_prob(targets[:,action_part + 2])
        else:
            height_loss = -height_dis.log_prob(self.transform_range(x=targets[:, action_part], from_=(self.output_height_min, self.output_height_max), to=(0, 1)))
            x_loss = -x_dis.log_prob(self.transform_range(x=targets[:, action_part + 1], from_=(self.output_x_min, self.output_x_max), to=(0, 1)))
            y_loss = -y_dis.log_prob(self.transform_range(x=targets[:, action_part + 2], from_=(self.output_y_min, self.output_y_max), to=(0, 1)))
        loss = 10*action_loss + height_loss + x_loss + y_loss
        loss_pos = sum(height_loss + x_loss + y_loss)
        loss_action = sum(action_loss)
        action, height, x, y = self._output_to_sample(temp_action=actions, temp_pos=params)
        if take_mean:
            return loss.mean(), loss_action, loss_pos, height_loss.sum(), x_loss.sum(), y_loss.sum(), action, height, x, y
        return loss, action, height, x, y

    def _output_to_sample(self, temp_action, temp_pos, deterministic=False):
        if deterministic:
            action =  torch.argmax(temp_action[:], dim=1)
            pos_prediction = temp_pos.view(-1,3,2)
            height = pos_prediction[:,0,0]
            x = pos_prediction[:,1,0]
            y = pos_prediction[:,2,0]
        else:
            action_dis, height_dis, x_dis, y_dis = self._output_to_dist(temp_action, temp_pos)
            action = action_dis.sample()
            height = height_dis.sample()
            x = x_dis.sample()
            y = y_dis.sample()
        height, x, y, action = self.clip_sample(height, x, y, action)
        return action, height, x, y

    def _output_to_dist(self, action, params):
        if not self.only_R1:
            action_dis = Categorical(action)
        else:
            if self.use_beta:
                action_dis = Beta(action[:, 0], action[:, 1])
            else:
                action_dis = Normal(loc=action[:,0], scale=self._make_positive(action[:,1]))

        height_params = params[:,:2]
        x_params = params[:, 2:4]
        y_params = params[:, 4:6]
        if self.use_beta:
            height_dis = Beta(height_params[:, 0], height_params[:, 1])
            x_dis = Beta(x_params[:, 0], x_params[:, 1])
            y_dis = Beta(y_params[:, 0], y_params[:, 1])
        else:
            height_dis = Normal(loc=height_params[:,0], scale=self._make_positive(height_params[:,1]))
            x_dis = Normal(loc=x_params[:, 0], scale=self._make_positive(x_params[:, 1]))
            y_dis = Normal(loc=y_params[:, 0], scale=self._make_positive(y_params[:, 1]))

        return action_dis, height_dis, x_dis, y_dis

    def get_prediction(self, states):
        actions, params = self(states)
        return actions, params

    def clip_sample(self, height, x, y, action):
        if not self.use_beta:
            height = torch.clamp(height, min=self.output_height_min, max=self.output_height_max)
            x = torch.clamp(x, min=self.output_x_min, max=self.output_x_max)
            y = torch.clamp(y, min=self.output_y_min, max=self.output_y_max)
            action = torch.clamp(action, min=0., max=1.) if self.only_R1 else action
        else:
            height = self.transform_range(x=height, from_=(0, 1), to=(self.output_height_min, self.output_height_max), eps=0.)
            x = self.transform_range(x=x, from_=(0, 1), to=(self.output_x_min, self.output_x_max), eps=0.)
            y = self.transform_range(x=y, from_=(0, 1), to=(self.output_y_min, self.output_y_max), eps=0.)
            action = self.transform_range(x=action, from_=(0, 1), to=(0, 1), eps=0.) if self.only_R1 else action
        return height, x, y, action

    @staticmethod
    def _make_positive(x: Tensor):
        x = torch.exp(x)
        x = x + 1.e-5
        return x

    @staticmethod
    def transform_range(x, from_: (float, float), to: (float, float), eps: float = 1e-7):
        a, b = from_
        c, d = to
        c, d = c + eps, d - eps
        return (x - a) / (b - a) * (d - c) + c


class S2APlanner(LowLevelPlanner):
    def __init__(self, cfg, config_length):
        super(S2APlanner, self).__init__(cfg, config_length)
        
        #init varibels
        self.s2a_path = self.cfg["STATE2ACTION_PARMS"]["path"]
        self.s2a_input_size = self.cfg["STATE2ACTION_PARMS"]["input_size"]
        self.s2a_output_size = self.cfg["STATE2ACTION_PARMS"]["output_size"]
        self.batch_size = int(1.5 * self.cfg["batch_size"])

        #init and load  NN
        self.init_state2action_nn()
        self.load_state2action_nn()

    def init_state2action_nn(self):
            height = self.cfg["STATE2ACTION_PARMS"]["output_ranges"]["height"]
            x = self.cfg["STATE2ACTION_PARMS"]["output_ranges"]["x"]
            y = self.cfg["STATE2ACTION_PARMS"]["output_ranges"]["y"]
            output_ranges = {
                "height": np.array(height),
                "x": np.array(x),
                "y": np.array(y),
            }
            self.s2a_model = stochastic_s2a_netowrk(input_size=self.s2a_input_size , output_size=self.s2a_output_size,\
                output_ranges=output_ranges, dropout=0)
    
    def load_state2action_nn(self):
        init = torch.load(self.s2a_path)
        update_model = {}
        try:
            
            model_state = init["model_state"]
            for i in model_state:
                new_i = i.replace('module.', '')
                update_model[new_i] = model_state[i]
        except Exception as e:
            model_state = init["state_dict"]
            for i in model_state:
                new_i = i.replace('model.', '')
                update_model[new_i] = model_state[i]
        self.s2a_model.load_state_dict(update_model)
        self.s2a_model = self.s2a_model.cuda()

    def generate_action(self, configuration, target_topology_state, state_idx, plan, physics, playground_physics):
        batch_size = self.batch_size
        set_physics_state(playground_physics, configuration)
        pos = get_position_from_physics(playground_physics)
        pos = np.reshape(pos, -1)
        topology_vector = convert_topology_state_to_input_vector(target_topology_state.points)
        x = np.zeros(self.s2a_input_size) #24 is action
        x[:47] = configuration[:47]
        x[47:113] = pos[:]
        x[113:] = topology_vector[:]
        #inference
        x = list(x)
        x = torch.tensor(x)

        x = torch.unsqueeze(x, 0)
        x = x.repeat(batch_size,1)

        if torch.cuda.is_available():
            x = x.cuda()

        actions_index, params = self.s2a_model.get_prediction(x.float())
        action_index, height, x_pos, y_pos = self.s2a_model._output_to_sample(actions_index, params)
        
        #get actions
        action_index = action_index.view(-1,1)
        height = height.view(-1,1)
        x_pos = x_pos.view(-1,1)
        y_pos = y_pos.view(-1,1)
        output = torch.cat((action_index, height, x_pos, y_pos),1)
        
        return output, [configuration for _ in range(batch_size)]

    def close(self):
        pass

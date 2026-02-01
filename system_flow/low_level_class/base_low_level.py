import torch
from torch import nn
import numpy as np
from abc import ABC, abstractmethod

from mujoco_infra.mujoco_utils.mujoco import calculate_number_of_crosses_from_topology_state

with torch.no_grad():
    class LowLevelPlanner(ABC):
        def __init__(self, cfg, config_length):
            self.cfg = cfg

            self.paths_to_models = self.cfg["STATE2STATE_PARMS"]["paths"]
            self.input_size = self.cfg["STATE2STATE_PARMS"]["input_size"]
            self.output_size = self.cfg["STATE2STATE_PARMS"]["output_size"]
            self.dropout = self.cfg["STATE2STATE_PARMS"]["dropout"]
            self.num_of_links = self.cfg["STATE2STATE_PARMS"]["num_of_links"]
            self.max_tries = self.cfg["max_tries"]
            self.config_length = config_length
            self.ensemble_prediction = self.cfg["STATE2STATE_PARMS"]["ensemble_prediction"]
            self.batch_size = self.cfg["batch_size"]

        @abstractmethod
        def generate_action(self, configuration, target_topology_state, state_idx, plan, physics, playground_physics):
            pass

        def find_curve(self, configuration, target_topology_state, state_idx, plan, physics, playground_physics):
            options = []
            results = self.generate_action(configuration, target_topology_state, state_idx, plan, physics, playground_physics)
            actions, configurations = results[0], results[1]
            episodes = results[2] if len(results) > 2 else None
            if not torch.is_tensor(actions):
                raise "need to change the output"
            actions = actions.cpu()
            for index in range(self.batch_size):
                sample = {
                    "action": torch.squeeze(actions[index]),
                    "configuration": configurations[index],
                    "episode": episodes[index] if episodes is not None else None,
                }
                options.append(sample)

            #select relevant samples
            selected_samples = self._select_samples(options)

            return selected_samples

        def _sort_actions(self,samples):

            #order ensemble_uncertainty from low to high
            if self.cfg["Select_Samples"]["Sort_Actions"]["sort_configuration_uncertainty"]: 
                samples = sorted(samples, key=lambda x: x['ensemble_uncertainty'],\
                    reverse=self.cfg["Select_Samples"]["Sort_Actions"]["sort_configuration_uncertainty_reverse"]) 
            
            #order topology_uncertainty from high to low and prediction_uncertainty from True to False
            samples = sorted(samples, key=lambda x: (x['topology_uncertainty'],x['prediction_uncertainty']),\
                 reverse=True)

            #print dis
            dis = []
            for sample in samples:
                dis.append(sample['topology_uncertainty'])
            number_of_crosses = calculate_number_of_crosses_from_topology_state(samples[0]['target_topology_state'])
            print("target_topology_state =", number_of_crosses, ", dis of topology uncertainty =", dis)
            
            return samples

        def _select_samples(self, samples):
            #cnt = len(samples)
            if self.cfg["Select_Samples"]["Sort_Actions"]["enable"]:
                samples = self._sort_actions(samples)
            num_samples = self.cfg["Select_Samples"]["num_samples"]
            selected_samples = samples[:num_samples]

            return selected_samples
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
from easydict import EasyDict as edict
from functools import partial
import os
from typing import List, Optional, Dict
import torch

from exploration.mdp.graph.problem_set import ProblemSet
from exploration.mdp.graph.high_level_graph import HighLevelGraph
from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.mdp.low_level_state import LowLevelState
from exploration.rl.cleanrl_scripts.sac_algorithm import SACAlgorithm
from exploration.rl.environment.exploration_gym_envs import ExplorationGymEnvs
from system_flow.low_level_class.base_low_level import LowLevelPlanner
from exploration.utils.config_utils import load_config, fix_env_config_backwards_compatibility, set_agents_by_ablation
from mujoco_infra.mujoco_utils.mujoco import convert_qpos_to_xyz_with_move_center, \
    convert_pos_to_topology, set_physics_state


class RLPlanner(LowLevelPlanner):
    def __init__(self, cfg, config_length, init_pool_context=True, env=None, init_env=False, init_agents=True):
        super().__init__(cfg, config_length)
        self.problem_set = ProblemSet()
        self.rl_config = edict(cfg["RL"])
        self.effective_batch_size = int(1.5 * self.batch_size)
        assert env is not None if not init_env else True, "Environment should be initialized if init_env is True"
        self.init_env = init_env
        self.max_steps = self.rl_config.max_steps
        self.init_pool_context = init_pool_context
        self.env = self.load_env(env)
        self.high_level_graph = HighLevelGraph.load_full_graph()
        assert self.env is not None, "Environment is not initialized"
        self.max_crosses = 10
        self.agents = self.get_agents() if init_agents else {}
        self.first_generate_action_occurred = {c: {move: False for move in self.env.high_level_graph.high_level_actions} for c in range(4)}
        self.episode_length = {a: {i: [] for i in range(self.max_crosses)} for a in ['R1', 'R2', 'cross']}
        self.current_plan_success = None

    def load_env(self, env):
        if env is None and self.init_env:
            agent_path = list(self.rl_config.agents.values())[0]
            print(f"{self.__class__.__name__} Loading environment from {agent_path}")
            config_path = os.path.join(os.path.dirname(agent_path), "config.yml")
            env_cfg = self.get_config(config_path).env
            env = ExplorationGymEnvs.from_cfg(cfg=env_cfg, init_pool_context=self.init_pool_context)
        return env

    def get_agents(self):
        agents = {}
        for problem_name, problem in self.problem_set.PROBLEMS.items():
            agent_path = self.rl_config.agents.get(problem_name)
            agent = self.get_agent(agent_path) if agent_path is not None else None
            agents[problem_name] = agent
        return agents

    def get_agent_by_crossing_number_and_move(self, crossing_number: int, move: str, to_print: bool) -> Optional[SACAlgorithm]:
        while crossing_number >= 0:
            problem_names = [f'G{crossing_number}_{move.capitalize()}', f'G{crossing_number}', f'G_{move.capitalize()}', 'G']
            for problem_name in problem_names:
                if self.agents.get(problem_name) is not None:
                    if to_print:
                        print(f"{self.__class__.__name__} Using agent for crossing number {crossing_number} and move {move} in {problem_name}")
                    return self.agents[problem_name]
            crossing_number -= 1
        raise ValueError(f"Agent for crossing number {crossing_number} and move {move} not found")

    def get_agent(self, path):
        config = self.get_config(os.path.join(os.path.dirname(path), "config.yml"))
        agent = SACAlgorithm(config.algorithm, self.env)
        agent.load(path)
        return agent

    def get_config(self, path):
        config = load_config(path)
        config.env = fix_env_config_backwards_compatibility(config.env)
        config.env.num_cpus = self.effective_batch_size
        config.env.env.reachable_configurations.replay_buffer_files_path = None
        config.env.env.min_crosses = 0
        config.env.env.max_crosses = self.max_crosses
        config.env.env.depth = self.max_crosses
        config.env.env.high_level_actions = ['R1', 'R2', 'cross']
        if self.max_steps is not None:
            config.env.env.max_steps = self.max_steps
        return config

    def play_episodes(self, goal_states, goal_actions, low_level_states, agent):
        queries = [{
            'goal_high_level_state': goal_high_level_state,
            'goal_high_level_action': goal_high_level_action,
            'start_low_level_state': start_low_level_state
        } for goal_high_level_state, goal_high_level_action, start_low_level_state in zip(goal_states, goal_actions, low_level_states)]
        get_actions = partial(agent.predict_action, deterministic=False)
        st = datetime.now()
        episodes, _, _ = self.env.play_episodes(get_actions=get_actions, apply_her=True, queries=queries)
        et = datetime.now()
        print(f"{self.__class__.__name__} Time to play episodes: {et - st}")
        return episodes

    def update_episode_containers(self, move, start_crossing_number, episodes,
                                  success_actions, success_configurations, success_episodes,
                                  success_actions_her, success_configurations_her, success_episodes_her,
                                  failed_actions, failed_configurations, failed_episodes):
        for episode in episodes:
            states, actions, raw_actions, rewards, done_flags, truncateds, infos = episode
            episode_length = len(actions)
            if episode_length == 0:
                continue
            last_experience = infos[-1]['experience']
            exception_occurred = last_experience.exception_occurred
            if exception_occurred:
                continue
            if last_experience.info.goal_reached:
                if last_experience.info.is_her:
                    success_actions_her.append(actions[-1])
                    success_configurations_her.append(last_experience.start_low_level_state_centered.configuration)
                    success_episodes_her.append(episode)
                else:
                    success_actions.append(actions[-1])
                    success_configurations.append(last_experience.start_low_level_state_centered.configuration)
                    success_episodes.append(episode)
                    self.episode_length[move][start_crossing_number].append(episode_length)
            else:
                failed_actions.append(actions[-1])
                failed_configurations.append(last_experience.start_low_level_state_centered.configuration)
                failed_episodes.append(episode)

    def get_episode_metadata(self, configuration, target_topology_state, physics, playground_physics):
        set_physics_state(playground_physics, configuration)
        state_pos = convert_qpos_to_xyz_with_move_center(playground_physics, configuration)
        low_level_state = LowLevelState(configuration)
        goal_state = HighLevelAbstractState.from_abstract_state(target_topology_state)
        state_topology = HighLevelAbstractState.from_abstract_state(convert_pos_to_topology(state_pos))
        start_crossing_number = state_topology.crossing_number
        assert self.env.high_level_graph.has_edge(state_topology, goal_state)
        goal_actions = self.env.high_level_graph.get_all_edge_variations(src=state_topology, dst=goal_state, from_graph=True)
        if start_crossing_number == 0:
            goal_actions = [a for a in goal_actions if a.data['move'] != 'cross']
        assert len(goal_actions) > 0, f"Goal actions are empty for {state_topology} and {goal_state}"
        goal_action = goal_actions[0]

        start_state_idx = self.high_level_graph.get_node_index(state_topology)
        goal_state_idx = self.high_level_graph.get_node_index(goal_state)
        goal_action_idx = self.high_level_graph.get_edge_index(goal_action)

        print(f'{self.__class__.__name__} {start_crossing_number=} {start_state_idx=} {goal_state_idx=} {goal_action_idx=} HLA={goal_action.data["move"]}')

        move = goal_action.data['move']

        return low_level_state, goal_state, goal_action, start_crossing_number, move

    def gather_results(self, move, start_crossing_number,
                       success_actions, success_configurations, success_episodes,
                       success_actions_her, success_configurations_her, success_episodes_her,
                       failed_actions, failed_configurations, failed_episodes):
        if len(success_actions) > 0:
            print(f"{self.__class__.__name__} Success actions: {len(success_actions)} for move {move} and start crossing number {start_crossing_number}")
        if len(success_actions_her) > 0:
            print(f"{self.__class__.__name__} Success actions HER: {len(success_actions_her)} for move {move} and start crossing number {start_crossing_number}")
        actions = success_actions + success_actions_her + failed_actions
        configurations = success_configurations + success_configurations_her + failed_configurations
        sorted_episodes = success_episodes + success_episodes_her + failed_episodes
        actions = np.stack(actions)
        torch_action = torch.from_numpy(actions)
        self.first_generate_action_occurred[start_crossing_number][move] = True
        return torch_action, configurations, sorted_episodes

    def generate_action(self, configuration, target_topology_state, state_idx, plan, physics, playground_physics):
        low_level_state, goal_state, goal_action, start_crossing_number, move = self.get_episode_metadata(
            configuration=configuration,
            target_topology_state=target_topology_state,
            physics=physics,
            playground_physics=playground_physics
        )
        low_level_states = [low_level_state for _ in range(self.effective_batch_size)]
        goal_states = [goal_state for _ in range(self.effective_batch_size)]
        goal_actions = [goal_action for _ in range(self.effective_batch_size)]

        if state_idx == 0:
            self.current_plan_success = [False] * len(plan)

        to_print = self.first_generate_action_occurred[start_crossing_number][move] is False
        agent = self.get_agent_by_crossing_number_and_move(
            crossing_number=start_crossing_number,
            move=move,
            to_print=True,
        )

        episodes = self.play_episodes(
            goal_states=goal_states,
            goal_actions=goal_actions,
            low_level_states=low_level_states,
            agent=agent
        )

        success_actions, success_actions_her, failed_actions = [], [], []
        success_configurations, success_configurations_her, failed_configurations = [], [], []
        success_episodes, success_episodes_her, failed_episodes = [], [], []
        kwargs = {
            'move': move,
            'start_crossing_number': start_crossing_number,
            'success_actions': success_actions,
            'success_configurations': success_configurations,
            'success_episodes': success_episodes,
            'success_actions_her': success_actions_her,
            'success_configurations_her': success_configurations_her,
            'success_episodes_her': success_episodes_her,
            'failed_actions': failed_actions,
            'failed_configurations': failed_configurations,
            'failed_episodes': failed_episodes,
        }
        self.update_episode_containers(**kwargs, episodes=episodes)
        if len(success_actions) > 0:
            self.current_plan_success[state_idx] = True
        torch_action, configurations, sorted_episodes = self.gather_results(**kwargs)

        return torch_action, configurations, sorted_episodes

    def close(self, print_stats=True, handle_env=True):
        if print_stats:
            for a, data in self.episode_length.items():
                for crossing_number, episode_lengths in data.items():
                    print(f"{self.__class__.__name__} {a=} {crossing_number=} average episode length: {np.mean(episode_lengths)}")
        if handle_env:
            if self.env is not None and self.init_env:
                self.env.close()


class MultiRLPlanner(RLPlanner):
    def __init__(self, cfg, config_length, init_pool_context=True, env=None, init_env=False):
        super().__init__(cfg=cfg, config_length=config_length, init_pool_context=init_pool_context, env=env,
                         init_env=init_env, init_agents=False)

        self.planners: List[RLPlanner] = []
        self.ablations = self.rl_config.ablations
        self.current_plan_success = [[] for _ in range(len(self.ablations))]

        for ablation in self.ablations:
            ablation_cfg = deepcopy(self.cfg)
            set_agents_by_ablation(cfg=ablation_cfg, ablation=ablation)
            ablation_rl_planner = RLPlanner(cfg=ablation_cfg, config_length=config_length, init_pool_context=False,
                                            env=self.env, init_env=False, init_agents=True)
            self.planners.append(ablation_rl_planner)
            print(f"{self.__class__.__name__} Loaded planner for ablation TWISTED-RL-{ablation}")

    def generate_action(self, configuration, target_topology_state, state_idx, plan, physics, playground_physics):
        low_level_state, goal_state, goal_action, start_crossing_number, move = self.get_episode_metadata(
            configuration=configuration,
            target_topology_state=target_topology_state,
            physics=physics,
            playground_physics=playground_physics
        )

        success_actions, success_actions_her, failed_actions = [], [], []
        success_configurations, success_configurations_her, failed_configurations = [], [], []
        success_episodes, success_episodes_her, failed_episodes = [], [], []
        kwargs = {
            'move': move,
            'start_crossing_number': start_crossing_number,
            'success_actions': success_actions,
            'success_configurations': success_configurations,
            'success_episodes': success_episodes,
            'success_actions_her': success_actions_her,
            'success_configurations_her': success_configurations_her,
            'success_episodes_her': success_episodes_her,
            'failed_actions': failed_actions,
            'failed_configurations': failed_configurations,
            'failed_episodes': failed_episodes
        }

        for idx, planner in enumerate(self.planners):
            ablation = self.ablations[idx]
            print(f"{self.__class__.__name__} Generating action using planner {ablation}")
            _, _, episodes = planner.generate_action(
                configuration, target_topology_state, state_idx, plan, physics, playground_physics
            )
            self.update_episode_containers(**kwargs, episodes=episodes)
            if state_idx == len(plan) - 1:
                self.current_plan_success[idx] = planner.current_plan_success

        torch_action, configurations, sorted_episodes = self.gather_results(**kwargs)

        return torch_action, configurations, sorted_episodes

    def close(self):
        for idx, planner in enumerate(self.planners):
            ablation = self.ablations[idx]
            print(f"Closing planner {ablation}")
            planner.close(print_stats=False, handle_env=False)
        super().close(print_stats=True, handle_env=True)

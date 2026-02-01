import pickle
import traceback
from collections import defaultdict
from uuid import uuid4

from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from typing import Optional, Tuple, List, Callable
from easydict import EasyDict as edict
import numpy as np
import os
from dm_control.mujoco import Physics
from datetime import datetime
import pytz
import shutil
from pytorch_lightning import seed_everything
from copy import deepcopy
from tqdm import tqdm

from exploration.preprocessing.preprocessor import Preprocessor
from exploration.preprocessing.preprocessor_factory import PreprocessorFactory
from exploration.goal_selector.goal_selector import GoalSelector
from exploration.goal_selector.goal_selector_factory import GoalSelectorFactory
from exploration.initial_state_selector.initial_state_selector import InitialStateSelector
from exploration.initial_state_selector.initial_state_selector_factory import InitialStateSelectorFactory
from exploration.mdp.state_mapping import StateVisitation
from exploration.reachable_configurations.reachable_configurations import IReachableConfigurations, ReachableConfigurationsState
from exploration.reachable_configurations.reachable_configurations_factory import ReachableConfigurationsFactory
from exploration.mdp.low_level_state import LowLevelState
from exploration.rl.environment.env_utils import get_next_state_from_experience, get_initial_state
from exploration.rl.environment.pool_episode_runner import run_queries_parallel, PoolContext
from exploration.rl.experience import Experience, EpisodeExperiences
from exploration.mdp.graph.high_level_graph import HighLevelGraph
from exploration.mdp.high_level_state import HighLevelAbstractState
from mujoco_infra.mujoco_utils.mujoco import physics_reset, get_current_primitive_state, set_physics_state, \
    convert_qpos_to_xyz_with_move_center, convert_pos_to_topology, get_link_segments
from exploration.utils.config_utils import load_env_config


class ExplorationGymEnvs:
    def __init__(self,
                 cfg: edict,
                 num_workers: int,
                 seed: int,
                 output_dir: str,
                 exceptions_dir: str,
                 save_videos: bool,
                 reachable_configurations: IReachableConfigurations,
                 goal_selector: GoalSelector,
                 initial_state_selector: InitialStateSelector,
                 preprocessor: Preprocessor,
                 env_path: str,
                 num_of_links: int,
                 goal_reward: int,
                 neg_reward: int,
                 stay_reward: int,
                 max_steps: int,
                 min_crosses: int,
                 max_crosses: int,
                 depth: int,
                 high_level_actions: List[str],
                 link_min: float,
                 link_max: float,
                 z_min: float,
                 z_max: float,
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float,
                 her: bool,
                 init_pool_context: bool = True
                 ):
        self.cfg = cfg
        self.num_workers = num_workers
        self.seed = seed
        self.save_videos = save_videos
        self.output_dir = output_dir
        self.exceptions_dir = exceptions_dir
        self.to_raise = False
        self.env_dir = None
        self.reachable_configurations_dir = None
        self.high_level_visitation_dir = None
        self.high_level_action_visitation_dir = None
        self.goal_generation_dir = None
        self.videos_dir = None
        self.R1_success_videos_dir = None
        self.R1_failure_videos_dir = None
        self.R2_success_videos_dir = None
        self.R2_failure_videos_dir = None
        self.cross_success_videos_dir = None
        self.cross_failure_videos_dir = None
        self.create_file_system_dirs()
        self.env_path = env_path
        self.goal_reward = goal_reward
        self.neg_reward = neg_reward
        self.stay_reward = stay_reward
        self.max_steps = max_steps
        self.min_crosses = min_crosses
        self.max_crosses = max_crosses
        self.depth = depth
        self.high_level_actions = high_level_actions
        self.reachable_configurations = reachable_configurations
        self.goal_selector = goal_selector
        self.initial_state_selector = initial_state_selector
        self.preprocessor = preprocessor
        self.num_of_links = num_of_links
        self.link_min = link_min
        self.link_max = link_max
        self.z_min = z_min
        self.z_max = z_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.her = her
        self.goal_states = []
        self.goal_actions = []
        self.high_level_graph = HighLevelGraph.load_full_graph()
        self.observation_space = self.create_observation_space(preprocessor=self.preprocessor, num_of_links=self.num_of_links)
        self.action_space = self.create_action_space(
            seed=self.seed,
            link_min=self.link_min, link_max=self.link_max,
            z_min=self.z_min, z_max=self.z_max,
            x_min=self.x_min, x_max=self.x_max,
            y_min=self.y_min, y_max=self.y_max
        )
        self.init_pool_context = init_pool_context
        if self.init_pool_context:
            self.pool_context = PoolContext(self.num_workers, **self.get_worker_env_params())
        else:
            self.pool_context = None
        self.high_level_visitation = StateVisitation()
        self.goal_generation = StateVisitation()
        self.high_level_action_visitation = StateVisitation()

    def create_file_system_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.exceptions_dir, exist_ok=True)
        self.env_dir = os.path.join(self.output_dir, 'env')
        os.makedirs(self.env_dir, exist_ok=True)

        # videos
        if self.save_videos:
            self.videos_dir = os.path.join(self.env_dir, 'videos')
            for action in ['R1', 'R2', 'cross']:
                for result in ['success', 'failure']:
                    action_result_dir = os.path.join(self.videos_dir, action, result)
                    os.makedirs(action_result_dir, exist_ok=True)
                    setattr(self, f'{action}_{result}_videos_dir', action_result_dir)

        # internals
        for internal_name in ['reachable_configurations', 'high_level_visitation', 'high_level_action_visitation', 'goal_generation']:
            internal_dir = os.path.join(self.env_dir, internal_name)
            os.makedirs(internal_dir, exist_ok=True)
            setattr(self, f'{internal_name}_dir', internal_dir)

    def get_worker_env_params(self) -> dict:
        return {
            'seed': self.seed,
            'env_path': self.env_path,
            'goal_reward': self.goal_reward,
            'neg_reward': self.neg_reward,
            'stay_reward': self.stay_reward,
            'max_steps': self.max_steps,
            'max_crosses': self.max_crosses,
            'observation_space': self.create_observation_space(
                preprocessor=self.preprocessor,
                num_of_links=self.num_of_links
            ),
            'action_space': self.create_action_space(
                seed=self.seed,
                link_min=self.link_min, link_max=self.link_max,
                z_min=self.z_min, z_max=self.z_max,
                x_min=self.x_min, x_max=self.x_max,
                y_min=self.y_min, y_max=self.y_max
            ),
            'preprocessor': self.preprocessor,
            'high_level_graph': self.high_level_graph,
            'save_episode_video': self.save_videos,
            'output_dir': self.output_dir,
            'exceptions_dir': self.exceptions_dir,
            'env_dir': self.env_dir,
            'videos_dir': self.videos_dir,
            'R1_success_videos_dir': self.R1_success_videos_dir,
            'R1_failure_videos_dir': self.R1_failure_videos_dir,
            'R2_success_videos_dir': self.R2_success_videos_dir,
            'R2_failure_videos_dir': self.R2_failure_videos_dir,
            'cross_success_videos_dir': self.cross_success_videos_dir,
            'cross_failure_videos_dir': self.cross_failure_videos_dir,
        }

    @staticmethod
    def create_observation_space(preprocessor: Preprocessor, num_of_links: int) -> DictSpace:
        state_space = Box(
            low=-np.inf * np.ones(preprocessor.observation_length, dtype=np.float32),
            high=np.inf * np.ones(preprocessor.observation_length, dtype=np.float32)
        )
        goal_space = Box(
            low=preprocessor.goal_representation_low,
            high=preprocessor.goal_representation_high,
        )
        observation_space = DictSpace({
            'observation': state_space,
            'desired_goal': goal_space
        })
        return observation_space

    @staticmethod
    def create_action_space(seed: int, link_min: float, link_max: float, z_min: float, z_max: float, x_min: float,
                            x_max: float, y_min: float, y_max: float) -> Box:
        return Box(
            low=np.array([link_min, z_min, x_min, y_min], dtype=np.float32),
            high=np.array([link_max, z_max, x_max, y_max], dtype=np.float32),
            seed=seed
        )

    def set_goals(self):
        st = datetime.now()
        goal_actions = self.goal_selector.select(reachable_configurations=self.reachable_configurations, total=self.num_workers)
        goal_states = [goal_action.dst for goal_action in goal_actions]
        et = datetime.now()
        goal_selection_time = (et - st).total_seconds()

        st = datetime.now()
        low_level_states, high_level_states = self.initial_state_selector.select(goal_actions=goal_actions, reachable_configurations=self.reachable_configurations)
        et = datetime.now()
        initial_state_selection_time = (et - st).total_seconds()

        goals_per_crossing_number = defaultdict(list)
        for goal in goal_states:
            goals_per_crossing_number[goal.crossing_number].append(goal)
        assert max(goals_per_crossing_number.keys()) <= self.max_crosses
        for crossing_number in range(1, self.max_crosses + 1):
            print(f'C={crossing_number}: Generated {len(goals_per_crossing_number[crossing_number])} goals')

        return low_level_states, high_level_states, goal_states, goal_actions, goal_selection_time, initial_state_selection_time

    def update_internals_from_experience(self, experience: Experience, is_first: bool):
        high_level_action_exists = False
        if experience.high_level_action is not None:
            if self.high_level_graph.has_edge(src=experience.start_high_level_state, dst=experience.end_high_level_state, num_extensions=0):
                self.high_level_action_visitation.visit(experience.high_level_action)
                high_level_action_exists = True

        if all([
            not experience.is_empty,
            not experience.exception_occurred,
            not (experience.moved_to_lower_goal_crossing_number is not None and experience.moved_to_lower_goal_crossing_number),
            not (experience.moved_to_higher_goal_crossing_number is not None and experience.moved_to_higher_goal_crossing_number),
            high_level_action_exists or experience.stayed_in_the_same_crossing_number
        ]):
            if self.high_level_graph.has_node(experience.end_high_level_state):
                self.high_level_visitation.visit(experience.end_high_level_state)
            self.goal_generation.visit(experience.goal_state)
            if is_first:
                rc_state_src = ReachableConfigurationsState(high_level_state=experience.start_high_level_state,
                                                            low_level_state=experience.start_low_level_state,
                                                            low_level_pos=experience.start_low_level_pos,
                                                            link_segments=experience.start_link_segments,
                                                            intersections=experience.start_intersections)
                self.reachable_configurations.add_node(state=rc_state_src)
            rc_state_dst = ReachableConfigurationsState(high_level_state=experience.end_high_level_state,
                                                        low_level_state=experience.end_low_level_state,
                                                        low_level_pos=experience.end_low_level_pos,
                                                        link_segments=experience.end_link_segments,
                                                        intersections=experience.end_intersections)
            self.reachable_configurations.add_node(state=rc_state_dst)
            # self.reachable_configurations.add_edge(src=rc_state_src, dst=rc_state_dst, action=experience.low_level_action)

    def step(self, actions: List[Optional[List]]):
        raise NotImplementedError('Use play_episodes instead')

    def close(self):
        if self.pool_context is not None:
            self.pool_context.close()

    def get_random_actions(self, states) -> (np.ndarray, None):
        raw_actions = np.array([self.action_space.sample() for _ in range(len(states))])
        return raw_actions, None

    def play_test_queries(self, apply_her: bool, get_actions: Callable = None, iterations: int = 1):
        episodes, times, is_her = [], [], []
        for i in tqdm(range(iterations), desc='Playing test queries', total=iterations):
            i_episodes, i_times, i_is_her = self.play_episodes(get_actions=get_actions, apply_her=apply_her)
            episodes.extend(i_episodes)
            times.extend(i_times)
            is_her.extend(i_is_her)
        return episodes, times, is_her

    def play_episodes(self, get_actions: Callable = None, apply_her: bool = False, queries: list = None):
        episodes, times, is_her = [], [], []

        if queries is None:
            low_level_states, high_level_states, self.goal_states, self.goal_actions,\
                goal_selection_time, initial_state_selection_time = self.set_goals()
            time_info = {
                'goal_selection_time': goal_selection_time,
                'initial_state_selection_time': initial_state_selection_time
            }
            queries = [{
                'goal_high_level_state': goal_high_level_state,
                'goal_high_level_action': goal_high_level_action,
                'start_low_level_state': start_low_level_state
            } for goal_high_level_state, goal_high_level_action, start_low_level_state in zip(self.goal_states, self.goal_actions, low_level_states)]
        else:
            time_info = {'goal_selection_time': 0., 'initial_state_selection_time': 0.}

        if get_actions is None:
            get_actions = self.get_random_actions

        if self.pool_context is None:
            raise Exception('Pool context is not initialized')

        results = run_queries_parallel(
            policy_func=get_actions, post_processing_function=None,
            queries=queries, pool_context=self.pool_context
        )
        episode_experiences_list = [result[1] for result in results]
        postprocessed_results = self.postprocess_play_episodes_results(
            episode_experiences_list=episode_experiences_list,
            apply_her=apply_her,
            time_info=time_info,
            times=times
        )
        return postprocessed_results

    def postprocess_play_episodes_results(self, episode_experiences_list, apply_her, time_info=None, times=None):
        time_info = time_info or {}
        times = times or []
        episodes, is_her = [], []

        for episode_experiences in episode_experiences_list:
            start_time = datetime.now()
            postprocess_exception_occurred = False
            times.append(episode_experiences.time)
            states, actions, raw_actions, rewards, dones, truncateds, infos = [], [], [], [], [], [], []
            is_first = True

            for experience in episode_experiences.experiences:
                self.update_internals_from_experience(experience, is_first)
                try:
                    self.extract_transition_and_add(experience, states, actions, raw_actions, rewards, dones, truncateds, infos)
                except Exception as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    postprocess_exception_occurred = True
                    if self.to_raise:
                        raise e
                    else:
                        path = os.path.join(self.exceptions_dir, 'postprocess_play_episodes_results' , f'{uuid4().hex}.pkl')
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, 'wb') as f:
                            pickle.dump(episode_experiences, f)
                        print(f'Exception occurred and saved to {path}')
                        break

                is_first = False

            if postprocess_exception_occurred:
                continue

            end_time = datetime.now()
            time_info['episode_postprocess_time'] = (end_time - start_time).total_seconds()
            if len(infos) > 0:
                infos[0].update(time_info)
            else:
                infos = [time_info]
            episodes.append((states, actions, raw_actions, rewards, dones, truncateds, infos))
            is_her.append(False)

            if apply_her and not episode_experiences.exception_occurred:
                her_episodes = self.apply_her(episode_experiences, time_info)
                for her_episode in her_episodes:
                    episodes.append(her_episode)
                    times.append(episode_experiences.time)
                    is_her.append(True)

        return episodes, times, is_her

    def _extract_transition(self, experience: Experience):
        state = get_initial_state(
            preprocessor=self.preprocessor,
            start_low_level_pos=experience.start_low_level_pos,
            start_low_level_state=experience.start_low_level_state,
            start_high_level_state=experience.start_high_level_state,
            goal=(experience.goal_state, experience.goal_action),
            link_segments=experience.start_link_segments,
            intersections=experience.start_intersections
        )
        next_state = get_next_state_from_experience(
            experience=experience,
            preprocessor=self.preprocessor,
        )
        action = experience.low_level_action.np_encoding
        raw_action = experience.raw_low_level_action
        reward = experience.reward
        done = experience.done
        truncated = False
        info = {'experience': experience}
        return state, next_state, action, raw_action, reward, done, truncated, info

    def extract_transition_and_add(self, experience, states, actions, raw_actions, rewards, dones, truncateds, infos):
        if experience.exception_occurred:
            states.append(get_initial_state(
                preprocessor=self.preprocessor,
                start_low_level_pos=experience.start_low_level_pos,
                start_low_level_state=experience.start_low_level_state,
                start_high_level_state=experience.start_high_level_state,
                goal=(experience.goal_state, experience.goal_action),
                link_segments=experience.start_link_segments,
                intersections=experience.start_intersections
            ))
            infos.append({'experience': experience})
        else:
            state, next_state, action, raw_action, reward, done, truncated, info = self._extract_transition(experience)
            if len(states) == 0:
                states.append(state)
            states.append(next_state)
            actions.append(action)
            raw_actions.append(raw_action)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)

    def apply_her(self, episode_experiences: EpisodeExperiences, time_info: dict) -> List[Tuple]:
        episodes = []
        exceptions = []
        assert not episode_experiences.exception_occurred
        last_experience = episode_experiences.get_last_experience()

        goal_not_reached = not last_experience.info.goal_reached
        moved_high_level_state = last_experience.info.moved_high_level_state
        high_level_action_exists = last_experience.high_level_action is not None

        if all([goal_not_reached, moved_high_level_state, high_level_action_exists]):
            high_level_actions = self.high_level_graph.get_all_edge_variations(
                src=last_experience.high_level_action.src,
                dst=last_experience.high_level_action.dst,
                from_graph=False,
            )
            for high_level_action in high_level_actions:
                states, actions, raw_actions, rewards, dones, truncateds, infos = [], [], [], [], [], [], []
                for idx, experience in enumerate(episode_experiences.experiences):
                    curr_experience = deepcopy(experience)
                    curr_experience.goal_state = last_experience.end_high_level_state
                    curr_experience.goal_action = high_level_action
                    curr_experience.info.is_her = True
                    if idx + 1 == episode_experiences.num_experiences:
                        curr_experience.info.goal_reached = True
                        curr_experience.reward = self.goal_reward
                        curr_experience.high_level_action = high_level_action
                    try:
                        self.extract_transition_and_add(curr_experience, states, actions, raw_actions, rewards, dones, truncateds, infos)
                    except Exception as e:
                        traceback.print_exception(type(e), e, e.__traceback__)
                        if self.to_raise:
                            raise e
                        else:
                            exceptions.append((episode_experiences, high_level_action))

                if len(infos) > 0:
                    infos[0].update(time_info)
                else:
                    infos = [time_info]
                episode = (states, actions, raw_actions, rewards, dones, truncateds, infos)
                episodes.append(episode)

        return episodes

    @classmethod
    def from_cfg(cls, cfg: edict, create_reachable_configurations: bool = True,
                 reachable_configurations: Optional[IReachableConfigurations] = None,
                 init_pool_context: bool = True) -> 'ExplorationGymEnvs':
        num_workers = min(os.cpu_count() - 2, cfg.num_cpus)
        min_crosses = cfg.env.min_crosses
        max_crosses = cfg.env.max_crosses
        depth = cfg.env.depth
        high_level_actions = cfg.env.high_level_actions

        # Initialize the reachable configurations and add the initial state.
        rc_cls_name = cfg.env.reachable_configurations.name
        if reachable_configurations is None:
            assert create_reachable_configurations, 'Need to create reachable configurations'
            if min_crosses == 0:
                rc_cls_name = 'ReachableConfigurations'
            if cfg.env.reachable_configurations.replay_buffer_files_path is None:
                if rc_cls_name == 'SumTreeReachableConfigurations':
                    reachable_configurations = ReachableConfigurationsFactory.create(class_name=rc_cls_name, crossing_number=min_crosses)
                else:
                    reachable_configurations = ReachableConfigurationsFactory.create(class_name=rc_cls_name)
            else:
                num_cpus = num_workers if num_workers > 1 else 0
                reachable_configurations = ReachableConfigurationsFactory.get_cls(rc_cls_name).from_replay_buffer_files(
                    replay_buffer_files_path=cfg.env.reachable_configurations.replay_buffer_files_path, num_cpus=num_cpus,
                    min_crosses=min_crosses,
                    max_crosses=max_crosses,
                    sample_size=cfg.env.reachable_configurations.sample_size
                )
        assert reachable_configurations is not None, 'Reachable configurations was not initialized'
        if min_crosses == 0:
            physics = Physics.from_xml_path(cfg.file_system.env_path)
            playground_physics = Physics.from_xml_path(cfg.file_system.env_path)
            physics_reset(physics)
            physics_reset(playground_physics)
            configuration = get_current_primitive_state(physics)
            low_level_state = LowLevelState(configuration)
            set_physics_state(playground_physics, configuration)
            low_level_pos = convert_qpos_to_xyz_with_move_center(playground_physics, configuration)
            link_segments, intersections = get_link_segments(np.array(low_level_pos))
            high_level_state = HighLevelAbstractState.from_abstract_state(convert_pos_to_topology(low_level_pos))
            rc_state = ReachableConfigurationsState(high_level_state=high_level_state,
                                                    low_level_state=low_level_state,
                                                    low_level_pos=low_level_pos,
                                                    link_segments=link_segments,
                                                    intersections=intersections)
            reachable_configurations.add_node(state=rc_state)

        # Initialize the initial state selector via factory.
        initial_state_selector_name, initial_state_selector_kwargs = cfg.env.initial_state_selector.name, cfg.env.initial_state_selector.kwargs
        initial_state_selector_kwargs = {} if initial_state_selector_kwargs is None else initial_state_selector_kwargs
        initial_state_selector = InitialStateSelectorFactory.create(class_name=initial_state_selector_name, **initial_state_selector_kwargs)

        # Initialize the preprocessor via factory.
        cfg.env.preprocessor.kwargs.num_of_links = cfg.num_of_links
        cfg.env.preprocessor.kwargs.categorical_link = False
        preprocessor = PreprocessorFactory.create_from_cfg(cfg.env)

        # Initialize the goal selector via factory.
        goal_selector_name, goal_selector_kwargs = cfg.env.goal_selector.name, cfg.env.goal_selector.kwargs
        goal_selector_kwargs = {} if goal_selector_kwargs is None else goal_selector_kwargs
        goal_selector_kwargs.min_crosses = min_crosses
        goal_selector_kwargs.max_crosses = max_crosses
        goal_selector_kwargs.depth = depth
        goal_selector_kwargs.high_level_actions = high_level_actions
        goal_selector = GoalSelectorFactory.create(class_name=goal_selector_name, **goal_selector_kwargs)

        obj = cls(
            cfg=cfg.env,
            num_workers=num_workers,
            seed=cfg.seed,
            output_dir=cfg.file_system.output_dir,
            exceptions_dir=cfg.file_system.exceptions_dir,
            reachable_configurations=reachable_configurations,
            goal_selector=goal_selector,
            initial_state_selector=initial_state_selector,
            preprocessor=preprocessor,
            save_videos=cfg.save_videos,
            env_path=cfg.file_system.env_path,
            num_of_links=cfg.num_of_links,
            goal_reward=cfg.env.goal_reward,
            neg_reward=cfg.env.neg_reward,
            stay_reward=cfg.env.stay_reward,
            max_steps=cfg.env.max_steps,
            min_crosses=min_crosses,
            max_crosses=max_crosses,
            depth=depth,
            high_level_actions=high_level_actions,
            link_min=cfg.env.link_min,
            link_max=cfg.env.link_max,
            z_min=cfg.env.z_min,
            z_max=cfg.env.z_max,
            x_min=cfg.env.x_min,
            x_max=cfg.env.x_max,
            y_min=cfg.env.y_min,
            y_max=cfg.env.y_max,
            her=cfg.env.her,
            init_pool_context=init_pool_context,
        )
        return obj


def plot_collection_summary(episodes, episode_times, is_her_list, start_time: datetime, end_time: datetime, save_dir: str = None):
    time_per_step_list, success_list, rewards_list, lengths, moved_list, max_crosses_list,\
        max_steps_list, goal_selection_time, initial_state_selection_time, episode_postprocess_time = [], [], [], [], [], [], [], [], [], []
    num_exceptions = 0
    step_times = []
    R1_success_list, R2_success_list, cross_success_list = [], [], []

    for episode, time, is_her in zip(episodes, episode_times, is_her_list):
        states, actions, raw_actions, rewards, dones, truncateds, infos = episode
        first_info = infos[0]
        last_experience = infos[-1]['experience']
        num_exceptions += last_experience.exception_occurred
        if last_experience.exception_occurred:
            continue
        if is_her:
            continue
        success = last_experience.info.goal_reached
        goal_action = last_experience.goal_action
        if goal_action.data['move'] == 'R1':
            R1_success_list.append(success)
        elif goal_action.data['move'] == 'R2':
            R2_success_list.append(success)
        elif goal_action.data['move'] == 'cross':
            cross_success_list.append(success)
        moved = last_experience.info.moved_high_level_state
        max_cross = last_experience.info.max_crosses_passed
        max_steps = last_experience.info.max_steps_reached
        step_time = [info['experience'].info.step_time for info in infos]
        total_reward = sum(rewards)
        time_per_step = time / len(rewards)
        time_per_step_list.append(time_per_step)
        success_list.append(success)
        rewards_list.append(total_reward)
        lengths.append(len(rewards))
        moved_list.append(moved)
        max_crosses_list.append(max_cross)
        max_steps_list.append(max_steps)
        step_times.extend(step_time)
        goal_selection_time.append(first_info['goal_selection_time'])
        initial_state_selection_time.append(first_info['initial_state_selection_time'])
        episode_postprocess_time.append(first_info['episode_postprocess_time'])

    texts = []
    texts.append('==================================================')
    texts.append(f'{len(episodes)} Episodes Summary:')
    texts.append(f'Total time [s]: {np.round((end_time - start_time).total_seconds(), 2)}')
    texts.append(f'Number of Exceptions: {num_exceptions}')
    texts.append(f'Average Time Per Step [s]: {np.mean(time_per_step_list).round(2)}')
    texts.append(f'Average Time Per Simulation Step [s]: {np.mean(step_times).round(2)}')
    texts.append(f'Average Time Per Episode [s]: {np.mean(episode_times).round(2)}')
    texts.append(f'Average Success Rate: {np.mean(success_list).round(2) * 100}%')
    texts.append(f'Average R1 Success Rate: {np.mean(R1_success_list).round(2) * 100 if len(R1_success_list) > 0 else -1}%')
    texts.append(f'Average R2 Success Rate: {np.mean(R2_success_list).round(2) * 100 if len(R2_success_list) > 0 else -1}%')
    texts.append(f'Average Cross Success Rate: {np.mean(cross_success_list).round(2) * 100 if len(cross_success_list) > 0 else -1}%')
    texts.append(f'Average Total Reward: {np.mean(rewards_list).round(2)}')
    texts.append(f'Average Length: {np.mean(lengths).round(2)}')
    texts.append(f'Average Moved High Level State: {np.mean(moved_list).round(2) * 100}%')
    texts.append(f'Average Passed Max Crosses: {np.mean(max_crosses_list).round(2) * 100}%')
    texts.append(f'Average Reached Max Steps: {np.mean(max_steps_list).round(2) * 100}%')
    texts.append(f'Number of HER Episodes: {sum(is_her_list)}/{len(episodes) - sum(is_her_list)}')
    texts.append(f'HER Ratio: {np.round(sum(is_her_list)/(len(episodes) - sum(is_her_list)), 2) * 100}%')
    texts.append(f'Average Goal Selection Time [s]: {np.mean(goal_selection_time).round(2)}')
    texts.append(f'Average Initial State Selection Time [s]: {np.mean(initial_state_selection_time).round(2)}')
    texts.append(f'Average Episode Postprocess Time [s]: {np.mean(episode_postprocess_time).round(2)}')
    texts.append('=====================================================')

    if save_dir is not None:
        # save the summary to txt file
        with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
            for text in texts:
                f.write(f'{text}\n')

    for text in texts:
        print(text)


if __name__ == '__main__':
    cfg_path = 'exploration/rl/environment/exploration_env.yaml'
    algo_name = 'ExplorationGymEnvs'
    israel_tz = pytz.timezone('Israel')
    now_in_israel = datetime.now(israel_tz)

    output_dir = os.path.join('exploration/outputs/debug', algo_name, now_in_israel.strftime("%Y-%m-%d_%H-%M"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    exceptions_dir = os.path.join(output_dir, 'exceptions')
    if os.path.exists(exceptions_dir):
        shutil.rmtree(exceptions_dir)
    os.makedirs(exceptions_dir)

    cfg = load_env_config(cfg_path)
    cfg.file_system.output_dir = output_dir
    cfg.file_system.exceptions_dir = exceptions_dir

    seed_everything(cfg.seed)
    env = ExplorationGymEnvs.from_cfg(cfg)

    validation_success_rates = [0.97] * 11 + [0.75, 0.9, 0.99, 1.]
    best_success_rates = [0.9] + [0.97] * 11 + [0.75, 0.9, 0.99]
    env_steps_list = range(1000, 1000 * (len(validation_success_rates) + 1), 1000)
    iterations = len(validation_success_rates)
    validation_rounds = range(iterations)
    assert len(env_steps_list) == len(validation_success_rates) == len(best_success_rates) == iterations

    try:
        for i in range(iterations):
            start_time = datetime.now()
            episodes, episode_times, is_her_list = env.play_episodes(apply_her=env.her)
            end_time = datetime.now()
            plot_collection_summary(episodes, episode_times, is_her_list, start_time=start_time, end_time=end_time)

            for episode, episode_time, is_her in zip(episodes, episode_times, is_her_list):
                states, actions, raw_actions, rewards, done_flags, truncateds, infos = episode

    finally:
        print('closing env')
        env.close()

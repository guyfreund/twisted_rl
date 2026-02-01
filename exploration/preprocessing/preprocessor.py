from typing import List, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod

from exploration.mdp.high_level_action import HighLevelAction
from exploration.mdp.low_level_state import LowLevelState
from exploration.mdp.high_level_state import HighLevelAbstractState


class IPreprocessor(ABC):
    @abstractmethod
    def preprocess_qG(self, low_level_poses: List[np.ndarray], low_level_states: List[LowLevelState],
                      goals: List[Tuple[HighLevelAbstractState, HighLevelAction]],
                      link_segments: list, intersections: list) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
        raise NotImplementedError


class Preprocessor(IPreprocessor):
    def __init__(self, num_of_links: int, return_with_init_position: bool,
                 categorical_link: bool, use_qpos: bool, use_qvel: bool, use_pos: bool,
                 use_action_type_for_goal_representation: bool = False):
        self.use_qpos = use_qpos
        self.use_qvel = use_qvel
        self.use_pos = use_pos
        self.qpos_length = 47
        self.qvel_length = 46
        self.pos_length = (num_of_links + 1) * 3
        self.observation_length = self.qpos_length * use_qpos + self.qvel_length * use_qvel + self.pos_length * use_pos
        self.R1_only = False
        self.config_length = (num_of_links - 1) * 2 + 7
        self.num_of_links = num_of_links
        self.return_with_init_position = return_with_init_position
        self.one_hot_low_level_action = categorical_link
        self.use_action_type_for_goal_representation = use_action_type_for_goal_representation
        self.magic_number = 4 + self.use_action_type_for_goal_representation

    @staticmethod
    def fix_yaw_problem(sample, config_length):
        for index in range(len(sample[7:config_length])):
            while sample[index + 7] > np.pi:
                sample[index + 7] -= np.pi * 2
            while sample[index + 7] < -np.pi:
                sample[index + 7] += np.pi * 2
        return sample

    @property
    def goal_representation_low(self):
        return np.concatenate([np.zeros(2 * self.num_of_links), -1 * np.ones(self.magic_number)])

    @property
    def goal_representation_high(self):
        return np.ones(2 * self.num_of_links + self.magic_number)

    def preprocess_high_level_action(self, start_pos: np.ndarray, high_level_action: HighLevelAction,
                                     link_segments: list, intersections: list) -> np.ndarray:
        if high_level_action is None:
            return None

        if start_pos is None:
            return None

        if high_level_action.data['move'] == 'R1':
            idx_start, idx_end = link_segments[high_level_action.data['idx']]
            # create a numpy array of zeros with ones from idx_start to idx_e
            one_hot_range = np.zeros(self.num_of_links)
            one_hot_range[idx_start:idx_end + 1] = 1
            under_one_hot_range = one_hot_range.copy()
            over_one_hot_range = one_hot_range.copy()
            # action_type = [0, 0]
            action_type = [0] if self.use_action_type_for_goal_representation else []

        elif high_level_action.data['move'] == 'R2' or high_level_action.data['move'] == 'cross':
            under_idx_start, under_idx_end = link_segments[high_level_action.data['under_idx']]
            over_idx_start, over_idx_end = link_segments[high_level_action.data['over_idx']]
            under_one_hot_range = np.zeros(self.num_of_links)
            under_one_hot_range[under_idx_start:under_idx_end + 1] = 1
            over_one_hot_range = np.zeros(self.num_of_links)
            over_one_hot_range[over_idx_start:over_idx_end + 1] = 1
            # action_type = [0, 1] if high_level_action.data['move'] == 'R2' else [1, 0]
            action_type = [1 if high_level_action.data['move'] == 'R2' else 2] if self.use_action_type_for_goal_representation else []
        else:
            raise NotImplementedError

        missing_representation = -1
        left = high_level_action.data['left'] if 'left' in high_level_action.data else missing_representation
        over_first = high_level_action.data['over_first'] if 'over_first' in high_level_action.data else missing_representation
        over_before_under = high_level_action.data['over_before_under'] if 'over_before_under' in high_level_action.data else missing_representation
        sign = high_level_action.data['sign'] if 'sign' in high_level_action.data else missing_representation
        binary_features = [left, over_first, over_before_under, sign]
        data = np.concatenate([under_one_hot_range, over_one_hot_range, binary_features, action_type])
        return data

    def preprocess_qG(self, low_level_poses: List[np.ndarray], low_level_states: List[LowLevelState],
                      goals: List[Tuple[HighLevelAbstractState, HighLevelAction]],
                      link_segments: list, intersections: list) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        states, goal_embs = [], []
        for low_level_pos, low_level_state, (GS, GA), lk, isc in zip(low_level_poses, low_level_states, goals, link_segments, intersections):
            qs = []
            if low_level_state is not None:
                cfg = low_level_state.configuration.copy()

                qpos, qvel = cfg[:self.qpos_length], cfg[self.qpos_length:]
                qpos = self.fix_yaw_problem(qpos, self.qpos_length)
                qpos[:2] = 0.0

                if low_level_pos is not None:
                    low_level_pos.copy()
                    pos_ravel = low_level_pos.copy()
                    pos_ravel[:, :2] -= qpos[:2]
                    pos_ravel = pos_ravel.ravel()
                else:
                    pos_ravel = None

            else:
                qpos, qvel = None, None

                if low_level_pos is not None:
                    pos_ravel = low_level_pos.copy().ravel()
                else:
                    pos_ravel = None

            if self.use_qpos:
                qs.append(qpos)
            if self.use_qvel:
                qs.append(qvel)
            if self.use_pos:
                qs.append(pos_ravel)

            q = np.concatenate(qs) if not any(q is None for q in qs) else None
            states.append(q)

            A = self.preprocess_high_level_action(low_level_pos, GA, lk, isc)
            goal_embs.append(A)

        return states, goal_embs

    def get_indices(self, low_level_poses: List[np.ndarray], goals: List[Tuple[HighLevelAbstractState, HighLevelAction]],
                    link_segments: list, intersections: list) -> List[Tuple[int, int]]:
        indices = []
        for low_level_pos, (GS, GA), lk, isc in zip(low_level_poses, goals, link_segments, intersections):
            if self.R1_only:
                idx_start, idx_end = lk[GA.data['idx']]
            else:
                idx_start, idx_end = 0, self.num_of_links - 1
            indices.append((idx_start, idx_end))
        return indices

    def postprocess_actions(self, raw_actions: List[np.ndarray], low_level_poses: List[np.ndarray],
                            goals: List[Tuple[HighLevelAbstractState, HighLevelAction]],
                            link_segments: list, intersections: list) -> np.ndarray:
        indices = self.get_indices(low_level_poses=low_level_poses, goals=goals, link_segments=link_segments, intersections=intersections)
        actions = []
        for raw_action, (start_idx, end_idx) in zip(raw_actions, indices):
            possible_link = np.arange(start_idx, end_idx + 1e-5, step=1, dtype=np.int32)
            possible_frac = (possible_link - start_idx + 1) / (end_idx - start_idx + 1)
            pred_frac_idx = np.argmin(np.abs(possible_frac - raw_action[0]))
            link = possible_link[pred_frac_idx]
            action = np.concatenate([[link], raw_action[1:]])
            actions.append(action)
        actions = np.vstack(actions)
        return actions

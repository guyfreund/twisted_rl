from typing import Optional
from dataclasses import dataclass, field
import numpy as np

from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.mdp.low_level_state import LowLevelState
from exploration.mdp.low_level_action import LowLevelAction
from exploration.mdp.high_level_action import HighLevelAction
from mujoco_infra.mujoco_utils.video_utils import Video


@dataclass
class ExceptionInfo:
    traceback: list
    exception: str
    exception_type: str
    exception_metadata_path: str


@dataclass
class Info:
    goal_reached: bool
    max_crosses_passed: bool
    moved_high_level_state: bool
    max_steps_reached: bool
    env_step: int
    step_time: float
    is_her: bool = False
    exception_info: Optional[ExceptionInfo] = None

    @property
    def exception_occurred(self) -> bool:
        return self.exception_info is not None


@dataclass
class Experience:
    start_low_level_state: LowLevelState
    start_low_level_state_centered: LowLevelState
    start_low_level_pos: np.ndarray
    start_high_level_state: HighLevelAbstractState
    start_link_segments: list
    start_intersections: list
    low_level_action: LowLevelAction
    raw_low_level_action: np.ndarray
    end_low_level_state: LowLevelState
    end_low_level_state_centered: LowLevelState
    end_low_level_pos: np.ndarray
    end_high_level_state: HighLevelAbstractState
    end_link_segments: list
    end_intersections: list
    reward: int
    done: bool
    info: Info
    goal_state: HighLevelAbstractState
    goal_action: HighLevelAction
    high_level_action: Optional[HighLevelAction]
    stddev_link: float = None
    stddev_z: float = None
    stddev_x: float = None
    stddev_y: float = None
    start_image: np.ndarray = None
    end_image: np.ndarray = None

    @property
    def moved_to_lower_goal_crossing_number(self) -> Optional[bool]:
        if self.exception_occurred or self.is_empty:
            return None
        if self.stayed_in_the_same_crossing_number:
            return None
        return self.end_high_level_state.crossing_number < self.goal_state.crossing_number

    @property
    def moved_to_higher_goal_crossing_number(self) -> Optional[bool]:
        if self.exception_occurred or self.is_empty:
            return None
        if self.stayed_in_the_same_crossing_number:
            return None
        return self.end_high_level_state.crossing_number > self.goal_state.crossing_number

    @property
    def stayed_in_the_same_crossing_number(self) -> Optional[bool]:
        if self.exception_occurred or self.is_empty:
            return None
        return self.start_high_level_state.crossing_number == self.end_high_level_state.crossing_number

    @property
    def diff_to_goal_crossing_number(self) -> Optional[int]:
        if self.exception_occurred or self.is_empty:
            return None
        if self.stayed_in_the_same_crossing_number:
            return None
        return abs(self.end_high_level_state.crossing_number - self.goal_state.crossing_number)

    @property
    def exception_occurred(self) -> bool:
        return self.info is not None and self.info.exception_occurred

    @property
    def is_empty(self) -> bool:
        return all([
            self.start_low_level_state is None,
            self.start_low_level_pos is None,
            self.start_high_level_state is None,
            self.low_level_action is None,
            self.end_low_level_state is None,
            self.end_low_level_pos is None,
            self.end_high_level_state is None,
            self.reward is None,
            self.done is None,
            self.info is None,
            self.goal_state is None,
            self.goal_action is None,
            self.high_level_action is None,
            self.start_image is None,
            self.end_image is None,
            self.stddev_link is None,
            self.stddev_z is None,
            self.stddev_x is None,
            self.stddev_y is None
        ])


@dataclass
class EpisodeExperiences:
    max_steps: int
    experiences: list = field(default_factory=list)
    time: float = None
    video: Video = None

    @property
    def exception_occurred(self) -> bool:
        return self.get_last_experience().exception_occurred

    @property
    def num_experiences(self):
        assert len(self.experiences) <= self.max_steps
        return len(self.experiences)

    def add_experience(self, experience: Experience):
        self.experiences.append(experience)

    def get_last_experience(self) -> Experience:
        return self.experiences[-1] if self.num_experiences > 0 else None

    def get_first_experience(self) -> Experience:
        return self.experiences[0] if self.num_experiences > 0 else None

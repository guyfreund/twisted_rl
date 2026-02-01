from typing import Dict
import numpy as np

from exploration.mdp.high_level_action import HighLevelAction
from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.mdp.low_level_state import LowLevelState
from exploration.rl.experience import Experience
from exploration.preprocessing.preprocessor import IPreprocessor


def get_state(preprocessor: IPreprocessor, low_level_state: LowLevelState,
              low_level_pos: np.ndarray, goal: (HighLevelAbstractState, HighLevelAction),
              link_segments: list, intersections: list):
    states, goals = preprocessor.preprocess_qG(
        low_level_poses=[low_level_pos],
        low_level_states=[low_level_state],
        goals=[goal],
        link_segments=[link_segments],
        intersections=[intersections]
    )
    return states, goals


def get_initial_state(preprocessor: IPreprocessor, start_low_level_state: LowLevelState,
                      start_low_level_pos: np.ndarray, start_high_level_state: HighLevelAbstractState,
                      goal: (HighLevelAbstractState, HighLevelAction),
                      link_segments: list, intersections: list):
    states, desired_goals = get_state(
        preprocessor=preprocessor,
        low_level_state=start_low_level_state,
        low_level_pos=start_low_level_pos,
        goal=goal,
        link_segments=link_segments,
        intersections=intersections
    )
    assert desired_goals[0] is not None
    state = {
        'observation': states[0],
        'desired_goal': desired_goals[0]
    }
    return state


def get_next_state_from_experience(experience: Experience, preprocessor: IPreprocessor) -> Dict:
    _, desired_goals = get_state(  # desired goal is always from the start state
        preprocessor=preprocessor,
        low_level_state=experience.start_low_level_state_centered,
        low_level_pos=experience.start_low_level_pos,
        goal=(experience.goal_state, experience.goal_action),
        link_segments=experience.start_link_segments,
        intersections=experience.start_intersections
    )
    next_states, _ = get_state(
        preprocessor=preprocessor,
        low_level_state=experience.end_low_level_state_centered,
        low_level_pos=experience.end_low_level_pos,
        goal=(experience.goal_state, None),
        link_segments=experience.end_link_segments,
        intersections=experience.end_intersections
    )
    assert desired_goals[0] is not None
    next_state = {
        'observation': next_states[0],
        'desired_goal': desired_goals[0]
    }
    return next_state

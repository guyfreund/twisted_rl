import numpy as np
import traceback
import os
from dm_control.mujoco import Physics
from uuid import uuid4
import pickle
from datetime import datetime
from typing import Optional, Callable
from gymnasium.spaces import Space
from pytorch_lightning import seed_everything

from exploration.preprocessing.preprocessor import Preprocessor
from exploration.mdp.graph.high_level_graph import HighLevelGraph
from exploration.mdp.high_level_state import HighLevelAbstractState
from exploration.mdp.high_level_action import HighLevelAction
from exploration.mdp.low_level_state import LowLevelState
from exploration.mdp.low_level_action import LowLevelAction
from exploration.rl.environment.env_utils import get_next_state_from_experience, get_initial_state
from exploration.rl.experience import Info, Experience, ExceptionInfo, EpisodeExperiences
from mujoco_infra.mujoco_utils.mujoco import convert_qpos_to_xyz_with_move_center, \
    convert_pos_to_topology, set_physics_state, get_current_primitive_state, \
    execute_action_in_curve_with_mujoco_controller_one_hand, physics_reset, get_env_image, get_link_segments
from mujoco_infra.mujoco_utils.mujoco import move_center
from mujoco_infra.mujoco_utils.video_utils import Video


class ExplorationWorkerEnv:
    def __init__(self,
                 seed: int,
                 env_path: str,
                 goal_reward: int,
                 neg_reward: int,
                 stay_reward: int,
                 max_steps: int,
                 max_crosses: int,
                 observation_space: Space,
                 action_space: Space,
                 preprocessor: Preprocessor,
                 high_level_graph: HighLevelGraph,
                 save_episode_video: bool,
                 output_dir: str,
                 exceptions_dir: str,
                 env_dir: str,
                 videos_dir: str,
                 R1_success_videos_dir: str,
                 R1_failure_videos_dir: str,
                 R2_success_videos_dir: str,
                 R2_failure_videos_dir: str,
                 cross_success_videos_dir: str,
                 cross_failure_videos_dir: str,
    ):
        self.seed = seed
        seed_everything(self.seed)
        self.env_path = env_path
        self.save_episode_video = save_episode_video
        if self.save_episode_video:
            # os.environ['MUJOCO_GL'] = 'egl'
            pass
        self.physics = Physics.from_xml_path(self.env_path)
        physics_reset(self.physics)
        self.playground_physics = Physics.from_xml_path(self.env_path)
        physics_reset(self.playground_physics)
        self.preprocessor = preprocessor
        self.output_dir = output_dir
        self.exceptions_dir = exceptions_dir
        self.env_dir = env_dir
        self.videos_dir = videos_dir
        self.R1_success_videos_dir = R1_success_videos_dir
        self.R1_failure_videos_dir = R1_failure_videos_dir
        self.R2_success_videos_dir = R2_success_videos_dir
        self.R2_failure_videos_dir = R2_failure_videos_dir
        self.cross_success_videos_dir = cross_success_videos_dir
        self.cross_failure_videos_dir = cross_failure_videos_dir
        self.goal_reward = goal_reward
        self.neg_reward = neg_reward
        self.stay_reward = stay_reward
        self.max_steps = max_steps
        self.max_crosses = max_crosses
        self.observation_space = observation_space
        self.action_space = action_space
        self.high_level_graph = high_level_graph
        self.to_raise = False

        # changing on demand params
        self.num_steps = 0
        self.goal_state = None
        self.goal_action = None
        self.start_low_level_state = None
        self.start_high_level_state = None
        self.start_low_level_pos = None
        self.start_link_segments = None
        self.start_intersections = None

    def set_goal(self, goal: (HighLevelAbstractState, HighLevelAction)):
        self.goal_state, self.goal_action = goal[0], goal[1]

    def set_initial_state(self, low_level_state: LowLevelState):
        set_physics_state(self.physics, low_level_state.configuration)
        move_center(self.physics)  # note this is changing the configuration (two first are zeros)
        configuration = get_current_primitive_state(self.physics)
        self.start_low_level_state = LowLevelState(configuration)
        set_physics_state(self.playground_physics, configuration)
        self.start_low_level_pos = convert_qpos_to_xyz_with_move_center(self.playground_physics, configuration)
        high_level_state = convert_pos_to_topology(self.start_low_level_pos)
        self.start_link_segments, self.start_intersections = get_link_segments(np.array(self.start_low_level_pos))
        self.start_high_level_state = HighLevelAbstractState.from_abstract_state(high_level_state)

    def reset(self, goal_high_level_state: HighLevelAbstractState, goal_high_level_action: HighLevelAction,
              start_low_level_state: LowLevelState = None):
        self.num_steps = 0
        physics_reset(self.physics)
        physics_reset(self.playground_physics)
        if start_low_level_state is None:
            start_low_level_state = LowLevelState(get_current_primitive_state(self.physics))
        self.set_initial_state(start_low_level_state)
        self.set_goal((goal_high_level_state, goal_high_level_action))

    def get_exception_experience(self, action: LowLevelAction, raw_action: np.ndarray, e: Exception, path: str) -> Experience:
        info = Info(
            goal_reached=False,
            max_crosses_passed=False,
            moved_high_level_state=False,
            max_steps_reached=False,
            env_step=self.num_steps + 1,
            exception_info=ExceptionInfo(
                traceback=traceback.format_exception(type(e), e, e.__traceback__),
                exception=str(e),
                exception_type=type(e).__name__,
                exception_metadata_path=path
            ),
            step_time=-1
        )
        experience = Experience(
            start_low_level_state=self.start_low_level_state,
            start_low_level_state_centered=self.start_low_level_state,
            start_high_level_state=self.start_high_level_state,
            start_low_level_pos=self.start_low_level_pos,
            start_link_segments=self.start_link_segments,
            start_intersections=self.start_intersections,
            low_level_action=action,
            raw_low_level_action=raw_action,
            high_level_action=None,
            end_low_level_state=None,
            end_low_level_state_centered=None,
            end_low_level_pos=None,
            end_high_level_state=None,
            end_link_segments=None,
            end_intersections=None,
            reward=None,
            done=True,
            info=info,
            goal_state=self.goal_state,
            goal_action=self.goal_action,
            start_image=None,
            end_image=None,
        )
        return experience

    def get_empty_experience(self) -> Experience:
        experience = Experience(
            start_low_level_state=None,
            start_low_level_state_centered=None,
            start_high_level_state=None,
            start_low_level_pos=None,
            start_link_segments=None,
            start_intersections=None,
            low_level_action=None,
            raw_low_level_action=None,
            high_level_action=None,
            end_low_level_state=None,
            end_low_level_state_centered=None,
            end_low_level_pos=None,
            end_high_level_state=None,
            end_link_segments=None,
            end_intersections=None,
            reward=None,
            done=None,
            info=None,
            goal_state=None,
            goal_action=None,
            start_image=None,
            end_image=None,
        )
        return experience

    def step(self, raw_action: Optional[np.ndarray]) -> Experience:
        if raw_action is None:
            experience = self.get_empty_experience()
            return experience

        start_image = get_env_image(self.physics) if self.save_episode_video else None

        postprocessed_action = self.preprocessor.postprocess_actions(
            raw_actions=[raw_action],
            low_level_poses=[self.start_low_level_pos],
            goals=[(self.goal_state, self.goal_action)],
            link_segments=[self.start_link_segments],
            intersections=[self.start_intersections]
        )[0]
        action = LowLevelAction(
            link=int(postprocessed_action[0]),
            z=postprocessed_action[1],
            y=postprocessed_action[2],
            x=postprocessed_action[3]
        )

        try:
            # os.environ['MUJOCO_GL'] = 'egl'
            start = datetime.now()
            self.physics, _ = execute_action_in_curve_with_mujoco_controller_one_hand(
                physics=self.physics, action=action.np_encoding, get_video=False, save_video=False, show_image=False,
                return_render=False, sample_rate=None, video=None, env_path=self.env_path, return_video=False,
                # to_raise=True, verbose=False, save_high_freq_video=False, spline_sample_rate=None
            )
            end = datetime.now()
        except Exception as e:
            obj = {
                'goal_state': self.goal_state,
                'goal_action': self.goal_action,
                'start_low_level_state': self.start_low_level_state,
                'start_high_level_state': self.start_high_level_state,
                'start_low_level_pos': self.start_low_level_pos,
                'low_level_action': action,
                'raw_low_level_action': raw_action,
                'traceback': traceback.format_exception(type(e), e, e.__traceback__),
                'exception': str(e),
                'exception_type': type(e).__name__,
                'seed': self.seed,
            }
            path = os.path.join(self.exceptions_dir, f'{uuid4().hex}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            traceback.print_exception(type(e), e, e.__traceback__)
            if self.to_raise:
                raise e
            else:
                # reset physics to avoid exceptions
                self.physics = Physics.from_xml_path(self.env_path)
                physics_reset(self.physics)
                set_physics_state(self.playground_physics, self.start_low_level_state.configuration)
                experience = self.get_exception_experience(action=action, raw_action=raw_action, e=e, path=path)
                return experience

        self.num_steps += 1
        end_cfg = get_current_primitive_state(self.physics)
        end_low_level_state = LowLevelState(end_cfg)
        set_physics_state(self.playground_physics, end_cfg)
        end_pos = convert_qpos_to_xyz_with_move_center(self.playground_physics, end_cfg)
        end_low_level_state_centered = LowLevelState(get_current_primitive_state(self.playground_physics))
        end_high_level_state = convert_pos_to_topology(end_pos)
        end_link_segments, end_intersections = get_link_segments(np.array(end_pos))
        end_high_level_state = HighLevelAbstractState.from_abstract_state(end_high_level_state)
        end_image = get_env_image(self.playground_physics) if self.save_episode_video else None

        goal_reached = end_high_level_state == self.goal_state
        max_steps_reached = self.num_steps == self.max_steps
        moved_high_level_state = self.start_high_level_state != end_high_level_state
        max_crosses_passed = end_high_level_state.crossing_number > self.max_crosses
        done = goal_reached or max_steps_reached or moved_high_level_state or max_crosses_passed

        info = Info(
            goal_reached=goal_reached,
            max_crosses_passed=max_crosses_passed,
            moved_high_level_state=moved_high_level_state,
            max_steps_reached=max_steps_reached,
            env_step=self.num_steps,
            step_time=(end - start).total_seconds()
        )

        high_level_actions = self.high_level_graph.get_all_edge_variations(src=self.start_high_level_state, dst=end_high_level_state, from_graph=False)
        if len(high_level_actions) == 0:
            high_level_action = None
        else:
            high_level_action = high_level_actions[-1]

        if goal_reached:
            reward = self.goal_reward
        else:
            start_crossing_number, end_crossing_number = self.start_high_level_state.crossing_number, end_high_level_state.crossing_number
            abs_diff = np.abs(start_crossing_number - end_crossing_number)
            moved_to_forbidden_state = any([
                high_level_action is None and abs_diff > 0,
                max_crosses_passed,
                moved_high_level_state
            ])
            if moved_to_forbidden_state:
                reward = self.neg_reward
            else:
                reward = self.stay_reward

        experience = Experience(
            start_low_level_state=self.start_low_level_state,
            start_low_level_state_centered=self.start_low_level_state,
            start_high_level_state=self.start_high_level_state,
            start_low_level_pos=self.start_low_level_pos,
            start_link_segments=self.start_link_segments,
            start_intersections=self.start_intersections,
            low_level_action=action,
            raw_low_level_action=raw_action,
            high_level_action=high_level_action,
            end_low_level_state=end_low_level_state,
            end_low_level_state_centered=end_low_level_state_centered,
            end_low_level_pos=end_pos,
            end_high_level_state=end_high_level_state,
            end_link_segments=end_link_segments,
            end_intersections=end_intersections,
            reward=reward,
            done=done,
            info=info,
            goal_state=self.goal_state,
            goal_action=self.goal_action,
            start_image=start_image,
            end_image=end_image
        )

        self.set_initial_state(end_low_level_state)

        return experience

    def save_video(self, video: Video, episode_experiences: EpisodeExperiences):
        first_experience = episode_experiences.get_first_experience()
        last_experience = episode_experiences.get_last_experience()
        goal_action = last_experience.goal_action
        goal_action_str = goal_action.data['move']
        success = last_experience.info.goal_reached
        success_str = 'success' if success else 'failure'
        path = self.__getattribute__(f'{goal_action_str}_{success_str}_videos_dir')
        start_crossing_number = first_experience.start_high_level_state.crossing_number
        goal_state = last_experience.goal_state
        goal_crossing_number = goal_state.crossing_number
        output_path = os.path.join(path, f'{start_crossing_number}to{goal_crossing_number}')
        os.makedirs(output_path, exist_ok=True)
        filename = f'{uuid4().hex}'
        video.save(output_path=output_path, filename=filename)

    def play_episode(self, get_action: Callable) -> EpisodeExperiences:
        assert self.goal_state is not None and self.goal_action is not None

        if self.save_episode_video:
            video = Video()
            try:
                start_image = get_env_image(self.physics)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                raise e
            video.add_frames([start_image])
        else:
            video = None

        done = False
        start = datetime.now()
        state = get_initial_state(
            preprocessor=self.preprocessor,
            start_low_level_pos=self.start_low_level_pos,
            start_low_level_state=self.start_low_level_state,
            start_high_level_state=self.start_high_level_state,
            goal=(self.goal_state, self.goal_action),
            link_segments=self.start_link_segments,
            intersections=self.start_intersections
        )
        episode_experiences = EpisodeExperiences(max_steps=self.max_steps)

        while not done:
            raw_action, stddev = get_action(state)
            experience = self.step(raw_action=raw_action)
            if stddev is not None:
                experience.stddev_link = stddev[LowLevelAction.arg_to_idx('link')]
                experience.stddev_z = stddev[LowLevelAction.arg_to_idx('z')]
                experience.stddev_x = stddev[LowLevelAction.arg_to_idx('x')]
                experience.stddev_y = stddev[LowLevelAction.arg_to_idx('y')]
            episode_experiences.add_experience(experience)
            if not experience.exception_occurred:
                try:
                    state = get_next_state_from_experience(
                        experience=experience,
                        preprocessor=self.preprocessor,
                    )
                except Exception as e:
                    # probably high level action is not compatible with the current state
                    experience.reward = self.neg_reward
                    experience.done = True
                    traceback.print_exception(type(e), e, e.__traceback__)
            if self.save_episode_video:
                video.add_frames([experience.end_image])
            done = experience.done

        if episode_experiences.exception_occurred:
            self.physics = Physics.from_xml_path(self.env_path)
        physics_reset(self.physics)

        if self.save_episode_video:
            video.min_frames = episode_experiences.num_experiences * 50
            self.save_video(video=video, episode_experiences=episode_experiences)
        episode_experiences.video = video

        end = datetime.now()
        episode_experiences.time = (end - start).total_seconds()

        return episode_experiences

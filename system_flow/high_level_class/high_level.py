import sys
import traceback
from typing import Optional
from collections import defaultdict
from pytorch_lightning import seed_everything
from datetime import datetime, timedelta
import random
import time
import os
import pickle
import numpy as np
import cv2
import torch
from dm_control import mujoco

from exploration.mdp.high_level_state import HighLevelAbstractState

sys.path.append(".")

from exploration.mdp.low_level_state import LowLevelState
from exploration.reachable_configurations.reachable_configurations import ReachableConfigurationsState, get_priority
from exploration.rl.environment.exploration_gym_envs import ExplorationGymEnvs
from mujoco_infra.mujoco_utils.topology import BFS, representation, reverse_BFS
from system_flow.low_level_class.s2a_planner import S2APlanner
from system_flow.low_level_class.rl_planner import RLPlanner, MultiRLPlanner
from system_flow.high_level_class.graph_manager import GraphManager
from mujoco_infra.mujoco_utils.general_utils import load_pickle
from mujoco_infra.mujoco_utils.topology.BFS import get_state_score
from mujoco_infra.mujoco_utils.mujoco import calculate_number_of_crosses_from_topology_state, convert_topology_to_str, \
    convert_pos_to_topology, convert_qpos_to_xyz_with_move_center, set_physics_state, get_current_primitive_state, \
    physics_reset, execute_action_in_curve_with_mujoco_controller, get_link_segments

with torch.no_grad():
    class HighLevelPlanner():
        def __init__(self, args, cfg):
            #save inputs
            self.args = args
            self.cfg = cfg
            self.to_save = True

            #save config
            self.set_all_cfg_varibels()

            #set initial and goal states
            self.topology_start = self._generate_state_from_high_level_actions(representation.AbstractState(),\
                self.high_level_cfg.initial_state)
            self.topology_goal = self._generate_state_from_high_level_actions(representation.AbstractState(),\
                self.high_level_cfg.goal_state)

            #set mujoco
            self.args.env_path = self.high_level_cfg.env_path
            self.physics = mujoco.Physics.from_xml_path(self.args.env_path)
            self._physics_reset(self.physics)
            self.playground_physics = mujoco.Physics.from_xml_path(self.args.env_path)
            self._physics_reset(self.playground_physics)

            #init all levels
            if self.low_level_cfg.NAME == "S2APlanner":
                self.init_low_level(self.low_level_cfg)
            self.init_graph(self.graph_cfg)

            #create log file
            self.creaate_log_file()

            #load h values
            self.load_H_values_from_data(self.cfg.HIGH_LEVEL.h_path)

            #update max depth based on the goal topology
            self.high_level_cfg.max_depth = calculate_number_of_crosses_from_topology_state(self.topology_goal)

            self.reachable = {
                convert_topology_to_str(self.topology_start): self.topology_start
            }

            #init all running times
            self._init_run_time()

            #set initial_topology_states bandit
            self.bandit_initial_topology_states = {}
            self.bandit_select_trejctory_from_topology_state = {}

        def set_all_cfg_varibels(self):
            self.high_level_cfg = self.cfg["HIGH_LEVEL"]
            #self.mid_level_cfg = self.cfg["MID_LEVEL"]
            self.low_level_cfg = self.cfg["LOW_LEVEL"]
            self.graph_cfg = self.cfg["GRAPH"]

            self.config_length = self.low_level_cfg['STATE2STATE_PARMS']["config_length"]
            self.show_image = self.cfg.GENERAL_PARAMS.SAVE.show_image
            self.get_video = self.cfg.GENERAL_PARAMS.SAVE.get_video
            self.frame_rate = self.cfg.GENERAL_PARAMS.SAVE.frame_rate
            self.video = []
            self.num_of_links = self.low_level_cfg.STATE2STATE_PARMS.num_of_links
            self.random_search_steps = self.low_level_cfg.STATE2STATE_PARMS["random_search_steps"]

            #Random action
            self.low_index = self.low_level_cfg["RANDOM_ACTION"]["low_index"]
            self.high_index = self.low_level_cfg["RANDOM_ACTION"]["high_index"]
            self.low_height = self.low_level_cfg["RANDOM_ACTION"]["low_height"]
            self.high_height = self.low_level_cfg["RANDOM_ACTION"]["high_height"]
            self.high_end_location = self.low_level_cfg["RANDOM_ACTION"]["high_end_location"]
            self.low_end_location = self.low_level_cfg["RANDOM_ACTION"]["low_end_location"]

            self.output_path = self.cfg.GENERAL_PARAMS.output_path
            
        def _generate_state_from_high_level_actions(self, state, actions):
            for item in actions:
                #[MS] need to fix the number of parameters I am sending
                #RT1
                if item[0] == "Reide1":
                    state.Reide1(item[1], item[2], item[3])
                    continue

                #RT2
                elif item[0] == "Reide2":
                    if len(item[0]) == 4:
                        state.Reide2(item[1], item[2], item[3], item[4])
                    else:
                        state.Reide2(item[1], item[2], item[3])
                    continue

                #Cross
                elif item[0] == "cross":
                    state.cross(item[1], item[2], item[3])
                    continue
            return state        

        def creaate_log_file(self):
            self.log_file = {}
            self.log_file["paths"] = []
            self.log_file["error_primitive"] = []
            self.log_file["actions"] = []
            self.log_file["uncertainty"] = []
            self.log_file["error_topology"] = []
            self.log_file["topolgy_states"] = []
            self.log_file["success"] = []
            self.log_file["configuration"] = []

        def save_log(self, path):
            a_file = open(path+"/log.pkl", "wb")
            pickle.dump(self.log_file, a_file)
            a_file.close()

        def write_log_instance(self, key, value):
            if key not in self.log_file.keys():
                self.log_file[key] = []
            self.log_file[key].append(value)

        def set_topology_start(self, start):
            self.topology_start = start
            # self.reachable[convert_topology_to_str(start)] = start

        def set_new_goal(self, goal):
            self.topology_goal = goal
            self.high_level_cfg.max_depth = calculate_number_of_crosses_from_topology_state(self.topology_goal)

        def save_log_actions(self, log_actions):
            self.write_log_instance("log_actions", log_actions)

        def _update_score(self, paths, h_score):
            """
            Get paths from "get_all_high_level_plan" and add score for eacxh trejctory

            Args:
                paths (list): paths
                h_score (list): topology score for each state

            Returns:
                update paths with score
            """
            for index in range(len(paths)):
                score = 0
                for state in paths[index][0]:
                    score += get_state_score(h_score, state)
                paths[index] = (paths[index][0], paths[index][1], score)
            return paths

        def get_all_high_level_plan(self,start, goal):
            if self.cfg.HIGH_LEVEL.h_path == "":
                with_h = False
            else:
                with_h = False
            max_depth = self.high_level_cfg.max_depth-calculate_number_of_crosses_from_topology_state(start)
            raw_paths = BFS.bfs_all_path_new(start, goal, max_depth=max_depth, with_h=with_h, h_path=self.cfg.HIGH_LEVEL.h_path)

            if self.high_level_cfg['only_R1']:
                raw_paths = [path for path in raw_paths if all([action['move'] == 'R1' for action in path[1]])]

            paths = []
            for raw_path in raw_paths:
                states_path = []
                actions_path = raw_path[1]
                for state in raw_path[0]:
                    states_path.append(HighLevelAbstractState.from_abstract_state(state))
                path = (states_path, actions_path)
                paths.append(path)
            #update score
            paths = self._update_score(paths, self.h_scores)
            print(f"Got {len(paths)} high level plans from start state with crossing number {calculate_number_of_crosses_from_topology_state(start)} to goal state with crossing number {calculate_number_of_crosses_from_topology_state(goal)}")

            return paths

        def get_all_reverse_high_level_plan(self, start, goal):
            max_depth = self.high_level_cfg.max_depth-calculate_number_of_crosses_from_topology_state(start)
            paths = reverse_BFS.bfs_all_path(goal, start, max_depth=max_depth)
            reverse_paths = []
            reverse_paths_action = []
            for path, path_action in paths:
                reverse_path = path[::-1]
                reverse_path_action = [representation.reverse_action(path_action[i], path[i], path[i+1])
                                    for i in range(len(path_action))]
                reverse_path_action = reverse_path_action[::-1]
                reverse_paths.append(reverse_path)
                reverse_paths_action.append(reverse_path_action)
            paths = reverse_paths
            paths_action = reverse_paths_action
            return paths, paths_action

        def load_H_values_from_data(self, path):
            self.h_scores = load_pickle(path)

        def init_low_level(self, low_cfg, env=None):
            if low_cfg.NAME == "S2APlanner":
                self.low_planner = S2APlanner(low_cfg, self.config_length)
            elif low_cfg.NAME in ["RLPlanner", "MultiRLPlanner"]:
                rl_planner_cls = RLPlanner if low_cfg.NAME == "RLPlanner" else MultiRLPlanner
                self.low_planner = rl_planner_cls(cfg=low_cfg, config_length=self.config_length, env=env,
                                                  init_env=env is None, init_pool_context=env is None)
            else:
                raise
            print("low level planner =", low_cfg.NAME)

        def init_graph(self, graph_cfg):
            self.graph_manager = GraphManager(config_length=self.config_length)
            self.graph_manager.add_node(self.topology_start, get_current_primitive_state(self.physics), None, None) 

        def _physics_reset(self, physics):
            physics_reset(physics)

        def _get_all_initial_topology_states(self, paths):
            """
            Concat all initial topology states and return set of them
        
            Args:
                paths (list): all high level plans
            """
            all_topology_states = []
            crossing_numbers = defaultdict(list)
            for path in paths:
                temp_state = HighLevelAbstractState.from_abstract_state(path[0][0])
                if not temp_state in all_topology_states:
                    all_topology_states.append(temp_state)
                crossing_number = temp_state.crossing_number
                crossing_numbers[crossing_number].append(temp_state)
            for crossing_number, states in crossing_numbers.items():
                print(f'Got {len(states)} topology states with crossing number {crossing_number}')
            return all_topology_states
            
        def _bandit_select_topology_state(self, initial_topology_states):
            """
            select topology states from a list using bandits

            Args:
                initial_topology_states (lsit): list with all topology states
            """
            #add new states to bandit_initial_topology_states
            for state in initial_topology_states:
                state_str = convert_topology_to_str(state)
                if state_str not in self.bandit_initial_topology_states.keys():
                    self.bandit_initial_topology_states[state_str] = {
                        "cnt": 0
                    }

            #print the current dis
            dis = []
            for key in self.bandit_initial_topology_states.keys():
                dis.append(self.bandit_initial_topology_states[key]["cnt"])
            #print("bandit_select_topology_state =", dis)

            new_initial_topology_states = initial_topology_states

            #select unused states
            if self.high_level_cfg["SELECT_UNUSED_STATE"]:
                new_initial_topology_states = self._select_unused_state(initial_topology_states, dis)

            #give priority to states with higher number of crosses
            prob = np.ones(len(new_initial_topology_states)) / len(new_initial_topology_states)
            if self.high_level_cfg["SELECT_HIGHER_CROSS_STATES"]:
                prob = self._select_higher_cross_states(new_initial_topology_states)

            #select state
            topology_state_np = np.random.choice(new_initial_topology_states, 1, p=prob)
            topology_state = topology_state_np[0]
            #topology_state = random.choice(new_initial_topology_states)

            #update the number of selection +=1
            self.bandit_initial_topology_states[convert_topology_to_str(topology_state)]["cnt"] +=1
            print(f'Selected topology state with crossing number {calculate_number_of_crosses_from_topology_state(topology_state)}')

            return topology_state

        def _select_unused_state(self, states, dis:list):
            new_states = []
            indexes = []
            for index, item in enumerate(dis):
                if item == 0:
                    indexes.append(index)
            for index in indexes:
                new_states.append(states[index])

            if len(new_states) == 0:
                return states
            else:
                return new_states

        def _select_higher_cross_states(self, states):
            prob = []
            for state in states:
                prob.append(calculate_number_of_crosses_from_topology_state(state)+1)
            prob = np.array(prob)
            return prob/sum(prob)

        def _bandit_select_trejctory_from_topology_state(self, topology_state, high_level_plans):
            """
            select high level plan based on the initial topology state

            Args:
                topology_state (topology_state): initial topology state
                high_level_plans (list): all high level plans
            """
            #extrect all high-level plans with initil topology state
            high_level_plans_with_topolgy_state = []
            for plan in high_level_plans:
                plan_str = ""
                for temp_state in plan[0]:
                    plan_str += convert_topology_to_str(temp_state)
                if plan_str not in self.bandit_select_trejctory_from_topology_state.keys():
                    self.bandit_select_trejctory_from_topology_state[plan_str] = {
                        "cnt": 0
                    }

                if topology_state == plan[0][0]:
                    high_level_plans_with_topolgy_state.append(plan)

            #print dis
            dis = []
            for key in self.bandit_select_trejctory_from_topology_state.keys():
                dis.append(self.bandit_select_trejctory_from_topology_state[key]["cnt"])
            #print("bandit_select_trejctory_from_topology_state =", dis)


            #select plan
            number_of_plans = len(high_level_plans_with_topolgy_state)
            prob = np.ones(number_of_plans) / number_of_plans
            options = np.arange(0, number_of_plans, 1, dtype=int)
            path_index = np.random.choice(options, 1, p=prob)
            path = high_level_plans_with_topolgy_state[path_index[0]]
            #path = random.choice(high_level_plans_with_topolgy_state)

            #update the number of visits
            plan_str = ""
            for temp_state in path[0]:
                plan_str += convert_topology_to_str(temp_state)
            self.bandit_select_trejctory_from_topology_state[plan_str]["cnt"] +=1

            return path

        def _bendit_select_config_from_topology_state(self, topology_state):
            configurations, number_of_visits = self.graph_manager.get_all_states_with_topology_state(topology_state)

            new_configurations = configurations

            if self.high_level_cfg["SELECT_UNUSED_STATE"]:
                new_configurations = self._select_unused_state(configurations, number_of_visits)

            number_of_configurations = len(new_configurations)
            crossing_number = HighLevelAbstractState.from_abstract_state(topology_state).crossing_number
            if crossing_number == 1:
                priorities = []
                for configuration in new_configurations:
                    pos = convert_qpos_to_xyz_with_move_center(self.playground_physics, configuration)
                    link_segments, intersections = get_link_segments(pos)
                    rc_state = ReachableConfigurationsState(
                        high_level_state=HighLevelAbstractState.from_abstract_state(topology_state),
                        low_level_state=LowLevelState(configuration),
                        low_level_pos=pos,
                        link_segments=link_segments,
                        intersections=intersections
                    )
                    priority = get_priority[crossing_number](rc_state, True)
                    priorities.append(priority)
                prob = np.array(priorities) / sum(priorities)
            else:
                prob = np.ones(number_of_configurations) / number_of_configurations
            options = np.arange(0, number_of_configurations, 1, dtype=int)
            configuration_index = np.random.choice(options, 1, p=prob)
            configuration = new_configurations[configuration_index[0]]
            #configuration = random.choice(new_configurations)

            self.graph_manager.update_number_of_visits(configuration)

            return configuration

        def execute_action_in_curve(self, action, physics):
            if self.video is not None:
                physics = execute_action_in_curve_with_mujoco_controller(physics=physics, action=action,\
                    get_video=self.get_video, show_image=self.show_image,return_render=False,\
                    sample_rate=self.frame_rate, video=self.video,num_of_links=self.num_of_links, env_path=self.args.env_path)
            else:
                physics = execute_action_in_curve_with_mujoco_controller(physics=physics, action=action, get_video=self.get_video,\
                    show_image=self.show_image,return_render=False, sample_rate=self.frame_rate,\
                        num_of_links=self.num_of_links, env_path=self.args.env_path)
            return physics

        def follow_plan(self, plan):
            #select configuration and set the physics
            new_config = self._bendit_select_config_from_topology_state(plan[0][0])
            set_physics_state(self.physics, new_config)
            
            #remove current state
            states_plan = plan[0][1:]
            actions_plan = plan[1]
            assert len(states_plan) == len(actions_plan), f"{len(states_plan)=}, {len(actions_plan)=}"
            success = True
            for i, state in enumerate(states_plan):
                high_level_action = actions_plan[i]['move']
                print(f'Following plan, trying to perform transition {i+1}/{len(states_plan)} with crossing number {calculate_number_of_crosses_from_topology_state(state)} by action {high_level_action}')
                configuration = new_config
                parent_id = self.graph_manager.get_parent_id(configuration)
                set_physics_state(self.physics, configuration)    
                found_action = False
                selected_samples = self.low_planner.find_curve(
                    configuration=configuration,
                    target_topology_state=state,
                    state_idx=i,
                    plan=states_plan,
                    physics=self.physics,
                    playground_physics=self.playground_physics
                )
                for index, sample in enumerate(selected_samples):
                    set_physics_state(self.physics, sample['configuration'])
                    self.physics = self.execute_action_in_curve(sample['action'], self.physics)
                    new_primitive_state = get_current_primitive_state(self.physics)
                    new_pos_state = convert_qpos_to_xyz_with_move_center(self.playground_physics, new_primitive_state)
                    new_topology_state_precidtion = convert_pos_to_topology(new_pos_state)
                    crossing_number = calculate_number_of_crosses_from_topology_state(state)

                    #if we found good action
                    # if comperae_two_high_level_states(new_topology_state_precidtion, state):
                    if HighLevelAbstractState.from_abstract_state(new_topology_state_precidtion) == HighLevelAbstractState.from_abstract_state(state):
                        found_action = True
                        x = HighLevelAbstractState.from_abstract_state(new_topology_state_precidtion) == HighLevelAbstractState.from_abstract_state(state)
                        new_config = new_primitive_state

                        print(f'Following plan, good action led to a state with crossing number {crossing_number} by action {high_level_action}, index = {index+1} out of {len(selected_samples)} samples')

                    state_str = convert_topology_to_str(new_topology_state_precidtion)
                    
                    #if we found new topology state
                    if len(new_topology_state_precidtion.points) >= len(state.points):
                        if state_str not in self.reachable:
                            self.reachable[state_str] = new_topology_state_precidtion
                            new_plans = self.get_all_high_level_plan(new_topology_state_precidtion, self.topology_goal)
                            print(f'Adding {len(new_plans)} high level plans that start with state with crossing number {calculate_number_of_crosses_from_topology_state(new_topology_state_precidtion)}')
                            self.high_level_plans.extend(new_plans)
                        
                        #if we found new configuration that has at least number of crosses as the target.
                        
                        _ = self.graph_manager.add_node(new_topology_state_precidtion, new_primitive_state, parent_id,\
                                sample['action'], sample['episode'])
                        if HighLevelAbstractState.from_abstract_state(new_topology_state_precidtion) == self.topology_goal:
                            print(f'Following plan, FOUND THE GOAL STATE with crossing number {calculate_number_of_crosses_from_topology_state(new_topology_state_precidtion)} by action {high_level_action}')
                    #print("AR - need to reset bandits")
                    #[MS] need to reset bandits
                if found_action == False:
                    print(f"Following plan, failed to perform transition {i+1}/{len(states_plan)} to crossing number {calculate_number_of_crosses_from_topology_state(state)} by action {high_level_action} => success=False:\n{state}")
                    success = False
                    break
            return success

        def _select_topology_state_from_reachable_options(self, all_states, states_with_plan, return_state_without_plan=True):

            states_options = list(all_states.values())
            
            if return_state_without_plan:
                unique_options = [state for state in states_options if (state not in states_with_plan)]
                states_options = unique_options

            prob = np.ones(len(states_options)) / len(states_options)
            state_np = np.random.choice(states_options, 1, p=prob)
            state = state_np[0]
            #state = random.choice(states_options)

            return state

        def _generate_random_action(self, size=1):
            int_part = torch.randint(self.low_index, self.high_index,(size,1))
            continues_part = torch.rand(size,3)
            #height
            continues_part[:,0] *= self.high_height - self.low_height
            continues_part[:,0] += self.low_height

            #x,y part
            continues_part[:,1] *= self.high_end_location - self.low_end_location
            continues_part[:,1] += self.low_end_location
            continues_part[:,2] *= self.high_end_location - self.low_end_location
            continues_part[:,2] += self.low_end_location

            #concat
            batch = torch.cat((int_part,continues_part), 1)
            return batch

        def expand(self, topology_state):
            #select configuration and set the physics
            configuration = self._bendit_select_config_from_topology_state(topology_state)
            set_physics_state(self.playground_physics, configuration)
            old_pos_state = convert_qpos_to_xyz_with_move_center(self.playground_physics, configuration)
            old_topology_state_precidtion = convert_pos_to_topology(old_pos_state)
            parent_id = self.graph_manager.get_parent_id(configuration)

            for _ in range(self.random_search_steps):
                set_physics_state(self.playground_physics, configuration)
                action = self._generate_random_action()
                self.playground_physics = self.execute_action_in_curve(action[0], self.playground_physics)
                new_primitive_state = get_current_primitive_state(self.playground_physics)
                new_pos_state = convert_qpos_to_xyz_with_move_center(self.playground_physics, new_primitive_state)
                new_topology_state_precidtion = convert_pos_to_topology(new_pos_state)
                if HighLevelAbstractState.from_abstract_state(new_topology_state_precidtion) == self.topology_goal:
                    print(f'Expand step: FOUND THE GOAL STATE with crossing number {calculate_number_of_crosses_from_topology_state(new_topology_state_precidtion)}')

                state_str = convert_topology_to_str(new_topology_state_precidtion)
                if state_str not in self.reachable and len(new_topology_state_precidtion.points) >=\
                     len(old_topology_state_precidtion.points):
                    self.reachable[state_str] = new_topology_state_precidtion
                    new_plans = self.get_all_high_level_plan(new_topology_state_precidtion, self.topology_goal)
                    self.high_level_plans.extend(new_plans)
                    _ = self.graph_manager.add_node(new_topology_state_precidtion, new_primitive_state, parent_id,\
                            action[0])
                    break

        def save_video(self, path):
            if not self.get_video:
                print("video == False, no vidoe was generated")
                return
            index = 0
            check = True
            name = "project.avi"
            while check:
                if os.path.isfile(path +"/"+ str(index)+"_"+name):
                    index += 1
                else:
                    check=False
            new_path = path +"/"+ str(index)+"_"+name

            out = cv2.VideoWriter(new_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (640,480))
            for i in range(len(self.video)):
                out.write(self.video[i])
            out.release()

        def save(self):
            if self.to_save:
                os.makedirs(self.output_path, exist_ok=True)
                self.save_log(self.output_path)
                self.save_video(self.output_path)
                self.save_graph(self.output_path)

        def save_graph(self, path):
            with open(path + "/graph.gpickle", 'wb') as f:
                pickle.dump(self.graph_manager.graph, f, pickle.HIGHEST_PROTOCOL)

        def _init_run_time(self):
            self.time_get_all_high_level_plan = 0
            self.time_get_all_initial_topology_states = 0
            self.time_bandit_select_topology_state = 0
            self.time_bandit_select_trejctory_from_topology_state = 0
            self.time_follow_plan = 0
            self.time_expand = 0

        def tie_knot(self, seed: int, running_time: timedelta = None, env: Optional[ExplorationGymEnvs] = None):
            seed_everything(seed)
            if self.low_level_cfg.NAME in ["RLPlanner", "MultiRLPlanner"]:
                self.init_low_level(self.low_level_cfg, env)

            global_start_time = datetime.now()

            try:
                st = time.time()
                self.high_level_plans = self.get_all_high_level_plan(self.topology_start, self.topology_goal)
                et = time.time()
                self.time_get_all_high_level_plan += et-st

                while True:
                    if running_time is not None and datetime.now() - global_start_time > running_time:
                        return False, datetime.now() - global_start_time

                    #get all inital topology states
                    st = time.time()
                    self.initial_topology_states = self._get_all_initial_topology_states(self.high_level_plans)
                    et = time.time()
                    self.time_get_all_initial_topology_states += et-st


                    #select topolgy state using bandits
                    st = time.time()
                    topology_state = self._bandit_select_topology_state(self.initial_topology_states)
                    et = time.time()
                    self.time_bandit_select_topology_state += et-st


                    #select plan with initial topology state, also bandits
                    st = time.time()
                    selected_plan = self._bandit_select_trejctory_from_topology_state(topology_state,\
                         self.high_level_plans)
                    et = time.time()
                    self.time_bandit_select_trejctory_from_topology_state += et-st



                    #follow plan
                    st = time.time()
                    _ = self.follow_plan(selected_plan)
                    et = time.time()
                    self.time_follow_plan += et-st


                    #solution was found
                    if self.graph_manager.check_topology_goal(self.topology_goal):
                        trajectory = self.graph_manager.trajctory_extractor(self.topology_goal)
                        self.write_log_instance("goal_found", True)
                        self.write_log_instance("trajectory", trajectory)
                        self.save()
                        return True, datetime.now() - global_start_time

                    #in some probability we will run it totaly random
                    elif random.random() < self.low_level_cfg.random_search_threshold:
                        topogly_state_for_expend = self._select_topology_state_from_reachable_options(self.reachable,\
                                self.initial_topology_states, return_state_without_plan=False)

                        #search randomly
                        st = time.time()
                        self.expand(topogly_state_for_expend)
                        et = time.time()
                        self.time_expand += et-st
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                raise e
            finally:
                self.low_planner.close()

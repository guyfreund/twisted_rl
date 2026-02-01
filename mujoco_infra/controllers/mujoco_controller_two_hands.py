from dm_control import mujoco
import numpy as np
from numpy import linalg as LA
import cv2
import copy

from mujoco_infra.mujoco_utils.mujoco import get_link_xyz, create_spline_from_points, get_number_of_links,\
    get_current_primitive_state, set_physics_state, enable_mocap, convert_name_to_index,\
    update_mocap_location, location_and_velocity_reached, get_env_image,\
    update_mocap_location_and_rotation, delete_mocap
    

class MujocoControllerTwoHands():
    def __init__(self, env_path:str, save_video:bool, create_video:bool,
                sample_rate:int,  show_image:bool, max_tries:int=1000
                ):
        self.env_path = env_path
        self.max_tries = max_tries
        self.create_video = create_video
        self.save_video = save_video
        self.sample_rate = sample_rate
        self.show_image = show_image
        
        physics = mujoco.Physics.from_xml_path(self.env_path)
        self.num_of_links = get_number_of_links(physics=physics, qpos=physics.get_state())

    @staticmethod
    def validate_action(physics:mujoco.Physics, action:list, link_size:float=0.04) -> bool:
        """_summary_
        Args:
            physics (mujoco.Physics): _description_
            action (list): _description_

        Returns:
            bool: _description_
        """
        #check the action is from two hands
        if len(action) <= 4:
            return True
        
        #get links XYZ
        active_link_index = action[0]
        passive_link_index = action[1]
        action_link_pos = get_link_xyz(link_number=active_link_index, physics=physics)
        passive_link_pos = get_link_xyz(link_number=passive_link_index, physics=physics)

        #max distance
        max_distance = abs(active_link_index-passive_link_index) * link_size

        #create spline
        start_point = action_link_pos
        end_point= [action[3] + start_point[0],action[4] + start_point[1], 0]
        max_height = action[2]
        splines = create_spline_from_points(start_point=start_point, max_height=max_height, end_point=end_point,\
            step_size=0.00001)
        
        #get critical_points
        critical_points = []
        critical_points.append(splines['location'][0][0])
        for spline in splines['location']:
            critical_points.append(spline[-1])

        #check each point
        for critical_point in critical_points:
            distance = LA.norm((np.array(passive_link_pos)-np.array(critical_point)), 2)
            if distance > max_distance:
                return False
            
        return True

    def execute_sub_action(self, physics:mujoco.Physics, action_index:int, velo_tol:int=5,
            loc_tol:float=0.0005, max_velo:float=0.6, position:bool=True, velocity:bool=True,
            reduce_tries:int=1
            ):
        tries = 0
        while not location_and_velocity_reached(physics=physics, index=action_index,
                position=position, velocity=velocity, velo_tol=velo_tol, loc_tol=loc_tol,
                max_velo=max_velo) and tries < int(self.max_tries/reduce_tries):
            physics.step()
            tries += 1
            self.save_image_and_video(physics=physics)
            self.sample_rate_cnt += 1

        if tries == self.max_tries:
            success = False
        else:
            success = True
        
        return physics, success

    def save_image_and_video(self, physics:mujoco.Physics):
        if self.create_video:
            if self.sample_rate_cnt%self.sample_rate==0:
                pixels = get_env_image(physics)
                self.video.append(pixels)
        if self.show_image and self.sample_rate_cnt%self.sample_rate==0:
            cv2.imshow("image", pixels)
            cv2.waitKey(2)

    def execute_action(self, physics:mujoco.Physics ,action:list, output_path:str):

        #check the action is valid
        if not self.validate_action(physics=physics, action=action):
            if self.create_video:
                return physics, [], False
            else:
                return physics, False
        
        #break action to params
        mocap_active_index, mocap_passive_index, max_height, max_x, max_y = action
        self.sample_rate_cnt = 0
        self.video = []

        current_state = get_current_primitive_state(physics)
        set_physics_state(physics, current_state)
        enable_mocap(physics)

        active_joint = "G"+str(int(mocap_active_index))
        action_index = convert_name_to_index(active_joint, num_of_links=self.num_of_links)

        passive_joint = "G"+str(int(mocap_passive_index))
        passive_index = convert_name_to_index(passive_joint, num_of_links=self.num_of_links)

        update_mocap_location(physics, action_index, mocap_index=0)
        update_mocap_location(physics, passive_index, mocap_index=1)

        physics, _ = self.execute_sub_action(physics=physics, action_index=action_index,
            velo_tol=2., loc_tol=0.0001, max_velo=0.1)

        #create waypoints from spline
        start_point = physics.data.xpos[action_index]
        end_point= [max_x + start_point[0], max_y + start_point[1], 0]
        splines = create_spline_from_points(start_point=start_point, max_height=max_height,
            end_point=end_point, step_size=0.00001)
        number_of_paths_in_curve = len(splines["location"])

        #run the spline
        for path_in_spline in range(number_of_paths_in_curve):
            position = splines["location"][path_in_spline]
            for index in range(len(position)-1):
                update_mocap_location_and_rotation(physics=physics, index=0, position=position[index],\
                                                    xquat=copy.copy(physics.data.xquat[action_index]))
                physics, _ = self.execute_sub_action(physics=physics, action_index=action_index,
                    position=True, velocity=False, loc_tol=0.0001, reduce_tries=10)
                
        #last action in the curve
        update_mocap_location_and_rotation(physics=physics, index=0, position=position[-1])

        physics, _ = self.execute_sub_action(physics=physics, action_index=action_index,
            position=True, velocity=True, velo_tol=5, loc_tol=0.0005, max_velo=0.6)
        
        #run steps to stable the system
        update_mocap_location_and_rotation(physics=physics, index=0,
            xquat=copy.copy(physics.data.xquat[action_index]))
        
        physics, success = self.execute_sub_action(physics=physics, action_index=action_index,
            position=True, velocity=True, velo_tol=2, loc_tol=0.0001, max_velo=0.1)  
        
        delete_mocap(physics, mocap_index=-1)
        physics.step()
        update_mocap_location_and_rotation(physics=physics, index=0,
            xquat=copy.copy(physics.data.xquat[action_index]))

        for _ in range(10):
            physics.step()
            self.save_image_and_video(physics=physics)
            self.sample_rate_cnt+=1


        if self.save_video:
            self.save_video_from_images(images=self.video, output_path=output_path, sample_rate=1)
        
        if self.create_video:
            return physics, self.video, success
        else:
            return physics, success
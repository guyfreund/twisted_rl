from dm_control import mujoco
import numpy as np
from numpy import linalg as LA
import cv2
import copy

from mujoco_infra.mujoco_utils.mujoco_cable import get_number_of_links, convert_name_to_index
from mujoco_infra.mujoco_utils.mujoco import get_current_primitive_state, set_physics_state, enable_mocap,\
    update_mocap_location, location_and_velocity_reached, create_spline_from_points, update_mocap_location_and_rotation,\
    delete_mocap
from mujoco_infra.controllers.mujoco_controller import MujocoController

class CabelController(MujocoController):
    def __init__(self, env_path:str, save_video:bool, create_video:bool,
                sample_rate:int,  show_image:bool, max_tries:int=1000
                ):
        super().__init__(
            env_path=env_path,
            save_video=save_video,
            create_video=create_video,
            sample_rate=sample_rate,
            show_image=show_image,
            max_tries=max_tries,
        )

        self.cnt = 0

        physics = mujoco.Physics.from_xml_path(self.env_path)
        self.num_of_links = get_number_of_links(physics=physics, qpos=physics.get_state())

    def execute_sub_action(self, physics:mujoco.Physics, action_index:int, velo_tol:int=5,
            loc_tol:float=0.0005, max_velo:float=0.6, position:bool=True, velocity:bool=True,
            reduce_tries:int=1
            ):
        tries = 0
        while not location_and_velocity_reached(physics=physics, index=action_index,
                position=position, velocity=velocity, velo_tol=velo_tol, loc_tol=loc_tol,
                max_velo=max_velo) and tries < int(self.max_tries/reduce_tries):
            physics.step()

            if self.cnt % 10 == 0:
                image = physics.render()
                cv2.imshow("image", image)
                cv2.waitKey(0)
            self.cnt +=1

            tries += 1
            self.save_image_and_video(physics=physics)
            self.sample_rate_cnt += 1

        if tries == self.max_tries:
            success = False
        else:
            success = True
        
        return physics, success

    def execute_action(self, physics:mujoco.Physics ,action:list, output_path:str):

        #break action to params
        mocap_index, max_height, max_x, max_y = action
        self.sample_rate_cnt = 0
        self.video = []

        current_state = get_current_primitive_state(physics)
        set_physics_state(physics, current_state)
        enable_mocap(physics)

        active_joint = "G"+str(int(mocap_index))
        action_index = convert_name_to_index(active_joint, num_of_links=self.num_of_links)

        update_mocap_location(physics, action_index, mocap_index=0)

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
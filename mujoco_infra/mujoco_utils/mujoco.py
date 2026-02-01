from dm_control import mujoco
import cv2
import copy
import numpy as np
from numpy import linalg as LA
import torch
from num2words import num2words
import os

from mujoco_infra.mujoco_utils.enums import convert_name_to_index
from mujoco_infra.mujoco_utils.general_utils import load_pickle
from mujoco_infra.mujoco_utils.topology.representation import AbstractState
from num2words import num2words
from mujoco_infra.mujoco_utils.video_utils import Video
from mujoco_infra.mujoco_utils.topology.state_2_topology import state2topology, find_new_intersections

#functions_list = getmembers(utils.mujoco, isfunction)
"""
calculate_number_of_crosses_from_topology_state
convert_joint_pos_to_link_pos
convert_name_to_index
convert_qpos_to_xyz_with_move_center
create_spline_from_points
delete_mocap
enable_mocap
execute_action_in_curve_with_mujoco_controller
execute_action_in_curve_with_mujoco_controller_two_hands
fix_anachor_state
get_current_primitive_state
get_env_image
get_link_xyz
get_number_of_links
get_position_from_physics
get_xanchor_indexes
load_mujoco_env
location_and_velocity_reached
move_center
physics_reset
save_video_from_images
set_physics_state
update_mocap_location
update_mocap_location_and_rotation
validate_action
"""


def execute_action_in_curve_with_mujoco_controller(physics, action, num_of_links, get_video=False, \
                                                   show_image=False, save_video=False, \
                                                   return_render=True, sample_rate=20, video=None,
                                                   output_path="outputs/videos/", env_path="", return_video=False):
    # reset physics
    current_state = get_current_primitive_state(physics)
    num_of_links = get_number_of_links(physics=physics, qpos=current_state)

    physics = mujoco.Physics.from_xml_path(env_path)
    set_physics_state(physics, current_state)

    enable_mocap(physics)
    video = []

    joint = "G" + str(int(action[0]))
    action_index = convert_name_to_index(joint, num_of_links=num_of_links)

    # update the locatino of the mocap according the link that will be move
    update_mocap_location(physics, action_index)
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=False, velocity=True, velo_tol=2, \
                                            loc_tol=0.0001, max_velo=0.1) and tries < 1000:
        physics.step()
        tries += 1
        if get_video:
            pixels = physics.render()
            video.append(pixels)
        if show_image and index % sample_rate == 0:
            cv2.imshow("image", pixels)
            cv2.waitKey(2)

    start_point = physics.data.xpos[action_index]
    end_point = [action[2] + start_point[0], action[3] + start_point[1], 0]
    max_height = action[1]
    if video is None:
        video = []

    splines = create_spline_from_points(start_point=start_point, max_height=max_height, end_point=end_point, \
                                        step_size=0.00001)

    number_of_paths_in_curve = len(splines["location"])

    for path_in_spline in range(number_of_paths_in_curve):
        position = splines["location"][path_in_spline]
        for index in range(len(position) - 1):
            physics._data.mocap_pos[:] = position[index]
            physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
            tries = 0
            while not location_and_velocity_reached(physics, action_index, position=True, velocity=False, \
                                                    loc_tol=0.0001) and tries < 1000:
                tries += 1
                physics.step()
                if get_video:  # and index%sample_rate==0:
                    pixels = physics.render()
                    video.append(pixels)
                if show_image and index % sample_rate == 0:
                    cv2.imshow("image", pixels)
                    cv2.waitKey(2)

        # last action in the curve
        physics._data.mocap_pos[0] = position[-1]
        tries = 0
        while not location_and_velocity_reached(physics, action_index, position=True, velocity=True) and tries < 1000:
            tries += 1
            physics.step()
            if get_video:
                pixels = physics.render()
                video.append(pixels)
            if show_image and index % sample_rate == 0:
                cv2.imshow("image", pixels)
                cv2.waitKey(2)
    physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=True, velocity=True, \
                                            velo_tol=2, loc_tol=0.0001, max_velo=0.1) and tries < 1000:
        physics.step()
        tries += 1
        if get_video:
            pixels = physics.render()
            video.append(pixels)
        if show_image and index % sample_rate == 0:
            cv2.imshow("image", pixels)
            cv2.waitKey(2)

    delete_mocap(physics)
    physics.step()
    physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=False, velocity=True, \
                                            velo_tol=1.5, loc_tol=0.0001, max_velo=0.05) and tries < 1000:
        for _ in range(10):
            physics.step()
            if get_video:
                pixels = physics.render()
                video.append(pixels)
            if show_image and index % sample_rate == 0:
                cv2.imshow("image", pixels)
                cv2.waitKey(2)
        tries += 1

    if save_video:
        video_index = 0
        free = True
        while free:
            if os.path.isfile(output_path + str(video_index) + "_project.avi"):
                video_index += 1
            else:
                free = False

        path = output_path + str(video_index) + "_project.avi"
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"DIVX"), 60, (640, 480))
        for i in range(len(video)):
            if i % sample_rate == 0:
                out.write(video[i])
        out.write(video[-1])
        out.release()

    if return_video:
        return physics, video
    else:
        return physics


def calculate_number_of_crosses_from_topology_state(topology_state):
    return int((len(topology_state.points)-2)/2)

def get_joints_indexes(num_of_links: int) -> dict:
    """return dict with index for each joints

    Args:
        num_of_links (int): number of links

    Returns:
        dict: index between joint and name
    """
    #num_links = int(num_of_links)
    index = {}
    for i in range(int(num_of_links/2)):
        for joint_index in [0,1]:
            key = "J"+str(joint_index)+"_"+str(i)
            index[key] = 6 + (num_of_links-1)*2 - i*2 - 1 + joint_index
    for i in range(int(num_of_links/2+1), num_of_links):
        for joint_index in [0,1]:
            key = "J"+str(joint_index)+"_"+str(i)
            index[key] = 6 + (i-int(num_of_links/2+1))*2 + 1 + joint_index
    return index


def convert_pos_to_topology(pos):
    if not torch.is_tensor(pos):
        pos = torch.tensor(pos)
    pos = pos.detach()
    topology = state2topology(pos, full_topology_representation=True)
    return topology


def convert_qpos_to_xyz(physics, qpos):
    current_state = copy.deepcopy(physics.get_state())
    set_physics_state(physics, qpos)
    move_center(physics)
    pos = get_position_from_physics(physics)
    set_physics_state(physics, current_state)
    move_center(physics)
    return pos


def convert_topology_to_number_of_crosses(topology):
    crosses = int((len(topology)-2)/2)
    return crosses

def load_mujoco_env(env_path:str) -> mujoco.Physics:
    """
    Load mujoco env from xml file
    """
    physics = mujoco.Physics.from_xml_path(env_path)
    return physics

def get_env_image(physics:mujoco.Physics):
    """
    Render env current state
    """
    image = physics.render(
        height=2400,
        width=3200,
        )
    return image

def set_physics_state(physics, state):
    with physics.reset_context():
        physics.set_state(state)

def get_current_primitive_state(physics:mujoco.Physics):
    """
    return qpos of current physics state
    """
    state = copy.copy(physics.get_state())
    return state

def get_number_of_links(physics:mujoco.Physics=None, qpos:np.ndarray=None)-> int:
    """return how much links the rope have

    Args:
        physics (dm control): physics of the SIM

    Returns:
        num_of_links(int): num of links
    """
    if physics is not None:
        xanchor = copy.deepcopy(physics.data.xanchor)
        num_of_links = int((len(xanchor)-1)/2+1)
    elif qpos is not None:
        if torch.is_tensor(qpos):
            qpos_length = qpos.shape[1]
            num_of_links = (qpos_length-7)/2+1
    else:
        #not input
        raise
    return int(num_of_links)

def set_physics_state(physics:mujoco.Physics, state:np.ndarray):
    """
    Set physics with state
    """
    with physics.reset_context():
        physics.set_state(state)

def enable_mocap(physics:mujoco.Physics, index:int=-1):
    """
    Turn on mocap
    """
    if index == -1:
        physics.model.eq_active[:] = 1
    else:
        physics.model.eq_active[index] = 1

def delete_mocap(physics:mujoco.Physics, mocap_index:int=-1):
    """
    Turn off mocap
    """
    if mocap_index == -1:
        physics.model.eq_active[:] = 0
    else:
        physics.model.eq_active[mocap_index] = 0

def update_mocap_location(physics:mujoco.Physics, action_index:int, mocap_index:int=0):
    """
    Update mocap location to given "action_index"
    """
    physics.model.eq_obj2id[mocap_index] = action_index
    physics._data.mocap_pos[mocap_index] = copy.copy(physics.data.xpos[action_index])
    physics._data.mocap_quat[mocap_index] = copy.copy(physics.data.xquat[action_index])
    physics.model.eq_data[mocap_index][:6] = [0,0,0,0,0,0] 

def location_and_velocity_reached(
        physics:mujoco.Physics, index:int, position:bool=True,\
        velocity:bool=True, velo_tol:int=5,\
        loc_tol:float=0.0005, max_velo:float = 0.6
        ) -> bool:
    
    if position:
        goal_location = copy.copy(physics._data.mocap_pos[0])
        current_location = copy.copy(physics._data.xpos[index])

        if pow(goal_location[0]-current_location[0],2) > loc_tol or\
            pow(goal_location[1]-current_location[1],2) > loc_tol or\
            pow(goal_location[2]-current_location[2],2) > loc_tol:
            return False
    if velocity:
        if sum(pow(physics.data.qvel[:],2)) > velo_tol:
            return False

        if max(physics.data.qvel[:]) > max_velo or abs(min(physics.data.qvel[:])) > max_velo:
            return False
    
    return True

def create_spline_from_points(
        start_point:list[float], max_height:float,\
        end_point:list[float], step_size:float=0.01
        ) -> list[float]:
    """
    This method will get a 3 points and step size and return location and velocity in each time stamp
    """
    # start and end point are in z=0

    second_point = copy.copy(start_point)
    second_point[2] = max_height
    third_point = copy.copy(end_point)
    third_point[2] = max_height

    points_1 = []
    points_2 = []
    points_3 = []
    num_of_points = max(int((max_height - start_point[2])/step_size), int(0.01/step_size))
    z_points = np.linspace(start_point[2], max_height, num_of_points)
    x_y_max_points = int(max(abs(end_point[0] - start_point[0]), abs(end_point[1] - start_point[1])) / step_size)

    #from start to second point
    points_1.append(list(start_point))
    for item in z_points:
        points_1.append([start_point[0], start_point[1], item])
    
    #from second point to third point
    x_points = np.linspace(start_point[0], end_point[0], x_y_max_points)
    y_points = np.linspace(start_point[1], end_point[1], x_y_max_points)
    for index in range(x_y_max_points):
        points_2.append([x_points[index], y_points[index], max_height])

    #from third to end
    for item in reversed(z_points):
        points_3.append([end_point[0], end_point[1], item])
    
    points = {}
    points["location"] = [points_1, points_2, points_3]
    return points

def update_mocap_location_and_rotation(physics:mujoco.Physics, index:int,\
                                       position:list=None, xquat:list=None):
    if position is not None:
        physics._data.mocap_pos[index] = position
    if xquat is not None:
        physics._data.mocap_quat[index] = copy.copy(xquat)

def move_center(physics:mujoco.Physics):
    """Move middle link to [0,0,0]"""
    state = get_current_primitive_state(physics)
    state[:2] = 0
    set_physics_state(physics, state)

def convert_qpos_to_xyz_with_move_center(physics:mujoco.Physics, qpos:list) -> list:
    """Convert qpos to xyz"""
    current_state = copy.deepcopy(physics.get_state())
    set_physics_state(physics, qpos)
    move_center(physics)
    pos = get_position_from_physics(physics)
    set_physics_state(physics, current_state)
    move_center(physics)
    return pos

def get_xanchor_indexes(num_of_links: int) -> dict:
    """return dict with index for each position

    Args:
        num_of_links (int): number of links

    Returns:
        dict: index between position and name
    """
    num_links = int(num_of_links)
    index = {}
    for i in range(int(num_links/2)):
        index[i] = (num_links-1-i)*2
    for i in range(int(num_links/2+1), num_links):
        index[i] = (1+(i-int(num_links/2+1)))*2
    return index

def fix_anachor_state(state, num_of_links=21):
    index = get_xanchor_indexes(num_of_links=num_of_links)
    update_state = []
    for key in index.keys():
        update_state.append(state[index[key]])
    return update_state

def get_position_from_physics(physics:mujoco.Physics) -> np.array:
    """return postion of rope

    Args:
        physics (dm control): physics of the SIM

    Returns:
        position (np.array): rope position relative to joints
    """
    xanchor = copy.deepcopy(physics.data.xanchor)
    num_of_links = get_number_of_links(physics=physics)
    position = np.zeros((num_of_links+1,3))
    state = np.array(fix_anachor_state(xanchor, num_of_links=num_of_links))
    position[1:-1] = state
    start_body_location = physics.named.data.xpos["B0"]
    position[0] = start_body_location-position[1] + start_body_location
    end_body_location = physics.named.data.xpos["B"+str(num_of_links-1)]
    position[-1] = end_body_location-position[num_of_links-1] + end_body_location
    
    return position

def get_link_xyz(link_number:int, physics:mujoco.Physics) -> list:
    """Return [x,y,z] of a given link"""
    joints_positions = get_position_from_physics(physics)
    links_positions = convert_joint_pos_to_link_pos(joints_positions)
    return links_positions[int(link_number)]

def convert_joint_pos_to_link_pos(joint_pos:list) -> list:
    """convert joint pos to link pos"""
    #create empty list
    links_positions = []

    #take average between two points
    for index in range(1,len(joint_pos)):
        links_positions.append((joint_pos[index]+joint_pos[index-1])/2)
    
    return links_positions

def physics_reset(physics:mujoco.Physics):
    physics.reset()
    num_of_links = get_number_of_links(physics)
    config_length = num_of_links*2 + 7 
    current_state = copy.copy(physics.get_state())
    for index in range(7,config_length):
        if index%2 == 1:
            continue
        if index > 7 + num_of_links:
            current_state[index] -= np.pi / ((config_length-7)/4)
        else:
            current_state[index] += np.pi / ((config_length-7)/4)
    set_physics_state(physics, current_state)   

def validate_action(physics:mujoco.Physics, action:list, link_size:float=0.04) -> bool:
    """Check if the action is in the rope area"""

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

def get_random_action(min_index:int, max_index:int, max_height:float, min_location:float, max_location:float,
                      num_of_indexes:int=1):

    #find indexes
    indexes_range = max_index-min_index
    indexes = np.random.choice(indexes_range, num_of_indexes, replace=False)
    indexes += min_index
    list_indexes = [int(index) for index in indexes]

    max_height_value = np.random.uniform(low=0.001, high=max_height)
    end_location = np.random.uniform(low=min_location, high=max_location, size=2)
    
    return np.array(list_indexes + [max_height_value, end_location[0], end_location[1]])

def save_image_and_video(get_video:bool, show_image:bool, physics:mujoco.Physics,
                         sample_rate_cnt:int, video:Video, sample_rate:int):
    if get_video:
        if sample_rate_cnt%sample_rate==0:
            pixels = get_env_image(physics)
            video.add_frames([pixels])
    if show_image and sample_rate_cnt%sample_rate==0:
        cv2.imshow("image", pixels)
        cv2.waitKey(2)

    return video

def execute_action_in_curve_with_mujoco_controller_two_hands(
    physics:mujoco.Physics, action, get_video=False,\
    show_image=False, save_video=False,\
    return_render=True, sample_rate=20, video=None,\
    output_path="outputs/videos/", env_path="", return_video=False, max_tries=1000, filename=None
    ):

    if not validate_action(physics=physics, action=action):
        if return_video:
            return physics, [], False
        else:
            return physics, False

    mocap_active_index, mocap_passive_index, max_height, max_x, max_y = action
    sample_rate_cnt = 0
    video = Video()

    #reset physics
    current_state = get_current_primitive_state(physics)
    num_of_links = get_number_of_links(physics=physics, qpos=current_state)
    
    physics = mujoco.Physics.from_xml_path(env_path)
    set_physics_state(physics, current_state)
    enable_mocap(physics)

    active_joint = "G"+str(int(mocap_active_index))
    action_index = convert_name_to_index(active_joint, num_of_links=num_of_links)

    passive_joint = "G"+str(int(mocap_passive_index))
    passive_index = convert_name_to_index(passive_joint, num_of_links=num_of_links)

    #update the locatino of the mocap according the link that will be move
    update_mocap_location(physics, action_index, mocap_index=0)
    update_mocap_location(physics, passive_index, mocap_index=1)
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=False, velocity=True, velo_tol=2,\
         loc_tol=0.0001, max_velo = 0.1) and tries < max_tries:
        physics.step()
        tries += 1
        save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                             sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
        sample_rate_cnt+=1

    #create waypoints from spline
    start_point = physics.data.xpos[action_index]
    end_point= [max_x + start_point[0], max_y + start_point[1], 0]
    splines = create_spline_from_points(start_point=start_point, max_height=max_height, end_point=end_point,\
         step_size=0.00001)
    number_of_paths_in_curve = len(splines["location"])

    #run the spline
    for path_in_spline in range(number_of_paths_in_curve):
        position = splines["location"][path_in_spline]
        for index in range(len(position)-1):
            update_mocap_location_and_rotation(physics=physics, index=0, position=position[index],\
                                                xquat=copy.copy(physics.data.xquat[action_index]))
            tries = 0
            while not location_and_velocity_reached(physics, action_index, position=True, velocity=False,\
                 loc_tol=0.0001) and tries < max_tries/10:
                tries += 1
                physics.step()
                save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                                     sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
                sample_rate_cnt+=1
        
        #last action in the curve
        update_mocap_location_and_rotation(physics=physics, index=0, position=position[-1])

        tries = 0
        while not location_and_velocity_reached(physics, action_index, position=True, velocity=True) and tries < max_tries:
            tries += 1
            physics.step()
            save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                                 sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
            sample_rate_cnt+=1


    #run steps to stable the system
    update_mocap_location_and_rotation(physics=physics, index=0, xquat=copy.copy(physics.data.xquat[action_index]))
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=True, velocity=True,\
         velo_tol=2, loc_tol=0.0001, max_velo = 0.1)  and tries < max_tries:
        physics.step()
        tries +=1
        save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                             sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
        sample_rate_cnt+=1

    #reached goal or not
    if tries == max_tries:
        success = False
    else:
        success = True


    #remove mocaps and let the system stop
    delete_mocap(physics, mocap_index=-1)
    physics.step()
    update_mocap_location_and_rotation(physics=physics, index=0, xquat=copy.copy(physics.data.xquat[action_index]))
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=False, velocity=True,\
         velo_tol=1.5, loc_tol=0.0001, max_velo = 0.05) and tries < max_tries:
        for _ in range(10):
            physics.step()
            save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                                 sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
            sample_rate_cnt+=1
        tries += 1

    if save_video:
        video.save(output_path=output_path, filename=filename)
    
    if return_video:
        return physics, video.frames, success
    else:
        return physics, success

def execute_action_in_curve_with_mujoco_controller_one_hand(
    physics:mujoco.Physics, action, get_video=False,\
    show_image=False, save_video=False,\
    return_render=True, sample_rate=20, video=None,\
    output_path="outputs/videos/", env_path="", return_video=False, max_tries=1000, filename=None,
    return_cfgs=False,
    ):

    mocap_active_index, max_height, max_x, max_y = action
    sample_rate_cnt = 0
    cfgs = []

    #reset physics
    current_state = get_current_primitive_state(physics)
    if return_cfgs:
        cfgs.append(current_state)
    num_of_links = get_number_of_links(physics=physics, qpos=current_state)
    
    physics = mujoco.Physics.from_xml_path(env_path)
    set_physics_state(physics, current_state)

    enable_mocap(physics)
    video = Video()

    joint = "G"+str(int(mocap_active_index))
    action_index = convert_name_to_index(joint,num_of_links=num_of_links)

    #update the locatino of the mocap according the link that will be move
    update_mocap_location(physics, action_index)
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=False, velocity=True, velo_tol=2,\
         loc_tol=0.0001, max_velo = 0.1) and tries < max_tries:
        physics.step()
        tries += 1
        if return_cfgs and sample_rate_cnt % sample_rate == 0:
            cfgs.append(get_current_primitive_state(physics))
        save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                             sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
        sample_rate_cnt+=1

    start_point = physics.data.xpos[action_index]  # [-0.06575469 -0.10038043  0.00991071]
    end_point= [max_x + start_point[0], max_y + start_point[1], 0]

    splines = create_spline_from_points(start_point=start_point, max_height=max_height, end_point=end_point,\
         step_size=0.00001)
    number_of_paths_in_curve = len(splines["location"])

    for path_in_spline in range(number_of_paths_in_curve):
        position = splines["location"][path_in_spline]
        for index in range(len(position)-1):
            physics._data.mocap_pos[:] = position[index]
            physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
            tries = 0
            while not location_and_velocity_reached(physics, action_index, position=True, velocity=False,\
                 loc_tol=0.0001) and tries < max_tries/10:
                tries += 1
                physics.step()
                if return_cfgs and sample_rate_cnt % sample_rate == 0:
                    cfgs.append(get_current_primitive_state(physics))
                save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                                     sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
                sample_rate_cnt+=1
        
        #last action in the curve
        physics._data.mocap_pos[0] = position[-1]
        tries = 0
        while not location_and_velocity_reached(physics, action_index, position=True, velocity=True) and tries < max_tries:
            tries += 1
            physics.step()
            if return_cfgs and sample_rate_cnt % sample_rate == 0:
                cfgs.append(get_current_primitive_state(physics))
            save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                                 sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
            sample_rate_cnt+=1
            
    physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=True, velocity=True,\
         velo_tol=2, loc_tol=0.0001, max_velo = 0.1)  and tries < max_tries:
        physics.step()
        tries +=1
        if return_cfgs and sample_rate_cnt % sample_rate == 0:
            cfgs.append(get_current_primitive_state(physics))
        save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                             sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
        sample_rate_cnt+=1


    #reached goal or not
    if tries == max_tries:
        success = False
    else:
        success = True

    delete_mocap(physics)
    physics.step()
    physics._data.mocap_quat[:] = copy.copy(physics.data.xquat[action_index])
    tries = 0
    while not location_and_velocity_reached(physics, action_index, position=False, velocity=True,\
         velo_tol=1.5, loc_tol=0.0001, max_velo = 0.05) and tries < max_tries:
        for _ in range(10):
            physics.step()
            if return_cfgs and sample_rate_cnt % sample_rate == 0:
                cfgs.append(get_current_primitive_state(physics))
            save_image_and_video(get_video=get_video, show_image=show_image, physics=physics,
                                 sample_rate_cnt=sample_rate_cnt, video=video, sample_rate=sample_rate)
            sample_rate_cnt+=1
        tries += 1

    if save_video:
        video.save(output_path=output_path, filename=filename)
    
    if return_video:
        return physics, video, success
    elif return_cfgs:
        return physics, cfgs, success
    else:
        return physics, success

    
def generate_video_from_plan(plan_path:str, env_path:str, output_path:str, filename = None):
    #load plan
    plan = load_pickle(plan_path)

    #create_mujoco env
    physics = mujoco.Physics.from_xml_path(env_path)

    #create_empty video
    video = Video()

    for sample in [plan]:
        #set initil state
        set_physics_state(physics, sample['start_config'])

        #get action
        action = sample['action']

        #execute action
        physics, temp_images, _ = execute_action_in_curve_with_mujoco_controller_two_hands(
            physics=physics,
            action=action, 
            return_render=False,
            sample_rate=10,
            env_path=env_path,
            get_video=True,
            return_video=True)
        
        video.add_frames(temp_images)

    video.save(output_path=output_path, filename=filename)

def convert_topology_state_to_input_vector(topology:AbstractState) -> np.array:
    """_summary_

    Args:
        topology (AbstractState): topological state

    Returns:
        np.array: topological vector
    """
    output_length = 300
    output = np.zeros(output_length)
    current_index = 0
    if isinstance(topology, AbstractState):
        topology_points = topology.points
    else:
        topology_points = topology
    for item in topology_points:
        if item is None:
            break
        if current_index >= output_length:
            break
        if item.over is not None:
            #assert item.over<9
            output[current_index+item.over-1] = 1
        current_index+=8
        if current_index >= output_length:
            break
        if item.sign is not None:
            assert item.sign==1 or item.sign==-1
            if item.sign == 1:
                output[current_index] = 1
            else:
                output[current_index] = -1
        current_index+=1
        if current_index >= output_length:
            break
        if item.under is not None:
            #assert item.under<9
            output[current_index+item.under-1] = 1
        current_index+=8
    return output[:144]

def convert_topology_to_str(toplogy):
    if isinstance(toplogy, list):
        temp_topology = toplogy
    else:
        temp_topology = toplogy.points

    string = ""
    for item_topology in temp_topology:
        string += str(item_topology)
    return string

def convert_topological_str_to_sentenceses(topological_str):
    topological_str = str(topological_str)
    topological_str = topological_str.split("\n")
    full_sentence = ""
    for index, item in enumerate(topological_str):
        if item == "End point" or item == "":
            continue
        sentence = ""
        sentence += num2words(index, to = 'ordinal')
        for char in item:
            if char == "O":
                sentence += " over"
            elif char == "U":
                sentence += " under"
            elif char == "+":
                sentence += " clockwise"
            elif char == "-":
                sentence += " counterclockwise"
            elif char == " ":
                continue
            else:
                sentence += " " + num2words(char, to = 'ordinal')
        sentence += ". "
        full_sentence += sentence

    if len(full_sentence) == 0:
        full_sentence = "loose rope"

    return full_sentence

def convert_action_from_index_to_one_hot_vector(action:np.array, num_of_links:int) -> np.array:
    """_summary_

    Args:
        action (np.array): _description_
        num_of_links (int): _description_

    Returns:
        _type_: _description_
    """
    one_hot = np.zeros(num_of_links+3)
    one_hot[int(action[0])] = 1
    one_hot[num_of_links] = action[1]
    one_hot[num_of_links+1] = action[2]
    one_hot[num_of_links+2] = action[3]

    return one_hot


def get_link_segments(pos: np.ndarray):
    num_links = pos.shape[0] - 1
    intersections = find_new_intersections(pos)
    points = list(set([it[0] for it in intersections] + [it[1] for it in intersections]))
    points.sort()
    points = [0] + points + [num_links - 1]
    link_segments = [(i, j) for i, j in zip(points[:-1], points[1:])]
    return link_segments, intersections

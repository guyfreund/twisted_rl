from dm_control import mujoco
import cv2
import copy
import numpy as np
from numpy import linalg as LA
import torch

from mujoco_infra.mujoco_utils.enums import convert_name_to_index
from mujoco_infra.mujoco_utils.image_utils import save_video_from_images
from mujoco_infra.mujoco_utils.general_utils import load_pickle
from mujoco_infra.mujoco_utils.topology.representation import AbstractState
from num2words import num2words

def get_number_of_links(physics:mujoco.Physics=None, qpos:np.ndarray=None)-> int:
    """return how much links the rope have

    Args:
        physics (dm control): physics of the SIM

    Returns:
        num_of_links(int): num of links
    """
    if physics is not None:
        xanchor = copy.deepcopy(physics.data.xanchor)
        num_of_links = int(len(xanchor))-1
        assert num_of_links == int((len(qpos)-13)/7)
    elif qpos is not None:
        if isinstance(qpos, np.ndarray):
            num_of_links = int((len(qpos)-2)/7)+1
        elif torch.is_tensor(qpos):
            raise #need to fix it
            qpos_length = qpos.shape[1]
            num_of_links = int((qpos_length-7)/2+1)
    else:
        #not input
        raise
    return num_of_links

def convert_name_to_index(name:str, num_of_links:int=None):
    """
    Convert G to index
    """
    if num_of_links == 39:
        names = {}
        for index in range(num_of_links):
            names["G"+str(index)] = index+1
    else:
        raise
    return names[name]
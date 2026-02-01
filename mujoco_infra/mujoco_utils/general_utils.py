import argparse

import numpy as np
import os
import pickle
import yaml
import json
from easydict import EasyDict as edict
from multiprocessing import Process
from datetime import datetime

from pytorch_lightning import seed_everything

from exploration.mdp.high_level_state import HighLevelAbstractState


def argparse_create():
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpus_to_use', type=int, default=0)
    parser.add_argument('--env_path', type=str, default=None) #'assets/rope_v4.xml')
    parser.add_argument('--cfg', type=str, default='system_flow/config/system_config.yml')
    parser.add_argument('--load_state2state', type=bool, default=True)
    parser.add_argument('--load_state2action', type=bool, default=True)
    parser.add_argument('--reverse_high_level_plan', type=bool, default=False)
    parser.add_argument('--step_size', type=int, default=0.00001)
    parser.add_argument('--location_k', type=int, default=1)
    parser.add_argument('--velocity_k', type=int, default=1)
    parser.add_argument('--location_z', type=int, default=100000)
    parser.add_argument('--location_x_y', type=int, default=7)
    parser.add_argument('--max_force_x_y', type=float, default=0.1)
    parser.add_argument('--max_force_z', type=int, default=5)
    parser.add_argument('--get_image', type=bool, default=False)
    parser.add_argument('--get_video', type=bool, default=False)
    parser.add_argument('--treshold', type=float, default=0.0010)
    parser.add_argument('--stage', type=bool, default=1)
    parser.add_argument('--problme_index', type=bool, default=0)
    parser.add_argument('--run_time_limit', type=bool, default=1800)
    parser.add_argument('--problems_file', type=str, default="system_flow/metrics/all_3_crosses_states.txt")
    parser.add_argument('-llp', '--low_level_planner', type=str, default=None)
    parser.add_argument('-st', '--states_type', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--timecap', type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs='+', default=None, help="seeds for evaluation")
    parser.add_argument('-a', '--ablation', type=str, default='C', choices=['A', 'G', 'M', 'C', 'H', 'AC', 'ALL'], help="ablation study")
    parser.add_argument('-raa', '--remove_all_ablation', type=str, default=None, choices=['G', 'A', 'C', 'AC'], help="remove all ablation study")
    parser.add_argument('-sf', '--search_factor', type=int, default=None, help="search factor for the planner")

    args = parser.parse_args()

    return args


def comperae_two_high_level_states(state_1, state_2):
    return HighLevelAbstractState.from_abstract_state(state_1) == HighLevelAbstractState.from_abstract_state(state_2)

    if not isinstance(state_1, list):
        state_1 = state_1.points
    if not isinstance(state_2, list):
        state_2 = state_2.points
    if len(state_1) != len(state_2):
        return False
    for index in range(len(state_2)):
        if state_1[index].over != state_2[index].over or\
        state_1[index].sign != state_2[index].sign or\
        state_1[index].under != state_2[index].under:
            return False
    return True


def edict2dict(data):
    if isinstance(data, (edict, dict)):
        return {key: edict2dict(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [edict2dict(item) for item in data]
    else:
        return data


def run_with_limited_time(func, args, kwargs, time):
    """Runs a function with time limit
    :param func: The function to run
    :param args: The functions args, given as tuple
    :param kwargs: The functions keywords, given as dict
    :param time: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    start_time = datetime.now()
    p = Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        return False, datetime.now() - start_time
    return True, datetime.now() - start_time

def run_with_limited_time_new(func, args=(), kwargs={}, time=1, default=False):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result

def get_file_name(number_of_crosses=None, dataset_path="datasets", prefix="")->str:
    """_summary_

    Args:
        number_of_crosses (_type_): _description_
        dataset_path (str, optional): _description_. Defaults to "datasets".
        prefix (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    number_of_crosses_str = str(number_of_crosses) if isinstance(number_of_crosses, int) else ""
    folder_path = os.path.join(dataset_path,number_of_crosses_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    index = 0
    ok = True
    while(ok):
        file_name = number_of_crosses_str+"_"+prefix+str(index)+".txt"
        if os.path.isfile(os.path.join(folder_path,file_name)):
            index +=1
        else:
            ok = False
    return os.path.join(folder_path,file_name)

def set_seed(seed:int):
    """_summary_

    Args:
        seed (int): seed to set
    """

    seed_everything(seed=seed)

def save_plan(plan:list, main_folder:str, num_of_crosses:int, prefix=""):
    """_summary_

    Args:
        plan (list): _description_
        main_folder (str): _description_
        num_of_crosses (int): _description_
        prefix (str, optional): _description_. Defaults to "".
    """
    name = get_file_name(num_of_crosses,dataset_path=main_folder, prefix=prefix)
    print("sample", name, " was saved")
    with open(name, "wb") as fp:
        pickle.dump(plan, fp)

def save_transition(transition:dict, main_folder:str, prefix=""):
    """_summary_

    Args:
        plan (list): _description_
        main_folder (str): _description_
        num_of_crosses (int): _description_
        prefix (str, optional): _description_. Defaults to "".
    """
    name = get_file_name(dataset_path=main_folder, prefix=prefix)
    with open(name, "wb") as fp:
        pickle.dump(transition, fp)

def save_pickle(path:str, object_to_save:dict):
    """_summary_

    Args:
        path (str): _description_
        object_to_save (dict): _description_
    """
    with open(path, "wb") as fp:
        pickle.dump(object_to_save, fp)
    fp.close()

def save_json(path:str, object_to_save:dict):
    """_summary_

    Args:
        path (str): _description_
        object_to_save (dict): _description_
    """
    with open(path, "w") as fp:
        json.dump(object_to_save, fp)
    fp.close()

def load_pickle(path:str) -> pickle:
    """_summary_

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(path, "rb") as fp:
        file = pickle.load(fp)
    fp.close()
    return file

def load_yaml(path:str):
    with open(path, 'r') as fp:
        file = yaml.safe_load(fp)
    fp.close()
    return file

def load_numpy(file_path:str) -> dict:
    raw_data = np.load(file_path, allow_pickle=True)
    sample = raw_data.f.a.tolist()

    return sample

def load_json(file_path:str) -> dict:
    f = open(file_path)
    sample = json.load(f)

    return sample

def convert_action_from_index_to_one_hot_vector(action:np.array, num_of_links:int) -> np.array:
    one_hot = np.zeros(num_of_links+3)
    one_hot[int(action[0])] = 1
    one_hot[num_of_links] = action[1]
    one_hot[num_of_links+1] = action[2]
    one_hot[num_of_links+2] = action[3]

    return one_hot

def get_time_string() -> str:
    """
    :return: string with the current time in the format 'month_day_hour_minute_second'
    """
    time = datetime.now()

    return f'{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}'
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tempfile


def prepare_tv_image_from_pos(poses:np.array) -> np.ndarray:
    """_summary_

    Args:
        pos (np.array): _description_

    Returns:
        _type_: _description_
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        for rope in poses:
            rope = rope.reshape(-1,3)
            plt.plot(rope[:,0], rope[:,1], marker = 'o')
        path = os.path.join(tmpdirname,'my_plot.png')
        plt.xlim([-0.4, 0.4])
        plt.ylim([-0.4, 0.4])
        plt.title("blue gt, orange prediction")
        plt.savefig(path)
        image = cv2.imread(path)
    plt.close()
    plt.cla()
    plt.clf()
    return image

def show_image(image:np.array):
    """_summary_

    Args:
        image (np.array): _description_
    """
    cv2.imshow("image", image)
    cv2.waitKey(0)

def save_video_from_images(images:list[np.ndarray], output_path:str,
                            sample_rate_from_data:int, ferquncy:int=60, resolution:tuple=(640,480)):
    """_summary_
    Args:
        images (list[np.ndarray]): _description_
        output_path (str): _description_
        sample_rate (int): _description_
    """
    video_index = 0
    free = True
    while free:
        name = str(video_index)+"_project.avi"
        full_path = os.path.join(output_path, name)
        if os.path.isfile(full_path):
            video_index += 1
        else:
            free=False
    os.makedirs(output_path, exist_ok=True)
    out = cv2.VideoWriter(full_path, cv2.VideoWriter_fourcc(*"DIVX"), ferquncy, resolution)
    for i in range(len(images)):
        if i%sample_rate_from_data ==0:
            out.write(images[i])
    out.write(images[-1])
    out.release()
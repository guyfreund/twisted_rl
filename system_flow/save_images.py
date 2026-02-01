import pickle
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pathlib

os.environ['MUJOCO_GL'] = 'egl'

from dm_control import mujoco
from mujoco_infra.mujoco_utils.mujoco import set_physics_state, physics_reset, get_env_image
from exploration.mdp.graph.high_level_graph import HighLevelGraph
from exploration.mdp.high_level_state import HighLevelAbstractState


def crop_to_rope(frame, slack_ratio=0.2):
    """Crop the frame to a square bounding box around the orange rope, with slack."""
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Define range for orange color
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # If no contours found, return original frame
        print("Warning: No orange rope detected!")
        return frame

    # Get bounding box around all contours
    x_min = y_min = float('inf')
    x_max = y_max = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Compute the square bbox
    width = x_max - x_min
    height = y_max - y_min
    side_length = max(width, height)

    # Add slack
    slack = int(slack_ratio * side_length)
    side_length_with_slack = side_length + 2 * slack

    # Center the box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    half_side = side_length_with_slack // 2

    # Compute final box coordinates with boundaries
    x_min_square = max(center_x - half_side, 0)
    x_max_square = min(center_x + half_side, frame.shape[1])
    y_min_square = max(center_y - half_side, 0)
    y_max_square = min(center_y + half_side, frame.shape[0])

    cropped = frame[y_min_square:y_max_square, x_min_square:x_max_square, :]
    return cropped


paths = [
    'exploration/outputs/evaluation/twisted_evaluation/05-05-2025_21-23/',
    'exploration/outputs/evaluation/twisted_evaluation/29-04-2025_13-14/',
]

for path in paths:
    base_paths = [x.parent for x in pathlib.Path(path).rglob('*.gpickle')]
    physics = mujoco.Physics.from_xml_path("mujoco_infra/assets/rope_v3_21_links_wow.xml")
    physics_reset(physics)
    high_level_graph = HighLevelGraph.load_full_graph()

    for base_path in tqdm(base_paths, total=len(base_paths), desc='Processing states'):
        path = os.path.join(base_path, 'log.pkl')
        graph_path = os.path.join(base_path, 'graph.gpickle')
        image_save_path = os.path.join(base_path, f'trajectory_{base_path.name}.png')

        with open(path, 'rb') as f:
            data = pickle.load(f)

        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)

        if len(data['trajectory']) > 1:
            print(f'state {base_path} has more than one trajectory')

        trajectory = data['trajectory'][0]
        fig, axs = plt.subplots(1, len(trajectory), figsize=(20, 5))
        title = ''
        success = True

        for idx, s in enumerate(trajectory):
            item = graph.nodes[s]
            state = HighLevelAbstractState.from_abstract_state(item['topology_state'])
            # title += f"{state.crossing_number}"
            # if idx < len(trajectory) - 1:
            #     next_state = HighLevelAbstractState.from_abstract_state(graph.nodes[trajectory[idx + 1]]['topology_state'])
            #     edges = high_level_graph.get_all_edge_variations(src=state, dst=next_state, from_graph=False)
            #     moves = set([edge.data['move'] for edge in edges])
            #     if len(moves) < 1:
            #         print(f'{len(moves)=} != 1 for {base_path=}')
            #         success = False
            #         break
            #     move = moves.pop()
            #     title += " \overset{" + move + "}{\longrightarrow} "
            set_physics_state(physics, item['configuration'])
            frame = get_env_image(physics)
            cropped_frame = crop_to_rope(frame)  # crop automatically
            axs[idx].imshow(cropped_frame)
            axs[idx].axis('off')
            # axs[idx].set_title(rf'$\phi = {state.crossing_number}$', fontsize=40, pad=20)

        if success:
            if title:
                fig.suptitle(rf'${title}$', fontsize=64)
            plt.tight_layout()
            plt.savefig(image_save_path)
            plt.close(fig)
            print(f'done {base_path}')

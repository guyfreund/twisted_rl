import pickle
import os
import pathlib
from tqdm import tqdm
import numpy as np


os.environ['MUJOCO_GL'] = 'egl'

from dm_control import mujoco
from mujoco_infra.mujoco_utils.video_utils import Video
from mujoco_infra.mujoco_utils.mujoco import physics_reset, move_center, \
    execute_action_in_curve_with_mujoco_controller_one_hand, get_current_primitive_state, set_physics_state, \
    get_env_image

env_path = "mujoco_infra/assets/rope_v3_21_links_wow.xml"
paths = [
    'exploration/outputs/evaluation/twisted_evaluation/04-05-2025_10-27/',  # TWISTED-RL-C
    'exploration/outputs/evaluation/twisted_evaluation/04-05-2025_10-29/',  # TWISTED-RL-C
]
paths = ['exploration/outputs/evaluation/twisted_evaluation/05-05-2025_21-54']
physics = mujoco.Physics.from_xml_path(env_path)
resume = True

for path in paths:
    base_paths = [x.parent for x in pathlib.Path(path).rglob('*.gpickle')]
    for base_path in tqdm(base_paths, total=len(base_paths), desc='Creating videos'):
        video = Video()
        path = os.path.join(base_path, 'log.pkl')
        graph_path = os.path.join(base_path, 'graph.gpickle')
        filename = f"trajectory_{base_path.name}{video.extension}"
        video_save_path = base_path / filename
        if resume and video_save_path.exists():
            print(f"Video {video_save_path} already exists, skipping...")
            continue

        with open(path, 'rb') as f:
            data = pickle.load(f)

        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)

        trajectory = data['trajectory'][0]
        frames = []
        cfgs = []
        dis = np.zeros(2)
        physics_reset(physics)
        success = True
        texts = []

        ep = 0
        for s_src, s_dst in tqdm(zip(trajectory[:-1], trajectory[1:]), total=len(trajectory)-1, desc='Playing actions'):
            ep += 1
            episode = graph.edges[(s_src, s_dst)]['episode']
            if episode is None:
                print(f'Found None episode in base path: {base_path}')
                success = False
                break
            experiences = [x['experience'] for x in episode[-1]]
            for step, experience in enumerate(experiences, start=1):
                configuration = get_current_primitive_state(physics)
                dis += configuration[:2]
                move_center(physics)
                action = experience.low_level_action.np_encoding
                physics, action_cfgs, _ = execute_action_in_curve_with_mujoco_controller_one_hand(
                    physics=physics,
                    action=action,
                    get_video=False,
                    return_video=False,
                    show_image=False,
                    return_render=False,
                    video=frames,
                    env_path=env_path,
                    return_cfgs=True
                )
                text = f'episode {ep} step {step}'
                for cfg in action_cfgs:
                    cfg[:2] += dis
                    cfgs.append(cfg)
                    texts.append(text)

        if success:
            physics_reset(physics)
            for cfg, text in tqdm(zip(cfgs, texts), total=len(cfgs), desc='Creating frames'):
                set_physics_state(physics, cfg)
                image = get_env_image(physics)
                # video.add_frames(frames=[image], texts=[text])
                video.add_frames(frames=[image])
            filename = f"trajectory_{base_path.name}{video.extension}"
            video.save(output_path=str(base_path), filename=filename, force=True)

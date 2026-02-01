import ruamel.yaml as ryaml
from easydict import EasyDict as edict
from typing import Union

from mujoco_infra.mujoco_utils.general_utils import edict2dict


def load_config(path: str) -> edict:
    """ Load a config file as an EasyDict """
    with open(path, 'r') as f:
        yaml = ryaml.YAML(typ='safe', pure=True)  # Load as a standard dict
        cfg = yaml.load(f)

    return edict(cfg)  # Manually convert dict to EasyDict


def dump_config(cfg: Union[edict, dict], path: str):
    """ Dump a config file """
    data = edict2dict(cfg) if isinstance(cfg, edict) else cfg
    with open(path, 'w') as f:
        yaml = ryaml.YAML()
        yaml.dump(data, f)


def fix_env_config_backwards_compatibility(env_cfg: edict) -> edict:
    """ Fix config for backwards compatibility """
    if 'curriculum_manager' in env_cfg:
        env_cfg.env.min_crosses = env_cfg.curriculum_manager.layers[0].kwargs.min_crosses
        env_cfg.env.max_crosses = env_cfg.curriculum_manager.layers[0].kwargs.max_crosses
        env_cfg.env.depth = env_cfg.curriculum_manager.layers[0].kwargs.depth
        env_cfg.env.high_level_actions = env_cfg.curriculum_manager.layers[0].kwargs.sub_layers[0].kwargs.high_level_actions

    env_cfg.env.preprocessor.name = 'Preprocessor'
    env_cfg.env.her = env_cfg.env.her if 'her' in env_cfg.env else False

    return env_cfg


def load_env_config(env_config_path: str) -> edict:
    """ Load the environment config file """
    env_cfg = load_config(env_config_path)
    env_cfg = fix_env_config_backwards_compatibility(env_cfg)
    return env_cfg


def set_agents_by_ablation(cfg: edict, ablation: str) -> dict:
    if ablation in ['A', 'G', 'C', 'AC']:
        if ablation == 'A':
            problems = ['G_R1', 'G_R2', 'G_Cross']
        elif ablation == 'G':
            problems = ['G']
        elif ablation == 'C':
            problems = ['G0', 'G1', 'G2']
        elif ablation == 'AC':
            problems = ['G0_R1', 'G0_R2', 'G1_R1', 'G1_R2', 'G1_Cross', 'G2_R1', 'G2_R2', 'G2_Cross']
        agents = {problem: path for problem, path in cfg.RL.agents.items() if problem in problems}
    elif ablation == 'H':
        agents = {problem: path for problem, path in cfg.RL.variants.h.items()}
    elif ablation == 'M':
        agents = {problem: path for problem, path in cfg.RL.variants.m.items()}
        cfg.RL.max_steps = 1
    elif ablation == 'ALL':
        agents = {problem: path for problem, path in cfg.RL.agents.items()}
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    print(f'agents:\n{agents}')
    cfg.RL.agents = agents

    return agents

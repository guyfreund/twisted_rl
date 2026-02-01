import os
import sys
import shutil
import traceback
from datetime import datetime
from functools import partial
from itertools import product
from pytorch_lightning import seed_everything

sys.path.append('.')

from exploration.mdp.graph.problem_set import ProblemSet
from exploration.reachable_configurations.reachable_configurations import ReachableConfigurations
from exploration.reachable_configurations.reachable_configurations_factory import ReachableConfigurationsFactory
from exploration.rl.cleanrl_scripts.sac_algorithm import SACAlgorithm
from exploration.rl.test_scripts.random_evaluation_analysis import run_analysis, view_results
from exploration.rl.environment.exploration_gym_envs import ExplorationGymEnvs, plot_collection_summary
from exploration.utils.config_utils import load_config, fix_env_config_backwards_compatibility
from exploration.utils.server_utils import SERVER_TO_PREFIX


def get_metadata(model_dir: str, server: str) -> (str, str, str, str):
    model_training_dir = model_dir.strip('/').replace('evaluation', 'training')
    model_evaluation_dir = model_dir.strip('/').replace('training', 'evaluation')
    mode = 'training' if 'training' in model_dir else 'evaluation'
    assert model_dir.startswith(model_evaluation_dir) or model_dir.startswith(model_training_dir)
    base_dir = SERVER_TO_PREFIX[server]
    model_dir = os.path.join(base_dir, model_dir.strip('/'))
    model_training_dir = os.path.join(base_dir, model_training_dir)
    model_evaluation_dir = os.path.join(base_dir, model_evaluation_dir)

    algo_name = model_dir.strip('/').split(f'twisted_rl/exploration/outputs/{mode}/')[1].split('/')[0]
    config_path = os.path.join(model_training_dir, 'config.yml')
    assert os.path.exists(config_path), f'Config file {config_path} does not exist'
    problem_set = ProblemSet()
    config = load_config(config_path)
    config.env = fix_env_config_backwards_compatibility(config.env)
    min_crosses = config.env.env.min_crosses
    max_crosses = config.env.env.max_crosses
    depth = config.env.env.depth
    high_level_actions = config.env.env.high_level_actions
    problem = problem_set.get_problem_by_kwargs(**{
        'min_crosses': min_crosses,
        'max_crosses': max_crosses,
        'depth': depth,
        'high_level_actions': high_level_actions,
    })
    assert problem is not None, f'Problem with min_crosses={min_crosses}, max_crosses={max_crosses}, depth={depth}, high_level_actions={high_level_actions} does not exist'
    return model_training_dir, model_evaluation_dir, algo_name, problem


def run_random_evaluation(model_dir, iterations_per_model, sample_size, num_cpus, save_videos, raise_exception,
                          initial_states_from_training, initial_states_from_dir, initial_states_path, deterministic_list, her_list,
                          plot, save_table, save_plot, show, algo_name, problem, resume):
    if save_videos:
        os.environ['MUJOCO_GL'] = 'egl'

    assert os.path.exists(model_dir), f'Model directory {model_dir} does not exist'
    best_model_paths = []
    mid_all_model_paths = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith('best_model_') and 'collection' not in f])
    versions = [int(f.split('_')[-1]) for f in mid_all_model_paths]
    min_version = 4
    all_model_paths = [os.path.join(model_dir, 'best_model')] + [mid_all_model_paths[i] for i in range(len(versions)) if versions[i] > min_version]
    mid_all_model_collection_paths = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith('best_model_collection_')])
    versions = [int(f.split('_')[-1]) for f in mid_all_model_collection_paths]
    all_model_collection_paths = [os.path.join(model_dir, 'best_model_collection')] + [mid_all_model_collection_paths[i] for i in range(len(versions)) if versions[i] > min_version]
    # best_model_paths.extend(all_model_paths)
    best_model_paths.extend(all_model_collection_paths)

    algo_suffix = model_dir.split('twisted_rl/exploration/outputs/training/')[1]
    evaluation_output_dir = os.path.join('exploration/outputs/evaluation', algo_suffix)
    os.makedirs(evaluation_output_dir, exist_ok=True)

    failed_models = []
    cfg_path = os.path.join(model_dir, 'config.yml')
    train_cfg = load_config(cfg_path)
    env_cfg = fix_env_config_backwards_compatibility(train_cfg.env)
    min_crosses = env_cfg.env.min_crosses
    max_crosses = env_cfg.env.max_crosses

    # create reachable_configurations from replay buffer
    if initial_states_from_training:
        reachable_configurations = ReachableConfigurations.from_replay_buffer_files(
            replay_buffer_files_path=[os.path.join(model_dir, 'replay_buffer_files')],
            num_cpus=num_cpus,
            min_crosses=min_crosses,
            max_crosses=max_crosses,
            sample_size=sample_size
        )
    else:
        if initial_states_from_dir:
            model_dir = env_cfg.env.reachable_configurations.replay_buffer_files_path
        elif initial_states_path is not None:
            model_dir = initial_states_path
        else:
            raise NotImplementedError('Initial states configuration is not provided')

        reachable_configurations = ReachableConfigurationsFactory.get_cls(env_cfg.env.reachable_configurations.name).from_replay_buffer_files(
            replay_buffer_files_path=[os.path.join(model_dir, 'replay_buffer_files')],
            num_cpus=num_cpus,
            min_crosses=min_crosses,
            max_crosses=max_crosses,
            sample_size=sample_size
        )

    # create env
    seed_everything(env_cfg.seed)
    env_cfg.num_cpus = num_cpus
    env = ExplorationGymEnvs.from_cfg(
        cfg=env_cfg,
        create_reachable_configurations=False,
        reachable_configurations=reachable_configurations
    )

    try:
        total_runs = len(best_model_paths) * len(deterministic_list) * len(her_list)
        for run_idx, (best_model_path, deterministic, apply_her) in enumerate(product(best_model_paths, deterministic_list, her_list)):
            deterministic_str = 'deterministic' if deterministic else 'stochastic'
            her_str = 'inference_her' if apply_her else 'no_inference_her'
            best_model_str = os.path.basename(best_model_path)
            print(f'========= Evaluating {run_idx + 1}/{total_runs} - {best_model_str} {deterministic_str} {her_str} =========')
            model_evaluation_output_dir = os.path.join(evaluation_output_dir, best_model_str, deterministic_str, her_str)
            if resume:
                files = [
                    os.path.join(model_evaluation_output_dir, 'goal_states.csv'),
                    os.path.join(model_evaluation_output_dir, 'goal_actions.csv'),
                    os.path.join(model_evaluation_output_dir, 'summary.txt'),
                ]
                if all([os.path.exists(f) for f in files]):
                    print(f'Skipping {best_model_str} {deterministic_str} {her_str} agent evaluation')
                    continue
            else:
                if os.path.exists(model_evaluation_output_dir):
                    shutil.rmtree(model_evaluation_output_dir)
            os.makedirs(model_evaluation_output_dir, exist_ok=True)
            exceptions_dir = os.path.join(model_evaluation_output_dir, 'exceptions')
            os.makedirs(exceptions_dir, exist_ok=True)

            env.output_dir = model_evaluation_output_dir
            env.exceptions_dir = exceptions_dir
            env.save_videos = save_videos
            env.create_file_system_dirs()

            # handle agent stuff
            agent = SACAlgorithm(config=train_cfg.algorithm, env=env)
            agent.load(best_model_path)
            agent.eval()
            get_actions = partial(agent.predict_action, deterministic=deterministic, epsilon=None)

            # run evaluation
            try:
                print(f'====================== {best_model_str} {deterministic_str} {her_str} agent evaluation ======================')
                start_time = datetime.now()
                episodes, episode_times, is_her_list = env.play_test_queries(
                    get_actions=get_actions,
                    iterations=iterations_per_model,
                    apply_her=apply_her,
                )
                end_time = datetime.now()

                plot_collection_summary(
                    episodes=episodes,
                    episode_times=episode_times,
                    is_her_list=is_her_list,
                    start_time=start_time,
                    end_time=end_time,
                    save_dir=model_evaluation_output_dir,
                )

                run_analysis(
                    episodes=episodes,
                    episode_times=episode_times,
                    is_her_list=is_her_list,
                    model_evaluation_output_dir=model_evaluation_output_dir,
                    her=apply_her,
                    plot=plot,
                    save_table=save_table,
                    save_plot=save_plot,
                    show=show,
                    deterministic_str=deterministic_str,
                    algo_name=algo_name,
                    problem=problem,
                    model_name=best_model_str,
                    to_print=True,
                    log_to_wandb=False,
                )

            except Exception as e:
                traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
                failed_models.append((best_model_str, deterministic_str, her_str, traceback_str))
                if raise_exception:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    raise e

        if failed_models:
            print('====================== Failed models ======================')
            for best_model_str, deterministic_str, her_str, traceback_str in failed_models:
                print(f'====================== {best_model_str} {deterministic_str} {her_str} agent evaluation ======================')
                print(''.join(traceback_str))
            raise Exception('Some models failed to evaluate')
    finally:
        env.close()


def main(model_dir):
    model_training_dir, model_evaluation_dir, algo_name, problem = get_metadata(model_dir=model_dir, server=server)

    if run_evaluation:
        run_random_evaluation(
            model_dir=model_training_dir,
            iterations_per_model=iterations_per_model,
            sample_size=sample_size,
            num_cpus=num_cpus,
            save_videos=save_videos,
            raise_exception=raise_exception,
            initial_states_from_training=initial_states_from_training,
            initial_states_from_dir=initial_states_from_dir,
            initial_states_path=initial_states_path,
            deterministic_list=deterministic_list,
            her_list=her_list,
            plot=plot,
            save_table=save_table,
            save_plot=save_plot,
            show=show,
            algo_name=algo_name,
            problem=problem,
            resume=resume,
        )
    if show_results:
        view_results(
            model_dir=model_evaluation_dir,
            plot=plot,
            save_table=save_table,
            save_plot=save_plot,
            show=show,
            print_success_only=print_success_only,
            algo_name=algo_name,
            problem_name=problem.name,
        )


if __name__ == '__main__':
    # ======================== Evaluation parameters ========================
    # =============== Global parameters ===============
    run_evaluation = True
    show_results = False
    server = 'server_new'

    # =============== Shared parameters ===============
    # train_dir should start with 'twisted_rl/exploration/outputs/training/' or 'twisted_rl/exploration/outputs/evaluation/'
    train_dir = 'twisted_rl/exploration/outputs/training/ExplorationSAC/11-04-2025_11-19'
    # train_dir = 'twisted_rl/exploration/outputs/training/ExplorationSAC/06-04-2025_09-39'
    plot = True
    save_table = True
    save_plot = True
    show = False
    skip_actions = ['R2']

    # =============== Show Results parameters ===============
    print_success_only = True

    # =============== Run Evaluation parameters ===============
    resume = True
    initial_states_from_training = False
    initial_states_from_dir = False
    initial_states_path = '/home/guyfreund/twisted_rl/exploration/outputs/training/ExplorationSAC/G2_R1/LR_1e-05_QLR_1e-05_GC_1.0_QGC_1.0_ES_ExplorationSchedule_TS_False_AHL_4_CHL_4_UPDT_2_BQV_OFF_ISS_RND_PP_POS/28-03-2025_15-04'
    deterministic_list = [False]
    her_list = [False]

    iterations_per_model = 100
    sample_size = 500000
    num_cpus = 60
    save_videos = False
    raise_exception = False
    # ======================== Evaluation parameters ========================

    for problem_name in os.listdir(os.path.join(SERVER_TO_PREFIX[server], train_dir)):
        if any(problem_name.endswith(skip_action) for skip_action in skip_actions):
            continue
        print(f'Running evaluation for model at path {os.path.join(SERVER_TO_PREFIX[server], train_dir, problem_name)}')
        main(model_dir=os.path.join(train_dir, problem_name))

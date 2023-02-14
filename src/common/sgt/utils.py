import numpy as np
import gym
from stable_baselines3.common.monitor import get_monitor_files


def make_ipd_env(
    env_name: str, seed: int = 0,
    one_hot: bool = True, sparse: bool = False
):
    from m3g.examples.games.wrappers import (
        OneHotObservationWrapper, SparseRewardWrapper, 
        PerformanceSparseRewardWrapper, TotalSparseRewardWrapper
    )
    env = gym.make(env_name)
    if sparse:
        env = SparseRewardWrapper(env)
    if one_hot:
        env = OneHotObservationWrapper(env)
    env.seed(seed)

    return env


def dual_evaluate(
    env: gym.Env,
    model,
    num_episodes: int = 100,
    deterministic=True,
):

    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_agent_rewards = []
    all_episode_nature_rewards = []
    for i in range(num_episodes):
        episode_agent_rewards = []
        episode_nature_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_agent_rewards.append(reward)
            episode_nature_rewards.append(info['nature_reward'])

        all_episode_agent_rewards.append(sum(episode_agent_rewards))
        all_episode_nature_rewards.append(sum(episode_nature_rewards))

    agent_mean_episode_reward = np.mean(all_episode_agent_rewards)
    nature_mean_episode_reward = np.mean(all_episode_nature_rewards)

    return agent_mean_episode_reward, nature_mean_episode_reward


def load_results_sb3(path: str, stage: str = 'train'):
    import os
    import json
    import pandas
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: the logged data
    """

    def get_algorithm_name(file_path):
        _, root_dir = os.path.split(os.path.dirname(file_path))
        return root_dir.split('-')[0]

    def get_run_seed(file_path):
        file_name = os.path.basename(file_path)
        return int(file_name.split('.')[0])
    
    def get_stage_name(file_path):
        file_name = os.path.basename(file_path)
        return file_name.split('.')[1]
    
    def get_run_name(file_path):
        dirname = os.path.dirname(file_path)
        return dirname.split('/')[-1].split('-')[2]

    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise Exception("no monitor files")
    data_frames = []

    for file_path in monitor_files:
        # if True:
        if stage in file_path:
            alg = get_algorithm_name(file_path)
            seed = get_run_seed(file_path)
            stage = get_stage_name(file_path)
            run_env = get_run_name(file_path)
            with open(file_path) as file_handler:
                first_line = file_handler.readline()
                assert first_line[0] == "#"
                header = json.loads(first_line[1:])
                data_frame = pandas.read_csv(file_handler, index_col=None)
                data_frame['ep'] = np.arange(0, len(data_frame))
                data_frame['seed'] = np.repeat(seed, len(data_frame))
                data_frame['alg'] = np.repeat(alg, len(data_frame))
                data_frame['env'] = np.repeat(header['env_id'], len(data_frame))
                data_frame['run-env'] = np.repeat(get_run_name(file_path), len(data_frame))
                data_frame['stage'] = np.repeat(stage, len(data_frame))
            data_frames.append(data_frame)
    return pandas.concat(data_frames)

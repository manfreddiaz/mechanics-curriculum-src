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


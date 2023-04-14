from typing import Iterable
import numpy as np
import gym


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class OneHotObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(env.observation_space.n,))

    def observation(self, observation):
        if observation is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[observation] = 1.0

        return obs


class SparseRewardWrapper(gym.Wrapper):

    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, **kwargs):
        self.agent_acc_rwd = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.agent_acc_rwd += reward
        info['agent_acc_reward'] = self.agent_acc_rwd
        if done:
            if self.agent_acc_rwd > info['nature_acc_reward']:
                reward = 1.0
            elif self.agent_acc_rwd == info['nature_acc_reward']:
                reward = 0.0
            else:
                reward = -1.0
            info['opponent'] = self.unwrapped.nature_strategy.name
            self.agent_acc_rwd = 0.0  # clean interactions
        else:
            reward = 0.0

        return obs, reward, done, info

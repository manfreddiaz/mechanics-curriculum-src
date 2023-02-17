from turtle import shape
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
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[observation] = 1.0
        return obs


class SparseRewardWrapper(gym.Wrapper):

    def __init__(self, env) -> None:
        super().__init__(env)
        self.agent_acc_rwd = 0

    def reset(self):
        self.agent_acc_rwd = 0
        return super().reset()

    def step(self, action):
        next_state, reward, done, info = super().step(action)
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
        else:
            reward = 0.0

        return next_state, reward, done, info


class PerformanceSparseRewardWrapper(SparseRewardWrapper):

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if done:
            reward = self.agent_acc_rwd - info['nature_acc_reward']
            reward = sigmoid(reward)
        else:
            reward = 0.0

        return next_state, reward, done, info


class TotalSparseRewardWrapper(SparseRewardWrapper):

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if done:
            reward = self.agent_acc_rwd / 200.0
        else:
            reward = 0.0

        return next_state, reward, done, info


class DifferenceRewardWrapper(gym.Wrapper):

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        reward = info['nature_reward'] + reward
        return next_state, reward, done, info


class ActionAsObservationWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * env.observation_space.shape[0],)
        )
        self.last_action = None

    def reset(self):
        observation = super().reset()

        return np.concatenate([
            observation, np.zeros(self.observation_space.shape)])

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        last_action = np.zeros(self.action_space.n)

        return np.concatenate([next_state, last_action]), reward, done, info

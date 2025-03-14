from typing import Dict, Tuple, Union

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

from learned.core import MetaTrainingEnvironment


class JointActionObservationWrapper(gym.Wrapper):

    def __init__(
        self,
        env: MetaTrainingEnvironment,
    ):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            high=1.0, low=-1.0, shape=[self.action_space.n * 2])

    def reset(self, **kwargs):
        """Reset semantics: None indicates that no training has been made.
        It is the indentifier of initialization

        Returns:
            _type_: _description_
        """
        super().reset(**kwargs)
        return np.zeros(self.observation_space.shape)

    def step(self, actions) -> Tuple[int, float, bool, dict]:
        obs, reward, done, info = super().step(actions)
        # NOTE: we store the original info (most likely the agent)
        info['o_obs'] = obs

        # obs = []
        # for action in actions:
        #     one_hot = np.zeros(
        #         self.action_space.n, dtype=np.float32)
        #     one_hot[action] = 1.0
        #     obs.append(one_hot)
        # obs = np.concatenate(obs, axis=-1)
        obs = np.zeros(self.observation_space.shape)

        return obs, reward, done, info


class FixedEvaluatorWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, eval_action: int = 0):
        super().__init__(env)
        self._eval_action = eval_action

    def step(self, action):
        action = [action, self._eval_action]
        
        return super().step(action)


class RewardLearningProgressionWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reward = None

    def reset(self, **kwargs):
        self._reward = 0.0
        self._agent_stats = 0.0
        self._steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._agent_stats += reward
        self._steps += 1
        info['agent_stats'] = self._agent_stats / self._steps
        info['cf_agent_stats'] = 0.0

        next_reward = reward - self._reward
        self._reward = reward

        return obs, next_reward, done, info


class TimeFeatureWrapper(gym.Wrapper):
    """
    From Stable Baselines 3 Contrib
    Add remaining, normalized time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    .. note::

        Only ``gym.spaces.Box`` and ``gym.spaces.Dict`` (``gym.GoalEnv``) 1D observation spaces
        are supported for now.

    :param env: Gym env to wrap.
    :param max_steps: Max number of steps of an episode
        if it is not wrapped in a ``TimeLimit`` object.
    :param test_mode: In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env: gym.Env, max_steps: int = 1000, test_mode: bool = False):
        assert isinstance(
            env.observation_space, (spaces.Box, spaces.Dict)
        ), "`TimeFeatureWrapper` only supports `gym.spaces.Box` and `spaces.Dict` (`gym.GoalEnv`) observation spaces."

        # Add a time feature to the observation
        if isinstance(env.observation_space, spaces.Dict):
            assert "observation" in env.observation_space.spaces, "No `observation` key in the observation space"
            obs_space = env.observation_space.spaces["observation"]
            assert isinstance(obs_space, spaces.Box), "`TimeFeatureWrapper` only supports `gym.spaces.Box` observation space."
            obs_space = env.observation_space.spaces["observation"]
        else:
            obs_space = env.observation_space

        assert len(obs_space.shape) == 1, "Only 1D observation spaces are supported"

        low, high = obs_space.low, obs_space.high
        low, high = np.concatenate((low, [0.0])), np.concatenate((high, [1.0]))
        self.dtype = obs_space.dtype

        if isinstance(env.observation_space, spaces.Dict):
            env.observation_space.spaces["observation"] = spaces.Box(low=low, high=high, dtype=self.dtype)
        else:
            env.observation_space = spaces.Box(low=low, high=high, dtype=self.dtype)

        super().__init__(env)

        # Try to infer the max number of steps per episode
        try:
            self._max_steps = env.spec.max_episode_steps
        except AttributeError:
            self._max_steps = None

        # Fallback to provided value
        if self._max_steps is None:
            self._max_steps = max_steps

        self._current_step = 0
        self._test_mode = test_mode

    def reset(self) -> GymObs:
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Concatenate the time feature to the current observation.

        :param obs:
        :return:
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        time_feature = np.array(time_feature, dtype=self.dtype)

        if isinstance(obs, dict):
            obs["observation"] = np.append(obs["observation"], time_feature)
            return obs
        return np.append(obs, time_feature)
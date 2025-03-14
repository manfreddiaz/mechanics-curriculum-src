import numpy as np
import gym
import gym.spaces as spaces


class MinAtarStandardObservation(gym.ObservationWrapper):

    def __init__(self, env) -> None:
        super().__init__(env)
        self._shape = list(reversed(env.observation_space.shape))
        self._shape[0] = 10
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=self._shape,
            dtype=np.float32
        )

    def _pad_pattern(self, shape):
        return (0, self._shape[0] - shape[0]), (0, 0), (0, 0)

    def observation(self, observation: np.array):
        obs = observation.astype(np.uint8) * 255.0
        obs = obs.transpose(2, 0, 1)
        return np.pad(
            obs, self._pad_pattern(obs.shape), 'constant', constant_values=0)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(reward)
from typing import Tuple
import gym

from learned.core import MetaTrainingEnvironment


class MetaActionIdentificationWrapper(gym.Wrapper):

    def __init__(
        self,
        env: MetaTrainingEnvironment,
        meta_player_id: int
    ):
        super().__init__(env)
        assert self.env.action_space.n > meta_player_id
        self._meta_player_id = meta_player_id

        self.observation_space = gym.spaces.Discrete(self.action_space.n)

    def reset(self, **kwargs):
        """Reset semantics: None indicates that no training has been made.
        It is the indentifier of initialization

        Returns:
            _type_: _description_
        """
        super().reset(**kwargs)
        return None

    def step(self, action) -> Tuple[int, float, bool, dict]:
        obs, reward, done, info = super().step(action)

        return action[self._meta_player_id], reward, done, info

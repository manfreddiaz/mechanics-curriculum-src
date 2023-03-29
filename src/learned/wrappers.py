from typing import Tuple
import gym

from learned.core import MetaTrainingEnvironment


class JointActionObservationWrapper(gym.Wrapper):

    def __init__(
        self,
        env: MetaTrainingEnvironment,
    ):
        super().__init__(env)

        self.observation_space = gym.spaces.Discrete(self.action_space.n ** 2)

    def reset(self, **kwargs):
        """Reset semantics: None indicates that no training has been made.
        It is the indentifier of initialization

        Returns:
            _type_: _description_
        """
        super().reset(**kwargs)
        return None

    def step(self, action) -> Tuple[int, float, bool, dict]:
        _, reward, done, info = super().step(action)
        trainer_action, evaluator_action = action
        observation_encoding = trainer_action * self.action_space.n + evaluator_action 
        return observation_encoding, reward, done, info


class FixMetaEvaluatorAction(gym.ActionWrapper):

    def __init__(
        self, 
        env: gym.Env,
        evaluator_action: int
    ):
        super().__init__(env)
        assert evaluator_action < self.action_space.n
        self._evaluator_action = evaluator_action

    def action(self, action):
        return [action, self._evaluator_action]
        # return super().action(action)

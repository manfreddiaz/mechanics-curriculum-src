from typing import Callable, List, Optional, Tuple

import gym
from ccgm.common.algs.core import Agent, AlgorithmPlayFn, AlgorithmOptimizeFn


class DiscreteMetaEnvironment():
    """
        Holds a discrete indexable collection of environments
    """
    pass


class MetaTrainingEnvironment(gym.Env[Agent, int]):

    def __init__(
        self,
        agent_fn: Callable[[gym.vector.VectorEnv], Agent],
        env_fn: Callable[..., List[gym.vector.VectorEnv]],
        alg_fn: Callable[..., Tuple[AlgorithmPlayFn, AlgorithmOptimizeFn]],
        eval_fn: Callable[[Agent], float]
    ) -> None:
        # factories
        self._env_fn = env_fn
        self._agent_fn = agent_fn
        self._alg_fn = alg_fn
        # state
        self._learning_step: int = None
        self._agent: Agent = None
        self._alg_play_fn, self._alg_optim_fn = alg_fn()
        # environments
        self._train_envs = env_fn()
        self._need_reset = [True] * len(self._train_envs) 
        self._eval_envs = env_fn()

        # evaluation function
        self._eval_fn = eval_fn

        self.action_space = gym.spaces.Discrete(len(self._train_envs))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Agent:

        # TODO: there is a better approach to this
        # we can just reset the agent parameters
        # and optimizer. The current approach will lead to
        # high memory consumption and potential leaks
        # if 'reset_agent' in options:
        self._agent = self._agent_fn(self._train_envs[0])
        self._learning_step = 0

        super().reset(
            seed=seed, return_info=return_info,
            options=options
        )
        return self._agent

    def step(self, action: Tuple[int]) -> Tuple[Agent, float, bool, dict]:

        info = dict()

        trainer_action, evaluator_action = action
        # NOTE: if envs use the `AutoResetWrapper`, it doesn't matter
        # if the episode (interaction with task) ends in the middle
        # of a play, `dones` in the RBs capture termination
        # and so do the algorithms? DQN explicitly does it.
        if self._agent.memory.full:
            self._agent.memory.reset()

        if self._need_reset[trainer_action]:
            rb = self._agent.memory
            rb.observations[rb.pos] = self._train_envs[trainer_action].reset()  # noqa
            self._need_reset[trainer_action] = False

        play_steps = self._alg_play_fn(
            self._agent,
            envs=self._train_envs[trainer_action],
            global_step=self._learning_step
        )
        self._learning_step += play_steps

        optim_steps = self._alg_optim_fn(
            self._agent,
            envs=self._train_envs[trainer_action],
            global_step=self._learning_step
        )
        # TODO: should include number of internal steps?

        reward, _ = self._eval_fn(
            self._agent,
            envs=self._eval_envs[evaluator_action]
        )

        info['meta_optimized'] = optim_steps > 0
        info['optim_steps'] = optim_steps
        info['play_steps'] = play_steps

        return self._agent, reward, False, info

    def seed(self, seed=None):
        for env in self._train_envs:
            env.seed(seed)
        for env in self._eval_envs:
            env.seed(seed)

        return super().seed(seed)

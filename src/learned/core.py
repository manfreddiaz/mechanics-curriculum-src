from typing import Callable, List, Optional, Tuple

import gym
import numpy as np
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
        _, _, self._alg_play_fn, _ = alg_fn()
        self._learning_step = 0
        # environments
        self._train_envs = env_fn()
        self._need_reset = [True] * len(self._train_envs)
        self._eval_envs = env_fn()

        # evaluation function
        self._eval_fn = eval_fn
        self._agent = self._agent_fn(self._train_envs[0])

        # action space interpretation
        # 0 - n train on tasks
        # n + 1 reset training agent
        # n + 2 noop
        self.action_space = gym.spaces.Discrete(len(self._train_envs))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[Agent, Agent]:

        # TODO: there is a better approach to this
        # we can just reset the agent parameters
        # and optimizer. The current approach will lead to
        # high memory consumption and potential leaks
        # if 'reset_agent' in options:

        super().reset(
            seed=seed, return_info=return_info,
            options=options
        )
        return [self._agent, self._agent]

    def step(self, action: Tuple[int]) -> Tuple[
            Tuple[Agent, Agent], Tuple[float, float], bool, dict]:

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

        play_steps, optim_steps = self._alg_play_fn(
            self._agent,
            envs=self._train_envs[trainer_action],
            global_step=self._learning_step
        )
        self._learning_step += play_steps

        reward, _ = self._eval_fn(
            self._agent,
            envs=self._eval_envs[evaluator_action]
        )

        info['meta_optimized'] = optim_steps > 0
        info['optim_steps'] = optim_steps
        info['play_steps'] = play_steps
        info['train_action'] = trainer_action
        info['eval_action'] = evaluator_action
        info['train_env'] = self._train_envs[trainer_action]
        info['eval_env'] = self._eval_envs[evaluator_action]
        
        return self._agent, reward, False, info

    def seed(self, seed=None):
        for env in self._train_envs:
            env.seed(seed)
        for env in self._eval_envs:
            env.seed(seed)

        return super().seed(seed)


class CounterfactualMetaTrainingEnvironment(MetaTrainingEnvironment):
    def __init__(
        self, 
        agent_fn: Callable[[gym.vector.VectorEnv], Agent], 
        env_fn: Callable[..., List[gym.vector.VectorEnv]], 
        alg_fn: Callable[..., Tuple[AlgorithmPlayFn, AlgorithmOptimizeFn]], 
        eval_fn: Callable[[Agent], float]
    ) -> None:
        super().__init__(agent_fn, env_fn, alg_fn, eval_fn)
        self._cf_agent = self._agent_fn(self._train_envs[0])
        self._cf_agent.copy(self._agent)
        self._cfa_learning_step = 0
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        return_info: bool = False, 
        options: Optional[dict] = None
    ) -> Tuple[Agent, Agent]:
        super().reset(
            seed=seed, return_info=return_info, options=options)
        return self._agent, self._cf_agent
    

    def step(self, action: Tuple[int]) -> Tuple[Tuple[Agent, Agent], Tuple[float, float], bool, dict]:
        info = dict()

        trainer_action, evaluator_action = action
        # NOTE: if envs use the `AutoResetWrapper`, it doesn't matter
        # if the episode (interaction with task) ends in the middle
        # of a play, `dones` in the RBs capture termination
        # and so do the algorithms? DQN explicitly does it.
        if self._agent.memory.full:
            self._agent.memory.reset()
        
        if self._cf_agent.memory.full:
            self._agent.memory.reset()

        if self._need_reset[trainer_action]:
            rb = self._agent.memory
            rb.observations[rb.pos] = self._train_envs[trainer_action].reset()  # noqa
            if trainer_action == evaluator_action:
                c_rb = self._cf_agent.memory
                c_rb.observations[c_rb.pos] = rb.observations[rb.pos]
            self._need_reset[trainer_action] = False
        # what if the two actions are the same.
        if self._need_reset[evaluator_action]:
            rb = self._cf_agent.memory
            rb.observations[rb.pos] = self._train_envs[evaluator_action].reset()  # noqa
            self._need_reset[evaluator_action]

        play_steps, optim_steps = self._alg_play_fn(
            self._agent,
            envs=self._train_envs[trainer_action],
            global_step=self._learning_step
        )
        self._learning_step += play_steps

        play_steps, optim_steps = self._alg_play_fn(
            self._cf_agent,
            envs=self._train_envs[evaluator_action],
            global_step = self._cfa_learning_step
        )
        self._cfa_learning_step += play_steps

        reward, _ = self._eval_fn(
            self._agent,
            envs=self._train_envs[evaluator_action]
        )
        cf_reward, _ = self._eval_fn(
            self._cf_agent,
            envs=self._train_envs[evaluator_action]
        )

        info['meta_optimized'] = optim_steps > 0
        info['optim_steps'] = optim_steps
        info['play_steps'] = play_steps
        info['train_action'] = trainer_action
        info['eval_action'] = evaluator_action
        info['train_env'] = self._train_envs[trainer_action]
        info['eval_env'] = self._eval_envs[evaluator_action]
        info['reward'] = reward
        info['cf_reward'] = cf_reward
        
        return [self._agent, self._cf_agent], [reward, cf_reward], False, info


class CounterfactualSelfPlayWrapper(gym.Wrapper):

    def __init__(
        self, 
        env: CounterfactualMetaTrainingEnvironment,
        learning_progression: bool = False
    ):
        super().__init__(env)
        self._learning_progression = learning_progression
        if learning_progression:
            self._agent_last_reward = 0.0
            self._cf_last_reward = 0.0


    def reset(self, **kwargs):
        self._agent_returns = 0.0
        self._cf_agent_returns = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, rewards, done, info = super().step(action)

        reward, cf_reward = rewards

        # TODO: exponential weighted moving average
        self._agent_returns += reward
        self._cf_agent_returns += cf_reward

        info['agent_returns'] = self._agent_returns
        info['cfr_agent_returns'] = self._cf_agent_returns

        if self._learning_progression:
            agent_lp = reward - self._agent_last_reward
            cf_agent_lp = cf_reward - self._cf_last_reward

            self._agent_last_reward = reward
            self._cf_last_reward = cf_reward

            rewards = [agent_lp, cf_agent_lp]


        if done:
            agent, cf_agent = obs
            if self._agent_returns >= self._cf_agent_returns:
                cf_agent.copy(agent)
                # NOTE: if learning progression is enabled, then switch
                # the last reward after the copy
                if self._learning_progression:
                    self._cf_last_reward = self._agent_last_reward
            else:
                agent.copy(cf_agent)
                # NOTE: if learning progression is enabled, then switch
                # the last reward after the copy
                if self._learning_progression:
                    self._agent_last_reward = self._cf_last_reward
                    
            self._agent_returns = self._cf_agent_returns = 0.0
        
        return obs, rewards, done, info


class CounterfactualRewardWrapper(gym.Wrapper):

    def __init__(self, env: CounterfactualMetaTrainingEnvironment):
        super().__init__(env)

    
    def reset(self, **kwargs):
        self._agent_stats = 0.0
        self._cf_agent_stats = 0.0
        return super().reset(**kwargs)

    
    def step(self, action):
        obs, rewards, done, info = super().step(action)
        self._agent_stats += info['reward']
        self._cf_agent_stats += info['cf_reward']
        
        info['agent_stats'] = self._agent_stats
        info['cf_agent_stats'] = self._cf_agent_stats

        reward, cf_reward = rewards
        return obs, reward - cf_reward, done, info
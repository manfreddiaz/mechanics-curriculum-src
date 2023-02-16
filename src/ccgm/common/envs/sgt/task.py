from typing import Callable, List, Union
import numpy as np
import gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO

from ccgm.common.coalitions import Coalition, OrderedCoalition

# from ccgm.common.envs.sgt.impl.prisioner_dilemma import pd_nature_strategy_factory, pd_principal_strategy_factory
from .impl.prisioner_dilemma import SPID_STRATEGIES
from ccgm.common.games import CooperativeMetaGame

from .strategies import NatureMemoryOneStrategy, PrincipalMarkovStrategy
from .wrappers import SparseRewardWrapper, OneHotObservationWrapper


class SequentialMatrixGameEnvironment(gym.Env):
    def __init__(
        self,
        nature_strategy: Union[NatureMemoryOneStrategy, str],
        principal_strategy: Union[PrincipalMarkovStrategy, str],
        nature_strategy_factory: Callable[
            [str], NatureMemoryOneStrategy] = None,
        principal_strategy_factory: Callable[
            [str], PrincipalMarkovStrategy] = None,
        with_memory: bool = False
    ) -> None:

        super().__init__()

        self.np_random = np.random.RandomState(0)
        self.with_memory = with_memory
        self.acc_rewards = 0.0

        if self.with_memory:
            self.memory = None
        self._last_obs = None

        if type(nature_strategy) == str:
            assert nature_strategy_factory is not None
            self.nature_strategy = nature_strategy_factory(nature_strategy)
        elif type(nature_strategy) == NatureMemoryOneStrategy:
            self.nature_strategy = nature_strategy
        else:
            raise NotImplementedError()

        if type(principal_strategy) == str:
            assert principal_strategy_factory is not None
            self.principal_strategy = principal_strategy_factory(
                principal_strategy)
        elif type(principal_strategy) == PrincipalMarkovStrategy:
            self.principal_strategy = principal_strategy
        else:
            raise NotImplementedError()

        self.action_space = gym.spaces.Discrete(
            self.nature_strategy.action_dim)
        self.observation_space = gym.spaces.Discrete(
            self.nature_strategy.action_dim
        )

    def seed(self, seed: int):
        self.np_random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.nature_strategy.seed(seed)
        return

    def reset(self, seed=None, options=None):
        if self.with_memory and self.memory is not None:
            self._last_obs = self.nature_strategy(*self.memory)
        else:
            self._last_obs = self.nature_strategy.first()
        self.acc_rewards = 0.0
        return self._last_obs

    def step(self, action):
        if self.with_memory:
            self.memory = (self._last_obs, action)
        next_obs = self.nature_strategy(self._last_obs, action)
        nature_reward, agent_reward = self.principal_strategy(
            self._last_obs, action)
        self.acc_rewards += nature_reward

        self._last_obs = next_obs
        done = False
        info = dict({
            'nature_reward': nature_reward,
            'nature_acc_reward': self.acc_rewards
        })
        return next_obs, agent_reward, done, info


def make_matrix_env(strategy_id, episode_time_limit, sparse, one_hot):
    env = gym.make(strategy_id)
    env = TimeLimit(
        env,
        episode_time_limit
    )
    if sparse:
        env = SparseRewardWrapper(env)
    
    if one_hot:
        env = OneHotObservationWrapper(env)
    
    return env


def make_coalition(
    team, 
    ordered, 
    episode_time_limit, 
    total_time_limit, 
    probs = None,
    sparse: bool = True,
    one_hot: bool = True
):

    if not ordered:
        return Coalition(
            strategies=[
                make_matrix_env(
                    strategy_id=strategy_id,
                    episode_time_limit=episode_time_limit,
                    sparse=sparse,
                    one_hot=one_hot
                ) for strategy_id in team
            ],
            probs=probs
        )
    else:
        return OrderedCoalition(
            strategies=[
                make_matrix_env(
                    strategy_id=strategy_id,
                    episode_time_limit=episode_time_limit,
                    sparse=sparse,
                    one_hot=one_hot
                ) for strategy_id in team
            ],
            probs=probs,
            time_limit=total_time_limit
        )


def make_cooperative_env(
    team: List,
    ordered: bool = False,
    probs: List = None,
    episode_time_limit: int = 200,
    num_episodes: int = 500,
    sparse: bool = True,
    one_hot: bool = True, 
) -> 'CooperativeMetaGame':

    env = CooperativeMetaGame(
        meta_strategy=make_coalition(
            team=team,
            ordered=ordered,
            episode_time_limit=episode_time_limit,
            total_time_limit=num_episodes,
            probs=probs,
            sparse=sparse,
            one_hot=one_hot
        )
    )

    return env


def make_policy():
    return "MlpPolicy"


def make_ppo(env, seed, args) -> 'PPO':
    return PPO(
        policy=make_policy(),
        env=env,
        verbose=0,
        seed=seed,
        n_steps=min(2048, args.episode_limit * args.num_episodes),
        device='cpu'
    )


def make_task(args):
    specs = {
        'players': SPID_STRATEGIES
    }

    def make_game(team, seed, args, probs=None):
        env = make_cooperative_env(
            team=team,
            ordered=args.ordered,
            probs=probs,
            episode_time_limit=args.episode_limit,
            num_episodes=args.num_episodes
        )
        algorithm = make_ppo(
            env=env,
            seed=seed,
            args=args
        )

        return env, algorithm

    return specs, make_game


import functools
from typing import List
import gym
from stable_baselines3 import DQN, PPO
from ccgm.common.coalitions import Coalition, OrderedCoalition
from ccgm.common.games import CooperativeMetaGame

from .impl.v1 import MINATAR_STRATEGIES_v1
from .utils import StableBaselinesCompatWrapper, MinAtarFeatureExtractor
from stable_baselines3.dqn.policies import CnnPolicy


def make_coalition(
    team, 
    ordered,
    total_time_limit: int,  
    probs = None,
):

    if not ordered:
        return Coalition(
            players=[
                gym.make(strategy_id) for strategy_id in team
            ],
            probs=probs
        )
    else:
        return OrderedCoalition(
            players=[
                gym.make(strategy_id) for strategy_id in team
            ],
            probs=probs,
            time_limit=total_time_limit
        )


def make_cooperative_env(
    team: List,
    ordered: bool = False,
    probs: List = None,
    episode_time_limit: int = 200,
    num_episodes: int = 500 
) -> 'CooperativeMetaGame':

    env = CooperativeMetaGame(
        meta_strategy=make_coalition(
            team=team,
            ordered=ordered,
            total_time_limit=num_episodes,
            probs=probs,
        )
    )

    env = StableBaselinesCompatWrapper(
        env=env
    )

    return env

def make_policy(env: gym.Env):
    return functools.partial(
        CnnPolicy,
        features_extractor_class=MinAtarFeatureExtractor,
        net_arch=[]
    )


def make_dqn(env, seed, args) -> 'PPO':
    return DQN(
        policy=make_policy(env),
        env=env,
        verbose=1,
        seed=seed,
    )


def make_task(args):
    game_spec = {
        'players': MINATAR_STRATEGIES_v1
    }

    def make_game(team, seed, args, probs=None):
        make_env = functools.partial( 
            make_cooperative_env,
            team=team,
            ordered=args.ordered,
            probs=probs,
            episode_time_limit=args.episode_limit,
            num_episodes=args.num_episodes
        )
        make_alg = functools.partial(
            make_dqn,
            seed=seed,
            args=args
        )

        return make_env, make_alg

    return game_spec, make_game

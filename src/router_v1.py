import os
from threading import Thread
from typing import Callable, List

import numpy as np

import gym
import gym.spaces as spaces

from stable_baselines3 import DQN, PPO  # , DQN, A2C
# from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from m3g.examples.games.env import SequentialMatrixGameEnvironment
from m3g.examples.games.impl.prisioner_dilemma import (
    pd_nature_strategy_factory, pd_principal_strategy_factory
)
from m3g.examples.games.wrappers import (
    OneHotObservationWrapper, SparseRewardWrapper
)

from utils import make_cooperative_env


class MetaGameInPlaceRouterEnvironment(gym.Env):

    def __init__(
        self,
        train_strategies: List[str],
        train_env: gym.Env,
        eval_env: gym.Env,
        strategies_factory: Callable[[str], Callable],
    ) -> None:
        super().__init__()
        self.strategies = train_strategies
        self.strategies_factory = strategies_factory

        self.train_env = train_env
        self.eval_env = eval_env

        self.action_space = spaces.Discrete(len(train_strategies))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=[len(train_strategies)])
        self.state = np.zeros(len(train_strategies))
        self.last_reward = 0.0

    def reset(self):
        self.state.fill(0)
        return self.state

    def _reward(self, learner):
        mean, std = evaluate_policy(
            learner, self.eval_env, n_eval_episodes=1,
            deterministic=False
        )
        return mean

    def _obs(self, action):
        self.state[action] += 1
        obs = (self.state + 1e-5)
        obs /= sum(obs)
        return obs
        # return np.zeros(len(self.strategies))

    def step(self, action):
        obs = self._obs(action)

        print(obs)
        self.train_env.nature_meta_strategy.probs = obs

        learner = PPO(
            policy="MlpPolicy",
            env=self.train_env,
            verbose=1,
            seed=0,
            device='cpu'
        )
        learner.learn(200 * 50)
        _reward = self._reward(learner)
        reward = _reward - self.last_reward
        self.last_reward = _reward

        return obs, reward, False, {'freqs': self.state}


def make_meta_learner_env(
    train_team: List[str],
    eval_team: List[str],
    base_dir: str,
    seed: int,
    episode_length: int
):
    meta_eval_env = make_cooperative_env(
        team=eval_team,
    )
    meta_eval_env = Monitor(
        meta_eval_env,
        os.path.join(base_dir, f'{seed}.meta-learner.eval'),
        info_keywords=(
            'nature_acc_reward', 'agent_acc_reward', 'opponent')
    )

    train_env = make_cooperative_env(
        train_team
    )
    train_env = Monitor(
        train_env,
        os.path.join(base_dir, f'{seed}.learner.train'),
        info_keywords=(
            'nature_acc_reward', 'agent_acc_reward', 'opponent')
    )

    meta_learner_env = MetaGameInPlaceRouterEnvironment(
        train_strategies=train_team,
        train_env=train_env,
        eval_env=meta_eval_env,
        strategies_factory=pd_nature_strategy_factory,
    )
    meta_learner_env = gym.wrappers.TimeLimit(meta_learner_env, episode_length)
    meta_learner_env = Monitor(
        meta_learner_env,
        os.path.join(base_dir, f'{seed}.meta-learner.train'),
        info_keywords=('freqs',)
    )
    meta_learner_env.seed(seed)

    return meta_learner_env


def main(
    seed: int,
    base_dir: str = 'logs/router-meta-game/',
    learner_episode_length: int = 200,
    meta_learner_episode_length: int = 10,
    meta_learner_steps: int = 3000
):

    os.makedirs(base_dir, exist_ok=True)

    meta_learner_env = make_meta_learner_env(
        train_team=[
            'tit_for_tat',
            'always_cooperate',
            'always_defect',
            'win_stay_lose_switch',
            'extortionate-zd-default'
        ],
        eval_team=[
            'always_defect'
        ],
        episode_length=meta_learner_episode_length,
        base_dir=base_dir,
        seed=seed
    )
    meta_learner = PPO(
        policy="MlpPolicy",
        n_steps=16,
        env=meta_learner_env,
        verbose=2,
        seed=seed,
        device='cpu'
    )

    runs = [
        Thread(target=meta_learner.learn, args=[meta_learner_steps]),
        # Thread(target=learner.learn, args=[
        #     learner_episode_length*meta_learner_steps])
    ]

    for run in runs:
        run.start()

    for run in runs:
        run.join()


if __name__ == '__main__':
    main(7)

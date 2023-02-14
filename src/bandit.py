import os
from threading import Thread
from typing import List

import gym

from stable_baselines3 import DQN, PPO  # , DQN, A2C
from sb3_contrib import ARS
# from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor


from m3g.examples.games.env import SequentialMatrixGameEnvironment
from m3g.examples.games.impl.prisioner_dilemma import (
    pd_nature_strategy_factory, pd_principal_strategy_factory
)
from m3g.examples.games.wrappers import (
    OneHotObservationWrapper, SparseRewardWrapper
)

from utils import make_cooperative_env
from common.games import MetaGameSyncWrapper, MetaGameBanditEnvironment


def make_learner_env(
    base_dir: str,
    seed: int,
    episode_length: int
):
    learner_env = SequentialMatrixGameEnvironment(
        nature_strategy='always_defect',
        principal_strategy='default',
        nature_strategy_factory=pd_nature_strategy_factory,
        principal_strategy_factory=pd_principal_strategy_factory
    )
    learner_env = gym.wrappers.TimeLimit(learner_env, episode_length)
    learner_env = OneHotObservationWrapper(learner_env)
    learner_env = SparseRewardWrapper(learner_env)
    learner_env = MetaGameSyncWrapper(learner_env)

    learner_env = Monitor(
        learner_env,
        os.path.join(base_dir, f'{seed}.learner.train'),
        info_keywords=('nature_acc_reward', 'agent_acc_reward', 'opponent')
    )
    learner_env.seed(seed)

    return learner_env


def make_meta_learner_env(
    train_team: List[str],
    eval_team: List[str],
    learner: PPO,
    learner_env: gym.Env,
    base_dir: str,
    seed: int,
    episode_length: int
):
    meta_eval_env = make_cooperative_env(
        team=eval_team,
        sparse=True,
        one_hot=True
    )
    meta_eval_env = Monitor(
        meta_eval_env,
        os.path.join(base_dir, f'{seed}.meta-learner.eval'),
        info_keywords=(
            'nature_acc_reward', 'agent_acc_reward', 'opponent')
    )

    meta_learner_env = MetaGameBanditEnvironment(
        env=learner_env,
        train_strategies=train_team,
        eval_env=meta_eval_env,
        strategies_factory=pd_nature_strategy_factory,
        learner=learner
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
    meta_learner_steps: int = 5000
):

    os.makedirs(base_dir, exist_ok=True)

    learner_env = make_learner_env(
        base_dir=base_dir,
        seed=seed,
        episode_length=learner_episode_length
    )

    learner = PPO(
        policy="MlpPolicy",
        env=learner_env,
        verbose=2,
        seed=seed,
        device='cpu'
    )

    meta_learner_env = make_meta_learner_env(
        train_team=[
            'extortionate-zd-default',
            'tit_for_tat',
            'win_stay_lose_switch',
            'always_defect',
            'always_cooperate'
        ],
        eval_team=[
            'extortionate-zd-default',
            'tit_for_tat',
            'win_stay_lose_switch',
            'always_defect',
            'always_cooperate'
        ],
        learner=learner,
        learner_env=learner_env,
        episode_length=meta_learner_episode_length,
        base_dir=base_dir,
        seed=seed
    )
    # meta_learner = DQN(
    #     policy="MlpPolicy",
    #     # n_steps=16,
    #     env=meta_learner_env,
    #     verbose=2,
    #     seed=seed,
    #     device='cpu',
    #     # max_grad_norm=40,
    #     learning_starts=5*10,
    #     target_update_interval=5*10,
    #     # exploration_fraction=0.45
    # )
    # meta_learner = PPO(
    #     policy='MlpPolicy',
    #     env=meta_learner_env,
    #     n_steps=16,
    #     batch_size=16,
    #     n_epochs=5,
    #     seed=seed,
    #     device='cpu'
    # )
    meta_learner = ARS(
        policy='MlpPolicy',
        env=meta_learner_env,
        device='cpu',
        verbose=2,
        seed=seed
    )

    runs = [
        Thread(target=meta_learner.learn, args=[meta_learner_steps]),
        Thread(target=learner.learn, args=[
            learner_episode_length*meta_learner_steps])
    ]

    for run in runs:
        run.start()

    for run in runs:
        run.join()


if __name__ == '__main__':
    main(2)

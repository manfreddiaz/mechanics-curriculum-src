# from concurrent.futures import (
#     ThreadPoolExecutor, as_completed, ProcessPoolExecutor
# )
import os
import traceback
from itertools import product
import torch.multiprocessing as tmp
from stable_baselines3 import PPO  # , DQN, A2C
# from sb3_contrib import QRDQN, ARS, TRPO
from stable_baselines3.common.monitor import Monitor
from m3g.examples.games.impl.prisioner_dilemma import (
    PD_NATURE_AVAILABLE_STRATEGIES
)

from utils import team_to_id, make_cooperative_env, form_teams


def play(team, seed, outdir):
    team_id = team_to_id(team)
    team_dir = os.path.join(outdir, team_id)
    os.makedirs(team_dir, exist_ok=True)

    env = make_cooperative_env(team)
    env = Monitor(
        env,
        os.path.join(team_dir, f'{seed}.train'),
        info_keywords=('nature_acc_reward', 'agent_acc_reward')
    )
    env.seed(seed)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=seed,
        device='cpu'
    )
    print(f"game with: {team_id}, seed: {seed}")
    model.learn(
        total_timesteps=100*1000,
    )
    model.save(
        os.path.join(os.path.join(team_dir, f'{seed}.model.ckpt'))
    )
    del model.policy
    del model
    del env
    print(f"completed game with: {team_id}, seed: {seed}")
    return 0


def main(
    seed: int = 0,
    num_seeds: int = 10,
    outdir: str = "logs/cooperative-metagame/"
):
    os.makedirs(outdir, exist_ok=True)

    games = {}
    teams = form_teams(
        players=PD_NATURE_AVAILABLE_STRATEGIES.keys())

    tmp.set_start_method('spawn')
    with tmp.Pool(10, maxtasksperchild=4) as ppe:
        for seed, team in product(range(seed, seed + num_seeds), teams):
            game_id = f'{team_to_id(team)}-{seed}'
            print(f"submitting game with id: {game_id}")
            games[ppe.apply_async(play, (team, seed, outdir))] = game_id

        print('reporting games results')
        for game in games:
            try:
                code = game.get()
                game_id = games[game]
                print(f'game {game_id}: finished with exit code {code}')
            except Exception:
                print(
                    f'game {game_id}: failed with the following exception: \n',
                    traceback.format_exc()
                )


if __name__ == '__main__':
    main()

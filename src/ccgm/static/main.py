# from concurrent.futures import (
#     ThreadPoolExecutor, as_completed, ProcessPoolExecutor
# )
import argparse
import os
import traceback
from itertools import product
import torch.multiprocessing as tmp
from stable_baselines3 import PPO  # , DQN, A2C
# from sb3_contrib import QRDQN, ARS, TRPO
from stable_baselines3.common.monitor import Monitor

from ccgm.utils import team_to_id, form_teams
from ccgm.common.envs.sgt import (
   make_task as ipd_make_task
)
# from ccgm.common.envs.rl.gym.miniatar.impl.v1 import MINATAR_STRATEGIES_v1
# from ccgm.common.envs.rl.gym.miniatar.task import (
#     make_tasks as miniatar_make_task
# )

TASKS = {
    'sipd': ipd_make_task
    # 'minatar': {
    #     'strat': MINATAR_STRATEGIES_v1,
    #     'factory': miniatar_make_task
    # }
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', default='ppo')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_seeds', default=5, type=int)
    parser.add_argument('--outdir', default="logs/static", type=str)
    parser.add_argument('--num-episodes', default=10, type=int)
    parser.add_argument('--episode-limit', default=50, type=int)
    parser.add_argument('--ordered', default=True, type=bool)
    parser.add_argument('--task', default='sipd', choices=[
        'sipd', 
        'srps',
        'minatar'
    ])

    return parser.parse_args()


def play(index, game_factory, team, seed, outdir, args):
    team_id = team_to_id(team)
    team_dir = os.path.join(outdir, f'game-{str(index)}')
    os.makedirs(team_dir, exist_ok=True)

    print(f"game {index} with: {team_id}, seed: {seed}")
    
    env, algorithm = game_factory(team, seed, args)
    algorithm.learn(
        total_timesteps=args.episode_limit * args.num_episodes,
    )
    algorithm.save(
        os.path.join(os.path.join(team_dir, f'{seed}.model.ckpt'))
    )
    with open(os.path.join(team_dir, 'game.info'), mode='w') as f:
        f.write(team_id)
    
    del algorithm.policy
    del algorithm
    del env
    print(f"completed game with: {team_id}, seed: {seed}")
    return 0


def main(
    args
):
    outdir = os.path.join(
        args.outdir, 
        f"{args.task}", 
        f"{'ordered' if args.ordered else 'random'}"
    )
    os.makedirs(outdir, exist_ok=True)

    task_factory = TASKS[args.task]
    task_spec, game_factory = task_factory(args)

    games = {}
    # TODO: Add empty player (initialization).
    
    players = [player for player in task_spec['players']]
    teams = form_teams(players=players, ordered=args.ordered)
    teams_idx = {
        team: idx for idx, team in enumerate(teams)
    }

    # tmp.set_start_method('spawn')
    # with tmp.Pool(10, maxtasksperchild=4) as ppe:
    for seed, team in product(
        range(args.seed, args.seed + args.num_seeds), teams):
        # game_id = f'{team_to_id(team)}-{seed}'
        # print(f"submitting game with id: {game_id}")
        # games[ppe.apply_async(play, (team, seed, outdir))] = game_id
                    # games[ppe.apply_async(play, (team, seed, outdir))] = game_id
        play(teams_idx[team], game_factory, team, seed, outdir, args)



        # print('reporting games results')
        # for game in games:
        #     try:
        #         code = game.get()
        #         game_id = games[game]
        #         print(f'game {game_id}: finished with exit code {code}')
        #     except Exception:
        #         print(
        #             f'game {game_id}: failed with the following exception: \n',
        #             traceback.format_exc()
        #         )


if __name__ == '__main__':
    main(parse_args())

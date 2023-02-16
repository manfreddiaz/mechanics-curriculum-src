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
    parser.add_argument('--outdir', default="logs/", type=str)
    parser.add_argument('--num-episodes', default=500, type=int)
    parser.add_argument('--episode-limit', default=200, type=int)
    parser.add_argument('--thread-pool-size', default=10, type=int)
    parser.add_argument('--sync', default=False, action='store_true')
    parser.add_argument('--ordered', default=False, type=bool)
    parser.add_argument('--task', default='sipd', choices=[
        'sipd', 
        'srps',
        'minatar'
    ])

    return parser.parse_args()


def play(index, task_factory, team, seed, outdir, args):
    team_id = team_to_id(team)

    # loggging and saving config
    team_dir = os.path.join(outdir, f'game-{str(index)}')
    os.makedirs(team_dir, exist_ok=True)
    
    model_file = os.path.join(team_dir, f'{seed}.model.ckpt')

    # avoid duplicated runs on restart
    if os.path.exists(model_file):
        print(f"AP: game {index} with: {team_id}, seed: {seed}")
        return 0

    # playing the actual game
    print(f"game {index} with: {team_id}, seed: {seed}")
    _, game_factory = task_factory(args)

    
    make_env, make_alg = game_factory(team, seed, args)
    env = make_env()
    env = Monitor(
        env,
        filename=os.path.join(team_dir, f'{seed}.train'),
        info_keywords=('meta-strategy',)
    )
    env.seed(seed)

    algorithm = make_alg(env=env)
    algorithm.learn(
        total_timesteps=args.episode_limit * args.num_episodes,
    )
    algorithm.save(
        os.path.join(model_file)
    )

    game_info_file = os.path.join(team_dir, 'game.info')
    # log game info
    if not os.path.exists(game_info_file):
        with open(game_info_file, mode='w') as f:
            f.write(team_id)
            f.write('\r\n')
            f.write(str(algorithm.__class__.__name__))

    del algorithm.policy
    del algorithm
    del env
    print(f"completed game with: {team_id}, seed: {seed}")
    return 0


def run_async(outdir, config):
    task_factory = TASKS[config.task]
    game_spec, _ = task_factory(config)

    games = {}
    # TODO: Add empty player (initialization).
    
    players = [player for player in game_spec['players']]
    teams = form_teams(players=players, ordered=config.ordered)
    teams_idx = {
        team: idx for idx, team in enumerate(teams)
    }

    tmp.set_start_method('spawn')
    with tmp.Pool(config.thread_pool_size, maxtasksperchild=4) as ppe:
        for seed, team in product(
            range(config.seed, config.seed + config.num_seeds), teams):
            game_id = f'{team_to_id(team)}-{seed}'
            print(f"submitting game with id: {game_id}")
            games[ppe.apply_async(play, (teams_idx[team], task_factory, team, seed, outdir, config))] = game_id

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

def run(outdir, config):
    # task spec
    task_factory = TASKS[config.task]
    task_spec, _ = task_factory(config)

    # game spec    
    players = [player for player in task_spec['players']]
    teams = form_teams(players=players, ordered=config.ordered)
    teams_idx = {
        team: idx for idx, team in enumerate(teams)
    }

    for seed, team in product(
        range(config.seed, config.seed + config.num_seeds), teams):
        game_id = f'{team_to_id(team)}-{seed}'
        print(f"submitting game with id: {game_id}")
        play(teams_idx[team], task_factory, team, seed, outdir, config)



def main(
    config
):
    outdir = os.path.join(
        config.outdir,
        "static/" 
        f"{config.task}", 
        f"{'ordered' if config.ordered else 'random'}"
    )
    os.makedirs(outdir, exist_ok=True)

    if config.sync:
        run(outdir=outdir, config=config)
    else:
        run_async(outdir=outdir, config=config)
    

if __name__ == '__main__':
    main(parse_args())

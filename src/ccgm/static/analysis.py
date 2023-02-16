# from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult
import os
import traceback
from typing import Dict, Tuple

import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from m3g.examples.games.impl.prisioner_dilemma import (
    PD_NATURE_AVAILABLE_STRATEGIES
)
from ccgm.utils import form_teams, make_cooperative_env, team_to_id  # , DQN, A2C


def get_seed(checkpint_name):
    return os.path.basename(checkpint_name).split('.')[0]


def eval(checkpoint: str, train_team: str, eval_team: Tuple):
    run_seed = int(get_seed(checkpoint))
    eval_team_id = team_to_id(eval_team)
    model = PPO.load(
        path=checkpoint,
        device='cpu',
        # print_system_info=True
    )
    env = make_cooperative_env(team=eval_team)
    env.seed(1234)

    print(f'playing: {train_team}.{run_seed} vs {eval_team_id}')
    mean, std = evaluate_policy(
        model,
        env
    )
    del env
    del model

    return {
        'train_team': train_team,
        'eval_team': team_to_id(eval_team),
        'seed': run_seed,
        'mean': mean,
        'std': std
    }


def main(
    indir: str = 'logs/cooperative-metagame',
    order: int = 1
):
    players = PD_NATURE_AVAILABLE_STRATEGIES.keys()
    evaluation_teams = form_teams(
        players=players,
        min_order=1,
    )

    results_ft: Dict[ApplyResult, str] = dict()
    with Pool(10, maxtasksperchild=4) as tpe:
        for odir in sorted(os.listdir(indir)):
            results_dir = os.path.join(indir, odir)
            for check_point in sorted(
                glob(os.path.join(results_dir, '*.ckpt'))
            ):
                for eval_team in evaluation_teams:
                    future = tpe.apply_async(
                        eval, (check_point, odir, eval_team))
                    results_ft[future] = f'{odir}-{team_to_id(eval_team)}'

        results = []
        print('reporting results')
        for game in results_ft:
            try:
                game_id = results_ft[game]
                entry = game.get(60)
                if game.successful():
                    results.append(entry)
                    print(f'game {results_ft[game]}: finished')
                else:
                    print(f'game {results_ft[game]}: failed')
            except:
                print(
                    f'game {game_id}: failed'
                )
                traceback.print_exc()

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(indir, 'results.csv')
    )


if __name__ == '__main__':
    main()

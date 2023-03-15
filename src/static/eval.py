import csv
import logging
import os
import random
from itertools import product
import traceback
import torch.multiprocessing as tmp
import gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ccgm.utils import CoalitionMetadata
from static.utils import hydra_custom_resolvers, make_xpt_coalition_dir, make_xpt_dir

log = logging.getLogger(__name__)


def load_coalition_models(
    training_coalition: CoalitionMetadata,
    seed: int,
    cfg: DictConfig
):
    team_dir = make_xpt_coalition_dir(training_coalition, cfg)
    
    # verify if the initial model was produced
    # fail otherwise
    initial_model = os.path.join(team_dir, f'{seed}i.model.ckpt')
    if not os.path.exists(initial_model):
        log.error(f"<failed> game {training_coalition.id} may have never started.")
        raise FileNotFoundError(initial_model)
    
    # verify if the final model was produced
    # fail otherwise
    final_model = os.path.join(team_dir, f'{seed}f.model.ckpt')
    if not os.path.exists(final_model):
        log.error(f"<failed> game {training_coalition.id} may have not completed!")
        raise FileNotFoundError(final_model)

    device = torch.device(cfg.torch.device) 
    i_model = torch.load(initial_model, map_location=device)
    f_model = torch.load(final_model, map_location=device)

    return i_model, f_model


def eval_model(
    model, 
    env: gym.vector.VectorEnv, 
    num_steps: int,
    cfg: DictConfig
):
    device = torch.device(cfg.torch.device) 
    rewards = np.zeros(shape=(env.num_envs, num_steps)) 
    
    obs = env.reset()
    for step in range(num_steps):
        action = model.predict(torch.tensor(obs, device=device))
        next_obs, reward, done, info = env.step(action.cpu().numpy())
        rewards[::, step] = reward
        if all(done):
            next_obs = env.reset()
        obs = next_obs
    
    return rewards


def eval(
    training_coalition: CoalitionMetadata,
    evaluation_coalition: CoalitionMetadata, 
    train_seed: int,
    eval_seed: int,
    cfg: DictConfig
):
    if GlobalHydra.instance() is not None:
        if not GlobalHydra.instance().is_initialized():
            hydra.initialize(version_base=None, config_path='conf')

    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    train_dir = make_xpt_coalition_dir(training_coalition, cfg)
    init_agent, final_agent = load_coalition_models(
        training_coalition=training_coalition,
        seed=train_seed,
        cfg=cfg
    )

    _, game_factory = hydra.utils.instantiate(cfg.eval.task)
    make_env = game_factory(evaluation_coalition)
    
    def monitored():
        env = make_env()
        return Monitor(
            env,
            filename=os.path.join(train_dir, f'{train_seed}.eval'),
            info_keywords=('meta-strategy',)
        )

    envs = gym.vector.SyncVectorEnv([
        monitored
    ])
    envs.seed(eval_seed)

    r_init = eval_model(
        init_agent, 
        envs, 
        num_steps=cfg.eval.total_timesteps, 
        cfg=cfg
    )
    r_final = eval_model(
        final_agent, 
        envs, 
        num_steps=cfg.eval.total_timesteps,
        cfg=cfg
    )

    log.info(f"<evaluation> game with: {training_coalition.id}, seed: {eval_seed}")

    envs.close()

    return {
        'train_team': training_coalition.id,
        'train_seed': train_seed,
        'eval_team': evaluation_coalition.id,
        'eval_seed': eval_seed,
        'r_0': np.mean(r_init),
        'r_0_std': np.std(r_init),
        'r_1': np.mean(r_final),
        'r_1_std': np.std(r_final),
    }


hydra_custom_resolvers()
@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(
    cfg: DictConfig
) -> None:
    
    log = logging.getLogger(__name__)
    
    torch.backends.cudnn.deterministic = cfg.torch.deterministic
    
    train_game_spec, _ = hydra.utils.instantiate(cfg.task)
    eval_game_spec, _ = hydra.utils.instantiate(cfg.eval.task)

    training_seeds = range(cfg.run.seed, cfg.run.seed + cfg.run.num_seeds)
    evaluation_seeds = range(cfg.eval.seed, cfg.eval.seed + cfg.eval.num_seeds)

    games = {}

    tmp.set_start_method('spawn')
    with tmp.Pool(
        processes=cfg.thread_pool.size, 
        maxtasksperchild=cfg.thread_pool.maxtasks
    ) as ppe:
        for train_seed, eval_seed, training_coalition, evaluation_coalition in product(
            training_seeds, evaluation_seeds, train_game_spec.coalitions, eval_game_spec.coalitions
        ):
            log.info(
                f"<submit> eval game {training_coalition.id}-{train_seed} vs {evaluation_coalition.id}-{eval_seed}" 
            )
            async_result = ppe.apply_async(
                eval, (training_coalition, evaluation_coalition, train_seed, eval_seed, cfg)
            )
            games[async_result] = (train_seed, eval_seed, training_coalition, evaluation_coalition)

        fieldnames = [
            'train_team', 'train_seed', 'eval_team', 'eval_seed', 
            'r_0', 'r_0_std', 'r_1', 'r_1_std'
        ]
        with open(f'{make_xpt_dir(cfg)}/results.csv', mode="w") as f:
            with open(f"{make_xpt_dir(cfg)}/failures.csv", mode="w") as g:
                
                results_writer = csv.DictWriter(f, fieldnames=fieldnames)
                results_writer.writeheader()
                fail_writer = csv.DictWriter(g, fieldnames=fieldnames[:4])
                for game_result in games:
                    try:
                        train_seed, eval_seed, training_coalition, evaluation_coalition = games[game_result]
                        result = game_result.get()
                        results_writer.writerow(result)
                        f.flush()
                        log.info(f"<finished> eval game {training_coalition.id}-{train_seed} vs {evaluation_coalition.id}-{eval_seed}")
                    except Exception as ex:
                        log.error(
                            f'<FAIL> eval game {training_coalition.id}-{train_seed} vs {evaluation_coalition.id}-{eval_seed} with the following exception: \n',
                            traceback.format_exc(),
                            exc_info=ex
                        )
                        fail_writer.writerow({
                            'train_team': training_coalition.id,
                             'train_seed':train_seed, 
                             'eval_team': evaluation_coalition.id,
                            'eval_seed': eval_seed, 
                        })
                        g.flush()


if __name__ == '__main__':
    main()

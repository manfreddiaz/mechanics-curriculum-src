# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
from dataclasses import dataclass

import random
import time
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ccgm.common.algs.core import Agent
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class QPolicyWithTarget:
    q_network: nn.Module
    target_network: nn.Module


@dataclass
class OffPolicyAgent(
    Agent[
        QPolicyWithTarget, ReplayBuffer, torch.optim.Optimizer
    ]
):
    pass


@dataclass
class HParamsDQN:
    start_e: int
    end_e: int
    exploration_fraction: int
    target_network_frequency: int
    gamma: int
    learning_starts: int
    train_frequency: int
    tau: float


@dataclass
class RParamsDQN:
    total_timesteps: int
    batch_size: int


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class DQN:

    @staticmethod
    def play(
        agent: OffPolicyAgent,
        obs: torch.Tensor,
        next_done: torch.Tensor,
        envs: gym.vector.SyncVectorEnv,
        hparams: HParamsDQN,
        rparams: RParamsDQN,
        logger,
        global_step: int,
        log_every: int,
        log_file_format,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = agent.policy
        memory = agent.memory

        epsilon = linear_schedule(
            hparams.start_e, hparams.end_e,
            hparams.exploration_fraction * rparams.total_timesteps,
            global_step
        )
        if random.random() < epsilon:
            actions = np.array([
                envs.single_action_space.sample() for _ in range(envs.num_envs)
            ])
        else:
            q_values = policy.q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)


        # handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        memory.add(obs, real_next_obs, actions, rewards, dones, infos)

        return 1 * envs.num_envs

    @staticmethod
    def optimize(
        agent: OffPolicyAgent,
        envs: gym.vector.VectorEnv,
        global_step: int,
        hparams: HParamsDQN,
        rparams: RParamsDQN,
        logger,
        log_every: int,
        log_file_format: str,
    ):
        # ALGO LOGIC: training.
        rb = agent.memory
        q_network = agent.policy.q_network
        target_network = agent.policy.target_network
        optimizer = agent.optimizer

        if global_step > hparams.learning_starts:
            if global_step % hparams.train_frequency == 0:
                data = rb.sample(rparams.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(
                        data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + hparams.gamma * target_max * \
                        (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(
                    1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    logger.add_scalar("losses/td_loss", loss, global_step)
                    logger.add_scalar("losses/q_values",
                                      old_val.mean().item(), global_step)
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % hparams.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        hparams.tau * q_network_param.data +
                        (1.0 - hparams.tau) * target_network_param.data
                    )

            return 1

        return 0

    def learn(
        agent: OffPolicyAgent,
        envs: gym.vector.VectorEnv,
        hparams: HParamsDQN,
        rparams: RParamsDQN,
        logger,
        device,
        log_every: int = -1,  # means no intermediate save log
        log_file_format: str = None
    ):
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs = envs.reset()
        for global_step in range(rparams.total_timesteps):
            # ALGO LOGIC: put action logic here
            obs, _ = DQN.play(
                agent=agent,
                obs=obs,
                next_done=None,
                envs=envs,
                hparams=hparams,
                rparams=rparams,
                logger=logger,
                global_step=global_step,
                log_every=log_every,
                log_file_format=log_file_format,
                device=device
            )

            DQN.optimize(
                agent=agent,
                envs=envs,
                hparams=hparams,
                rparams=rparams,
                logger=logger,
                global_step=global_step,
                log_every=log_every,
                log_file_format=log_file_format
            )

            logger.add_scalar("charts/SPS", int(global_step /
                              (time.time() - start_time)), global_step)

            if log_every != -1 and global_step % log_every == 0:
                assert log_file_format is not None
                torch.save(
                    agent,
                    log_file_format.format(global_step)
                )

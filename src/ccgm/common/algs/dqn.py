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

    def predict(self, obs: torch.Tensor):
        q_values = self.q_network(obs)
        return torch.argmax(q_values, dim=1)


@dataclass
class OffPolicyAgent(
    Agent[
        QPolicyWithTarget, ReplayBuffer, torch.optim.Optimizer
    ]
):
    def copy(self, other: 'OffPolicyAgent'):
        if not isinstance(other, type(self)):
            print(f'copy attempt mistmatch -> from {type(other)} to {type(self)}')
            return
        self.policy.q_network.load_state_dict(
            other.policy.q_network.state_dict()
        )
        self.policy.target_network.load_state_dict(
            other.policy.target_network.state_dict()
        )
        self.optimizer.load_state_dict(
            other.optimizer.state_dict()
        )
        self.memory.observations = np.array(other.memory.observations, copy=True)
        self.memory.actions = np.array(other.memory.actions, copy=True)
        self.memory.next_observations = np.array(other.memory.next_observations, copy=True)
        self.memory.rewards = np.array(other.memory.rewards, copy=True)
        self.memory.dones = np.array(other.memory.dones, copy=True)
        self.memory.full = other.memory.full
        self.memory.pos = other.memory.pos
        self.memory.timeouts = np.array(other.memory.timeouts, copy=True)
        # TODO: Should memory be also copied? it should be
        # self.memory.


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
    num_steps: int = 1


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
        obs: np.array,
        global_step: int,
        hparams: HParamsDQN,
        rparams: RParamsDQN,
        device: torch.device
    ):
        policy = agent.policy
        memory = agent.memory

        epsilon = linear_schedule(
            hparams.start_e, hparams.end_e,
            hparams.exploration_fraction * rparams.total_timesteps,
            global_step
        )
        if random.random() < epsilon:
            actions = np.array([
                np.random.choice(memory.action_space.n) for _ in range(len(obs))
            ])
        else:
            q_values = policy.q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        def memory_fn(obs, rewards, next_obs, dones, infos):
            # handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            memory.add(obs, real_next_obs, actions, rewards, dones, infos)

        return actions, memory_fn

    @staticmethod
    def repeat_play(
        agent: OffPolicyAgent,
        envs: gym.vector.SyncVectorEnv,
        hparams: HParamsDQN,
        rparams: RParamsDQN,
        logger,
        global_step: int,
        log_every: int,
        log_file_format,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        rb = agent.memory

        obs = rb.observations[rb.pos % rb.buffer_size]

        for step in range(hparams.num_steps):
            actions, memory_fn = DQN.play(
                agent, obs, global_step + step, hparams, rparams, device)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = envs.step(actions)

            memory_fn(obs, rewards, next_obs, dones, infos)

            DQN.optimize(
                agent, envs, global_step + step, hparams, rparams, logger,
                device, log_every, log_file_format
            )
            obs = next_obs

        return hparams.num_steps * 1 * envs.num_envs, hparams.num_steps

    @staticmethod
    def optimize(
        agent: OffPolicyAgent,
        envs: gym.vector.VectorEnv,
        global_step: int,
        hparams: HParamsDQN,
        rparams: RParamsDQN,
        logger,
        device: torch.device,
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
                    if logger:
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
        log_every: int = -1, # means no intermediate save log
        log_file_format: str = None,
        eval_fn = None,
        log_every: int = -1,  # means no intermediate save log
        log_file_format: str = None
    ):
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs = envs.reset()
        for global_step in range(rparams.total_timesteps):
            # ALGO LOGIC: put action logic here
            obs, _ = DQN.repeat_play(
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

            # DQN.optimize(
            #     agent=agent,
            #     envs=envs,
            #     hparams=hparams,
            #     rparams=rparams,
            #     logger=logger,
            #     global_step=global_step,
            #     log_every=log_every,
            #     log_file_format=log_file_format
            # )

            logger.add_scalar("charts/SPS", int(global_step /
                              (time.time() - start_time)), global_step)

            if log_every != -1 and global_step % log_every == 0:
                eval_value = eval_fn(agent.policy)
                logger.add_scalar("eval/returns", np.mean(eval_value), global_step)
                # assert log_file_format is not None
                # torch.save(
                #     agent,
                #     log_file_format.format(global_step)
                # )

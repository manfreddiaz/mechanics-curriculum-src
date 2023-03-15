# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import functools
import logging
import os
import random
import time
from distutils.util import strtobool
from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class QNetwork:
    pass

log = logging.getLogger(__name__)

class DQN:

    def __init__(
        self,        
        envs: List,
        rparams,
        hparams,
    ) -> None:
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        self._envs = envs
        self._rparams = rparams
        self._hparams = hparams


    def learn(
        self,
        agent: QNetwork,
        task,
        logger,
        device,
        log_every: int = -1, # means no intermediate save log
        log_file_format: str = None
    ):
        hparams = self._hparams
        rparams = self._rparams

        agent = agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=hparams.learning_rate)
        target_network = agent.__class__(self._envs)
        target_network = target_network.to(device)
        target_network.load_state_dict(agent.state_dict())

        rb = ReplayBuffer(
            hparams.buffer_size,
            self._envs.single_observation_space,
            self._envs.single_action_space,
            device,
            handle_timeout_termination=True,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs = self._envs.reset()
        for global_step in range(rparams.total_timesteps):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(
                hparams.start_e, hparams.end_e, 
                hparams.exploration_fraction * rparams.total_timesteps, 
                global_step
            )
            if random.random() < epsilon:
                actions = np.array([self._envs.single_action_space.sample() for _ in range(self._envs.num_envs)])
            else:
                q_values = agent(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, dones, infos = self._envs.step(actions)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            for info in infos:
                if "episode" in info.keys():
                    log.info(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    logger.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    logger.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    logger.add_scalar("charts/epsilon", epsilon, global_step)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            rb.add(obs, real_next_obs, actions, rewards, dones, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > hparams.learning_starts:
                if global_step % hparams.train_frequency == 0:
                    data = rb.sample(rparams.batch_size)
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + hparams.gamma * target_max * (1 - data.dones.flatten())
                    old_val = agent(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        logger.add_scalar("losses/td_loss", loss, global_step)
                        logger.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        log.info(f"SPS: {int(global_step / (time.time() - start_time))}")
                        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if global_step % hparams.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), agent.parameters()):
                        target_network_param.data.copy_(
                            hparams.tau * q_network_param.data + (1.0 - hparams.tau) * target_network_param.data
                        )
            
            if global_step % log_every == 0:
                assert log_file_format is not None
                torch.save(
                    agent,
                    log_file_format.format(global_step)
                )



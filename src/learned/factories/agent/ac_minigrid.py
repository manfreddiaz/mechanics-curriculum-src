
import logging
import gym
from omegaconf import DictConfig

import torch
from ccgm.common.algs.networks.ac_network import MiniGridActorCritic
from ccgm.common.algs.ppo import OnPolicyAgent, RolloutBuffer


def make_agent(
    id: str
):
    def agent_fn(
        envs: gym.vector.VectorEnv,
        hparams: DictConfig,
        device: torch.device
    ):
        policy = MiniGridActorCritic(envs).to(device)
        agent = OnPolicyAgent(
            policy=policy,
            memory=RolloutBuffer(
                buffer_size=hparams.num_steps, 
                observation_space=envs.single_observation_space,
                action_space=envs.single_action_space,
                gae_lambda=hparams.gae_lambda,
                gamma=hparams.gamma,
                device=device,
                n_envs=envs.num_envs
            ),
            optimizer=torch.optim.Adam(
                policy.parameters(), lr=hparams.learning_rate, eps=1e-5
            )
        )
        return agent
    
    return agent_fn

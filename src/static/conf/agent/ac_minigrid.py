
import logging
import gym
from omegaconf import DictConfig

import torch
from ccgm.common.algs.networks.ac_network import MiniGridActorCritic
from ccgm.common.algs.ppo import OnPolicyAgent, OnPolicyReplayBuffer


def make_agent(
    id: str
):
    def agent_fn(
        envs:gym.vector.VectorEnv,
        hparams: DictConfig,
        device: torch.device
    ):
        policy = MiniGridActorCritic(envs).to(device)
        agent = OnPolicyAgent(
            policy=policy,
            memory=OnPolicyReplayBuffer(
                obs=torch.zeros((hparams.num_steps, envs.num_envs) + envs.single_observation_space.shape).to(device),
                actions=torch.zeros((hparams.num_steps, envs.num_envs) + envs.single_action_space.shape).to(device),
                logprobs=torch.zeros((hparams.num_steps, envs.num_envs)).to(device),
                rewards=torch.zeros((hparams.num_steps, envs.num_envs)).to(device),
                dones=torch.zeros((hparams.num_steps, envs.num_envs)).to(device),
                values=torch.zeros((hparams.num_steps, envs.num_envs)).to(device),
                advantages=torch.zeros((hparams.num_steps, envs.num_envs)).to(device),
                returns=torch.zeros((hparams.num_steps, envs.num_envs)).to(device)
            ),
            optimizer=torch.optim.Adam(policy.parameters(), lr=hparams.learning_rate, eps=1e-5)
        )
        return agent
    
    return agent_fn

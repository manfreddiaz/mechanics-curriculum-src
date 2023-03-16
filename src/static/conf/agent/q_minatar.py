
import logging
import gym
import numpy as np
from omegaconf import DictConfig
import torch

from stable_baselines3.common.buffers import ReplayBuffer
from ccgm.common.algs.dqn import OffPolicyAgent, QPolicyWithTarget
from ccgm.common.algs.networks.q_networks import MinAtarQNetwork


log = logging.getLogger(__name__)


def make_agent(id: str):
    def agent_fn(
        envs:gym.vector.VectorEnv,
        hparams: DictConfig,
        device: torch.device
    ):
        q_network = MinAtarQNetwork(envs).to(device)
        target_network = MinAtarQNetwork(envs).to(device)
        target_network.load_state_dict(q_network.state_dict())
        agent = OffPolicyAgent(
            policy=QPolicyWithTarget(
                q_network=q_network,
                target_network=target_network
            ),
            memory=ReplayBuffer(
                hparams.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                handle_timeout_termination=True,
            ),
            optimizer=torch.optim.Adam(q_network.parameters(), lr=hparams.learning_rate, eps=1e-5)
        )
        return agent
    
    return agent_fn

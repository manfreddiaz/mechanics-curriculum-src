from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar

import gym  # noqa
import torch
from stable_baselines3.common.buffers import BaseBuffer

PolicyType = TypeVar("PolicyType")
ReplayBufferType = TypeVar("ReplayBufferType", bound=BaseBuffer)
OptimizerType = TypeVar("OptimizerType", bound=torch.optim.Optimizer)


@dataclass
class Agent(
    Protocol, Generic[PolicyType, ReplayBufferType, OptimizerType]
):
    policy: PolicyType
    memory: ReplayBufferType
    optimizer: OptimizerType


AlgorithmPlayFn = Callable[
    [Agent, gym.vector.VectorEnv, int],
    None
]

AlgorithmOptimizeFn = Callable[
    [Agent, gym.vector.VectorEnv, int],
    bool
]

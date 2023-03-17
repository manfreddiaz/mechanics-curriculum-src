
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, Tuple, TypeVar
import torch
import gym  # noqa

PolicyType = TypeVar("PolicyType")
ReplayBufferType = TypeVar("ReplayBufferType")
OptimizerType = TypeVar("OptimizerType", bound=torch.optim.Optimizer)


@dataclass
class Agent(Generic[PolicyType, ReplayBufferType, OptimizerType]):
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

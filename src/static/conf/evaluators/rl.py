import numpy as np
import torch
import gym
from stable_baselines3.common.evaluation import evaluate_policy

def eval_model(
    model, 
    env: gym.vector.VectorEnv, 
    num_steps: int,
    device: torch.device
):
   pass


def make_evaluator(
    id: str,
):
    return eval_model


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

    # accumulators
    rewards = np.zeros(env.num_envs, dtype=np.float32)
    steps = np.zeros(env.num_envs)
    run_episodes = np.zeros(env.num_envs)
    episodes_rewards, episodes_lengths = [], []

    next_obs = env.reset()
    while (run_episodes < num_steps).any():
        with torch.no_grad():
            action = model.predict(torch.tensor(next_obs).to(device))
        next_obs, reward, done, info = env.step(action.cpu().numpy())
        rewards += reward
        steps += 1
        for i in range(env.num_envs):
            if done[i]:
                run_episodes[i] += 1
                episodes_rewards.append(rewards[i])
                episodes_lengths.append(steps[i])
                rewards[i] = 0.0
                steps[i] = 0

    return episodes_rewards


def make_evaluator(
    id: str,
):
    return eval_model


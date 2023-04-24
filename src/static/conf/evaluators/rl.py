import numpy as np
import torch
import gym


def eval_model(
    model, 
    env: gym.vector.VectorEnv, 
    num_steps: int,
    device: torch.device
):
    device = torch.device(device) 
    rewards = np.zeros(shape=(env.num_envs, num_steps)) 
    
    obs = env.reset()
    for step in range(num_steps):
        action = model.predict(torch.tensor(obs, device=device))
        next_obs, reward, done, info = env.step(action.cpu().numpy())
        rewards[::, step] = reward
        if all(done):
            next_obs = env.reset()
        obs = next_obs
    
    return rewards


def make_evaluator(
    id: str,
):
    return eval_model


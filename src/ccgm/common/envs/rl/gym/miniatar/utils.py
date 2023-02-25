import numpy as np
import gym
import gym.spaces as spaces
import torch.nn as nn




class ChannelFirstFloatObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=tuple(reversed(env.observation_space.shape)), 
            dtype=np.float32
        )

    def observation(self, observation: np.array):
        obs = observation.astype(np.uint8) * 255.0
        return obs.transpose(2, 0, 1)



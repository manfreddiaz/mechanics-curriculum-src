import gym
import torch


class TorchObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, device: torch.DeviceObjType):
        super().__init__(env)
        self._device = device

    
    def observation(self, observation):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation)
        observation = observation.to(self._device)
        return observation

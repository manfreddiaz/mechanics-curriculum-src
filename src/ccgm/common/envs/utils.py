
import gym


class AutoResetWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done:
            next_obs = self.reset()
            info['terminal_observation'] = obs
            obs = next_obs

        return obs, reward, done, info

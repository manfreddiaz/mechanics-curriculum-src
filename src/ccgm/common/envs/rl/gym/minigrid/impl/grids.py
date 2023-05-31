import gym
import gym_minigrid

_CURRICULUM = [
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-MultiRoom-N6-v0"
]

MINIGRIDS_GRIDS = {
    id: env_spec 
        for id, env_spec in gym.envs.registry.env_specs.items() if id in _CURRICULUM
}


if __name__ == '__main__':
    print(f"available strategies: {', '.join(MINIGRIDS_GRIDS.keys())}")

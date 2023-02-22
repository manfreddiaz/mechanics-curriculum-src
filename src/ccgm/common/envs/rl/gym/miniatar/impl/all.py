import minatar  # noqa
from gym import envs


MINATAR_STRATEGIES_all = {
    id: env_spec 
        for id, env_spec in envs.registry.env_specs.items() if "MinAtar" in id
}


if __name__ == '__main__':
    print(f"available strategies: {', '.join(MINATAR_STRATEGIES_all.keys())}")

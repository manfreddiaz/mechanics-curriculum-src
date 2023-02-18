import minatar  # noqa
from gym import envs


MINATAR_STRATEGIES_ALL = {
    id: env_spec 
        for id, env_spec in envs.registry.env_specs.items() if "MinAtar" in id
}


def nature_strategy_factory(strategy_name: str):
    pass



def principal_strategy_factory(strategy_name: str):
    pass


if __name__ == '__main__':
    print(f"available strategies: {', '.join(MINATAR_STRATEGIES_ALL.keys())}")

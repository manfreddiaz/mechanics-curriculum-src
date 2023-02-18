from gym import envs


MINATAR_STRATEGIES_v1 = {
    id: env_spec 
        for id, env_spec in envs.registry.env_specs.items() if "MinAtar" in id and "v1" in id
}


def nature_strategy_factory(strategy_name: str):
    pass


def principal_strategy_factory(strategy_name: str):
    pass


if __name__ == '__main__':
    print(f"available strategies: {', '.join(MINATAR_STRATEGIES_v1.keys())}")

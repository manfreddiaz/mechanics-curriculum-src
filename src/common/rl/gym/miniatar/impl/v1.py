from gym import envs


STRATEGIES = {
    id: env_spec 
        for id, env_spec in envs.registry.items() if "MinAtar" in id and "v1" in id
}


def nature_strategy_factory(strategy_name: str):
    pass


def principal_strategy_factory(strategy_name: str):
    pass


if __name__ == '__main__':
    print(f"available strategies: {', '.join(STRATEGIES.keys())}")

from gym import envs

MINATAR_STRATEGIES_V0 = {
    id: env_spec 
        for id, env_spec in envs.registry.items() if "MinAtar" in id and "v0" in id
}


def nature_strategy_factory(strategy_name: str):
    pass


def principal_strategy_factory(strategy_name: str):
    pass



if __name__ == '__main__':
    print(f"available strategies: {', '.join(MINATAR_STRATEGIES_V0.keys())}")

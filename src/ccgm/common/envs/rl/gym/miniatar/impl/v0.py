from gym import envs

MINATAR_STRATEGIES_v0 = {
    id: env_spec 
        for id, env_spec in envs.registry.env_specs.items() if "MinAtar" in id and "v0" in id
}


if __name__ == '__main__':
    print(f"available strategies: {', '.join(MINATAR_STRATEGIES_v0.keys())}")

from gymnasium import envs


STRATEGIES = {
    id: env_spec 
        for id, env_spec in envs.registry.items() if "ALE" in id and "v5" in id and "ram" not in id
}


if __name__ == '__main__':
    print(f"available strategies: {', '.join(STRATEGIES.keys())}")
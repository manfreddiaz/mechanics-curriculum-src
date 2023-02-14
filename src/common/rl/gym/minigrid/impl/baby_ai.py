from gymnasium import envs


STRATEGIES = {
    id: env_spec 
        for id, env_spec in envs.registry.items() if "BabyAI" in id
}

if __name__ == '__main__':
    print(f"available strategies: {', '.join(STRATEGIES.keys())}")

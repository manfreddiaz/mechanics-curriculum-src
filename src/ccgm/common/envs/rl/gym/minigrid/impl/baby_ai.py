from gymnasium import envs


CURRICULUM = [
    "BabyAI-GoToLocal-v0",
    "GoToObjMaze",
    ""
]

STRATEGIES = {
    id: env_spec 
        for id, env_spec in envs.registry.items() if "BabyAI" in id
}

if __name__ == '__main__':
    print(f"available strategies: {', '.join(STRATEGIES.keys())}")

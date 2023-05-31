

def make_alg(
    variant: str,
    gamma: float,
    alpha: float
):
    if variant not in ["exp3, exp3s, exp3r"]:
        raise ValueError()
    
    def make_exp3():
            
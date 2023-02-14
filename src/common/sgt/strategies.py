from math import sqrt
import numpy as np


class NatureMemoryOneStrategy:

    def __init__(
        self,
        name: str,
        initial_action_dist: np.array,
        memory_one_action_dist: np.array
    ) -> None:
        '''
        unconditional_action_prob:
            a probability distribution over actions on
            the first move.

            Examples
              PD:
                Cooperate (C)   1.0
                Defect (D)      0.0
              always plays cooperate on the first move.

              RPS:
                Rock    (R)   1.0
                Paper   (P)   0.0
                Scissor (S)   0.0
              always plays rock on the first move.

        conditional_act_prob:
            a transition summarizing the probability of next action
            provided the last two observed actions.

            Example
                PD:
                                        Last Step
                                    CC  CD  DC  DD
                    Cooperate (C)   1.0 1.0 0.0 0.0
                    Defect    (D)   0.0 0.0 1.0 1.0
                RPS:
                                        Last Steps
                               RR  RP  RS PR  PP  PS  SR  SP  SS
                Rock    (R)   0.5 1.0 0.0 0.0 1.0 0.7 1.0 0.1 0.5
                Paper   (P)   0.2 0.0 1.0 1.0 0.0 0.2 0.0 0.8 0.3
                Scissor (S)   0.3 0.0 0.0 0.0 0.0 0.1 0.0 0.1 0.2
        '''
        self.initial_prob = initial_action_dist
        self.action_prob = memory_one_action_dist
        self.np_random = np.random.RandomState(0)
        self.action_dim = initial_action_dist.shape[0]
        self.name = name

    def seed(self, seed):
        self.np_random.seed(seed)

    def __str__(self) -> str:
        return self.name

    def first(self):
        return self.np_random.choice(
            np.arange(self.action_dim), p=self.initial_prob)

    def __call__(self, self_action: int, other_action: int):
        last_step = self_action * self.action_dim + other_action
        strategy = self.action_prob[:, last_step]

        return self.np_random.choice(np.arange(self.action_dim), p=strategy)


class PrincipalMarkovStrategy:

    def __init__(
        self,
        name,
        payoff_matrix: np.array
    ) -> None:
        '''
            payoff_matrix: payoff for each actions pair of the two players.

                Example
                    PD:
                                C,C C,D D,C D,D
                        nature  3.0 0.0 5.0 1.0
                        agent   3.0 5.0 0.0 1.0
                    RPS:
                                R,R  R,P  R,S  P,R P,P  P,S  S,R  S,P S,S
                        nature  0.0 -1.0  1.0  1.0 0.0 -1.0 -1.0  1.0 0.0
                        agent   0.0  1.0 -1.0 -1.0 0.0  1.0  1.0 -1.0 0.0
        '''
        self.strategy = payoff_matrix
        self.action_dim = int(sqrt(payoff_matrix.shape[1]))
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __call__(self, nature_action: int, agent_action: int):
        return self.strategy[:, nature_action * self.action_dim + agent_action]

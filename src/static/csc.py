from collections import deque
from itertools import permutations
import math
from typing import Any, Dict, List
import numpy as np

from ccgm.utils import CoalitionMetadata


class CooperativeGame:
    def __init__(
        self,
        values: dict[str, float],
        players: str
    ) -> None:
        pass

    def is_convex(self):
        pass

    def is_superadditive(self):
        pass


def shapley(values: Dict[str, Any], players: list[str], ordered: bool = False) -> List[float]:
    value = {player: 0.0 for player in players}
    players_idx = np.arange(len(players))
    players = np.array(players)

    def forward_dynamics(coalition: list):
        
        for player_idx in filter(lambda x: x not in coalition, players_idx):
            next_coalition = coalition + [player_idx]
            if not ordered:
                next_coalition_idx = sorted(next_coalition)
                coalition_idx = sorted(coalition)
            else:
                next_coalition_idx = next_coalition
                coalition_idx = coalition            
            # maintain permutation invariance by order
            next_coalition_id = CoalitionMetadata.to_id(players[next_coalition_idx])
            coalition_id = CoalitionMetadata.to_id(players[coalition_idx])
            
            if len(coalition) > 0:
                value[players[player_idx]] += values[next_coalition_id] - values[coalition_id]
            else:
                value[next_coalition_id] += (players.shape[0] - 1) * values[next_coalition_id]
            
            forward_dynamics(next_coalition)

    forward_dynamics([])
    
    norm = math.factorial(players.shape[0])
    value = {player: value * norm ** -1 for player, value in value.items()}

    return value


def nowak_radzik(values: Dict[str, Any], players: list[str]) -> List[float]:
    return shapley(values, players, True)


def sanchez_bergantinos(values: Dict[str, Any], players: list[str]) -> List[float]:
    value = {player: 0.0 for player in players}
    players_idx = np.arange(len(players))
    players = np.array(players)

    def backward_dynamics(coalition: list):
        for player in coalition:
            next_coalition = list(coalition)
            next_coalition.remove(player)
            coalition_id = CoalitionMetadata.to_id(players[coalition])
            # log.info((coalition, next_coalition))
            if len(next_coalition) > 0:
                next_coalition_id = CoalitionMetadata.to_id(players[next_coalition])
                value[players[player]] += values[coalition_id] - values[next_coalition_id]
            else:
                value[players[player]] += values[coalition_id]
            
            backward_dynamics(next_coalition)


    for coalition in permutations(players_idx):
        backward_dynamics(list(coalition))

    norm = math.factorial(players.shape[0]) ** 2
    value = {player: value * norm ** -1 for player, value in value.items()}

    return value


def vpop(values: Dict[str, Any], players: list[str], ordered: bool = False) -> np.array:
    """
        Computes the Value of a Player to Another Player from Hausken and Mohr, 2021
    """
    players_idx = np.arange(len(players))
    players = np.array(players)
    
    if '' not in values:
        values[''] = 0.0

    def subgames_shapley():
        """
            Computes each subgame Shapley value.
        """
        cache = dict()
        cache[''] = np.zeros_like(players_idx, dtype=float)
        
        # computes subgames Shapley values
        subg = dict()
        subg[''] = np.zeros_like(players_idx, dtype=float)

        # BFS traversal of the coalition formation graph
        queue = deque()
        queue.append([]) # start from the empty coalition
        while len(queue):
            next_coalition = queue.popleft()
            for player_idx in filter(lambda x: x not in next_coalition, players_idx):
                queue.append(next_coalition + [player_idx])
            
            if len(next_coalition):
                coalition = list(next_coalition)
                player = coalition.pop()
                
                # whether the characteristic is permutation invariant or not
                o_coalition = sorted(coalition) if not ordered else next_coalition
                o_next_coalition = sorted(next_coalition) if not ordered else next_coalition
                o_next_coalition_id = CoalitionMetadata.to_id(players[o_next_coalition])
                o_coalition_id = CoalitionMetadata.to_id(players[o_coalition])

                next_coalition_id =  CoalitionMetadata.to_id(players[next_coalition])
                coalition_id =  CoalitionMetadata.to_id(players[coalition])

                if next_coalition_id not in cache:
                    cache[next_coalition_id] = np.array(cache[coalition_id])
                
                if o_next_coalition_id not in subg:
                    subg[o_next_coalition_id] = np.zeros_like(subg[o_coalition_id])
                
                cache[next_coalition_id][player] += values[o_next_coalition_id] - values[o_coalition_id]
                subg[o_next_coalition_id] += math.factorial(len(next_coalition)) ** -1 * cache[next_coalition_id]

        return subg
    
    vpop = shapley(
        values=subgames_shapley(),
        players=players,
        ordered=ordered
    )

    return np.array([vpop[key] for key in vpop])


def core(values: Dict[str, Any], players: list[str]):
    pass


def ecore(values: Dict[str, Any], players: list[str]):
    pass


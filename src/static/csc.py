from collections import deque
from itertools import permutations
import itertools
import math
from typing import Any, Dict, List
import numpy as np

from ccgm.utils import CoalitionMetadata


class functional():

    def shapley(
        characteristic_fn: Dict[str, Any], players: list[str], 
        ordered: bool = False
    ) -> List[float]:
        """

        """
        value = {player: 0.0 for player in players}
        players_idx = np.arange(len(players))
        players = np.array(players)

        for permutation in itertools.permutations(players_idx):
            for idx, player in enumerate(permutation):
                if idx == 0:
                    coalition_id = None
                else:
                    coalition = permutation[:idx] if ordered else sorted(permutation[:idx]) 
                    coalition_id = CoalitionMetadata.to_id(players[coalition])
                
                next_coalition = permutation[: idx+1] if ordered else sorted(permutation[: idx+1])
                next_coalition_id = CoalitionMetadata.to_id(players[next_coalition])
                
                if coalition_id is None:
                    value[players[player]] += characteristic_fn[next_coalition_id]
                else:
                    value[players[player]] += characteristic_fn[next_coalition_id] - characteristic_fn[coalition_id]
        
        norm = math.factorial(players.shape[0])
        value = {player: value * norm ** -1 for player, value in value.items()}

        return value


    def nowak_radzik(
        characteristic_fn: Dict[str, Any], players: list[str]
    ) -> List[float]:
        """
            Computes Nowak & Radzik Shapley-based solution concept for
            ordered coalitions. 
        """
        return functional.shapley(characteristic_fn, players, True)


    def sanchez_bergantinos(
        characteristic_fn: Dict[str, Any], players: list[str]
    ) -> List[float]:
        """
            Computes Sanchez & Bergatinos Shapley-based solution concept for
            ordered coalitions.
        """
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
                    value[players[player]] += characteristic_fn[coalition_id] - characteristic_fn[next_coalition_id]
                else:
                    value[players[player]] += characteristic_fn[coalition_id]
                
                backward_dynamics(next_coalition)


        for coalition in permutations(players_idx):
            backward_dynamics(list(coalition))

        norm = math.factorial(players.shape[0]) ** 2
        value = {player: value * norm ** -1 for player, value in value.items()}

        return value

    def subgames_shapley(
        characteristic_fn: Dict[str, Any], players: List[str], 
        ordered: bool = False
    ):
        """
            Computes all subgames Shapley value from the game described by
            `charactersitic_fn`
        """
        players_idx = np.arange(len(players))
        players = np.array(players)
        
        if '' not in characteristic_fn:
            characteristic_fn[''] = 0.0

        cache = dict()
        cache[''] = np.zeros_like(players_idx, dtype=float)
        
        # computes subgames Shapley values
        subgames_values = dict()
        subgames_values[''] = np.zeros_like(players_idx, dtype=float)

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
                
                if o_next_coalition_id not in subgames_values:
                    subgames_values[o_next_coalition_id] = np.zeros_like(subgames_values[o_coalition_id])
                
                cache[next_coalition_id][player] += characteristic_fn[o_next_coalition_id] - characteristic_fn[o_coalition_id]
                subgames_values[o_next_coalition_id] += math.factorial(len(next_coalition)) ** -1 * cache[next_coalition_id]

        return subgames_values

    def vpop(characteristic_fn: Dict[str, Any], players: list[str], ordered: bool = False) -> np.array:
        """
            Computes Hausken and Mohr, 2001 Value of a Player to Another Player (vPoP) 
        """
        subgames_values = functional.subgames_shapley(
            characteristic_fn=characteristic_fn,
            players=players,
            ordered=ordered
        )
        vpop = functional.shapley(
            characteristic_fn=subgames_values,
            players=players,
            ordered=ordered
        )

        return np.array([vpop[key] for key in vpop])

    def core(characteristic_fn: Dict[str, Any], players: list[str]):
        """
          Computes the core of a the cooperative game described by
          `characteristic_fn` among players in `players`.
        """
        raise NotImplementedError()

    def ecore(characteristic_fn: Dict[str, Any], players: list[str], epsilon: float):
        """
          Computes the e-core of a the cooperative game described by
          `characteristic_fn` among players in `players`.
        """
        raise NotImplementedError()

    def dividends(characteristic_fn: Dict[str, any], players: list[str]):
        """
          Compute Harsanyi dividends
        """
        raise NotImplementedError()

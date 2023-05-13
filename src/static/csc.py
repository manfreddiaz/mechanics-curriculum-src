from collections import deque
from itertools import permutations
import itertools
import math
from typing import Any, Callable, Dict, List
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
                    coalition = list(permutation[:idx]) if ordered else sorted(permutation[:idx]) 
                    coalition_id = CoalitionMetadata.to_id(players[coalition])
                
                next_coalition = list(permutation[: idx+1]) if ordered else sorted(permutation[: idx+1])
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

    def subgames(
        characteristic_fn: Dict[str, Any], players: List[str], 
        solution_concept: Callable[[dict[str, float], list], dict],
        ordered: bool = False
    ):
        generator = itertools.permutations if ordered else itertools.combinations

        players_idx = np.arange(len(players))
        players = np.array(players)
        subgames_solutions = {'': np.zeros_like(players, dtype=np.float32)}
        for rank in range(len(players)):
            for combination in generator(players_idx, r=rank+1):
                combination = list(combination)
                subgame_solution = solution_concept(
                    characteristic_fn=characteristic_fn,
                    players=players[combination],
                )
                subgame_id = CoalitionMetadata.to_id(players[combination])
                subgames_solutions[subgame_id] = np.zeros_like(players, dtype=np.float32)
                subgames_solutions[subgame_id][combination] = np.array([subgame_solution[key] for key in subgame_solution])
        
        return subgames_solutions

    def vpop(
            characteristic_fn: Dict[str, Any], players: list[str],
            solution_concept: Callable[[dict, list], dict], ordered: bool = False) -> np.array:
        """
            Computes Hausken and Mohr, 2001 Value of a Player to Another Player (vPoP) 
        """

        subgames_values = functional.subgames(
            characteristic_fn=characteristic_fn,
            players=players, solution_concept=solution_concept,
            ordered=ordered
        )

        vpop = solution_concept(
            characteristic_fn=subgames_values,
            players=players
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

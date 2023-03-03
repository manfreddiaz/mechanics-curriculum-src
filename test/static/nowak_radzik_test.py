import unittest
import pandas as pd
from ccgm.utils import form_coalitions

import numpy as np

import src.static.core as core

class TestNowakRadzik(unittest.TestCase):

    def test_nowak_radzik_ex(self):
        # Nowak and Radzik, pp 152
        players = ['1', '2', '3']
        characteristic = { coalition.id: 0.0 for coalition in form_coalitions(players, ordered=True)}
        characteristic['1+3+2'] = 4
        characteristic['3+1+2'] = 4
        characteristic['2+3+1'] = 4
        characteristic['3+2+1'] = 4
        characteristic['1+2+3'] = 3
        characteristic['2+1+3'] = 3
        characteristic['1+2'] = 2
        characteristic['2+1'] = 2
        characteristic['1+3'] = 3
        characteristic['3+1'] = 3

        s_shapley = {
            '1': 39/18,
            '2': 12/18,
            '3': 15/18
        }
        shapley = core.nowak_radzik(
            pd.Series(characteristic),
            players=players,
        )

        self.assert_(all(np.isclose(s_shapley[key], shapley[key]) for key in shapley))
    
    def test_sanchez_bergantinos_ex_1(self):
        # Nowak and Radzik, pp 152
        players = ['1', '2']
        characteristic = { coalition.id: 0.0 for coalition in form_coalitions(players, ordered=True)}
        characteristic['1+2'] = 1.0

        s_shapley = {
            '1': 0,
            '2': 0.5,
        }
        shapley = core.nowak_radzik(
            pd.Series(characteristic),
            players=players,
        )

        self.assert_(all(np.isclose(s_shapley[key], shapley[key]) for key in shapley))
    
    def test_sanchez_bergantinos_ex_2(self):
        # Nowak and Radzik, pp 152
        players = ['1', '2', '3']
        characteristic = { coalition.id: 0.0 for coalition in form_coalitions(players, ordered=True)}
        characteristic['1+2'] = 1
        characteristic['1+3'] = 1
        characteristic['2+1+3'] = 1
        characteristic['3+1+2'] = 1
        characteristic['1+2+3'] = 2
        characteristic['1+3+2'] = 2

        s_shapley = {
            '1': 0.0,
            '2': 0.5,
            '3': 0.5
        }
        shapley = core.nowak_radzik(
            pd.Series(characteristic),
            players=players,
        )

        self.assert_(all(np.isclose(s_shapley[key], shapley[key]) for key in shapley))



if __name__ == "__main__":
    unittest.main()
import unittest
import pandas as pd
from ccgm.utils import form_coalitions

import numpy as np

import src.static.core as core 

class TestShapley(unittest.TestCase):

    def test_shapley_wiki(self):
        players = ['1', '2', '3']
        characteristic = {
            '': 0.0,
            '1': 0.0,
            '2': 0.0,
            '3': 0.0,
            '1+2': 0.0,
            '1+3': 1.0,
            '2+3': 1.0,
            '1+2+3': 1.0
        }
        s_shapley = {
            '1': 1/6,
            '2': 1/6,
            '3': 2/3
        }
        shapley = core.shapley(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        self.assert_(all(np.isclose(s_shapley[key], shapley[key]) for key in shapley))

    def test_hausken_mohr_1(self):
        players = ['1', '2', '3']
        characteristic = {
            '': 0.0,
            '1': 180.0,
            '2': 0.0,
            '3': 0.0,
            '1+2': 360.0,
            '1+3': 540.0,
            '2+3': 0.0,
            '1+2+3': 540.0
        }
        s_shapley = {
            '1': 390.0,
            '2': 30.0,
            '3': 120.0
        }
        shapley = core.shapley(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        self.assert_(all(np.isclose(s_shapley[key], shapley[key]) for key in shapley))
    
    def test_gto_understanding(self):
        # from https://youtu.be/9OFMRiAVH-w?t=815
        players = ['1', '2']
        characteristic = {
            '1': 1.0,
            '2': 2.0,
            '1+2': 4.0,
        }
        s_shapley = {
            '1': 1.5,
            '2': 2.5,
        }
        shapley = core.shapley(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        self.assert_(all(np.isclose(s_shapley[key], shapley[key]) for key in shapley))

    def test_shapley_calc_5(self):
        # http://shapleyvalue.com/?example=5
        players = ['1', '2']
        characteristic = {
            '1': 1000000.0,
            '2': 200000.0,
            '1+2': 1400000.0,
        }
        s_shapley = {
            '1': 1100000.,
            '2': 300000,
        }
        shapley = core.shapley(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        self.assert_(all(np.isclose(s_shapley[key], shapley[key]) for key in shapley))
    
    def test_shapley_calc_10(self):
        # http://shapleyvalue.com/?example=10
        players = ['1', '2', '3', '4', '5', '6', '7']
        
        characteristic = { coalition.id: 0.0 for coalition in form_coalitions(players)}
        characteristic[''] = 0.0
        characteristic['2+3+4+5+6+7'] = 3000000.00
        characteristic['1+2+3+4+5+6+7'] = 3500000.00

        s_shapley = {
            '1': 71428.571428571,
            '2':  571428.57142857,
            '3':  571428.57142857,
            '4':  571428.57142857,
            '5':  571428.57142857,
            '6':  571428.57142857,
            '7':  571428.57142857
        }
        shapley = core.shapley(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        self.assert_(all(np.isclose(s_shapley[key], shapley[key]) for key in shapley))


if __name__ == "__main__":
    unittest.main()
import unittest
import pandas as pd
from ccgm.utils import form_coalitions

import numpy as np

import src.static.core as core


class vPoPTest(unittest.TestCase):

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
        s_vpop = np.array([
            [295.0,  25.0,  70.0],
            [ 25.0,  25.0, -20.0],
            [ 70.0, -20.0,  70.0]
        ])
        vpop = core.vpop(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        self.assert_(np.isclose(vpop, s_vpop).astype(float).mean() == 1.0)
    

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
        s_shapley = [1/6, 1/6, 2/3]
        vpop = core.vpop(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        shapley = vpop.sum(axis=1)

        self.assert_(np.isclose(shapley, s_shapley).astype(float).mean() == 1.0)
    
    def test_gto_understanding(self):
        # from https://youtu.be/9OFMRiAVH-w?t=815
        players = ['1', '2']
        characteristic = {
            '1': 1.0,
            '2': 2.0,
            '1+2': 4.0,
        }
        s_shapley = [1.5, 2.5]
        
        vpop = core.vpop(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        shapley = vpop.sum(axis=1)

        self.assert_(np.isclose(shapley, s_shapley).astype(float).mean() == 1.0)

    def test_shapley_calc_5(self):
        # http://shapleyvalue.com/?example=5
        players = ['1', '2']
        characteristic = {
            '1': 1000000.0,
            '2': 200000.0,
            '1+2': 1400000.0,
        }
        s_shapley = [1100000., 300000]

        vpop = core.vpop(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        shapley = vpop.sum(axis=1)

        self.assert_(np.isclose(shapley, s_shapley).astype(float).mean() == 1.0)
    
    def test_shapley_calc_10(self):
        # http://shapleyvalue.com/?example=10
        players = ['1', '2', '3', '4', '5', '6', '7']
        
        characteristic = { coalition.id: 0.0 for coalition in form_coalitions(players)}
        characteristic[''] = 0.0
        characteristic['2+3+4+5+6+7'] = 3000000.00
        characteristic['1+2+3+4+5+6+7'] = 3500000.00

        s_shapley = [
            71428.571428571,
            571428.57142857,
            571428.57142857,
            571428.57142857,
            571428.57142857,
            571428.57142857,
            571428.57142857
        ]
        vpop = core.vpop(
            pd.Series(characteristic),
            players=players,
            ordered=False
        )

        shapley = vpop.sum(axis=1)

        self.assert_(np.isclose(shapley, s_shapley).astype(float).mean() == 1.0)

if __name__ == "__main__":
    unittest.main()
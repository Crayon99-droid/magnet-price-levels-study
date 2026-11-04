#!/usr/bin/env python3
"""Small deterministic guardrail tests for DC + trade backtest."""
import unittest
import numpy as np
from btc_magnet_dc_v2 import first_target,sim_trade,dc_detector
from btc_overshoot_management import sim_dynamic
class TestCausality(unittest.TestCase):
    def test_bar_ambiguity_stops_first(self):
        o=np.array([100.,100.]);h=np.array([101.,101.]);l=np.array([99.,99.]);c=np.array([100.,100.])
        r,bp,gross,i,reason=sim_trade(o,h,l,c,1,0,100.,.5,2,13.)
        self.assertEqual(reason,-1)
        self.assertAlmostEqual(r,-1.13)
    def test_double_target_is_ambiguous(self):
        h=np.array([105.,107.]);l=np.array([95.,93.]);v,i=first_target(h,l,0,2,104.,96.)
        self.assertEqual((v,i),(2,0))
    def test_dc_reversal_only_after_threshold(self):
        p=np.array([100.,101.,102.,101.5,100.9,100.7,100.4,101.5]);th=np.full(len(p),.01)
        ix,side,_,_=dc_detector(p,th)
        self.assertTrue(np.any(side==1))
        self.assertTrue(np.any(side==-1))
        self.assertTrue(np.all(np.diff(ix)>0))
    def test_profit_lock_only_after_close(self):
        o=np.array([100.,100.45,100.2]);h=np.array([100.6,100.5,100.3]);l=np.array([100.02,100.1,100.1]);c=np.array([100.5,100.2,100.2])
        r,net,gross,ei,why,protected=sim_dynamic(o,h,l,c,0,1,100.,3,13.,1)
        self.assertEqual((ei,why,protected),(1,-1,1))
        self.assertAlmostEqual(net,5.0,places=6)
    def test_scheduled_early_exit_occurs_later(self):
        o=np.full(20,100.);h=np.full(20,100.05);l=np.full(20,99.70);c=np.full(20,99.70)
        r,net,gross,ei,why,protected=sim_dynamic(o,h,l,c,0,1,100.,20,13.,2)
        self.assertEqual((ei,why),(10,3))
    def test_dynamic_fixed_matches_base(self):
        o=np.array([100.,100.,100.]);h=np.array([100.4,100.6,100.7]);l=np.array([99.9,99.9,99.9]);c=np.array([100.3,100.5,100.6]);
        d=sim_dynamic(o,h,l,c,0,1,100.,3,13.,0)
        b=sim_trade(o,h,l,c,1,0,100.,.5,3,13.)
        self.assertAlmostEqual(d[0],b[0]);self.assertEqual(d[3],b[3])
if __name__=='__main__':unittest.main(verbosity=2)

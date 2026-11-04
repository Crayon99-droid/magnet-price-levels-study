import unittest,numpy as np,pandas as pd
from magnet_dc_h5 import simulate_event_both,choose
class H5Tests(unittest.TestCase):
    def test_cost_and_tp_at_next_open(self):
        # From close confirmation at bar 0, enter at open[1]=100, 100bp risk => 1.0 distance, target 100.5, stop 99.
        o=np.array([100.,100.,100.,100.]);h=np.array([100.,100.6,100.,100.]);l=np.array([100.,99.7,100.,100.]);c=np.array([100.,100.3,100.,100.])
        ret,ei,why=simulate_event_both(o,h,l,c,np.array([0]),np.array([100.]),deadline=2,cost=13.,tp=.5)
        self.assertAlmostEqual(ret[0,0],.37,places=8)
        self.assertEqual(why[0,0],1)
    def test_stop_first_ambiguous_bar(self):
        o=np.array([100.,100.,100.]);h=np.array([100.,100.6,100.]);l=np.array([100.,98.9,100.]);c=np.array([100.,100.,100.]);ret,ei,why=simulate_event_both(o,h,l,c,np.array([0]),np.array([100.]),2,13,.5)
        self.assertAlmostEqual(ret[0,0],-1.13,places=8)
        self.assertEqual(why[0,0],-1)
    def test_abstention_and_nonoverlap(self):
        d=pd.DataFrame([{'idx':1,'symbol':'X','time':pd.Timestamp('2024-01-01',tz='UTC'),'trade_side':1,'dc_side':1,'near_side':1,'ratio':.3,'room_R':.9,'exit_idx':10,'net_R':.37,'held_minutes':4},
                        {'idx':1,'symbol':'X','time':pd.Timestamp('2024-01-01',tz='UTC'),'trade_side':-1,'dc_side':1,'near_side':1,'ratio':.3,'room_R':.9,'exit_idx':10,'net_R':-1.13,'held_minutes':4},
                        {'idx':2,'symbol':'X','time':pd.Timestamp('2024-01-01',tz='UTC'),'trade_side':1,'dc_side':1,'near_side':1,'ratio':.3,'room_R':.9,'exit_idx':11,'net_R':.37,'held_minutes':3},
                        {'idx':2,'symbol':'X','time':pd.Timestamp('2024-01-01',tz='UTC'),'trade_side':-1,'dc_side':1,'near_side':1,'ratio':.3,'room_R':.9,'exit_idx':11,'net_R':-1.13,'held_minutes':3}])
        v=choose(d,scores=np.array([-.1,-.2,.2,-.1]),policy='model')
        self.assertEqual(len(v),1);self.assertEqual(int(v.iloc[0].idx),2)
        v=choose(d,scores=np.array([.6,.8,.2,.1]),policy='model')
        self.assertEqual(len(v),1);self.assertEqual(int(v.iloc[0].trade_side),-1)
    def test_theta_is_previous_day(self):
        # Uses original source: any direct change to the CURRENT day cannot change current-day theta.
        import btc_magnet_dc_v2 as prev
        times=pd.date_range('2024-01-01',periods=4*1440,freq='min',tz='UTC');c=100*np.exp(np.cumsum(np.random.default_rng(0).normal(0,.0005,len(times))))
        d=pd.DataFrame({'minute':times,'close':c});before=prev.compute_theta(d)
        d.loc[d.minute.dt.day==3,'close'] *= 1.2
        after=prev.compute_theta(d)
        mask=(d.minute.dt.day==3)
        self.assertTrue(np.allclose(before[mask],after[mask],equal_nan=True))
if __name__=='__main__':unittest.main()

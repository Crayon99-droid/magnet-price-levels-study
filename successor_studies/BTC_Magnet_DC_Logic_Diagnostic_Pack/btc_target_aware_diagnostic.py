#!/usr/bin/env python3
"""BTC magnet-target diagnostic. Use saved causal events + original Bybit M1 klines.
For comparison ONLY. No parameter search, no live trading. Both exit choices
use exactly the same events separated by 120min to avoid position-overlap bias.
"""
from pathlib import Path
import zipfile, pandas as pd, numpy as np
from numba import njit
ROOT=Path('/mnt/data')
OUT=ROOT/'btc_magnet_target_diagnostic'
OUT.mkdir(exist_ok=True)
FEE_BP=13.
HOLD=120

@njit
def simulate(op, hi, lo, cl, entry_i, side, target, stop, maxbars):
    stop_hit=0
    final=min(len(op),entry_i+maxbars)
    exitp=cl[final-1];exit_idx=final-1
    for j in range(entry_i,final):
        s = lo[j]<=stop if side==1 else hi[j]>=stop
        t = hi[j]>=target if side==1 else lo[j]<=target
        if s:
            stop_hit=-1;exitp=stop;exit_idx=j;break
        if t:
            stop_hit=1;exitp=target;exit_idx=j;break
    return side*(exitp/op[entry_i]-1)*10000 -FEE_BP,stop_hit,exit_idx

def main():
    bars=[]
    for y in range(2021,2025):
        paths=list(ROOT.glob(f'BTCUSDT_1m_{y}*.zip'))
        if len(paths)!=1:raise ValueError((y,paths))
        with zipfile.ZipFile(paths[0]) as z:
            mem=[m for m in z.namelist() if m.endswith('.csv')]
            if len(mem)!=1:raise ValueError(mem)
            with z.open(mem[0]) as f:
                bars.append(pd.read_csv(f,usecols=['open','high','low','close']))
    a=pd.concat(bars,ignore_index=True)
    op=a.open.to_numpy(dtype=np.float64);hi=a.high.to_numpy(dtype=np.float64);lo=a.low.to_numpy(dtype=np.float64);cl=a.close.to_numpy(dtype=np.float64)
    ev=pd.read_csv(ROOT/'btc_magnet_dc_v2'/'dc_magnet_events_all.csv.gz')
    ev=ev[ev.aligned & (ev.ratio<=.5)].sort_values('idx')
    results=[];selected=[];blocked=0;no_room=0;expiry=0
    for mode in ('all_aligned','cost_and_0p2R_room'):
        previous=-HOLD-100
        for z in ev.itertuples(index=False):
            idx=int(z.idx)+1
            if idx>=len(op) or idx+HOLD>len(op):expiry+=1;continue
            side=int(z.dc_side);entry=op[idx];risk_bp=z.theta*10000
            magnet=float(z.upper if side==1 else z.lower)
            reward_bp=side*(magnet/entry-1)*10000
            if reward_bp<=0:continue
            if mode=='cost_and_0p2R_room' and reward_bp < FEE_BP+0.2*risk_bp:
                no_room+=1;continue
            if idx<=previous+HOLD:
                blocked+=1;continue
            previous=idx # fixed event schedule independent of exit
            stop=entry*(1-side*risk_bp/10000)
            for label,tp in [('magnet',magnet),('fixed_0p5R',entry*(1+side*0.5*risk_bp/10000))]:
                net,reason,exit_idx=simulate(op,hi,lo,cl,idx,side,tp,stop,HOLD)
                results.append((mode,label,int(z.year),idx,side,risk_bp,reward_bp,net,net/risk_bp,reason,exit_idx-idx+1))
    tab=pd.DataFrame(results,columns=['cohort','exit','year','entry_index','side','risk_bp','magnet_room_bp','net_bp','net_R','exit_reason','held_minutes'])
    tab.to_csv(OUT/'trade_comparison.csv',index=False)
    summ=[]
    for (mode,label,year),g in tab.groupby(['cohort','exit','year'],sort=True):
        wins=g.net_R[g.net_R>0].sum();losses=-g.net_R[g.net_R<0].sum();summ.append(dict(cohort=mode,exit=label,year=year,trades=len(g),mean_net_R=round(g.net_R.mean(),4),mean_net_bp=round(g.net_bp.mean(),3),win_percent=round(100*(g.net_R>0).mean(),2),profit_factor=round(wins/losses,3) if losses else np.nan,median_holding_minutes=g.held_minutes.median(),take_profit_pct=round(100*(g.exit_reason==1).mean(),2),stop_pct=round(100*(g.exit_reason==-1).mean(),2)))
    sumx=pd.DataFrame(summ);sumx.to_csv(OUT/'yearly_summary.csv',index=False)
    print('STRICT matched-entry comparison (120m time embargo), not a tuned strategy')
    print(sumx.to_string(index=False))
    print('diagnostic:',{'skipped_cost_room':no_room,'blocked_by_fixed_120m_embargo':blocked})
if __name__=='__main__':main()

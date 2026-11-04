#!/usr/bin/env python3
"""Prespecified weekly block uncertainty and additional 5bp-cost shocks on existing BTC results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('btc_magnet_dc_v2'); rng=np.random.default_rng(20261012)
def boot_weekly_mean(df,col,time_col='time',iters=3000):
    df=df.copy(); df['week']=pd.to_datetime(df[time_col],utc=True).dt.tz_localize(None).dt.to_period('W').astype(str)
    agg=df.groupby('week')[col].agg(['sum','count']).to_numpy(dtype='float64')
    if len(agg)<8:return None
    picks=rng.integers(0,len(agg),size=(iters,len(agg)))
    sampled=agg[picks]
    estimates=sampled[:,:,0].sum(axis=1)/sampled[:,:,1].sum(axis=1)
    return [round(float(x),5) for x in np.percentile(estimates,[2.5,97.5])]

e=pd.read_csv(ROOT/'dc_magnet_events_all.csv.gz')
r=e[e.target_first.isin([-1,1])].copy()
r['near_excess']=r.near_first-r.distance_only_p_near
res={'near_first':{'n':len(r),'actual_pct':float(r.near_first.mean()*100),'geometry_heuristic_pct':float(r.distance_only_p_near.mean()*100),
                   'excess_pp':float(r.near_excess.mean()*100),'weekly_block_CI95_excess_pp': [v*100 for v in boot_weekly_mean(r,'near_excess')]}}
for name in ['DC_only_0p5R_120m','Magnet_only_0p5R_120m','Aligned_ratio0p5_0p5R_120m','Aligned_room_0p5R_30m']:
    t=pd.read_csv(ROOT/f'trades_{name}.csv')
    res[name]={}
    for grp,x in [('2021-23',t[t.year<=2023]),('2024',t[t.year==2024]),('all',t)]:
        x=x.copy();x['shock_R']=x.net_R-5/x.risk_bp
        res[name][grp]={'n':len(x),'mean_net_R':float(x.net_R.mean()),'CI95':boot_weekly_mean(x.rename(columns={'event_time':'time'}),'net_R'),
                        'additional_5bp_cost_mean_R':float(x.shock_R.mean())}
fixed=pd.read_csv(ROOT/'overshoot_matched_trade_log.csv')
for grp,z in [('2021-23',fixed[fixed.year<=2023]),('2024',fixed[fixed.year==2024])]:
    pv=z.pivot(index='entry_idx',columns='mode',values='net_R')
    extra=z[z['mode']=='fixed_0p5R'][['entry_idx','event_time']]
    pv=pv.merge(extra,on='entry_idx')
    pv['protected_delta']=pv.protected_1R-pv.fixed_0p5R
    res['protected_runner_delta_'+grp]={'mean_R':float(pv.protected_delta.mean()),'CI95':boot_weekly_mean(pv.rename(columns={'event_time':'time'}),'protected_delta')}
(ROOT/'statistical_audit.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))

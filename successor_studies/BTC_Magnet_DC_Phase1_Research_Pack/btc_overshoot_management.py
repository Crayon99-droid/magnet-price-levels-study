#!/usr/bin/env python3
"""Matched-opportunity (same entries) causal profit-protection experiment for BTC DC+Magnet.
Uses BTC raw ZIP verified by btc_magnet_dc_v2.py and its frozen trade event list.
Not a tuned or forward-validated strategy. Funding is not available; quotes are M1 OHLCV.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit
import btc_magnet_dc_v2 as base

@njit(cache=True)
def sim_dynamic(o,h,l,c,entry_idx,side,risk_bp,horizon,fee_bp,mode):
    entry=o[entry_idx]; dist=entry*risk_bp/10000.0
    stop=entry-side*dist; tp=entry+side*dist*(0.5 if mode==0 or mode==2 else 1.0)
    limit=min(len(o),entry_idx+horizon)
    exitp=c[limit-1];reason=0; exit_idx=limit-1;protection=False; strongest_close=0.0
    early_scheduled=False
    for j in range(entry_idx,limit):
        # Exit scheduled by previous bar's close; executes at present OPEN.
        if early_scheduled:
            exitp=o[j];exit_idx=j;reason=3;break
        # Stop gap check, then conservative intrabar stop first, then TP.
        if side*(o[j]-stop)<=0:
            exitp=o[j];exit_idx=j;reason=-2;break
        if (l[j]<=stop if side==1 else h[j]>=stop):
            exitp=stop;exit_idx=j;reason=-1;break
        if (h[j]>=tp if side==1 else l[j]<=tp):
            exitp=tp;exit_idx=j;reason=1;break
        gross_close=side*(c[j]/entry-1)*10000
        if gross_close>strongest_close:strongest_close=gross_close
        # All updates caused by this closed bar take effect no earlier than NEXT bar.
        if j+1<limit:
            if mode==2 or mode==3:
                if j-entry_idx>=9 and not protection and gross_close<=-0.25*risk_bp:
                    early_scheduled=True
            if mode==1 or mode==3:
                # Directly-observed excursion since entry; causal proxy of *ongoing* OS strength.
                activation=max(0.35*risk_bp, fee_bp+0.10*risk_bp)
                if gross_close>=activation and strongest_close>=0.35*risk_bp:
                    requested=max(fee_bp+0.05*risk_bp,0.12*risk_bp)
                    if strongest_close>=.70*risk_bp:requested=max(requested,0.35*risk_bp)
                    # Do not set a stop on the wrong side of observed last close.
                    if requested<gross_close:
                        locked=entry+side*requested*entry/10000.0
                        if side*(locked-stop)>0:stop=locked
                        protection=True
    gross=side*(exitp/entry-1)*10000; net=gross-fee_bp
    return net/risk_bp,net,gross,exit_idx,reason, int(protection)

def summary(t):
    r=t.net_R.values; win=r[r>0].sum();loss=-r[r<0].sum()
    return {'trades':len(t),'net_R_mean':round(float(r.mean()),4),'net_bp_mean':round(float(t.net_bp.mean()),3),
            'win_pct':round(100*float((r>0).mean()),1),'PF':round(float(win/loss),3) if loss>0 else None,
            'median_holding_min':float(t.minutes.median()),'green_exit_pct':round(100*float((t.net_R>0).mean()),1),
            'early_stop_pct':round(100*float(t.reason.isin([-1,-2,3]).mean()),1)}

def run(root,out):
    d,_=base.load_crypto(root)
    o=d.open.to_numpy();h=d.high.to_numpy();l=d.low.to_numpy();c=d.close.to_numpy()
    candidates=pd.read_csv(out/'trades_Aligned_ratio0p5_0p5R_120m.csv',parse_dates=['event_time'])
    # Matched executable candidate set from original 120m baseline. All candidate entries remain fixed.
    res=[]
    for z in candidates.itertuples(index=False):
        for mode,name in enumerate(['fixed_0p5R','protected_1R','early_fail_0p5R','protected_plus_early_fail_1R']):
            r,net,gross,ex,why,pro=sim_dynamic(o,h,l,c,int(z.entry_idx),int(z.side),float(z.risk_bp),120,13.0,mode)
            res.append({'event_time':z.event_time,'year':int(z.year),'entry_idx':int(z.entry_idx),'mode':name,
                        'net_R':r,'net_bp':net,'gross_bp':gross,'exit_idx':ex,'reason':why,
                        'minutes':int(ex)-int(z.entry_idx)+1,'protection_activated':pro,'risk_bp':z.risk_bp})
    t=pd.DataFrame(res);t.to_csv(out/'overshoot_matched_trade_log.csv',index=False)
    ag=[]
    for (name,year),g in t.groupby(['mode','year']):ag.append({'policy':name,'year':int(year),**summary(g)})
    agg=pd.DataFrame(ag);agg.to_csv(out/'overshoot_matched_summary.csv',index=False)
    base0=t[t['mode']=='fixed_0p5R'].sort_values('entry_idx').reset_index(drop=True)
    orig=candidates.sort_values('entry_idx').reset_index(drop=True)
    err=np.max(np.abs(base0.net_R.to_numpy()-orig.net_R.to_numpy()))
    if err>1e-7:raise RuntimeError(f'DYNAMIC baseline mismatch {err}')
    import json
    json.dump({'fixed_baseline_max_R_diff':float(err),'policy':'aligned <=0.5 distance ratio; entry schedule remains identical to 120m fixed TP baseline',
               'profit_lock':'After a *completed bar* close profits exceed maximum of 0.35R and fee+0.10R, trailing stop for NEXT bar locks fee+0.05R, then +0.35R on strong continuation','early_fail':'After 10 completed bars at gross -0.25R exit at next minute open',
               'interpretation':'Matched-entry comparison: no extra trading of freed capacity; no true DC reversal classifier fitted; tests causal realized post-entry OS proxy',
               'funding':'Not included; historical funding required to certify fully net results','ohlcv':'No historical quotes, 1bp slippage allowance each fill'},open(out/'overshoot_methods.json','w'),indent=2)
    print(agg.to_string(index=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='/mnt/data');p.add_argument('--out',default='/mnt/data/btc_magnet_dc_v2');a=p.parse_args();run(a.data_dir,Path(a.out))

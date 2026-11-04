#!/usr/bin/env python3
"""Compute frozen policy trade CIs, source audit completion, and intrinsic-state diagnostics."""
import json, hashlib,glob,zipfile
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
OUT=HERE/'Magnet_DC_H5'

def weekly_ci(frame,seed=204,draws=5000):
    if len(frame)<5:return (None,None)
    weeks=frame.time.dt.tz_localize(None).dt.to_period('W').astype(str)
    g=frame.groupby(weeks).net_R.agg(['sum','count']).to_numpy('float64')
    if len(g)<5:return (None,None)
    rng=np.random.default_rng(seed);r=[]
    for _ in range(draws):
        ix=rng.integers(0,len(g),size=len(g));z=g[ix];r.append(z[:,0].sum()/z[:,1].sum())
    return (round(float(np.quantile(r,.025)),4),round(float(np.quantile(r,.975)),4))

if __name__=='__main__':
    known=json.load(open(HERE/'crypto_source_dc_audit.json'))['symbols']
    files=[]
    for sym in ['BTCUSDT','ETHUSDT','XRPUSDT']:
        for x in known[sym]['files']:
            ys=x['year'];paths=list(HERE.glob(f'{sym}_1m_{ys}*.zip'))
            if len(paths)!=1:raise ValueError(paths)
            p=paths[0]; sha=hashlib.sha256(p.read_bytes()).hexdigest()
            if sha!=x['sha256']:raise ValueError(f'hash mismatch {sym} {ys}')
            with zipfile.ZipFile(p) as z:
                if z.testzip() is not None:raise ValueError('crc')
            files.append({'symbol':sym,'year':ys,'archive':p.name,'rows':x['rows'],'sha256':sha,
                          'reproduces_prior_local_preflight_hash':True,
                          'independent_manifest_match':sym=='BTCUSDT'})
    audit={'archives':files,'archive_count':len(files),'minute_candles':sum(x['rows'] for x in files),
           'provenance':'BTC independently agrees with preexisting Bybit source manifest; ETH/XRP agree with previously supplied local audit but no independent manifest supplied.',
           'status':'PASS archive hashes and ZIP CRC; raw full data chronology/OHLC tests performed in initial complete research run'}
    (OUT/'source_audit.json').write_text(json.dumps(audit,indent=2))
    results=pd.read_csv(OUT/'yearly_results.csv')
    trades=pd.read_csv(OUT/'model_trade_logs.csv.gz',parse_dates=['time'])
    uncertainty=[]
    for (sym,year,mod),p in trades.groupby(['symbol','year','policy']):
        a,b=weekly_ci(p);uncertainty.append({'symbol':sym,'year':year,'model':mod,'n':len(p),'mean_net_R':float(p.net_R.mean()),'weekly_bootstrap_lower':a,'weekly_bootstrap_upper':b})
    pd.DataFrame(uncertainty).to_csv(OUT/'model_weekly_uncertainty.csv',index=False)
    out=[]
    for sym in ['BTCUSDT','ETHUSDT','XRPUSDT']:
        d=pd.read_pickle(OUT/f'features_{sym}.pkl');events=d[d.trade_side==d.dc_side].copy()
        for y,g in events.groupby('year'):
            out.append({'symbol':sym,'year':y,'events':len(g),
                        'small_and_large_dc_agree':round(float(g.fast_aligned_slow.mean()),6),
                        'median_R_magnet_room':round(float(g.room_R.median()),5),
                        'median_dc_risk_bp':round(float(g.slow_theta_bp.median()),2),
                        'mean_DC_direction_outcome_net_R_overlapping_events':round(float(g.net_R.mean()),5),
                        'cost_R_at_median_stop':round(float(13/g.slow_theta_bp.median()),4)})
    pd.DataFrame(out).to_csv(OUT/'state_diagnostics.csv',index=False)
    print('AUDIT',len(files),sum(x['rows'] for x in files))
    print(pd.DataFrame(uncertainty).to_string(index=False))
    print(pd.DataFrame(out).to_string(index=False))

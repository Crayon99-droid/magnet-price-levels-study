#!/usr/bin/env python3
"""H5 multi-scale causal DC + magnetic feature tests across BTC/ETH/XRP.
No live trading. Uses public Bybit OHLCV; 13 bp modeled round trip (no funding).
Run: python magnet_dc_h5.py --data-dir /mnt/data --out /mnt/data/Magnet_DC_H5
"""
import os,glob,zipfile,json,hashlib,argparse,time,sys
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import btc_magnet_dc_v2 as prior

THETA_L=1.25
THETA_S=.50
TOTAL_COST_BP=13.0
HORIZON=120
TP_R=.5
EXPECTED_COUNTS={'BTCUSDT':{2021:525600,2022:525600,2023:525600,2024:527040},
                 'ETHUSDT':{2021:420480,2022:525600,2023:525600,2024:527040,2025:525600,2026:336677},
                 'XRPUSDT':{2021:334946,2022:525600,2023:525600,2024:527040,2025:525600,2026:336677}}

@njit(cache=True)
def simulate_event_both(o,h,l,c,inds,risks,deadline=120,cost=13.,tp=.5):
    m=len(inds)
    outs=np.empty((m,2),np.float64)
    exit_i=np.empty((m,2),np.int64)
    exit_code=np.empty((m,2),np.int8)
    for k in range(m):
        idx=inds[k]+1
        p=o[idx];rb=risks[k];dist=p*rb/10000
        fin=min(len(o),idx+deadline)
        for a in range(2):
            side=1 if a==0 else -1
            tgt=p+side*dist*tp;stp=p-side*dist
            ep=c[fin-1];ei=fin-1;reason=0
            for t in range(idx,fin):
                stopped=(l[t]<=stp) if side==1 else (h[t]>=stp)
                profit=(h[t]>=tgt) if side==1 else (l[t]<=tgt)
                if stopped:ep=stp;ei=t;reason=-1;break
                if profit:ep=tgt;ei=t;reason=1;break
            netbp=side*(ep/p-1)*10000-cost
            outs[k,a]=netbp/rb
            exit_i[k,a]=ei
            exit_code[k,a]=reason
    return outs,exit_i,exit_code

def load_symbol(name, data_dir,expected_hashes):
    years=sorted(EXPECTED_COUNTS[name]); dd=[];audit=[]
    for year in years:
        paths=glob.glob(str(data_dir/f'{name}_1m_{year}*.zip'))
        if len(paths)!=1:raise RuntimeError(f'{name} {year} file missing/ambiguous: {paths}')
        p=Path(paths[0]);hh=hashlib.sha256(p.read_bytes()).hexdigest()
        if year in expected_hashes and hh != expected_hashes[year]:raise RuntimeError(f'Hash mismatch {p.name}')
        with zipfile.ZipFile(p) as z:
            if z.testzip():raise RuntimeError('ZIP CRC failure '+p.name)
            members=[m for m in z.namelist() if not m.endswith('/')]
            if members!=[f'{name}_1m_{year}.csv']:raise RuntimeError('Unexpected members '+str(members))
            with z.open(members[0]) as f:d=pd.read_csv(f)
        if list(d.columns)!=prior.REQUIRED:raise RuntimeError('Schema not expected')
        if len(d)!=EXPECTED_COUNTS[name][year]:raise RuntimeError(f'Count wrong {name}{year}')
        t=d.timestamp_ms.to_numpy('int64');
        if not np.all(np.diff(t)==60000):raise RuntimeError('Internal chronology error')
        if (d[['open','high','low','close']].min(axis=1)<=0).any() or (d.low>np.minimum(d.open,d.close)).any() or (d.high<np.maximum(d.open,d.close)).any():raise RuntimeError('Invalid OHLC')
        if not (pd.to_datetime(d.timestamp_utc,utc=True).astype('int64')//1000000 ==t).all():raise RuntimeError('UTC mismatch')
        audit.append(dict(symbol=name,year=year,rows=len(d),first=str(d.timestamp_utc.iloc[0]),last=str(d.timestamp_utc.iloc[-1]),sha256=hh,sha_match=(year in expected_hashes)))
        dd.append(d)
        print('AUDIT',name,year,len(d),'verifiedhash?',year in expected_hashes,flush=True)
    d=pd.concat(dd,ignore_index=True)
    if not np.all(np.diff(d.timestamp_ms)==60000):raise RuntimeError('Gap at year boundary')
    d['minute']=pd.to_datetime(d.timestamp_ms,utc=True,unit='ms')
    return d,audit

def make_features(d,symbol):
    tstart=time.monotonic()
    th=prior.compute_theta(d) # k=1.25 trailing UTC previous completed daily median rolling 240m realized vol
    cl=d.close.to_numpy('float64'); ids,side,os_prev,age=prior.dc_detector(cl,th)
    fast_th=th*(THETA_S/THETA_L)
    fastids,fastside,fastos,fastage=prior.dc_detector(cl,fast_th)
    lv=prior.build_levels(d)
    ev=prior.make_event_frame(d,lv,ids,side,os_prev,age,th)
    # nearest fast confirmation known before or at slow close, and completed fast DC leg sequences
    eidx=ev.idx.to_numpy('int64'); fi=np.searchsorted(fastids,eidx,side='right')-1
    valid=fi>=0
    ev=ev[valid].copy().reset_index(drop=True);eidx=ev.idx.to_numpy('int64');fi=fi[valid]
    fs=fastside[fi]; fprev=fastos[fi];fage=fastage[fi];f_last=fastids[fi]
    ev['fast_side']=fs
    ev['fast_aligned_slow']=fs==ev.dc_side.to_numpy()
    ev['fast_events_last_120m']=np.searchsorted(fastids,eidx,'right')-np.searchsorted(fastids,eidx-120,'left')
    ev['fast_event_age_minutes']=eidx-f_last
    ev['fast_prev_os_R']=fprev
    ev['fast_prev_event_length']=fage
    for j in (1,2,3):
        ix=fi-j
        ev[f'fast_os_lag{j}_R']=np.where(ix>=0,fastos[np.maximum(ix,0)],np.nan)
        ev[f'fast_age_lag{j}']=np.where(ix>=0,fastage[np.maximum(ix,0)],np.nan)
    for k in (5,30,120,360):
        shifted=np.maximum(0,eidx-k)
        ev[f'past_return_{k}m_bp']=np.where(eidx>=k,10000*(cl[eidx]/cl[shifted]-1),np.nan)
    rets=np.log(d.close).diff()
    ev['past_vol_60m_bp']=1e4*np.sqrt(rets.pow(2).rolling(60,min_periods=60).sum().to_numpy()[eidx])
    ev['past_vol_240m_bp']=1e4*np.sqrt(rets.pow(2).rolling(240,min_periods=240).sum().to_numpy()[eidx])
    ev['fast_theta_bp']=fast_th[eidx]*10000
    ev['slow_theta_bp']=th[eidx]*10000
    ev['hour_sin']=np.sin(2*np.pi*ev.time.dt.hour/24)
    ev['hour_cos']=np.cos(2*np.pi*ev.time.dt.hour/24)
    ev['weekend']=ev.time.dt.dayofweek>=5
    ev['symbol']=symbol
    # No target outcomes used as ML features; trade labels separately after features frozen
    risks=ev.slow_theta_bp.to_numpy('float64')
    take=np.isfinite(risks)&(risks>0)&(eidx+121<len(d))
    ev=ev[take].copy().reset_index(drop=True);eidx=ev.idx.to_numpy('int64');risks=ev.slow_theta_bp.to_numpy('float64')
    returns,exits,reasons=simulate_event_both(d.open.to_numpy('float64'),d.high.to_numpy('float64'),d.low.to_numpy('float64'),cl,eidx,risks)
    # Explicit action rows; same event yields highly dependent two actions, grouped by year/time.
    f=[]
    for col,asign in [('up',1),('down',-1)]:
        tab=ev.copy();tab['trade_side']=asign
        tab['room_to_trade_magnet_bp']=np.where(asign==1,tab.up_distance_bp,tab.down_distance_bp)
        tab['room_R']=tab.room_to_trade_magnet_bp / tab.slow_theta_bp
        tab['action_dc_alignment']=tab.dc_side==asign
        tab['action_fast_alignment']=tab.fast_side==asign
        tab['action_magnet_near']=tab.near_side==asign
        tab['action_past_5m']=tab.past_return_5m_bp*asign
        tab['action_past_30m']=tab.past_return_30m_bp*asign
        tab['action_past_120m']=tab.past_return_120m_bp*asign
        tab['action_past_360m']=tab.past_return_360m_bp*asign
        actionidx=0 if asign==1 else 1
        tab['net_R']=returns[:,actionidx]
        tab['exit_idx']=exits[:,actionidx]
        tab['exit_reason']=reasons[:,actionidx]
        tab['held_minutes']=exits[:,actionidx]-eidx
        tab['idx']=eidx
        f.append(tab)
    out=pd.concat(f,ignore_index=True)
    print('FEATURED',symbol,len(ev),'events',len(out),'action rows',len(lv),'levels',round(time.monotonic()-tstart,1),'seconds',flush=True)
    return out,dict(symbol=symbol,slow_events=len(ids),fast_events=len(fastids),levels=len(lv),dual_magnet_events=len(ev))

BASE_FEATURES=['slow_theta_bp','previous_completed_os','previous_event_age','fast_aligned_slow','fast_events_last_120m','fast_event_age_minutes','fast_prev_os_R','fast_prev_event_length','fast_os_lag1_R','fast_os_lag2_R','fast_os_lag3_R','fast_age_lag1','fast_age_lag2','fast_age_lag3','past_vol_60m_bp','past_vol_240m_bp','hour_sin','hour_cos','weekend',
 'trade_side','room_R','room_to_trade_magnet_bp','ratio','cluster_count','cluster_sources','action_dc_alignment','action_fast_alignment','action_magnet_near','action_past_5m','action_past_30m','action_past_120m','action_past_360m']
CONTEXT_FEATURES=['slow_theta_bp','previous_completed_os','previous_event_age','fast_aligned_slow','fast_events_last_120m','fast_event_age_minutes','fast_prev_os_R','fast_prev_event_length','fast_os_lag1_R','fast_os_lag2_R','fast_os_lag3_R','fast_age_lag1','fast_age_lag2','fast_age_lag3','past_vol_60m_bp','past_vol_240m_bp','hour_sin','hour_cos','weekend','trade_side','action_dc_alignment','action_fast_alignment','action_past_5m','action_past_30m','action_past_120m','action_past_360m']
MAGNET_FEATURES=['slow_theta_bp','hour_sin','hour_cos','weekend','trade_side','room_R','room_to_trade_magnet_bp','ratio','cluster_count','cluster_sources','action_magnet_near']

def choose(ev,scores=None,policy='model',threshold=0.0):
    if ev.empty:return pd.DataFrame()
    x=ev.copy()
    if scores is not None: x['score']=np.asarray(scores)[x.index.to_numpy('int64')]
    else:x['score']=0.0
    if policy=='dc': x=x[x.trade_side==x.dc_side]
    elif policy=='invert':x=x[x.trade_side!=x.dc_side]
    elif policy=='near':x=x[x.trade_side==x.near_side]
    elif policy=='align_room':x=x[(x.trade_side==x.dc_side)&(x.trade_side==x.near_side)&(x.ratio<=.5)&(x.room_R>=.5)]
    elif policy=='model':
        x=x[x.score>threshold].sort_values(['idx','score'],ascending=[True,False]).drop_duplicates(['symbol','idx'],keep='first')
    else:raise Exception(policy)
    x=x.sort_values(['idx','trade_side'])
    eligible=[];end=-1
    for row in x.itertuples(index=True):
        if row.idx<=end:continue
        eligible.append(row.Index)
        end=int(row.exit_idx)
    return x.loc[eligible]

def summary(t):
    if t is None or len(t)==0:return dict(n=0,net_R=None,pf=None,win=None,med_minutes=None,compounded_naive_R=None)
    rr=t.net_R.to_numpy('float64');win=rr[rr>0].sum();loss=-rr[rr<0].sum()
    return dict(n=len(rr),net_R=round(float(rr.mean()),5),pf=round(float(win/loss),4) if loss else None,win=round(float((rr>0).mean()*100),2),med_minutes=round(float(t.held_minutes.median()),1),sum_net_R=round(float(rr.sum()),2))

def evaluate(df,models,configs,out):
    lines=[];trades=[]
    # score model after each fit for whole df; indexing local per year.
    for (sym,year),part in df.groupby(['symbol','year'],sort=True):
        part=part.copy().reset_index(drop=True)
        for baseline in ['dc','invert','near','align_room']:
            chosen=choose(part,policy=baseline);lines.append(dict(symbol=sym,year=int(year),policy=baseline,**summary(chosen)))
        for model_name,model in models.items():
            features=configs[model_name];pred=np.asarray(model.predict(part[features].astype(float)))
            for hurdle in [0.]:
                chosen=choose(part,scores=pred,policy='model',threshold=hurdle)
                lines.append(dict(symbol=sym,year=int(year),policy=f'{model_name}_positiveEV',**summary(chosen)))
                if len(chosen):
                    chosen=chosen[['symbol','time','year','idx','trade_side','net_R','exit_reason','held_minutes','room_R','ratio','score']].copy()
                    chosen['policy']=model_name
                    trades.append(chosen)
    summary_df=pd.DataFrame(lines);summary_df.to_csv(out/'yearly_results.csv',index=False)
    if trades:pd.concat(trades).to_csv(out/'model_trade_logs.csv.gz',index=False,compression='gzip')
    print(summary_df.to_string(index=False),flush=True)
    return summary_df

def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    old_path=Path(args.data_dir)/'crypto_source_dc_audit.json'
    existing=json.loads(old_path.read_text()) if old_path.exists() else {}
    expected_hashes={sym:{int(y['year']):y['sha256'] for y in existing.get('symbols',{}).get(sym,{}).get('files',[])} for sym in EXPECTED_COUNTS}
    expected_hashes['BTCUSDT']=prior.KNOWN_HASH
    audit=[];metadata=[]; allf=[]
    for sym in ['BTCUSDT','ETHUSDT','XRPUSDT']:
        d,au=load_symbol(sym,Path(args.data_dir),expected_hashes[sym]);audit+=au
        dat,m=make_features(d,sym);metadata.append(m);allf.append(dat)
        dat.to_pickle(out/f'features_{sym}.pkl')
        del d,dat
    data=pd.concat(allf,ignore_index=True);del allf
    data['year']=data.time.dt.year
    audit_path=out/'source_audit.json';audit_path.write_text(json.dumps({'archives':audit,'method':metadata},indent=2))
    # fixed holdout boundaries: BTC 2024 already researched (nonvirgin); ETH/XRP 2025/26 were in older user research
    train=data[data.year<=2023];eval_data=data[data.year>=2024].copy()
    # One training-only sanity split: report 2023 from model trained <= 2022
    def train_models(tr):
        mods={};conf={}
        for name,features in [('RIDGE_DC',CONTEXT_FEATURES),('RIDGE_MAGNET',MAGNET_FEATURES),('RIDGE_COMBINED',BASE_FEATURES),('HGB_COMBINED',BASE_FEATURES)]:
            if name.startswith('RIDGE'):
                model=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler()),('reg',Ridge(alpha=250.))])
            else:
                model=HistGradientBoostingRegressor(max_iter=115,max_leaf_nodes=7,learning_rate=.05,min_samples_leaf=100,l2_regularization=20,random_state=42,early_stopping=False)
            X=tr[features].astype(float);y=tr.net_R.astype(float)
            model.fit(X,y); mods[name]=model;conf[name]=features
            print('TRAINED',name,len(tr),'prediction mean',round(float(np.mean(model.predict(X))),4),flush=True)
        return mods,conf
    train2022=data[data.year<=2022];ev2023=data[data.year==2023].copy()
    print('TRAIN 2021-22',len(train2022),'DEV PROBE 2023',len(ev2023),flush=True)
    mods22,cn=train_models(train2022)
    # evaluation 2023 in-sample-model design diagnostic, not used for selection in this script
    evaluate(ev2023,mods22,cn,out/'..' if False else out) # will overwrite later, rename immediately
    (out/'yearly_results.csv').rename(out/'development_2023_results.csv')
    (out/'model_trade_logs.csv.gz').rename(out/'development_2023_trades.csv.gz')
    print('TRAIN 2021-23',len(train),'CHECK 2024-26',len(eval_data),flush=True)
    mods,cn=train_models(train)
    results=evaluate(eval_data,mods,cn,out)
    results['stage']=np.where(results.year==2024,'2024 historical checkpoint','2025-26 retrospective later-year check')
    results.to_csv(out/'yearly_results.csv',index=False)
    cfg={'hypothesis':'H5: Two-scale close-confirmed DC and current unresolved magnetic geometry predict cost-adjusted signed payoff of next 120m 0.5R/1R trade, enabling trade or abstain',
         'slow_theta':'1.25 * previous UTC daily median of rolling 240m realized log-return vol','fast_theta':'0.5 * same sigma',
         'entry':'next M1 open following slow DC confirmation','fee_slippage_bp':TOTAL_COST_BP,'slip_assumption':'11bp taker fees + 2bp total slippage proxy, no explicit bid-ask history',
         'funding':'NOT modeled; trades crossing funding boundary may be misstated','horizon_minutes':HORIZON,'TP_R':TP_R,'SL_R':1.0,'same_bar':'stop-first',
         'directions':'both long and short labels at each event; select one model-predicted direction if predicted net_R > 0; otherwise abstain',
         'base_models':'ridge DC-only, ridge magnet-only, ridge combined, HistGradientBoostingRegressor combined (fixed hyperparameters)',
         'train_2023_eval':'trained 2021-22, evaluated 2023','later_fit':'trained 2021-23, evaluated on 2024, 2025, 2026',
         'false_confirmations':'2024 BTC/ETH/XRP and 2025-26 ETH/XRP previously evaluated by this project in other studies; they are retrospective chronological tests not pristine fresh holdout',
         'limitation':'OHLC candles: cannot determine intraminute path or maker fill queue; no funding schedule; no actual spread; 1m approximations',
         'feature_count':len(BASE_FEATURES),'features':BASE_FEATURES,'no_stale_future_features':'past only, causal pivots, previous completed day sigma, confirmed M15 swings; avoid using realized event labels as features'}
    (out/'METHODS.json').write_text(json.dumps(cfg,indent=2))
    print('DONE',out,flush=True)

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--data-dir',default='/mnt/data');pa.add_argument('--out',default='/mnt/data/Magnet_DC_H5');main(pa.parse_args())

#!/usr/bin/env python3
"""BTC Bybit M1 magnet + close-confirmed DC research engine v2.0.
Audited 2021-24 data. DOES NOT replicate original missing H4.2 event file.
Run: python btc_magnet_dc_v2.py --data-dir /mnt/data --out /mnt/data/btc_magnet_dc_v2
Requirements: numpy pandas numba. Deterministic research, no live orders.
"""
from __future__ import annotations
import argparse, hashlib, json, zipfile, re, math
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

KNOWN_HASH = {
2021:'620a6f5f7814db6cf30feab71b52cd3851d3e5afd2425983a6c1160e263c7ded',
2022:'65604ac53fcb89f95ea142cb47278d018917fb28400e187592b50997c794613a',
2023:'778e89c855d26559c57d1368a730d516debec0aca118f08e7b18de7826180cb0',
2024:'710ae80850facbd60faa0d42e57d2004a446f54dde15aac271773d516d8ea7f6'}
REQUIRED=['timestamp_utc','timestamp_ms','open','high','low','close','volume_base','turnover_usdt']
FEES_BP = 11.0
SLIPPAGE_BP = 2.0
K_THETA=1.25
HORIZON_MIN=240

@njit(cache=True)
def dc_detector(close, theta):
    n=len(close)
    ids=np.empty(n,dtype=np.int64); sides=np.empty(n,dtype=np.int8)
    previous_os=np.empty(n); pre_age=np.empty(n,dtype=np.int32)
    direction=0; lo=close[0];hi=close[0];confirm_price=close[0];confirm_i=0; leg_th=np.nan; count=0
    for i in range(1,n):
        p=close[i]; th=theta[i]
        if not np.isfinite(th) or th<=0: continue
        if direction==0:
            if p<lo:lo=p
            if p>hi:hi=p
            if p>=lo*(1+th):
                direction=1; hi=p; confirm_price=p;confirm_i=i;leg_th=th
                ids[count]=i;sides[count]=1;previous_os[count]=np.nan;pre_age[count]=0;count+=1
            elif p<=hi*(1-th):
                direction=-1;lo=p;confirm_price=p;confirm_i=i;leg_th=th
                ids[count]=i;sides[count]=-1;previous_os[count]=np.nan;pre_age[count]=0;count+=1
        elif direction==1:
            if p>hi:hi=p
            if p<=hi*(1-th):
                os=max(0.0,(hi/confirm_price-1.0)/leg_th) if leg_th>0 else np.nan
                ids[count]=i;sides[count]=-1;previous_os[count]=os;pre_age[count]=i-confirm_i;count+=1
                direction=-1;lo=p;confirm_price=p;confirm_i=i;leg_th=th
        else:
            if p<lo:lo=p
            if p>=lo*(1+th):
                os=max(0.0,(1.0-lo/confirm_price)/leg_th) if leg_th>0 else np.nan
                ids[count]=i;sides[count]=1;previous_os[count]=os;pre_age[count]=i-confirm_i;count+=1
                direction=1;hi=p;confirm_price=p;confirm_i=i;leg_th=th
    return ids[:count],sides[:count],previous_os[:count],pre_age[:count]

@njit(cache=True)
def first_sweeps(high,low,start,expire,level):
    m=len(level);out=np.empty(m,dtype=np.int64)
    for j in range(m):
        end=expire[j]; target=level[j]
        for i in range(start[j],end):
            if high[i]>=target and low[i]<=target:
                end=i;break
        out[j]=end
    return out

@njit(cache=True)
def first_target(high,low,lo,hi,upper,lower):
    for i in range(lo,hi):
        a=high[i]>=upper;b=low[i]<=lower
        if a and b:return 2,i
        if a:return 1,i
        if b:return -1,i
    return 0,hi-1

@njit(cache=True)
def sim_trade(open_,high,low,close,side,entry_i,risk_bp,tp_R,horizon,fee_bp):
    p0=open_[entry_i];dist=p0*risk_bp/10000.0
    target=p0+side*dist*tp_R;stop=p0-side*dist
    end=min(entry_i+horizon,len(open_)); exitp=close[end-1];why=0;ei=end-1
    for i in range(entry_i,end):
        if side==1:
            s=low[i]<=stop;t=high[i]>=target
        else:
            s=high[i]>=stop;t=low[i]<=target
        if s:
            exitp=stop;why=-1;ei=i;break
        if t:
            exitp=target;why=1;ei=i;break
    gross=side*(exitp/p0-1.0)*10000
    net=gross-fee_bp
    return net/risk_bp,net,gross,ei,why

def sha256_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()

def load_crypto(data_dir):
    pieces=[];audit=[]
    for year in (2021,2022,2023,2024):
        files=list(Path(data_dir).glob(f'BTCUSDT_1m_{year}*.zip'))
        if len(files)!=1:raise RuntimeError(f'Expected exactly one BTCUSDT {year} ZIP, found {files}')
        p=files[0];h=sha256_file(p)
        if h!=KNOWN_HASH[year]:raise RuntimeError(f'SHA256 mismatch for {p.name}: {h}')
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:raise RuntimeError('ZIP CRC failed '+str(p))
            members=[x for x in z.infolist() if not x.is_dir()]
            if len(members)!=1 or members[0].filename!=f'BTCUSDT_1m_{year}.csv':raise RuntimeError('Unexpected archive member '+str(p))
            with z.open(members[0]) as f:d=pd.read_csv(f)
        if list(d.columns)!=REQUIRED:raise RuntimeError('Wrong columns')
        if d.isna().any().any():raise RuntimeError('NaN data '+str(p))
        ts=d.timestamp_ms.to_numpy(dtype='int64')
        expected_first=int(pd.Timestamp(f'{year}-01-01',tz='UTC').timestamp()*1000)
        expected_last=int(pd.Timestamp(f'{year+1}-01-01',tz='UTC').timestamp()*1000)-60000
        if ts[0]!=expected_first or ts[-1]!=expected_last:raise RuntimeError('incomplete year '+str(year))
        if not np.all(np.diff(ts)==60000):raise RuntimeError('missing or duplicate minutes '+str(year))
        textts=pd.to_datetime(d.timestamp_utc,utc=True,errors='raise').astype('int64').to_numpy()//10**6
        if not np.array_equal(textts,ts):raise RuntimeError('timestamp disagree '+str(year))
        v=d[['open','high','low','close','volume_base','turnover_usdt']].to_numpy(dtype='float64')
        if not np.all(np.isfinite(v)):raise RuntimeError('nonfinite')
        if np.any(v[:,0:4]<=0) or np.any(v[:,4:]<0) or np.any(v[:,2]>np.minimum(v[:,0],v[:,3])) or np.any(v[:,1]<np.maximum(v[:,0],v[:,3])):raise RuntimeError('bad OHLCV')
        audit.append({'year':year,'bars':len(d),'first_utc':str(d.timestamp_utc.iloc[0]),'last_utc':str(d.timestamp_utc.iloc[-1]),'sha256':h,'zip_crc':'PASS','chronology':'PASS','ohlcv':'PASS'})
        pieces.append(d)
        print(f'AUDIT PASS {year} {len(d):,} minutes sha256={h[:16]}...',flush=True)
    allx=pd.concat(pieces,ignore_index=True)
    if not np.all(np.diff(allx.timestamp_ms.to_numpy())==60000):raise RuntimeError('year boundary gap')
    allx['minute']=pd.to_datetime(allx.timestamp_ms,unit='ms',utc=True)
    return allx,audit

def compute_theta(d):
    c=d.close
    rr=np.log(c).diff()
    vol=np.sqrt(rr.pow(2).rolling(240,min_periods=240).sum())
    days=d.minute.dt.floor('D')
    daily_median=vol.groupby(days).median().shift(1)
    theta=days.map(daily_median).to_numpy(dtype='float64')*K_THETA
    return theta

def build_levels(d):
    ts=d.timestamp_ms.to_numpy('int64'); high=d.high.to_numpy();low=d.low.to_numpy()
    levels=[]
    def insert(available,hi,lo,source,expiry_days):
        if not np.isfinite(hi) or not np.isfinite(lo):return
        start=int(np.searchsorted(ts,int(pd.Timestamp(available).timestamp()*1000)))
        end=int(np.searchsorted(ts,int(pd.Timestamp(available).timestamp()*1000+expiry_days*86400000)))
        if start<len(ts) and end>start:
            levels.append((start,end,float(hi),source))
            levels.append((start,end,float(lo),source))
    minute=d.minute
    period_start=minute.dt.floor('D')
    groups=d.groupby(period_start).agg(high=('high','max'),low=('low','min'),count=('high','size'))
    for start,g in groups.iterrows():
        if g['count']>=1200:insert(start+pd.Timedelta(days=1),g['high'],g['low'],'PD',7)
    period_start=minute.dt.floor('8h')
    groups=d.groupby(period_start).agg(high=('high','max'),low=('low','min'),count=('high','size'))
    for start,g in groups.iterrows():
        if g['count']>=460:insert(start+pd.Timedelta(hours=8),g['high'],g['low'],'8H',3)
    week_start=minute.dt.floor('D')-pd.to_timedelta(minute.dt.weekday,unit='D')
    groups=d.groupby(week_start).agg(high=('high','max'),low=('low','min'),count=('high','size'))
    for start,g in groups.iterrows():
        if g['count']>=9000:insert(start+pd.Timedelta(days=7),g['high'],g['low'],'PW',28)
    # Fully completed 15m bars timestamped at right boundary
    k=d.set_index('minute');b=k[['high','low','close']].resample('15min',closed='left',label='right').agg({'high':'max','low':'min','close':'last'})
    mhi=b.high.to_numpy();mlo=b.low.to_numpy()
    for i in range(2,len(b)-2):
        a=mhi[i-2:i+3];f=mlo[i-2:i+3]
        if not np.isfinite(a).all() or not np.isfinite(f).all():continue
        available=b.index[i+2]
        st=int(np.searchsorted(ts,int(available.timestamp()*1000)));en=min(len(ts),st+5*1440)
        if st>=len(ts):continue
        if a[2]>max(a[0],a[1],a[3],a[4]):levels.append((st,en,float(a[2]),'PIVOT15'))
        if f[2]<min(f[0],f[1],f[3],f[4]):levels.append((st,en,float(f[2]),'PIVOT15'))
    levels.sort(key=lambda x:x[0]); L=pd.DataFrame(levels,columns=['start','expiry','price','source'])
    L['end']=first_sweeps(high,low,L.start.to_numpy('int64'),L.expiry.to_numpy('int64'),L.price.to_numpy('float64'))
    print('LEVELS',len(L),L.source.value_counts().to_dict(),flush=True)
    return L

def make_event_frame(d,lev,ids,sides,os,age,theta):
    prices=lev.price.to_numpy();start=lev.start.to_numpy();end=lev.end.to_numpy();source=lev.source.to_numpy()
    c=d.close.to_numpy(); rows=[];active=set();j=0
    for k,i in enumerate(ids):
        while j<len(lev) and start[j]<=i:
            if end[j]>i:active.add(j)
            j+=1
        active={a for a in active if end[a]>i}
        if not active:continue
        a=np.array(list(active),dtype='int64'); pp=prices[a];now=c[i]
        ups=pp[pp>now];downs=pp[pp<now]
        if len(ups)==0 or len(downs)==0:continue
        upper=ups.min();lower=downs.max();du=10000*(upper/now-1);dl=10000*(1-lower/now)
        if du<=0 or dl<=0:continue
        near=1 if du<dl else -1; dn=min(du,dl);df=max(du,dl);ratio=dn/df
        near_price=upper if near==1 else lower
        # Exact or +/-0.01% grouping as rough density proxy, not proof liquidity.
        cluster_idx=a[np.abs(prices[a]-near_price)/now<=.0001]
        rows.append((i,d.minute.iloc[i],int(sides[k]),float(theta[i]),float(os[k]),int(age[k]),upper,lower,near,ratio,du,dl,
                     len(cluster_idx),len(set(source[cluster_idx]))))
    cols=['idx','time','dc_side','theta','previous_completed_os','previous_event_age','upper','lower','near_side','ratio','up_distance_bp','down_distance_bp','cluster_count','cluster_sources']
    ev=pd.DataFrame(rows,columns=cols)
    if len(ev):ev['year']=ev.time.dt.year;ev['aligned']=ev.near_side==ev.dc_side;ev['distance_only_p_near']=1/(1+ev.ratio)
    print('EVENTS',len(ev), 'DC total',len(ids),flush=True)
    return ev

def targets(d,ev):
    hi=d.high.to_numpy();lo=d.low.to_numpy();n=len(d);arr=[]
    for z in ev.itertuples():
        begin=z.idx+1;finish=min(n,begin+1440)
        if begin>=n or finish-begin<1440:continue
        hit,idx=first_target(hi,lo,begin,finish,z.upper,z.lower)
        arr.append((z.idx,hit,idx))
    out=ev.merge(pd.DataFrame(arr,columns=['idx','target_first','target_hit_idx']),on='idx',how='inner')
    out['near_first']=np.where(out.target_first.isin([-1,1]),out.target_first.eq(out.near_side),np.nan)
    out['dc_first']=np.where(out.target_first.isin([-1,1]),out.target_first.eq(out.dc_side),np.nan)
    return out

def sim_cohort(d,ev,policy,fee=FEES_BP+SLIPPAGE_BP):
    # Shared event universe. Standalone policies each enforce a single open position.
    a=d.open.to_numpy();b=d.high.to_numpy();lo=d.low.to_numpy();cl=d.close.to_numpy()
    trades=[];occupied_until=-1;eligible=0;skip_overlap=0;skip_small=0
    for z in ev.itertuples(index=False):
        if policy['cohort']=='aligned' and not z.aligned:continue
        if policy['cohort']=='disagree' and z.aligned:continue
        if policy['cohort']=='ratio_half' and not z.ratio<=.5:continue
        if policy['cohort']=='aligned_ratio_half' and (not z.aligned or z.ratio>.5):continue
        if policy['cohort']=='aligned_ratio_half_room' and (not z.aligned or z.ratio>.5):continue
        i=int(z.idx);entry_i=i+1
        if entry_i>=len(a):continue
        if policy['signal']=='dc':side=int(z.dc_side)
        elif policy['signal']=='near':side=int(z.near_side)
        elif policy['signal']=='inverse':side=-int(z.near_side)
        else:raise ValueError('signal')
        risk_bp=10000*z.theta # causal; relative to signal price
        if not np.isfinite(risk_bp) or risk_bp<=0:continue
        nearroom=(z.up_distance_bp if z.near_side==1 else z.down_distance_bp)
        if policy['cohort']=='aligned_ratio_half_room' and (nearroom<risk_bp*policy.get('minimum_room_R',.5) or risk_bp<26):
            skip_small+=1;continue
        eligible+=1
        if entry_i<=occupied_until:
            skip_overlap+=1;continue
        r,ret,gross,ei,reason=sim_trade(a,b,lo,cl,side,entry_i,risk_bp,policy['tp_R'],policy['horizon'],fee)
        # A fixed 8h settlement-time flag; funding data not available, and time slots can change.
        settle= ((d.timestamp_ms.iloc[entry_i:ei+1].to_numpy()//60000)%(480)==0).any()
        trades.append((z.idx,z.time,i,entry_i,ei,side,z.ratio,z.aligned,risk_bp,nearroom,policy['tp_R'],policy['horizon'],r,ret,gross,reason,ei-entry_i+1, bool(settle)))
        occupied_until=ei
    names=['idx','event_time','event_idx','entry_idx','exit_idx','side','ratio','aligned','risk_bp','near_room_bp','tp_R','horizon','net_R','net_bp','gross_bp','exit_reason','held_minutes','funding_may_apply']
    return pd.DataFrame(trades,columns=names),{'events_eligible':eligible,'skipped_overlap':skip_overlap,'skipped_insufficient_room':skip_small}

def stats(t):
    if t.empty:return {'n':0}
    R=t.net_R.to_numpy();positive=R[R>0].sum();negative=-R[R<0].sum()
    return {'n':len(t),'mean_net_R':round(float(R.mean()),5),'mean_net_bp':round(float(t.net_bp.mean()),3),
            'win_pct':round(100*float((R>0).mean()),2),'PF':round(float(positive/negative),3) if negative>0 else None,
            'median_minutes':float(t.held_minutes.median()),'percent_30m_or_less':round(100*float((t.held_minutes<=30).mean()),1),
            'max_drawdown_R':round(float((np.cumsum(R)-np.maximum.accumulate(np.r_[0,np.cumsum(R)])[1:]).min()),2),
            'funding_flag_pct':round(100*float(t.funding_may_apply.mean()),2)}

def weekly_bootstrap(t,iterations=2000,seed=52):
    if t.empty:return None
    weeks=t.event_time.dt.tz_convert(None).dt.to_period('W').astype(str)
    groups=t.groupby(weeks).net_R.agg(['sum','count']).to_numpy()
    if len(groups)<5:return None
    rng=np.random.default_rng(seed);res=[]
    for _ in range(iterations):
        a=groups[rng.integers(0,len(groups),len(groups))];res.append(a[:,0].sum()/a[:,1].sum())
    return [round(float(x),4) for x in np.percentile(res,[2.5,97.5])]

def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    d,audit=load_crypto(args.data_dir); print('AUDITED TOTAL',len(d),flush=True)
    theta=compute_theta(d); c=d.close.to_numpy();idx,side,os,age=dc_detector(c,theta)
    lv=build_levels(d);ev=make_event_frame(d,lv,idx,side,os,age,theta);ev=targets(d,ev)
    ev.to_csv(out/'dc_magnet_events_all.csv.gz',index=False,compression='gzip')
    print('CENSORED',ev.target_first.value_counts().to_dict(),flush=True)
    rows=[];trades={}
    # The primary comparison is FIXED BEFORE RESULTS ARE REVIEWED; 2024 reserved for chronological check.
    policies=[
      {'name':'DC_only_0p5R_120m','signal':'dc','cohort':'all','tp_R':.5,'horizon':120},
      {'name':'Magnet_only_0p5R_120m','signal':'near','cohort':'all','tp_R':.5,'horizon':120},
      {'name':'Aligned_0p5R_120m','signal':'dc','cohort':'aligned','tp_R':.5,'horizon':120},
      {'name':'Aligned_ratio0p5_0p5R_120m','signal':'dc','cohort':'aligned_ratio_half','tp_R':.5,'horizon':120},
      {'name':'Aligned_room_0p5R_120m','signal':'dc','cohort':'aligned_ratio_half_room','minimum_room_R':.5,'tp_R':.5,'horizon':120},
      {'name':'Aligned_room_0p5R_30m','signal':'dc','cohort':'aligned_ratio_half_room','minimum_room_R':.5,'tp_R':.5,'horizon':30},
      {'name':'Aligned_room_0p25R_120m','signal':'dc','cohort':'aligned_ratio_half_room','minimum_room_R':.5,'tp_R':.25,'horizon':120},
      {'name':'Aligned_room_1R_120m','signal':'dc','cohort':'aligned_ratio_half_room','minimum_room_R':.5,'tp_R':1.0,'horizon':120},
      {'name':'Inverse_near_0p5R_120m','signal':'inverse','cohort':'all','tp_R':.5,'horizon':120},
    ]
    for pol in policies:
        name=pol['name'];parts=[]
        for year in (2021,2022,2023,2024):
            ee=ev[ev.year==year]
            tr,aud=sim_cohort(d,ee,pol)
            if len(tr):tr['year']=year
            parts.append(tr)
            row={'policy':name,'year':year,**stats(tr),**aud};rows.append(row)
            print('RESULT',name,year,row['n'],row.get('mean_net_R'),row.get('PF'),flush=True)
        trades[name]=pd.concat(parts,ignore_index=True)
        trades[name].to_csv(out/f'trades_{name}.csv',index=False)
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'policy_summary_by_year.csv',index=False)
    filtered=ev[ev.target_first.isin([-1,1])].copy()
    measure=[]
    for name,group in [('ALL',filtered),('RATIO<=0.5',filtered[filtered.ratio<=.5]),('ALIGNED',filtered[filtered.aligned]),('DISAGREE',filtered[~filtered.aligned])]:
        for year in (2021,2022,2023,2024):
            g=group[group.year==year]
            if not len(g):continue
            measure.append({'group':name,'year':year,'n':len(g),'near_first_pct':round(g.near_first.mean()*100,2),
                            'geometry_prediction_pct':round(g.distance_only_p_near.mean()*100,2),
                            'excess_vs_geometry_pp':round((g.near_first-g.distance_only_p_near).mean()*100,3),
                            'dc_first_pct':round(g.dc_first.mean()*100,2)})
    pd.DataFrame(measure).to_csv(out/'destination_first_passage.csv',index=False)
    boot={name:{str(y):weekly_bootstrap(t[t.year==y]) for y in (2021,2022,2023,2024)} for name,t in trades.items()}
    json.dump({'archive_audit':audit,'bars':len(d),'dc_events':len(idx),'levels':len(lv),'dual_magnet_events':len(ev),
               'fixed_research_family':policies,'fee_bp':FEES_BP,'entry_exit_slippage_bp':SLIPPAGE_BP,
               'funding_cost':'NOT MODELED; flagged potential 8h settlement crossing','quoted_spread':'NOT AVAILABLE in OHLCV; 1bp/side total execution allowance assumption',
               'timing':'Signal closes M1, entry next M1 open','pivots':'M15 2-left / 2-right confirmed at close of second right candle',
               'magnet_levels':'last completed D/W/8H highs/lows and M15 pivots, no original H4.2 file',
               'stop_risk_bp':'theta_t * 10000 from previous-day sigma, stop 1R','exit_rule':'stop first when TP and SL same M1',
               'nonoverlap':'one position at a time per policy/year','no_trade':'ignored DC events outside selected cohort',
               'sampling':'2021-23 exploratory, 2024 chronological checkpoint already explored in older broader research',
               'bootstrap_CI95_by_year':boot},open(out/'audit_and_methods.json','w'),indent=2)
    print('\nDESTINATIONS:\n',pd.DataFrame(measure).to_string(index=False),flush=True)
    print('\nPOLICIES:\n',summary[['policy','year','n','mean_net_R','win_pct','PF','median_minutes','funding_flag_pct']].to_string(index=False),flush=True)
    print('END',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='/mnt/data');p.add_argument('--out',default='/mnt/data/btc_magnet_dc_v2');main(p.parse_args())

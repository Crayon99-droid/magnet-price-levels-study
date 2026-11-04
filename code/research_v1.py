#!/usr/bin/env python3
r"""Auditable crypto pullback research, version 0.1.3 (partial-start coverage support).

Windows PowerShell, from the directory containing this file:
  py -m venv .venv
  .\.venv\Scripts\python.exe -m pip install "numpy>=1.26,<3" "pandas>=2.2,<3"
  .\.venv\Scripts\python.exe research_v1.py --self-test
  .\.venv\Scripts\python.exe research_v1.py --data . --audit-only
  .\.venv\Scripts\python.exe research_v1.py --data . --fee-bps 5 --impact-bps 1

If ZIPs are in a data subfolder, replace '--data .' with '--data .\data'.
The example costs are illustrative, PER SIDE, not verified exchange fees.

RESEARCH CONTRACT (same eight hypotheses as the chat version):
- Only BTCUSDT and ETHUSDT 2021-2023 files are opened. No optimizer.
- 4h filter: close relative to EMA50 AND EMA slope over three closed bars.
- 30m pullback: close relative to its close three closed bars earlier.
- 5m trigger: close beyond previous six closed bars' high/low.
- Optional zone: previous 48 closed 30m bars' low/high, excluding newest
  closed 30m bar; +/- 0.25 simple ATR14. This is a support/resistance proxy,
  NOT institutional supply/demand or tape absorption.
- Zones judged against a 5m candle existed at that candle's OPEN.
  A touch during any of the latest six closed 5m candles qualifies.
- Stop: latest six closed 5m candles' extreme +/- 0.1 simple ATR14.
- Entry: minute open one minute after signal availability, adverse execution.
- Exit: 1R or UTC daily VWAP frozen at signal. VWAP needs >=0.5 gross R.
- No signals during UTC hour 00; identical indicator warmup for all variants.
- Max holding: 120 minutes; one position per asset per variant.
- Stop first for ambiguous intraminute stop/target hits. Stop gaps use open;
  favorable target gaps receive only target price. All exits incur costs.
- Features use only completed bars. No candle gap filling.
- Actual quote/base volumes give venue VWAP; no typical-price substitute.
- Funding, borrowing, liquidation, leverage, and size-dependent impact are
  EXCLUDED. Outputs explicitly say before funding. No portfolio claim.
- Bootstrap: calendar-day clusters, circular 7/28-day blocks, zero-trade days
  included. Adjusted bounds cover this run's 32 comparisons ONLY, not the
  wider research history. These are exploratory, not confirmation results.

0.1.3 allows leading missing minutes only in the first discovery year (2021),
with no internal gaps or missing year-end. Indicators warm up from actual data.
--assets ETHUSDT runs ETH alone; the original 32-comparison correction remains.
Actual coverage is recorded; unequal asset date ranges are not matched tests.

0.1.2 adds volume_base/turnover_usdt mapping and checks timestamp_utc against
the selected epoch timestamp column. Strategy calculations are unchanged.

0.1.1 packaging changes: shorter implementation, persistent failure report,
header-only CSVs for empty results, progress messages, more synthetic checks.
The predefined strategy rules and cost arithmetic are unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    raise SystemExit('Install dependencies first: py -m pip install "numpy>=1.26,<3" "pandas>=2.2,<3"')

VERSION = '0.1.3'
YEARS = (2021, 2022, 2023)
ASSETS = ('BTCUSDT', 'ETHUSDT')
MINUTE = pd.Timedelta(minutes=1)
PARAMS = dict(ema_4h=50, ema_slope_bars=3, pullback_30m_bars=3,
              breakout_5m_bars=6, atr_bars=14, zone_30m_bars=48,
              zone_halfwidth_atr=0.25, touch_memory_5m_bars=6,
              stop_5m_bars=6, stop_buffer_atr=0.1, entry_delay_minutes=1,
              maximum_holding_minutes=120, vwap_minimum_gross_R=0.5,
              first_signal_hour_UTC=1)
VARIANTS = [dict(name=f'bias{int(b)}_zone{int(z)}_{t}', bias=b, zone=z, target=t)
            for b,z,t in itertools.product((False,True),(False,True),('R1','VWAP'))]
BINANCE_COLUMNS = ['timestamp','open','high','low','close','volume','close_time',
                   'quote_volume','trade_count','taker_buy_volume',
                   'taker_buy_quote_volume','ignore']
ALIASES = {
 'timestamp':['timestamp','open_time','opentime','open_timestamp','datetime','date','timestamp_ms','timestamp_us'],
 'open':['open','o'], 'high':['high','h'], 'low':['low','l'], 'close':['close','c'],
 'volume':['volume','base_volume','volume_base','base_asset_volume','v'],
 'quote_volume':['quote_volume','turnover_usdt','quote_asset_volume','quoteassetvolume','quote_asset_vol','q'],
 'taker_buy_volume':['taker_buy_volume','taker_buy_base_asset_volume','taker_buy_base_volume','takerbuybaseassetvolume']}
BP = 'execution_net_bp_before_funding'
RR = 'execution_net_R_before_funding'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, obj):
    path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')


def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_name(name):
    return re.sub(r'[^a-z0-9]+','_',str(name).strip().lower()).strip('_')


def parse_time(series, unit='auto'):
    numeric=pd.to_numeric(series,errors='coerce')
    if numeric.notna().all():
        require(np.isfinite(numeric.to_numpy(dtype=float)).all(),'Nonfinite timestamp.')
        require((numeric%1==0).all(),'Epoch timestamps must be integers.')
        selected=unit
        if selected=='auto':
            x=float(numeric.abs().median())
            selected='s' if x<1e11 else 'ms' if x<1e14 else 'us' if x<1e17 else 'ns'
        result=pd.to_datetime(numeric.astype('int64'),unit=selected,utc=True,errors='raise')
    else:
        require(numeric.isna().all(),'Mixed numeric/textual timestamps.')
        result=pd.to_datetime(series,utc=True,errors='raise',format='mixed')
        selected='text_interpreted_as_UTC'
    index=pd.DatetimeIndex(result)
    require(not index.hasnans,'Missing timestamps.')
    require((index==index.floor('min')).all(),
            'Timestamps must be minute-aligned candle OPEN times; verify source semantics.')
    return index,selected


def select_file(directory, asset, year):
    pattern=re.compile(rf'^{asset}_1m_{year}(?:\.csv)?(?:\(\d+\))?\.(?:zip|csv)$',re.I)
    files=sorted(p for p in directory.iterdir() if p.is_file() and pattern.fullmatch(p.name))
    require(len(files)==1,f'Need exactly one {asset}_1m_{year} CSV/ZIP in {directory}; found {[p.name for p in files]}.')
    return files[0]


def read_csv_source(path):
    def read_handle(handle):
        first=handle.readline()
        if isinstance(first,bytes):
            first=first.decode('utf-8-sig')
        tokens=next(csv.reader([first.strip()]))
        # A known price-field name identifies named CSVs, including o/h/l/c.
        header=any(normalize_name(x) in ('open','o') for x in tokens)
        handle.seek(0)
        frame=pd.read_csv(handle,header=0 if header else None,encoding='utf-8-sig')
        original=[str(c) for c in frame.columns]
        if header:
            frame.columns=[normalize_name(c) for c in frame.columns]
        else:
            require(frame.shape[1]==12,f'{path.name}: unknown headerless schema ({frame.shape[1]} columns).')
            frame.columns=BINANCE_COLUMNS
        require(not frame.columns.duplicated().any(),'Duplicate normalized columns.')
        return frame,dict(header_present=header,original_columns=original)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            members=[m for m in z.namelist() if m.lower().endswith('.csv')
                     and not m.startswith('__MACOSX/') and not Path(m).name.startswith('.')]
            require(len(members)==1,f'{path.name}: expected exactly one CSV, found {members}.')
            with z.open(members[0]) as f:
                raw,info=read_handle(f)
            info['zip_member']=members[0]
    else:
        with path.open('rb') as f:
            raw,info=read_handle(f)
        info['zip_member']=None
    return raw,info


def load_year(path,asset,year,timestamp_unit):
    raw,info=read_csv_source(path)
    require(len(raw)>0,f'{path.name}: empty file.')
    mapped={}
    for target,aliases in ALIASES.items():
        hits=[c for c in aliases if c in raw.columns]
        require(len(hits)<=1,f'Ambiguous {target}: {hits}')
        if hits:
            mapped[target]=hits[0]
    if 'timestamp' not in mapped and 'timestamp_utc' in raw.columns:
        mapped['timestamp']='timestamp_utc'
    required={'timestamp','open','high','low','close','volume'}
    require(required<=mapped.keys(),f'{path.name}: missing recognized columns {sorted(required-mapped.keys())}; found {list(raw.columns)}')
    index,unit=parse_time(raw[mapped['timestamp']],timestamp_unit)
    timestamp_crosschecked=False
    if 'timestamp_utc' in raw.columns and mapped['timestamp']!='timestamp_utc':
        utc_index,_=parse_time(raw['timestamp_utc'])
        require((index==utc_index).all(),
                f'{path.name}: timestamp_utc disagrees with {mapped["timestamp"]}; refusing ambiguous times.')
        timestamp_crosschecked=True
    require((index.year==year).all(),f'{path.name}: timestamp year mismatch.')
    require(not index.has_duplicates,f'{path.name}: duplicate timestamps.')
    frame=pd.DataFrame(index=index)
    for target,col in mapped.items():
        if target!='timestamp':
            frame[target]=pd.to_numeric(raw[col],errors='raise').to_numpy(dtype=float)
    sorted_source=frame.index.is_monotonic_increasing
    frame=frame.sort_index()
    require(np.isfinite(frame.to_numpy()).all(),f'{path.name}: nonfinite numbers.')
    o,h,l,c=(frame[x] for x in ('open','high','low','close'))
    require(((l>0)&(h>=l)&(o>=l)&(o<=h)&(c>=l)&(c<=h)).all(),f'{path.name}: invalid OHLC.')
    require((frame.volume>=0).all(),'Negative base volume.')
    if 'quote_volume' in frame:
        q,v=frame.quote_volume,frame.volume
        require((q>=0).all() and (q[v==0]==0).all(),'Invalid quote volume.')
        positive=v>0
        ratio=q[positive]/v[positive]
        require(((ratio>=l[positive]*(1-1e-5))&(ratio<=h[positive]*(1+1e-5))).all(),
                'Quote/base volume outside price range. Check source schema and units.')
    if 'taker_buy_volume' in frame:
        require(((frame.taker_buy_volume>=0)&(frame.taker_buy_volume<=frame.volume*(1+1e-8)+1e-10)).all(),'Invalid taker-buy volume.')
    expected=pd.date_range(f'{year}-01-01',f'{year+1}-01-01',tz='UTC',freq='min',inclusive='left')
    missing=expected.difference(frame.index)
    leading=int((frame.index[0]-expected[0])/MINUTE)
    trailing=int((expected[-1]-frame.index[-1])/MINUTE)
    internal=int((frame.index[-1]-frame.index[0])/MINUTE)+1-len(frame)
    info.update(leading_missing_minutes=leading,trailing_missing_minutes=trailing,internal_missing_minutes=internal)
    info.update(asset=asset,year=year,filename=path.name,sha256=sha256_file(path),mapping=mapped,
                timestamp_unit=unit,timestamp_utc_crosschecked=timestamp_crosschecked,rows=len(frame),expected_rows=len(expected),missing_minutes=len(missing),
                first_missing_minutes=[str(t) for t in missing[:20]],first_open=str(frame.index[0]),
                last_open=str(frame.index[-1]),source_was_sorted=sorted_source,
                has_quote_volume='quote_volume' in frame,has_taker_buy_volume='taker_buy_volume' in frame)
    problem=coverage_problem(info)
    info.update(backtest_coverage_eligible=problem is None,coverage_problem=problem)
    return frame,info


def coverage_problem(audit):
    if audit['internal_missing_minutes']:
        return f"{audit['internal_missing_minutes']} internal missing minutes; no gap filling allowed."
    if audit['trailing_missing_minutes']:
        return 'Missing year-end data; repair the source or explicitly revise the research window.'
    if audit['leading_missing_minutes'] and audit['year'] != YEARS[0]:
        return 'Leading missing minutes outside the first discovery year; this would create a multi-year gap.'
    if pd.Timestamp(audit['first_open']) != pd.Timestamp(audit['first_open']).floor('D'):
        return 'Partial-day dataset start; require a midnight start for daily VWAP.'
    return None


def aggregate_bars(minutes,n):
    # [00:00,00:05) is available at 00:05, never at 00:00.
    g=minutes.resample(f'{n}min',closed='left',label='right',origin='start_day')
    bars=g.agg(dict(open='first',high='max',low='min',close='last',volume='sum'))
    return bars.loc[g.close.count()==n].copy()


def simple_atr(bars,length=14):
    previous=bars.close.shift(1)
    tr=pd.concat([bars.high-bars.low,(bars.high-previous).abs(),(bars.low-previous).abs()],axis=1).max(axis=1)
    return tr.rolling(length,min_periods=length).mean()


def map_available(series,times):
    return series.reindex(times,method='ffill').to_numpy()


def make_features(minutes):
    b5,b30,b4=(aggregate_bars(minutes,n) for n in (5,30,240))
    f=b5.copy()
    f['atr5']=simple_atr(b5,PARAMS['atr_bars'])
    n=PARAMS['breakout_5m_bars']
    f['previous_high']=b5.high.shift(1).rolling(n).max()
    f['previous_low']=b5.low.shift(1).rolling(n).min()
    f['stop_low']=b5.low.rolling(PARAMS['stop_5m_bars']).min()
    f['stop_high']=b5.high.rolling(PARAMS['stop_5m_bars']).max()
    ema=b4.close.ewm(span=PARAMS['ema_4h'],adjust=False,min_periods=PARAMS['ema_4h']).mean()
    slope=ema-ema.shift(PARAMS['ema_slope_bars'])
    bias=pd.Series(np.where((b4.close>ema)&(slope>0),1,np.where((b4.close<ema)&(slope<0),-1,0)),index=b4.index)
    for key,series in dict(bias4=bias,close4=b4.close,ema4=ema,ema4_slope=slope,
                           close30=b30.close,reference30=b30.close.shift(PARAMS['pullback_30m_bars'])).items():
        f[key]=map_available(series,f.index)
    f['pullback30_return']=f.close30/f.reference30-1
    n=PARAMS['zone_30m_bars']
    demand=b30.low.shift(1).rolling(n).min()
    supply=b30.high.shift(1).rolling(n).max()
    width=simple_atr(b30,PARAMS['atr_bars'])*PARAMS['zone_halfwidth_atr']
    for key,series in dict(demand_at_bar_open=demand,supply_at_bar_open=supply,zone_width_at_bar_open=width).items():
        f[key]=map_available(series,f.index-pd.Timedelta(minutes=5))
    d,s,w=f.demand_at_bar_open,f.supply_at_bar_open,f.zone_width_at_bar_open
    for key,touch in [('recent_demand_touch',(f.low<=d+w)&(f.high>=d-w)),
                      ('recent_supply_touch',(f.low<=s+w)&(f.high>=s-w))]:
        f[key]=touch.astype(int).rolling(PARAMS['touch_memory_5m_bars']).max().fillna(0).astype(bool)
    day=minutes.index.floor('D')
    vwap=minutes.quote_volume.groupby(day).cumsum()/minutes.volume.groupby(day).cumsum().replace(0,np.nan)
    f['vwap']=vwap.reindex(f.index-MINUTE).to_numpy()
    if 'taker_buy_volume' in minutes:
        buys=minutes.taker_buy_volume.resample('5min',closed='left',label='right',origin='start_day').sum()
        f['taker_buy5']=buys.reindex(f.index)
        f['delta5']=2*f.taker_buy5-f.volume
    f.index.name='signal_time'
    return f


def signals_for(f,variant):
    long=(f.pullback30_return<0)&(f.close>f.previous_high)
    short=(f.pullback30_return>0)&(f.close<f.previous_low)
    if variant['bias']:
        long &= f.bias4==1
        short &= f.bias4==-1
    if variant['zone']:
        long &= f.recent_demand_touch
        short &= f.recent_supply_touch
    ready=f[['ema4','ema4_slope','reference30','atr5','previous_high','previous_low','stop_low','stop_high',
             'demand_at_bar_open','supply_at_bar_open','zone_width_at_bar_open','vwap']].notna().all(axis=1)
    ready &= f.index.hour>=PARAMS['first_signal_hour_UTC']
    side=pd.Series(0,index=f.index,dtype=int)
    side.loc[ready&long]=1
    side.loc[ready&short]=-1
    result=f.loc[side!=0].copy()
    result['side']=side.loc[result.index]
    return result


def exit_on_bar(side,stop,target,o,h,l):
    if side==1:
        if o<=stop: return o,'stop_gap',False,True
        if o>=target: return target,'target_gap',False,True
        stop_hit,target_hit=l<=stop,h>=target
    else:
        if o>=stop: return o,'stop_gap',False,True
        if o<=target: return target,'target_gap',False,True
        stop_hit,target_hit=h>=stop,l<=target
    if stop_hit: return stop,'stop',bool(target_hit),False
    if target_hit: return target,'target',False,False
    return None


def return_components(side,entry_raw,exit_raw,fee_bps,impact_bps):
    impact,fee=impact_bps/10000,fee_bps/10000
    entry=entry_raw*(1+side*impact)
    exit=exit_raw*(1-side*impact)
    fees=fee*(entry+exit)
    bp=10000*(side*(exit-entry)-fees)/entry
    return entry,exit,fees,bp


def backtest(minutes,signals,variant,fee_bps,impact_bps):
    a={c:minutes[c].to_numpy() for c in ('open','high','low','close')}
    index=minutes.index
    hold=PARAMS['maximum_holding_minutes']
    busy_until=index[0]
    trades,decisions=[],[]
    for timestamp,row in signals.iterrows():
        side=int(row.side)
        entry_time=timestamp+pd.Timedelta(minutes=PARAMS['entry_delay_minutes'])
        decision=dict(signal_time=timestamp,entry_time=entry_time,side=side,status='')
        decisions.append(decision)
        if entry_time<busy_until:
            decision['status']='position_already_open'; continue
        j=index.get_indexer([entry_time])[0]
        if j<0 or j+hold>len(index):
            decision['status']='insufficient_execution_horizon'; continue
        raw=float(a['open'][j])
        entry=raw*(1+side*impact_bps/10000)
        stop=(row.stop_low-PARAMS['stop_buffer_atr']*row.atr5 if side==1
              else row.stop_high+PARAMS['stop_buffer_atr']*row.atr5)
        risk=side*(entry-stop)
        if stop<=0 or risk<=0 or side*(raw-stop)<=0:
            decision['status']='entry_beyond_stop_or_invalid_stop'; continue
        target=entry+side*risk if variant['target']=='R1' else float(row.vwap)
        if target<=0:
            decision['status']='invalid_target'; continue
        reward=side*(target-entry)
        if reward<=0:
            decision['status']='target_wrong_side_of_entry'; continue
        if variant['target']=='VWAP' and reward/risk<PARAMS['vwap_minimum_gross_R']:
            decision['status']='vwap_reward_below_minimum'; continue
        result=None
        for k in range(j,j+hold):
            result=exit_on_bar(side,stop,target,a['open'][k],a['high'][k],a['low'][k])
            if result is not None: break
        if result is None:
            k=j+hold-1
            result=float(a['close'][k]),'time_exit',False,False
        exit_raw,reason,ambiguous,at_open=result
        exit_bound=index[k] if at_open else index[k]+MINUTE
        busy_until=exit_bound
        entry,exit,fees,bp=return_components(side,raw,exit_raw,fee_bps,impact_bps)
        decision.update(status='traded',trade_number=len(trades))
        trade=dict(signal_time=timestamp,entry_time=entry_time,exit_bar_open=index[k],exit_time_upper_bound=exit_bound,
                   side=side,entry_raw=raw,entry_fill=entry,stop_raw=float(stop),target_raw=float(target),
                   target_gross_R=float(reward/risk),initial_price_risk=float(risk),exit_raw=float(exit_raw),
                   exit_fill=exit,fee_per_base_unit=fees,exit_reason=reason,stop_and_target_same_minute=ambiguous)
        trade[BP]=bp
        trade[RR]=(side*(exit-entry)-fees)/risk
        for field in ('close','previous_high','previous_low','close30','reference30','bias4','close4','ema4','ema4_slope','vwap','atr5'):
            trade['signal_'+('close5' if field=='close' else field)]=row[field]
        trade['recent_demand_touch']=row.recent_demand_touch
        trade['recent_supply_touch']=row.recent_supply_touch
        trades.append(trade)
    t=pd.DataFrame(trades) if trades else pd.DataFrame(columns=['signal_time','entry_time','side',BP,RR])
    d=pd.DataFrame(decisions) if decisions else pd.DataFrame(columns=['signal_time','entry_time','side','status'])
    return t,d


def statistics(trades):
    if trades.empty: return dict(trades=0)
    bp=trades[BP]
    wins,losses=bp[bp>0].sum(),-bp[bp<0].sum()
    cumulative=np.r_[0.,trades[RR].cumsum().to_numpy()]
    return dict(trades=len(trades),win_rate_before_funding=float((bp>0).mean()),
                mean_execution_net_bp_before_funding=float(bp.mean()),median_execution_net_bp_before_funding=float(bp.median()),
                profit_factor_before_funding=float(wins/losses) if losses>0 else None,
                max_additive_closed_trade_drawdown_R=float((np.maximum.accumulate(cumulative)-cumulative).max()),
                ambiguous_exit_fraction=float(trades.stop_and_target_same_minute.mean()))


def bootstrap_mean(trades,first_day,last_day,block,repetitions,family):
    result=dict(block_days=block,mean_ci_low=None,mean_ci_high=None,family_lower_bound=None)
    if trades.empty: return result
    dates=pd.date_range(first_day,last_day,freq='D')
    sums=pd.Series(0.,index=dates); counts=pd.Series(0.,index=dates)
    g=trades.groupby(pd.to_datetime(trades.entry_time,utc=True).dt.floor('D'))[BP]
    gs,gc=g.sum(),g.count()
    sums.loc[gs.index]=gs.to_numpy(); counts.loc[gc.index]=gc.to_numpy()
    v,c=sums.to_numpy(),counts.to_numpy()
    n=len(dates); rng=np.random.default_rng(20260914+block)
    draws=[]; offsets=np.arange(block)
    for _ in range(repetitions):
        starts=rng.integers(0,n,size=math.ceil(n/block))
        sampled=((starts[:,None]+offsets)%n).ravel()[:n]
        count=c[sampled].sum()
        if count>0: draws.append(v[sampled].sum()/count)
    if draws:
        result.update(mean_ci_low=float(np.quantile(draws,.025)),mean_ci_high=float(np.quantile(draws,.975)),
                      family_lower_bound=float(np.quantile(draws,.05/family)),valid_bootstrap_draws=len(draws))
    return result


def selftest():
    for side,end,expected in [(1,110,1000),(-1,90,1000)]:
        np.testing.assert_allclose(return_components(side,100,end,0,0)[3],expected)
    np.testing.assert_allclose(return_components(1,100,100,10,0)[3],-20)
    # Independently computed adverse execution + both notional fees.
    for side in (1,-1):
        ei=100*(1+side*.0002); eo=101*(1-side*.0002)
        expected=10000*(side*(eo-ei)-.0005*(ei+eo))/ei
        np.testing.assert_allclose(return_components(side,100,101,5,2)[3],expected)
    require(exit_on_bar(1,95,105,100,106,94)==(95,'stop',True,False),'Long ambiguity failed.')
    require(exit_on_bar(-1,105,95,100,106,94)==(105,'stop',True,False),'Short ambiguity failed.')
    require(exit_on_bar(1,95,105,93,100,92)==(93,'stop_gap',False,True),'Long gap failed.')
    require(exit_on_bar(-1,105,95,107,108,99)==(107,'stop_gap',False,True),'Short gap failed.')
    require(exit_on_bar(-1,105,95,94,96,93)==(95,'target_gap',False,True),'Target gap failed.')
    idx=pd.date_range('2021-01-01',periods=18000,freq='min',tz='UTC')
    x=np.arange(len(idx),dtype=float); c=100+x*.0005+np.sin(x/37)
    volume=1+(x%7)
    synthetic=pd.DataFrame(dict(open=c,high=c+.2,low=c-.2,close=c,volume=volume,quote_volume=c*volume),index=idx)
    bars=aggregate_bars(synthetic,5)
    require(bars.index[0]==idx[5],'Completed bar timestamp failed.')
    np.testing.assert_allclose(bars.iloc[0].close,c[4])
    require(len(aggregate_bars(synthetic.iloc[:7],5))==1,'Partial bar used.')
    known=pd.Series([10.,20.],index=pd.to_datetime(['2021-01-01 04:00','2021-01-01 08:00'],utc=True))
    mapped=map_available(known,pd.to_datetime(['2021-01-01 03:59','2021-01-01 04:00','2021-01-01 07:59'],utc=True))
    require(np.isnan(mapped[0]),'Premature higher timeframe information.')
    np.testing.assert_allclose(mapped[1:],[10,10])
    for unit,multiplier in [('s',1),('ms',1000),('us',1000000),('ns',1000000000)]:
        parsed,detected=parse_time(pd.Series([1609459200*multiplier]))
        require(detected==unit and parsed[0]==idx[0],'Timestamp detection failed.')
    full=make_features(synthetic)
    for cutoff in (15000,15003):
        prefix=make_features(synthetic.iloc[:cutoff])
        pd.testing.assert_frame_equal(prefix,full.loc[prefix.index],check_freq=False,check_exact=False,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(full.iloc[0].vwap,np.average(c[:5],weights=volume[:5]))
    np.testing.assert_allclose(full.loc[idx[1445],'vwap'],np.average(c[1440:1445],weights=volume[1440:1445]))
    ix=idx[:130]
    execution=pd.DataFrame(dict(open=100.,high=100.2,low=99.8,close=100.,volume=1.),index=ix)
    row=dict(side=1,stop_low=95.,stop_high=105.,atr5=0.,vwap=105.,close=100.,previous_high=99.,previous_low=94.,
             close30=100.,reference30=101.,bias4=1,close4=100.,ema4=99.,ema4_slope=1.,recent_demand_touch=True,recent_supply_touch=False)
    sig=pd.DataFrame([row],index=pd.DatetimeIndex([ix[5]]))
    execution.loc[ix[6],'low']=94.
    trades,_=backtest(execution,sig,dict(target='R1'),0,0)
    require(len(trades)==1 and trades.iloc[0].entry_time==ix[6] and trades.iloc[0].exit_reason=='stop','Entry-minute stop failed.')
    np.testing.assert_allclose(trades.iloc[0][BP],-500)
    execution.loc[ix[6],'low']=99.8
    trades,_=backtest(execution,sig,dict(target='R1'),0,0)
    require(trades.iloc[0].exit_reason=='time_exit' and trades.iloc[0].exit_time_upper_bound==ix[126],'Holding horizon failed.')
    row['side']=-1; row['vwap']=95.
    sig=pd.DataFrame([row],index=pd.DatetimeIndex([ix[5]]))
    execution.loc[ix[6],'high']=106.
    trades,_=backtest(execution,sig,dict(target='R1'),0,0)
    np.testing.assert_allclose(trades.iloc[0][BP],-500)
    # Real ZIP parsing and holdout exclusion using temporary synthetic files.
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary); path=folder/'BTCUSDT_1m_2021.csv(1).zip'
        line='1609459200000,100,101,99,100,2,1609459259999,200,3,1,100,0\n'
        with zipfile.ZipFile(path,'w') as z: z.writestr('sample.csv',line)
        for year in (2024,2025,2026):
            (folder/f'BTCUSDT_1m_{year}.csv.zip').write_bytes(b'NOT A ZIP: MUST NOT BE OPENED')
        selected=select_file(folder,'BTCUSDT',2021)
        parsed,audit=load_year(selected,'BTCUSDT',2021,'auto')
        require(len(parsed)==1 and audit['missing_minutes']==525599,'ZIP audit failed.')
        require(audit['mapping']['quote_volume']=='quote_volume','Schema mapping failed.')
        # Regression for the actual column names in the user's screenshot.
        custom=('timestamp_utc,timestamp_ms,open,high,low,close,volume_base,turnover_usdt\n'
                '2021-01-01 00:00:00+00:00,1609459200000,100,101,99,100,2,200\n'
                '2021-01-01 00:01:00+00:00,1609459260000,102,103,101,102,3,306\n')
        with zipfile.ZipFile(path,'w') as z: z.writestr('sample.csv',custom)
        parsed,audit=load_year(path,'BTCUSDT',2021,'auto')
        require(audit['mapping']['volume']=='volume_base','Custom base volume mapping failed.')
        require(audit['mapping']['quote_volume']=='turnover_usdt','Custom turnover mapping failed.')
        require(audit['timestamp_utc_crosschecked'],'Dual timestamps were not checked.')
        np.testing.assert_allclose(parsed.volume,[2,3])
        np.testing.assert_allclose(parsed.quote_volume,[200,306])
        np.testing.assert_allclose(parsed.quote_volume.sum()/parsed.volume.sum(),101.2)
        bad=custom.replace('2021-01-01 00:01:00+00:00','2021-01-01 00:02:00+00:00')
        with zipfile.ZipFile(path,'w') as z: z.writestr('sample.csv',bad)
        try:
            load_year(path,'BTCUSDT',2021,'auto')
        except ValueError as error:
            require('disagrees' in str(error),'Unexpected timestamp mismatch failure.')
        else:
            raise AssertionError('Inconsistent dual timestamps were accepted.')
    # Boundary missing minutes are different from internal gaps.
    coverage=dict(year=2021,leading_missing_minutes=105120,trailing_missing_minutes=0,
                  internal_missing_minutes=0,first_open='2021-03-15 00:00:00+00:00')
    require(coverage_problem(coverage) is None,'Continuous partial first year rejected.')
    for changes in (dict(internal_missing_minutes=1),dict(trailing_missing_minutes=1),dict(year=2022)):
        require(coverage_problem(dict(coverage,**changes)) is not None,'Unsafe missing data accepted.')
    require(int((pd.Timestamp('2022-01-01',tz='UTC')-pd.Timestamp(coverage['first_open']))/MINUTE)==420480,
            'Partial-year expected row count failed.')
    print('PASS: leading-year coverage accepted; internal/trailing/later-year gaps rejected.',flush=True)
    print('PASS: custom volume_base/turnover_usdt schema and dual timestamp consistency.',flush=True)
    print('PASS: arithmetic, fees, long/short exits, gaps, ambiguity, entry timing, time exit,',flush=True)
    print('      bar availability, partial bars, timestamp units, VWAP/reset, prefix invariance, ZIP audit.',flush=True)


def run(args,output):
    source=Path(__file__).read_text(encoding='utf-8')
    (output/'research_v1_source.txt').write_text(source,encoding='utf-8')
    write_json(output/'run_manifest.json',dict(version=VERSION,stage='exploratory_discovery_only',
        years_allowed=YEARS,assets=args.assets,original_comparison_assets=ASSETS,coverage_policy='leading_2021_only_no_internal_or_trailing_gaps',parameters=PARAMS,variants=VARIANTS,
        fee_bps_per_side=args.fee_bps,adverse_execution_bps_per_side=args.impact_bps,
        funding_included=False,borrowing_included=False,portfolio_simulation=False,
        bootstrap_draws=args.bootstrap,python=sys.version,platform=platform.platform(),
        pandas=pd.__version__,numpy=np.__version__,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        research_contract=__doc__))
    audits,summary,intervals=[],[],[]
    family=len(ASSETS)*len(VARIANTS)*2
    for asset in args.assets:
        frames=[]
        for year in YEARS:
            path=select_file(args.data,asset,year)
            print(f'Auditing {path.name}',flush=True)
            frame,audit=load_year(path,asset,year,args.timestamp_unit)
            audits.append(audit); write_json(output/'data_audit.json',audits)
            print(f"  rows={audit['rows']:,}; missing={audit['missing_minutes']:,}; quote_volume={audit['has_quote_volume']}",flush=True)
            print(f"  first={audit['first_open']}; last={audit['last_open']}; internal_gaps={audit['internal_missing_minutes']:,}; coverage_eligible={audit['backtest_coverage_eligible']}",flush=True)
            if audit['coverage_problem']:
                print(f"  COVERAGE BLOCK: {audit['coverage_problem']}",flush=True)
            if not args.audit_only:
                require(audit['backtest_coverage_eligible'],f"{path.name}: {audit['coverage_problem']} Audit saved.")
                require(audit['has_quote_volume'],f'{path.name}: quote volume missing; exact VWAP unavailable. Audit saved.')
                frames.append(frame)
            else: del frame
        if args.audit_only: continue
        minutes=pd.concat(frames).sort_index(); del frames,frame
        require(not minutes.index.has_duplicates,'Combined duplicate timestamps.')
        require((minutes.index[1:]-minutes.index[:-1]==MINUTE).all(),'Combined data are not continuous.')
        if 'taker_buy_volume' in minutes and minutes.taker_buy_volume.isna().any():
            minutes=minutes.drop(columns='taker_buy_volume')
        require(not minutes.isna().any().any(),'Inconsistent required columns across years.')
        features=make_features(minutes)
        features.to_csv(output/f'{asset}_features_5m.csv.gz',compression='gzip')
        for variant in VARIANTS:
            name=variant['name']; prefix=f'{asset}_{name}'
            print(f'Running {prefix}',flush=True)
            signals=signals_for(features,variant)
            trades,decisions=backtest(minutes,signals,variant,args.fee_bps,args.impact_bps)
            signals.to_csv(output/f'{prefix}_signals.csv.gz',compression='gzip')
            trades.to_csv(output/f'{prefix}_trades.csv',index=False)
            decisions.to_csv(output/f'{prefix}_decisions.csv',index=False)
            summary.append(dict(asset=asset,variant=name,period='2021-2023_available_coverage',
                                data_start=str(minutes.index[0]),data_end=str(minutes.index[-1]),**statistics(trades)))
            for year in YEARS:
                subset=trades if trades.empty else trades.loc[pd.to_datetime(trades.entry_time,utc=True).dt.year==year]
                year_start=max(minutes.index[0],pd.Timestamp(f'{year}-01-01',tz='UTC'))
                year_end=min(minutes.index[-1],pd.Timestamp(f'{year+1}-01-01',tz='UTC')-MINUTE)
                summary.append(dict(asset=asset,variant=name,period=str(year),data_start=str(year_start),data_end=str(year_end),**statistics(subset)))
            for number,trade in trades.head(3).iterrows():
                begin=trade.signal_time-pd.Timedelta(hours=2)
                end=trade.exit_time_upper_bound+pd.Timedelta(minutes=5)
                minutes.loc[begin:end].to_csv(output/f'{prefix}_replay_{number}.csv')
            for block in (7,28):
                print(f'  {len(trades)} trades; bootstrap {block}-day blocks ({args.bootstrap:,} draws)',flush=True)
                intervals.append(dict(asset=asset,variant=name,units=BP,family_size=family,
                    **bootstrap_mean(trades,minutes.index[0].floor('D'),minutes.index[-1].floor('D'),block,args.bootstrap,family)))
            pd.DataFrame(summary).to_csv(output/'summary_before_funding.csv',index=False)
            pd.DataFrame(intervals).to_csv(output/'bootstrap_before_funding.csv',index=False)
        del minutes,features
    write_json(output/'completion.json',dict(completed=True,audit_only=args.audit_only,
        holdout_price_files_opened=False,message='Discovery only. No profitability confirmation. Funding and borrowing excluded.'))
    bundle=output.with_suffix('.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in output.iterdir():
            if path.is_file(): z.write(path,arcname=path.name)
    print(f'\nComplete. Share: {bundle.resolve()}',flush=True)
    print('Results are BEFORE FUNDING; no strategy has been selected.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,default=Path('.'),help='Folder directly containing discovery ZIP/CSV files.')
    p.add_argument('--output',type=Path,default=Path('results'))
    p.add_argument('--assets',choices=ASSETS,nargs='+',default=list(ASSETS),help='Assets to run; select ETHUSDT to preserve existing BTC results.')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--fee-bps',type=float,help='Fee per side, basis points; explicitly required for backtest.')
    p.add_argument('--impact-bps',type=float,help='Adverse half-spread + slippage per side, basis points.')
    p.add_argument('--timestamp-unit',choices=['auto','s','ms','us','ns'],default='auto')
    p.add_argument('--bootstrap',type=int,default=10000)
    args=p.parse_args()
    require(len(args.assets)==len(set(args.assets)),'Duplicate assets requested.')
    selftest()
    if args.self_test: return
    require(args.data.is_dir(),f'Data folder not found: {args.data}')
    if not args.audit_only:
        require(args.fee_bps is not None and args.impact_bps is not None,'Supply --fee-bps and --impact-bps, or use --audit-only.')
        require(0<=args.fee_bps<1000 and 0<=args.impact_bps<1000,'Costs must be finite, nonnegative and below 1000 bps per side.')
        require(args.bootstrap>=10000,'Use at least 10000 bootstrap draws for adjusted tails.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.output/f'discovery_{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    print(f'Output: {output.resolve()}',flush=True)
    try:
        run(args,output)
    except Exception as error:
        write_json(output/'failure.json',dict(completed=False,error_type=type(error).__name__,message=str(error)))
        print(f'Failure details saved: {(output/"failure.json").resolve()}',file=sys.stderr)
        raise


if __name__=='__main__':
    main()

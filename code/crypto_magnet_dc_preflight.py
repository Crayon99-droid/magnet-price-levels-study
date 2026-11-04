#!/usr/bin/env python3
"""Bybit USDT-perpetual M1 ZIP validator + DC threshold preflight.

Designed for the user's BTCUSDT, ETHUSDT, XRPUSDT yearly CSV ZIPs.
This is an INPUT/DETECTOR audit, NOT a magnet strategy backtest.

The expected source CSV fields are:
 timestamp_utc,timestamp_ms,open,high,low,close,volume_base,turnover_usdt

Runs with Python standard library only; no Bybit access/API key required.
Examples:
    py crypto_magnet_dc_preflight.py --data-dir "C:\\TradingResearch"
    py crypto_magnet_dc_preflight.py --data-dir . --symbols BTCUSDT ETHUSDT XRPUSDT

Stops on any non-consecutive minute, invalid OHLC or timestamp mismatch.
Emits an audit JSON in the chosen directory. Missing years are reported, not fabricated.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import zipfile
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

REQUIRED = ("timestamp_utc", "timestamp_ms", "open", "high", "low", "close", "volume_base", "turnover_usdt")
FILENAME = re.compile(r"^(BTCUSDT|ETHUSDT|XRPUSDT)_1m_(20\d\d)\.csv\.zip$", re.I)
PERIOD_MS = 60_000
K = 1.25
VOL_WINDOW = 240


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def timestamp_text_to_ms(value):
    text = value.strip()
    try:
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt=dt.replace(tzinfo=timezone.utc)
        return round(dt.timestamp()*1000)
    except ValueError:
        raise ValueError('Invalid timestamp_utc format: '+repr(value))


class DCCounter:
    """Causal close-only detector with yesterday's completed 240m-vol median."""
    def __init__(self):
        self.date = None
        self.prev_date = None
        self.prev_median = None
        self.day_sigmas=[]
        self.theta=None
        self.ret_window=deque()
        self.sum_sq=0.0
        self.prev_close=None
        self.last_ts=None
        self.direction=0
        self.high=None
        self.low=None
        self.total=0
        self.up=0
        self.down=0
        self.initialization_minutes=0
        self.theta_min=math.inf
        self.theta_max=-math.inf

    def feed(self, ts_ms, close):
        dt=datetime.fromtimestamp(ts_ms/1000,timezone.utc)
        today=dt.date()
        if today!=self.date:
            if self.date is not None:
                self.prev_median=statistics.median(self.day_sigmas) if self.day_sigmas else None
                self.prev_date=self.date
            self.day_sigmas=[]
            self.theta=(K*self.prev_median if (self.prev_date is not None and today-self.prev_date == timedelta(days=1) and self.prev_median) else None)
            self.date=today
        contiguous = self.last_ts is None or ts_ms - self.last_ts == PERIOD_MS
        if not contiguous:
            # No invisible interpolation across an unknown price path.
            self.ret_window.clear();self.sum_sq=0.0
            self.prev_close=None
            self.direction=0;self.high=None;self.low=None
        self.last_ts=ts_ms
        if self.prev_close is not None:
            ret=math.log(close/self.prev_close)
            s=ret*ret
            self.ret_window.append(s);self.sum_sq+=s
            if len(self.ret_window)>VOL_WINDOW:
                self.sum_sq-=self.ret_window.popleft()
            if len(self.ret_window)==VOL_WINDOW:
                self.day_sigmas.append(math.sqrt(max(0.0,self.sum_sq)))
        self.prev_close=close
        th=self.theta
        if th is None or not (math.isfinite(th) and th>0):
            self.initialization_minutes+=1
            return
        self.theta_min=min(self.theta_min, th);self.theta_max=max(self.theta_max,th)
        if self.high is None:
            self.high=close;self.low=close;return
        if self.direction==0:
            self.high=max(self.high,close);self.low=min(self.low,close)
            if close >= self.low*(1+th):
                self.direction=1;self.high=close;self.up+=1;self.total+=1
            elif close <= self.high*(1-th):
                self.direction=-1;self.low=close;self.down+=1;self.total+=1
        elif self.direction==1:
            self.high=max(self.high,close)
            if close<=self.high*(1-th):
                self.direction=-1;self.low=close;self.down+=1;self.total+=1
        else:
            self.low=min(self.low,close)
            if close>=self.low*(1+th):
                self.direction=1;self.high=close;self.up+=1;self.total+=1


def manifest_expected(directory, symbol):
    candidate=directory/f'{symbol}_source_manifest.json'
    if not candidate.exists():return {}, None
    with candidate.open('r',encoding='utf-8') as f: data=json.load(f)
    return {x['file']:x['sha256'] for x in data.get('archives',[])},str(candidate)


def audit_year(path, symbol, year, detector):
    print(f'  {path.name}: checking SHA256, ZIP, timestamps, candles and DC...', flush=True)
    digest=file_sha256(path)
    rows=0;first=None;last=None;prior=None
    start_of_year=int(datetime(year,1,1,tzinfo=timezone.utc).timestamp()*1000)
    end_of_year=int(datetime(year+1,1,1,tzinfo=timezone.utc).timestamp()*1000)
    errors=[]
    with zipfile.ZipFile(path) as z:
        csv_names=[x.filename for x in z.infolist() if x.filename.lower().endswith('.csv') and not x.is_dir()]
        if len(csv_names)!=1:
            raise ValueError(f'{path.name}: expected 1 CSV member, found {csv_names!r}')
        with z.open(csv_names[0]) as raw, io.TextIOWrapper(raw,encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            missing=[key for key in REQUIRED if key not in (reader.fieldnames or [])]
            if missing: raise ValueError(f'{path.name}: missing {missing}. Actual header={reader.fieldnames!r}')
            for record in reader:
                rows+=1
                try:
                    stamp=int(record['timestamp_ms'])
                    if stamp % PERIOD_MS:raise ValueError('timestamp not on minute boundary')
                    if timestamp_text_to_ms(record['timestamp_utc']) !=stamp:
                        raise ValueError('timestamp_utc differs from timestamp_ms')
                    if stamp<start_of_year or stamp>=end_of_year:raise ValueError('outside calendar year')
                    if prior is not None and stamp-prior != PERIOD_MS:
                        raise ValueError(f'expected UTC +60s but delta={(stamp-prior)/1000:g}s')
                    o,h,l,c=(float(record[key]) for key in ('open','high','low','close'))
                    v=float(record['volume_base']);turn=float(record['turnover_usdt'])
                    if not all(map(math.isfinite,(o,h,l,c,v,turn))):raise ValueError('nonfinite OHLCV')
                    if l<=0 or min(o,h,c)<=0 or not (l<=o<=h and l<=c<=h and l<=h):
                        raise ValueError('invalid OHLC ordering or prices')
                    if v<0 or turn<0:raise ValueError('negative volume or turnover')
                except (ValueError,TypeError,OverflowError) as exc:
                    errors.append({'line':rows+1,'error':str(exc)})
                    if len(errors)>=5:break
                    continue
                if first is None:first=stamp
                detector.feed(stamp,c)
                prior=last=stamp
    if errors:raise ValueError(f'{path.name}: first errors: {errors!r}')
    if rows<1:raise ValueError(f'{path.name}: empty CSV')
    return {'file':path.name,'year':year,'rows':rows,'first_utc':datetime.fromtimestamp(first/1000,timezone.utc).isoformat(),
            'last_utc':datetime.fromtimestamp(last/1000,timezone.utc).isoformat(), 'sha256':digest,'zip_member':csv_names[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir',default='.',type=Path)
    p.add_argument('--symbols',nargs='+',default=['BTCUSDT','ETHUSDT','XRPUSDT'],choices=['BTCUSDT','ETHUSDT','XRPUSDT'])
    p.add_argument('--output',default='crypto_source_dc_audit.json')
    args=p.parse_args()
    base=args.data_dir.expanduser().resolve()
    if not base.is_dir():p.error('data directory does not exist: '+str(base))
    paths={}
    for f in base.iterdir():
        m=FILENAME.match(f.name)
        if m and m.group(1).upper() in args.symbols:
            key=(m.group(1).upper(),int(m.group(2)))
            if key in paths:raise RuntimeError('duplicate archive '+str(key))
            paths[key]=f
    result={'status':'IN_PROGRESS','source_dir':str(base),'purpose':'data and causal DC preflight; no trades simulated',
            'dc_theta':'1.25 * prior-completed-day median of rolling-240m realized volatility',
            'symbols':{},'warnings':[]}
    if not paths:raise SystemExit('No supported crypto ZIPs found in '+str(base))
    print(f'Discovered {len(paths)} supported yearly ZIP archives in {base}',flush=True)
    for symbol in args.symbols:
        years=sorted(year for sym,year in paths if sym==symbol)
        if not years:
            result['warnings'].append(f'{symbol}: no ZIP files found')
            continue
        expected,manifest_path=manifest_expected(base,symbol)
        detector=DCCounter();files=[];prior_end=None
        print(f'\n{symbol}: years {years[0]}-{years[-1]}',flush=True)
        for year in years:
            path=paths[symbol,year]
            info=audit_year(path,symbol,year,detector)
            if manifest_path and path.name in expected:
                if info['sha256'].lower()!=expected[path.name].lower():
                    raise ValueError(f'{path.name}: SHA256 does not match source manifest')
                info['manifest_hash_match']=True
            else:
                info['manifest_hash_match']=None
            current_start=int(datetime.fromisoformat(info['first_utc']).timestamp()*1000)
            current_end=int(datetime.fromisoformat(info['last_utc']).timestamp()*1000)
            if prior_end is not None and current_start-prior_end != PERIOD_MS:
                result['warnings'].append(f'{symbol}: data gap across archive boundary before {year}: '
                                          f'{(current_start-prior_end)//PERIOD_MS-1:,} missing minutes')
                # Detector itself resets continuity on next feed.
            prior_end=current_end
            files.append(info)
        yr_miss=sorted(set(range(min(years),max(years)+1))-set(years))
        if yr_miss:result['warnings'].append(f'{symbol}: missing intermediate years {yr_miss}')
        result['symbols'][symbol]={'archive_years':years,'missing_intermediate_years':yr_miss,
          'manifest_file':manifest_path,'minute_bars':sum(x['rows'] for x in files),
          'dc_events':detector.total,'dc_up':detector.up,'dc_down':detector.down,
          'dc_theta_min':detector.theta_min if math.isfinite(detector.theta_min) else None,
          'dc_theta_max':detector.theta_max if math.isfinite(detector.theta_max) else None,
          'insufficient_volatility_warmup_minutes':detector.initialization_minutes,'files':files}
        print(f'  PASS: {sum(x["rows"] for x in files):,} minute candles; '
              f'{detector.total:,} preliminary close-confirmed DC events',flush=True)
        if not manifest_path:
            result['warnings'].append(f'{symbol}: original SHA256 source manifest not found; cannot independently verify archive provenance')
    result['status']='PASS_DATA_INTEGRITY'
    out=base/args.output
    with out.open('w',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print('\nDATA INTEGRITY AUDIT PASSED for supplied files; strategic profitability NOT tested.',flush=True)
    for w in result['warnings']: print('WARNING:',w,flush=True)
    print('Saved:',out,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('AUDIT FAILED:',str(exc),file=sys.stderr)
        sys.exit(1)

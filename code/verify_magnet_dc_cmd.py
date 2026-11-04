#!/usr/bin/env python3
"""Reproduce the Magnet+DC EURUSD pilot from raw M1 bid/ask and study EARLIER exits.

Run on Windows Command Prompt:
  py -m pip install numpy pandas numba
  py verify_magnet_dc_cmd.py --data-dir "." --pack "Magnet_DC_Research_Pilot_2024_2026.zip"

Put the original EURUSD_2024.zip, EURUSD_2025.zip and EURUSD_2026.zip
(and the research ZIP) in --data-dir. The script rebuilds missing minute CSVs.
Already converted EURUSD_YYYY.m1.csv files work too.

IMPORTANT: Full verification recomputes using the *same original research engine*.
That tests repeatability and catches file/data mismatches, but is NOT an
independent audit of possible conceptual/code bugs.

2024-25: candidate rule plus exploratory earlier-exit policies.
2026: frozen ORIGINAL policy only; do not use for new exit-rule selection.

No order-book queue/slippage/funding/financing included except an optional
additional all-in round-trip slippage penalty in pips.
"""
from __future__ import annotations
import argparse
import hashlib
import csv
import time
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Missing packages. Run: py -m pip install numpy pandas numba", file=sys.stderr)
    raise

PIP = 10000.0
CANDIDATE = dict(aligned=True, max_ratio=0.50, min_room_pips=15.0,
                 take_profit_R=0.5, stop_R=1.0, horizon_minutes=1440)
DEV_REFERENCES = {2024: (23, 0.03536659377961335, 1.130266539699268),
                  2025: (37, 0.013294417979263394, 1.0429851292258079)}
HOLD_REFERENCES = {2026: (8, 0.22785739921396592, 2.7700226953432714)}


def convert_ticks_to_minute(src: Path, dst: Path):
    """Replicate the original streaming tick-to-M1 preprocessing (no extra packages)."""
    t0=time.time(); total=0; rejected=0; valid=0; minutes=0
    prev_bid=None;prev_ask=None;prior_key=None;bar=None
    with zipfile.ZipFile(src) as z:
        members=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(members)!=1:
            raise ValueError(f'Expected one tick CSV inside {src.name}, found {len(members)}')
        with z.open(members[0]) as inp, dst.open('w',newline='',encoding='utf-8') as out:
            w=csv.writer(out)
            w.writerow(['minute_utc','bid_open','ask_open','bid_high','bid_low','ask_high',
                        'ask_low','bid_close','ask_close','mid_high','mid_low',
                        'spread_mean_pips','quote_count'])
            next(inp)
            def emit(b):
                return [b[0],*b[1:11],round(b[11]/b[12],5),b[12]]
            for line in inp:
                total+=1
                fields=line.decode('ascii').rstrip('\r\n').split('\t')
                if len(fields)<4:
                    rejected+=1; continue
                key=fields[0].replace('.','-')+' '+fields[1][:5]
                if fields[2]: prev_bid=float(fields[2])
                if fields[3]: prev_ask=float(fields[3])
                if prev_bid is None or prev_ask is None or prev_bid<=0 or prev_ask<=prev_bid:
                    rejected+=1; continue
                bid=prev_bid; ask=prev_ask
                if prior_key!=key:
                    if bar is not None:
                        w.writerow(emit(bar));minutes+=1
                    mid=(bid+ask)/2
                    bar=[key,bid,ask,bid,bid,ask,ask,bid,ask,mid,mid,(ask-bid)*10000,1]
                    prior_key=key
                else:
                    mid=(bid+ask)/2
                    bar[3]=max(bar[3],bid);bar[4]=min(bar[4],bid)
                    bar[5]=max(bar[5],ask);bar[6]=min(bar[6],ask)
                    bar[7]=bid;bar[8]=ask
                    bar[9]=max(bar[9],mid);bar[10]=min(bar[10],mid)
                    bar[11]+=(ask-bid)*10000;bar[12]+=1
                valid+=1
            if bar:
                w.writerow(emit(bar)); minutes+=1
    print(f'  Converted {src.name}: {total:,} records, {valid:,} quotes, '
          f'{rejected:,} rejected, {minutes:,} M1 bars in {time.time()-t0:.1f}s -> {dst.name}',flush=True)


def engine_and_zip(zip_path: Path, dest: Path):
    if not zip_path.is_file():
        raise FileNotFoundError(f"Research ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        if 'magnet_dc_v1.py' not in z.namelist():
            raise RuntimeError('ZIP lacks the original magnet_dc_v1.py')
        code = z.read('magnet_dc_v1.py')
    # Only extract the known engine member, not arbitrary paths from ZIP.
    path = dest / 'research_engine_v1.py'
    path.write_bytes(code)
    spec = importlib.util.spec_from_file_location('research_engine_v1', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    print('Engine SHA256:', hashlib.sha256(code).hexdigest())
    return mod


def reference_csv(z, name):
    if name not in z.namelist():
        return None
    return pd.read_csv(io.BytesIO(z.read(name)))


def regenerated_events(mod, raw, year):
    theta = mod.build_theta(raw)
    ix, side, prev_os, age = mod.dc_events(raw.mid_close.to_numpy(), theta)
    bars15 = mod.build_m15(raw)
    levels = mod.level_candidates(raw, bars15)
    events = mod.calculate_magnet_events(raw, levels, ix, side, prev_os, age, theta, bars15)
    events = mod.attach_outcomes(raw, events)
    print(f'  {year}: {len(raw):,} M1 rows | {len(ix):,} DC | '
          f'{len(levels):,} levels | {len(events):,} eligible evaluated events')
    return events, len(ix), len(levels)


def compare_event_files(calculated, reference, year):
    if reference is None:
        print(f'  {year}: no published event CSV in research ZIP')
        return False
    match = len(calculated) == len(reference)
    cols = ['idx', 'dc_side', 'near_side', 'near', 'far', 'distance_ratio',
            'upper', 'lower', 'actual_first', 'near_net_R', 'near_risk_pips']
    if match:
        for col in cols:
            aa = calculated[col].to_numpy(dtype='float64')
            bb = reference[col].to_numpy(dtype='float64')
            if not np.allclose(aa, bb, rtol=1e-9, atol=1e-8, equal_nan=True):
                print(f'  {year}: MISMATCH in {col} (original vs rebuilt)')
                match = False
                break
    print(f'  {year}: event reproduction against published CSV = '
          f'{"PASS" if match else "FAIL"} ({len(calculated)} vs {len(reference)} rows)')
    return match


def take_profit_or_time(i, side, magnet, market, tp_R, horizon, fee_pips, extra_slip,
                        lock=False):
    """Per-bar executable stop-first simulator. Signal at bar i close; entry i+1 OPEN.

    Stop distance = original (executed) entry to magnet. For a long, stop/TP
    use BID highs/lows; for a short, ASK highs/lows. If both stop and TP appear
    in a minute, stop is assumed to occur first. No intrabar path look-ahead.

    lock=True: after a COMPLETED bar close first reaches +0.35 gross R,
    use a +0.15 gross R protective stop FROM THE FOLLOWING BAR. This is only
    an exploratory variant; still can lose due to unmodeled gaps/slippage.
    """
    tm = market['time']
    s = i + 1
    if s >= len(tm) or (tm[s] - tm[i]) > np.timedelta64(90, 's'):
        return None
    cap = np.searchsorted(tm, tm[i] + np.timedelta64(int(horizon), 'm'), side='right')
    if cap <= s:
        return None
    entry = float(market['ask_open'][s] if side == 1 else market['bid_open'][s])
    risk = side * (magnet - entry)
    if risk <= 0 or risk * PIP < 0.25:
        return None
    original_stop = entry - side * risk
    stop = original_stop
    target = entry + side * risk * tp_R
    trigger = entry + side * risk * .35
    lock_price = entry + side * risk * .15
    activated = False
    exit_idx = cap - 1
    exitp = float(market['bid_close'][exit_idx] if side == 1 else market['ask_close'][exit_idx])
    reason = 'time'
    for j in range(s, cap):
        if side == 1:
            adverse = market['bid_low'][j] <= stop
            tp_hit = market['bid_high'][j] >= target
            close_exe = market['bid_close'][j]
        else:
            adverse = market['ask_high'][j] >= stop
            tp_hit = market['ask_low'][j] <= target
            close_exe = market['ask_close'][j]
        if adverse:
            exit_idx, exitp, reason = j, stop, 'lock_stop' if activated else 'stop'
            break
        if tp_hit:
            exit_idx, exitp, reason = j, target, 'target'
            break
        # Only COMPLETED-bar information can change stop for NEXT bar.
        if lock and not activated and side * (float(close_exe) - trigger) >= 0:
            stop = lock_price
            activated = True
    risk_pips = risk * PIP
    net_pips = side * (exitp - entry) * PIP - fee_pips - extra_slip
    return dict(entry_idx=s, exit_idx=int(exit_idx), entry_time=str(tm[s]),
                exit_time=str(tm[exit_idx]), side=side, risk_pips=risk_pips,
                gross_R=(net_pips+fee_pips+extra_slip)/risk_pips,
                net_R=net_pips/risk_pips, net_pips=net_pips, exit_reason=reason,
                minutes_held=float((tm[exit_idx]-tm[s])/np.timedelta64(1, 'm')))


def trade_events(events, market, year, *, horizon=1440, tp_R=.5, fee=.8,
                 slip=0., lock=False, minimum_room=15., nonoverlap=True):
    eligible = events[(events['near_dc_aligned'].astype(bool)) &
                      (events['distance_ratio'] <= .50) &
                      (events['near_risk_pips'] >= minimum_room)]
    eligible = eligible.sort_values('idx')
    rec = []
    last_exit_idx = -1
    for row in eligible.itertuples(index=False):
        i = int(row.idx)
        if nonoverlap and i <= last_exit_idx:
            continue
        t = take_profit_or_time(i, int(row.near_side), float(row.near),market,
                                tp_R, horizon, fee, slip, lock=lock)
        if t is None:
            continue
        if nonoverlap:
            last_exit_idx = t['exit_idx']
        t.update(year=year, signal_idx=i, signal_time=str(row.time),
                 target_magnet=float(row.near), prior_os_ratio=float(row.prior_os_ratio))
        rec.append(t)
    return pd.DataFrame(rec)


def metrics(trades, year, policy, horizon):
    if trades.empty:
        return dict(year=year, policy=policy, max_hold_min=horizon,n=0)
    a = trades.net_R.to_numpy()
    loss = -a[a < 0].sum()
    wins = a[a > 0].sum()
    return dict(year=year, policy=policy, max_hold_min=horizon,
                n=len(a), mean_net_R=float(a.mean()), total_net_R=float(a.sum()),
                profit_factor=float(wins/loss) if loss else float('inf'),
                win_pct=float(100*(a > 0).mean()),
                median_hold_min=float(trades.minutes_held.median()),
                p90_hold_min=float(trades.minutes_held.quantile(.9)),
                target=int((trades.exit_reason == 'target').sum()),
                stop=int((trades.exit_reason == 'stop').sum()),
                locked_stop=int((trades.exit_reason == 'lock_stop').sum()),
                time_exit=int((trades.exit_reason == 'time').sum()))


def compare_original(trades, year, z):
    m = metrics(trades, year, 'tp0.5', 1440)
    ref = DEV_REFERENCES.get(year, HOLD_REFERENCES.get(year))
    if m.get('n',0)==0:
        return False
    agrees = (m['n'] == ref[0] and
              abs(m['mean_net_R']-ref[1]) < 1e-8 and
              abs(m['profit_factor']-ref[2]) < 1e-7)
    if year in (2024,2025):
        old = reference_csv(z, 'exits_trades.csv')
        if old is not None:
            old = old[(old.year==year)&(old.minrisk==15)&
                      (old['exit']=='tp0.5')&(old.nonoverlap==True)].sort_values('entry_idx')
            curr=trades.sort_values('signal_idx')
            if len(curr)!=len(old) or not np.array_equal(curr.signal_idx.to_numpy(),old.entry_idx.to_numpy()) or \
               not np.allclose(curr.net_R.to_numpy(),old.net_R.to_numpy(),rtol=0,atol=1e-9):
                agrees=False
    if year==2026:
        old=reference_csv(z,'holdout_2026_frozen_trades.csv')
        if old is not None:
            curr=trades.sort_values('signal_idx')
            old=old.sort_values('entry_idx')
            if len(curr)!=len(old) or not np.array_equal(curr.signal_idx.to_numpy(),old.entry_idx.to_numpy()) or \
               not np.allclose(curr.net_R.to_numpy(),old.net_R.to_numpy(),rtol=0,atol=1e-9):
                agrees=False
    print(f"  {year}: frozen-rule replication = {'PASS' if agrees else 'FAIL'} | "
          f"{m['n']} trades, {m['mean_net_R']:+.6f} R/trade, PF {m['profit_factor']:.4f}")
    return agrees


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir',type=Path,default=Path('.'),help='directory with EURUSD_YYYY.m1.csv')
    ap.add_argument('--pack',type=Path,default=Path('Magnet_DC_Research_Pilot_2024_2026.zip'),help='original research ZIP')
    ap.add_argument('--out',type=Path,default=Path('magnet_dc_verification'),help='output directory')
    ap.add_argument('--fast',action='store_true',help='use published event CSVs; skip rebuilding DC/magnets (not independent verification)')
    ap.add_argument('--fee-pips',type=float,default=.8,help='hypothetical round-trip commission in pips, default 0.8')
    ap.add_argument('--extra-slippage-pips',type=float,default=0.,help='extra round-trip execution penalty in pips')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    zip_path=args.pack.resolve()
    if not zip_path.exists() and not args.pack.is_absolute():
        zip_path=(args.data_dir/args.pack).resolve()
    core=engine_and_zip(zip_path,args.out)
    engine_tests = None
    summary=[]; alltrades=[]; validations=[]
    with zipfile.ZipFile(zip_path) as z:
        for year in (2024,2025,2026):
            csv=args.data_dir/f'EURUSD_{year}.m1.csv'
            srczip=args.data_dir/f'EURUSD_{year}.zip'
            if not csv.exists() and srczip.exists():
                print(f'\nMinute CSV for {year} is missing; converting original tick archive...',flush=True)
                convert_ticks_to_minute(srczip,csv)
            if not csv.exists():
                print(f'  {year}: SKIPPED: {csv} not found')
                continue
            print(f'\nLoading {csv} ...',flush=True)
            raw=core.load_fx(csv)
            market={k:raw[k].to_numpy() for k in ['bid_open','ask_open','bid_high','bid_low',
                'ask_high','ask_low','bid_close','ask_close']}
            market['time']=raw.minute_utc.to_numpy(dtype='datetime64[ns]')
            prev=reference_csv(z,f'events_{year}.csv')
            if args.fast:
                events=prev
                if events is None:raise RuntimeError(f'event CSV for {year} not found in ZIP')
                print(f'  {year}: FAST MODE: {len(events)} published events (signals NOT rebuilt)')
                valid_events=None
            else:
                events,dc_count,level_count=regenerated_events(core,raw,year)
                valid_events=compare_event_files(events,prev,year)
            # Replay original exactly: no added slippage, commission .8 pips.
            frozen=trade_events(events,market,year,fee=.8,slip=0.)
            match=compare_original(frozen,year,z)
            validations.append(dict(year=year,event_replication=valid_events,frozen_rule_match=match,
                                    original_trades=len(frozen)))
            label='original_tp0.5'
            m=metrics(frozen,year,label,1440)
            summary.append(m)
            alltrades.append(frozen.assign(policy=label,max_hold_min=1440))
            if year==2026:
                print('  2026: frozen policy only; exit variants intentionally not evaluated on consumed holdout.')
                continue
            # Exploratory comparisons on 2024-25; policy-specific nonoverlap recalculated.
            for minutes in (30,60,120,240,480,1440):
                for tp in (.25,.5,.75,1.0):
                    if minutes==1440 and tp==.5:continue # already done above
                    t=trade_events(events,market,year,horizon=minutes,tp_R=tp,
                        fee=args.fee_pips,slip=args.extra_slippage_pips)
                    policy=f'tp{tp:g}'
                    summary.append(metrics(t,year,policy,minutes))
                    if not t.empty: alltrades.append(t.assign(policy=policy,max_hold_min=minutes))
                # Experimental stop-lock: after +0.35R executable close, stop at +0.15R next minute; TP +1R
                t=trade_events(events,market,year,horizon=minutes,tp_R=1.,fee=args.fee_pips,
                               slip=args.extra_slippage_pips,lock=True)
                policy='runner_lock_0.15_after_0.35_close'
                summary.append(metrics(t,year,policy,minutes))
                if not t.empty: alltrades.append(t.assign(policy=policy,max_hold_min=minutes))
            print(f'  {year}: shorter-exit variants done.',flush=True)
    if not validations:
        print('\nNo input year files found. Put EURUSD_2024.m1.csv and 2025 there or set --data-dir.')
        raise SystemExit(2)
    summary=pd.DataFrame(summary)
    summary.to_csv(args.out/'exit_comparison.csv',index=False)
    if alltrades:
        pd.concat(alltrades,ignore_index=True).to_csv(args.out/'trade_log.csv',index=False)
    (args.out/'verification.json').write_text(json.dumps(dict(
        purpose='reproduction of original engine and exploratory shorter exits, not independent audit',
        original_rule=CANDIDATE, original_commission_pips=.8,
        exploratory_commission_pips=args.fee_pips, extra_slippage_pips=args.extra_slippage_pips,
        fast_mode=args.fast, verified=validations, original_archive=str(zip_path),
        files=['exit_comparison.csv','trade_log.csv']),indent=2),encoding='utf8')
    print('\nORIGINAL FROZEN POLICY (24h maximum, +0.5 gross R):')
    orig=summary[summary.policy=='original_tp0.5']
    print(orig[['year','n','mean_net_R','profit_factor','win_pct','median_hold_min',
                'p90_hold_min','target','stop','time_exit']].round(3).to_string(index=False))
    dev=summary[(summary.year.isin([2024,2025]))&(summary.policy!='original_tp0.5')]
    if len(dev):
        combined=dev.groupby(['policy','max_hold_min'],as_index=False).agg(
            trades=('n','sum'),net_R_sum=('total_net_R','sum'),
            # Weighted by number of ACTUAL accepted trades, not by year.
            target=('target','sum'),stops=('stop','sum'),time_exits=('time_exit','sum'))
        combined['mean_net_R']=combined.net_R_sum / combined.trades.replace(0,np.nan)
        combined.sort_values(['max_hold_min','policy']).to_csv(args.out/'development_combined.csv',index=False)
        print('\nEXPLORATORY SHORTER EXITS (2024+2025 combined, profit based on NON-overlapping trades):')
        print(combined[['policy','max_hold_min','trades','mean_net_R']].round(4).to_string(index=False))
    print(f'\nSaved: {args.out.resolve()}')
    print('NOTE: Reproducing the old code proves repeatability, NOT lack of coding bias or live profitability.')
    print('NOTE: Only original 24h policy is compared against prior published results. Other exits are NEW exploratory tests.')

if __name__=='__main__':
    main()

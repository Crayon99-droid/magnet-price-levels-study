from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.G1A — FRESH GMMA STOP FRONTIER + WINNER MAE AUDIT
# ============================================================
#
# PURPOSE
# -------
# The user asked the key question:
#
#   If the intended target is D away, why must the stop be the far
#   barrier? Can a much closer stop convert first-passage geometry
#   into a profitable trade?
#
# This diagnostic uses the already-frozen Fresh GMMA entries and asks:
#
#   1) How far do eventual target-hitting GMMA trades move adversely
#      BEFORE first hitting the target?
#
#   2) What happens with fixed stop fractions:
#         0.10D, 0.25D, 0.50D, 0.75D, 1.00D
#
#   3) Does the empirical target-first probability exceed the exact
#      zero-drift Brownian break-even probability:
#
#         P_BE = s / (1 + s)
#
#      where target = +1.0D and stop = -sD?
#
#   4) Does the gross edge survive transaction-cost sensitivity?
#
# IMPORTANT
# ---------
# This is a DEVELOPMENT-SAMPLE diagnostic. Fresh GMMA itself was
# discovered on BTC 2021-2024. No stop fraction should be promoted
# to a final trading rule solely because it looks best here.
#
# A separate external/locked validation remains required.
# ============================================================

BASE = Path(__file__).resolve().parent

TRADE_FILE_CANDIDATES = [
    BASE / "H4_6A2_fresh_GMMA_trades.csv",
    BASE / "H4_6A2_results" / "H4_6A2_fresh_GMMA_trades.csv",
]

BTC_FILES = [
    BASE / f"BTCUSDT_1m_{y}.csv.zip"
    for y in (2021, 2022, 2023, 2024)
]

OUT = BASE / "H4_G1A_results"
OUT.mkdir(exist_ok=True)

STOP_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
ROUND_TRIP_COST_BPS = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
HORIZON_HOURS = 24

BOOT_REPS = 10000
BOOT_SEED = 20260908


def save(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    return p


def find_trade_file():
    for p in TRADE_FILE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find H4_6A2_fresh_GMMA_trades.csv.\n"
        "Put it beside this script, or keep it inside H4_6A2_results."
    )


def load_btc_zip(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}")

    print("Loading", path.name)

    with zipfile.ZipFile(path, "r") as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"No CSV inside {path.name}")
        with z.open(csvs[0]) as f:
            raw = pd.read_csv(f)

    req = ["timestamp_utc", "open", "high", "low", "close"]
    miss = [c for c in req if c not in raw.columns]
    if miss:
        raise ValueError(f"{path.name} missing columns {miss}")

    x = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce"),
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "high": pd.to_numeric(raw["high"], errors="coerce"),
        "low": pd.to_numeric(raw["low"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
    }).dropna()

    return x.sort_values("timestamp").drop_duplicates("timestamp")


def ns(ts):
    return int(pd.Timestamp(ts).as_unit("ns").value)


def first_hit(mask):
    h = np.flatnonzero(mask)
    return int(h[0]) if h.size else None


def profit_factor(values):
    v = np.asarray(values, dtype=float)
    gp = v[v > 0].sum()
    gl = -v[v < 0].sum()
    return float(gp / gl) if gl > 0 else np.inf


def month_block_bootstrap_mean(frame, value_col):
    x = frame[["entry_time", value_col]].dropna().copy()
    if x.empty:
        return {
            "observations": 0,
            "month_blocks": 0,
            "mean": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "fraction_le_0": np.nan,
        }

    x["month"] = x["entry_time"].dt.strftime("%Y-%m")
    b = x.groupby("month")[value_col].agg(["sum", "count"])

    sums = b["sum"].to_numpy(float)
    counts = b["count"].to_numpy(float)

    rng = np.random.default_rng(BOOT_SEED)
    vals = np.empty(BOOT_REPS)

    for i in range(BOOT_REPS):
        j = rng.integers(0, len(b), len(b))
        vals[i] = sums[j].sum() / counts[j].sum()

    lo, hi = np.quantile(vals, [0.025, 0.975])

    return {
        "observations": len(x),
        "month_blocks": len(b),
        "mean": float(x[value_col].mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "fraction_le_0": float((vals <= 0).mean()),
    }


# ============================================================
# LOAD TRADES
# ============================================================

trade_file = find_trade_file()
print("Using trade file:", trade_file)

trades = pd.read_csv(trade_file)

required = ["entry_time", "entry_price", "target_price", "direction"]
missing = [c for c in required if c not in trades.columns]
if missing:
    raise ValueError(
        f"Fresh GMMA trade file missing required columns: {missing}\n"
        f"Available: {list(trades.columns)}"
    )

trades["entry_time"] = pd.to_datetime(
    trades["entry_time"], utc=True, errors="coerce"
)

if "decision_time" in trades.columns:
    trades["decision_time"] = pd.to_datetime(
        trades["decision_time"], utc=True, errors="coerce"
    )
else:
    trades["decision_time"] = pd.NaT

trades["entry_price"] = pd.to_numeric(
    trades["entry_price"], errors="coerce"
)
trades["target_price"] = pd.to_numeric(
    trades["target_price"], errors="coerce"
)

trades = trades.dropna(
    subset=["entry_time", "entry_price", "target_price", "direction"]
).reset_index(drop=True)

# ============================================================
# LOAD BTC M1
# ============================================================

parts = [load_btc_zip(p) for p in BTC_FILES]

m1 = (
    pd.concat(parts, ignore_index=True)
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

m1["timestamp"] = m1["timestamp"].dt.as_unit("ns")

tns = m1["timestamp"].astype("int64").to_numpy()
opens = m1["open"].to_numpy(float)
highs_all = m1["high"].to_numpy(float)
lows_all = m1["low"].to_numpy(float)
closes_all = m1["close"].to_numpy(float)

print(f"BTC M1 rows: {len(m1):,}")


# ============================================================
# EVENT-LEVEL PATH AUDIT
# ============================================================

event_rows = []
stop_rows = []

for n, tr in enumerate(trades.itertuples(index=False), 1):

    direction = str(tr.direction).upper()
    side = 1 if direction == "LONG" else -1

    entry_time = pd.Timestamp(tr.entry_time)
    entry = float(tr.entry_price)
    target = float(tr.target_price)

    target_D = side * (target - entry)

    if not np.isfinite(target_D) or target_D <= 0:
        continue

    # Match actual entry minute.
    i0 = int(np.searchsorted(tns, ns(entry_time), side="left"))

    if i0 >= len(tns):
        continue

    # H4.6A2 horizon was 24h from the original H4.2 decision.
    # Fall back to 24h from entry only if decision_time is unavailable.
    if pd.notna(tr.decision_time):
        horizon_time = pd.Timestamp(tr.decision_time) + pd.Timedelta(hours=HORIZON_HOURS)
    else:
        horizon_time = entry_time + pd.Timedelta(hours=HORIZON_HOURS)

    i1 = int(np.searchsorted(tns, ns(horizon_time), side="left"))
    i1 = min(i1, len(tns))

    if i1 <= i0:
        continue

    h = highs_all[i0:i1]
    l = lows_all[i0:i1]
    c = closes_all[i0:i1]

    if side == 1:
        target_hits = np.flatnonzero(h >= target)
    else:
        target_hits = np.flatnonzero(l <= target)

    target_rel = int(target_hits[0]) if target_hits.size else None

    # MAE before first target, independent of any hypothetical stop.
    if target_rel is not None:
        h_to_target = h[:target_rel + 1]
        l_to_target = l[:target_rel + 1]

        if side == 1:
            adverse_price = max(0.0, entry - float(np.min(l_to_target)))
        else:
            adverse_price = max(0.0, float(np.max(h_to_target)) - entry)

        mae_D = adverse_price / target_D
        minutes_to_target = target_rel + 1
    else:
        mae_D = np.nan
        minutes_to_target = np.nan

    base = {
        "event_id": getattr(tr, "event_id", n),
        "decision_time": tr.decision_time,
        "entry_time": entry_time,
        "year": entry_time.year,
        "direction": direction,
        "entry_price": entry,
        "target_price": target,
        "target_distance_price": target_D,
        "target_distance_bps": target_D / entry * 10000.0,
        "eventual_target_hit_within_horizon": target_rel is not None,
        "mae_before_target_D": mae_D,
        "minutes_to_target": minutes_to_target,
    }
    event_rows.append(base)

    for s in STOP_FRACTIONS:
        stop = entry - side * s * target_D

        if side == 1:
            stop_hits = np.flatnonzero(l <= stop)
        else:
            stop_hits = np.flatnonzero(h >= stop)

        stop_rel = int(stop_hits[0]) if stop_hits.size else None

        # Conservative same-M1 ordering: stop first.
        if target_rel is not None and (stop_rel is None or target_rel < stop_rel):
            outcome = "TP"
            exit_rel = target_rel
            exit_price = target
            pnl_D = 1.0

        elif stop_rel is not None and (target_rel is None or stop_rel <= target_rel):
            outcome = "SL"
            exit_rel = stop_rel
            exit_price = stop
            pnl_D = -s

        else:
            outcome = "TIMEOUT"
            exit_rel = len(c) - 1
            exit_price = float(c[-1])

            pnl_price = side * (exit_price - entry)
            pnl_D = pnl_price / target_D

        gross_bps = side * (exit_price / entry - 1.0) * 10000.0
        risk_bps = (s * target_D / entry) * 10000.0

        stop_rows.append({
            **base,
            "stop_fraction_D": s,
            "stop_price": stop,
            "brownian_break_even_tp_rate": s / (1.0 + s),
            "outcome": outcome,
            "exit_minutes": exit_rel + 1,
            "exit_price": exit_price,
            "pnl_target_D_units": pnl_D,
            "pnl_R_units": pnl_D / s,
            "gross_return_bps": gross_bps,
            "risk_bps": risk_bps,
        })

    if n % 100 == 0:
        print(f"Processed {n}/{len(trades)} GMMA trades")


events = pd.DataFrame(event_rows)
frontier = pd.DataFrame(stop_rows)

if events.empty or frontier.empty:
    raise RuntimeError("No G1A results generated.")


# ============================================================
# WINNER MAE DISTRIBUTION
# ============================================================

winners = events[
    events["eventual_target_hit_within_horizon"]
].copy()

q_levels = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

mae_summary = {
    "eventual_target_hits": len(winners),
    "all_gmma_entries": len(events),
    "eventual_target_hit_rate": len(winners) / len(events),
}

for q in q_levels:
    mae_summary[f"mae_D_p{int(q*100):02d}"] = (
        winners["mae_before_target_D"].quantile(q)
    )

for s in STOP_FRACTIONS:
    mae_summary[f"share_eventual_winners_surviving_{s:.2f}D_stop"] = (
        winners["mae_before_target_D"] < s
    ).mean()

mae_summary = pd.DataFrame([mae_summary])


# ============================================================
# STOP FRONTIER SUMMARY
# ============================================================

summary_rows = []

for s in STOP_FRACTIONS:
    g = frontier[frontier["stop_fraction_D"] == s].copy()

    tp_rate = (g["outcome"] == "TP").mean()
    sl_rate = (g["outcome"] == "SL").mean()
    timeout_rate = (g["outcome"] == "TIMEOUT").mean()

    be = s / (1.0 + s)

    boot = month_block_bootstrap_mean(
        g,
        "gross_return_bps",
    )

    summary_rows.append({
        "stop_fraction_D": s,
        "trades": len(g),

        "tp_rate": tp_rate,
        "sl_rate": sl_rate,
        "timeout_rate": timeout_rate,

        "brownian_break_even_tp_rate": be,
        "tp_rate_minus_brownian_break_even_pp": tp_rate - be,

        "mean_pnl_target_D_units": g["pnl_target_D_units"].mean(),
        "mean_pnl_R_units": g["pnl_R_units"].mean(),

        "gross_profit_factor_bps": profit_factor(g["gross_return_bps"]),
        "mean_gross_bps": g["gross_return_bps"].mean(),

        "gross_bps_block_ci95_low": boot["ci95_low"],
        "gross_bps_block_ci95_high": boot["ci95_high"],
        "gross_bps_bootstrap_fraction_le_0": boot["fraction_le_0"],

        "median_risk_bps": g["risk_bps"].median(),
        "median_exit_minutes": g["exit_minutes"].median(),
    })

stop_summary = pd.DataFrame(summary_rows)


# ============================================================
# COST SENSITIVITY
# ============================================================

cost_rows = []

for s in STOP_FRACTIONS:
    g = frontier[frontier["stop_fraction_D"] == s].copy()

    for cost in ROUND_TRIP_COST_BPS:

        net_bps = g["gross_return_bps"] - cost

        # Fixed-risk R representation:
        # notional transaction cost as a multiple of initial stop risk.
        net_R = (
            g["pnl_R_units"]
            -
            cost / g["risk_bps"].replace(0, np.nan)
        )

        cost_rows.append({
            "stop_fraction_D": s,
            "round_trip_cost_bps": cost,
            "trades": len(g),

            "mean_net_bps": net_bps.mean(),
            "net_profit_factor_bps": profit_factor(net_bps),

            "mean_net_R": net_R.mean(),
            "net_profit_factor_R": profit_factor(net_R),
        })

cost_summary = pd.DataFrame(cost_rows)


# ============================================================
# YEAR STABILITY
# ============================================================

year_rows = []

for s in STOP_FRACTIONS:
    sg = frontier[frontier["stop_fraction_D"] == s]

    for year in sorted(sg["year"].unique()):
        g = sg[sg["year"] == year]

        tp_rate = (g["outcome"] == "TP").mean()
        be = s / (1.0 + s)

        year_rows.append({
            "year": int(year),
            "stop_fraction_D": s,
            "trades": len(g),
            "tp_rate": tp_rate,
            "brownian_break_even_tp_rate": be,
            "tp_rate_minus_brownian_break_even_pp": tp_rate - be,
            "mean_gross_bps": g["gross_return_bps"].mean(),
            "gross_profit_factor_bps": profit_factor(g["gross_return_bps"]),
            "mean_pnl_R_units": g["pnl_R_units"].mean(),
        })

year_summary = pd.DataFrame(year_rows)


# ============================================================
# SAVE
# ============================================================

save(events, "H4_G1A_eventual_target_MAE.csv")
save(frontier, "H4_G1A_stop_frontier_event_level.csv")
save(mae_summary, "H4_G1A_winner_MAE_summary.csv")
save(stop_summary, "H4_G1A_stop_frontier_summary.csv")
save(cost_summary, "H4_G1A_cost_sensitivity.csv")
save(year_summary, "H4_G1A_stop_frontier_by_year.csv")

config = f"""H4.G1A — FRESH GMMA STOP FRONTIER + WINNER MAE AUDIT

INPUT:
{trade_file.name}

STOP FRACTIONS OF ACTUAL GMMA ENTRY-TO-TARGET DISTANCE D:
{STOP_FRACTIONS}

TARGET:
Original H4.6A2 target, +1.0D from actual Fresh GMMA entry.

STOP:
-sD adverse from actual Fresh GMMA entry.

ZERO-DRIFT BROWNIAN BREAK-EVEN:
P(TP before SL) = s / (1+s)

HORIZON:
Original H4.2 decision + {HORIZON_HOURS}h where decision_time is available.

SAME-M1 AMBIGUITY:
Stop first.

TIMEOUT:
Marked to final available M1 close at horizon.

COST SENSITIVITY:
{ROUND_TRIP_COST_BPS} round-trip bp.

IMPORTANT:
- Development-sample diagnostic only.
- No stop fraction is to be declared final from BTC 2021-2024.
- Fresh GMMA itself was discovered on this sample.
- External/locked validation remains mandatory.
"""

(OUT / "H4_G1A_config.txt").write_text(config, encoding="utf-8")

print("\nWINNER MAE SUMMARY")
print(mae_summary.to_string(index=False))

print("\nSTOP FRONTIER")
print(stop_summary.to_string(index=False))

print("\nCOST SENSITIVITY")
print(cost_summary.to_string(index=False))

print("\nSaved to:", OUT)
print("Done.")

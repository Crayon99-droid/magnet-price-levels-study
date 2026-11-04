from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.6A2 — BTC MAGNET + FRESH GMMA TRANSITION ONLY
# ============================================================
#
# PURPOSE
# -------
# H4.6A mixed two different situations:
#
#   A) GMMA was already fully aligned in the magnet direction
#      before the H4.2 signal, then later re-crossed.
#
#   B) GMMA was NOT aligned before the H4.2 signal, then a genuinely
#      fresh full-ribbon transition occurred after the signal.
#
# The audit showed that B was the promising population.
#
# H4.6A2 formally defines the GMMA entry strategy as B only.
#
# THIS IS A SEMANTIC CORRECTION, NOT A PARAMETER OPTIMIZATION.
#
# ------------------------------------------------------------
# MAGNET LAYER — FROZEN
# ------------------------------------------------------------
# Uses H4_2_magnet_events.csv unchanged.
#
# Only strong-confidence events:
#
#       nearer_distance / farther_distance <= 0.50
#
# Direction + target:
#   nearer upper magnet -> LONG candidate
#   nearer lower magnet -> SHORT candidate
#
# ------------------------------------------------------------
# GMMA LAYER — FROZEN
# ------------------------------------------------------------
# M1 EMA ribbons:
#
# Fast: 3, 5, 8, 10, 12, 15
# Slow: 30, 35, 40, 45, 50, 60
#
# Existing alignment is assessed on the LAST COMPLETED M1 candle
# immediately before the H4.2 decision.
#
# If already fully aligned in the H4.2 direction:
#       EXCLUDE from H4.6A2.
#
# For the remaining eligible signals:
#
# Bullish fresh transition:
#       min(fast EMAs) > max(slow EMAs)
# on a completed post-signal M1 candle,
# while this condition was false on the previous M1 candle.
#
# Bearish fresh transition:
#       max(fast EMAs) < min(slow EMAs)
# on a completed post-signal M1 candle,
# while this condition was false on the previous M1 candle.
#
# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------
# Enter at the NEXT M1 OPEN after the fresh GMMA confirmation.
#
# GMMA confirmation must occur strictly before the original H4.2
# first-magnet touch. If magnet touch and GMMA confirmation happen
# in the same M1 candle, no trade.
#
# ------------------------------------------------------------
# TARGET / STOP
# ------------------------------------------------------------
# Target = original nearer H4.2 magnet band edge.
#
# Stop = symmetric 1:1 stop around ACTUAL GMMA entry:
#
#       stop distance = entry-to-target distance
#
# This keeps H4.6A2 focused on ENTRY quality rather than stop tuning.
#
# ------------------------------------------------------------
# HORIZON
# ------------------------------------------------------------
# Target / stop / time exit at 24 hours from original H4.2 signal.
#
# ------------------------------------------------------------
# COSTS
# ------------------------------------------------------------
# Gross + round-trip cost sensitivity at:
# 1, 2, 3, 4, 5, 7.5, 10 bp
#
# NO PARAMETER OPTIMIZATION.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

EVENT_FILE = SCRIPT_DIR / "H4_2_magnet_events.csv"

FILES = [
    "BTCUSDT_1m_2021.csv.zip",
    "BTCUSDT_1m_2022.csv.zip",
    "BTCUSDT_1m_2023.csv.zip",
    "BTCUSDT_1m_2024.csv.zip",
]

OUTPUT_FOLDER = SCRIPT_DIR / "H4_6A2_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

STRONG_CONFIDENCE_MAX_RATIO = 0.50

FAST_EMAS = [3, 5, 8, 10, 12, 15]
SLOW_EMAS = [30, 35, 40, 45, 50, 60]

TRADE_HORIZON_HOURS = 24

ROUND_TRIP_COST_BPS = [
    1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0
]

CHECKPOINT_EVERY_SIGNALS = 500


# ============================================================
# HELPERS
# ============================================================

def safe_to_csv(frame, filename):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_FOLDER / filename
    frame.to_csv(path, index=False)
    return path


def read_zip_csv(zip_path):
    print(f"\nLoading: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise ValueError(
                f"No CSV found inside {zip_path}"
            )

        csv_name = csv_files[0]
        print(f"  CSV inside ZIP: {csv_name}")

        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    print(f"  Rows loaded: {len(df):,}")
    return df


def clean_dataframe(df):
    required = [
        "timestamp_utc",
        "open", "high", "low", "close"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing expected BTC columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()

    out["timestamp"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    return (
        out.dropna()
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def timestamp_ns(value):
    if pd.isna(value):
        return None

    return int(
        pd.Timestamp(value)
        .as_unit("ns")
        .value
    )


def first_true_index(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def target_correct(direction, first_touch):
    if first_touch not in ["UPPER", "LOWER"]:
        return np.nan

    if direction == "LONG":
        return first_touch == "UPPER"

    if direction == "SHORT":
        return first_touch == "LOWER"

    return np.nan


def summarize_trades(g):
    if g.empty:
        return None

    pnl_r = g["pnl_R_conservative"].astype(float)
    pnl_bps = g[
        "gross_return_bps_conservative"
    ].astype(float)

    gross_profit_r = pnl_r[pnl_r > 0].sum()
    gross_loss_r = abs(
        pnl_r[pnl_r < 0].sum()
    )

    return {
        "trades": len(g),

        "wins": int(
            (g["outcome"] == "WIN").sum()
        ),

        "losses": int(
            (g["outcome"] == "LOSS").sum()
        ),

        "time_exits": int(
            (g["outcome"] == "TIME_EXIT").sum()
        ),

        "ambiguous_same_bar": int(
            (
                g["outcome"]
                == "AMBIGUOUS_SAME_BAR"
            ).sum()
        ),

        "win_rate_raw": float(
            (g["outcome"] == "WIN").mean()
        ),

        "mean_R_conservative": float(
            pnl_r.mean()
        ),

        "profit_factor_R_conservative": (
            float(
                gross_profit_r /
                gross_loss_r
            )
            if gross_loss_r > 0
            else np.inf
        ),

        "mean_gross_bps": float(
            pnl_bps.mean()
        ),

        "median_gross_bps": float(
            pnl_bps.median()
        ),

        "mean_minutes_signal_to_entry": float(
            g["minutes_signal_to_entry"].mean()
        ),

        "median_minutes_signal_to_entry": float(
            g["minutes_signal_to_entry"].median()
        ),
    }


# ============================================================
# LOAD H4.2 EVENTS
# ============================================================

if not EVENT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{EVENT_FILE}\n\n"
        "Put H4_2_magnet_events.csv beside this script."
    )

events = pd.read_csv(EVENT_FILE)

required_event_columns = [
    "event_id",
    "decision_time",
    "entry_price",
    "atr15",

    "upper_band_low",
    "upper_band_high",
    "upper_distance",

    "lower_band_low",
    "lower_band_high",
    "lower_distance",

    "nearer_side",
    "first_touch",
    "touch_time",
]

missing = [
    c for c in required_event_columns
    if c not in events.columns
]

if missing:
    raise ValueError(
        f"H4.2 file missing columns: {missing}"
    )

events["decision_time"] = pd.to_datetime(
    events["decision_time"],
    utc=True,
    errors="coerce"
).dt.as_unit("ns")

events["touch_time"] = pd.to_datetime(
    events["touch_time"],
    utc=True,
    errors="coerce"
).dt.as_unit("ns")

events = (
    events
    .dropna(
        subset=[
            "decision_time",
            "entry_price",
            "upper_distance",
            "lower_distance",
        ]
    )
    .sort_values("decision_time")
    .reset_index(drop=True)
)

events["near_far_ratio"] = (
    np.minimum(
        events["upper_distance"],
        events["lower_distance"],
    )
    /
    np.maximum(
        events["upper_distance"],
        events["lower_distance"],
    )
)

strong_events = events[
    (
        events["nearer_side"]
        .isin(["UPPER", "LOWER"])
    )
    &
    (
        events["near_far_ratio"]
        <= STRONG_CONFIDENCE_MAX_RATIO
    )
].copy()

print("\n============================================")
print("H4.2 MAGNET SIGNALS")
print("============================================")
print(f"All H4.2 events:      {len(events):,}")
print(
    f"Strong-confidence:   {len(strong_events):,} "
    f"(near/far <= {STRONG_CONFIDENCE_MAX_RATIO})"
)

if strong_events.empty:
    raise RuntimeError(
        "No strong-confidence H4.2 events found."
    )


# ============================================================
# LOAD BTC M1
# ============================================================

all_data = []

for filename in FILES:
    path = SCRIPT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing BTC file:\n{path}\n"
            "Put the four BTC ZIP files beside this script."
        )

    raw = read_zip_csv(path)
    all_data.append(
        clean_dataframe(raw)
    )

m1 = pd.concat(
    all_data,
    ignore_index=True
)

m1 = (
    m1
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

m1["timestamp"] = (
    m1["timestamp"]
    .dt.as_unit("ns")
)

ts = (
    m1["timestamp"]
    .astype("int64")
    .to_numpy()
)

open_arr = (
    m1["open"]
    .to_numpy(dtype=float)
)

high_arr = (
    m1["high"]
    .to_numpy(dtype=float)
)

low_arr = (
    m1["low"]
    .to_numpy(dtype=float)
)

close_arr = (
    m1["close"]
    .to_numpy(dtype=float)
)

print("\n============================================")
print("BTC M1 DATA")
print("============================================")
print(f"Rows:  {len(m1):,}")
print(f"Start: {m1['timestamp'].min()}")
print(f"End:   {m1['timestamp'].max()}")


# ============================================================
# BUILD GMMA AGGREGATES
# ============================================================

print("\n============================================")
print("BUILDING GMMA")
print("============================================")

n_rows = len(m1)

close_series = pd.Series(
    close_arr,
    copy=False
)

fast_min = np.full(
    n_rows,
    np.inf,
    dtype=float
)

fast_max = np.full(
    n_rows,
    -np.inf,
    dtype=float
)

fast_sum = np.zeros(
    n_rows,
    dtype=float
)

slow_min = np.full(
    n_rows,
    np.inf,
    dtype=float
)

slow_max = np.full(
    n_rows,
    -np.inf,
    dtype=float
)

slow_sum = np.zeros(
    n_rows,
    dtype=float
)

for span in FAST_EMAS:
    print(f"  Fast EMA {span}")

    ema = (
        close_series
        .ewm(
            span=span,
            adjust=False,
            min_periods=span,
        )
        .mean()
        .to_numpy()
    )

    valid = ~np.isnan(ema)

    fast_min[valid] = np.minimum(
        fast_min[valid],
        ema[valid],
    )

    fast_max[valid] = np.maximum(
        fast_max[valid],
        ema[valid],
    )

    fast_sum[valid] += ema[valid]

    del ema


for span in SLOW_EMAS:
    print(f"  Slow EMA {span}")

    ema = (
        close_series
        .ewm(
            span=span,
            adjust=False,
            min_periods=span,
        )
        .mean()
        .to_numpy()
    )

    valid = ~np.isnan(ema)

    slow_min[valid] = np.minimum(
        slow_min[valid],
        ema[valid],
    )

    slow_max[valid] = np.maximum(
        slow_max[valid],
        ema[valid],
    )

    slow_sum[valid] += ema[valid]

    del ema


fast_mean = (
    fast_sum /
    len(FAST_EMAS)
)

slow_mean = (
    slow_sum /
    len(SLOW_EMAS)
)

gmma_valid = (
    np.arange(n_rows)
    >= (max(SLOW_EMAS) - 1)
)

bull_aligned = (
    gmma_valid
    &
    (fast_min > slow_max)
)

bear_aligned = (
    gmma_valid
    &
    (fast_max < slow_min)
)

bull_cross = np.zeros(
    n_rows,
    dtype=bool
)

bear_cross = np.zeros(
    n_rows,
    dtype=bool
)

bull_cross[1:] = (
    bull_aligned[1:]
    &
    ~bull_aligned[:-1]
)

bear_cross[1:] = (
    bear_aligned[1:]
    &
    ~bear_aligned[:-1]
)

print(
    f"Bullish full-ribbon transitions: "
    f"{bull_cross.sum():,}"
)

print(
    f"Bearish full-ribbon transitions: "
    f"{bear_cross.sum():,}"
)


# ============================================================
# PREFLIGHT
# ============================================================

first_signal_time = (
    strong_events[
        "decision_time"
    ].iloc[0]
)

first_signal_ns = timestamp_ns(
    first_signal_time
)

first_idx = int(
    np.searchsorted(
        ts,
        first_signal_ns,
        side="left"
    )
)

print("\n============================================")
print("TIMESTAMP SANITY CHECK")
print("============================================")
print(f"First signal:   {first_signal_time}")
print(f"Search index:   {first_idx}")

if first_idx >= len(ts):
    raise RuntimeError(
        "Timestamp alignment failed."
    )

print(
    f"Matched minute: "
    f"{m1['timestamp'].iloc[first_idx]}"
)

print("Timestamp alignment: OK")


# ============================================================
# FRESH-GMMA FILTER + TRADE SIMULATION
# ============================================================

signal_rows = []
trade_rows = []

print("\n============================================")
print("RUNNING H4.6A2")
print("============================================")

for counter, ev in enumerate(
    strong_events.itertuples(index=False),
    start=1
):

    decision_time = ev.decision_time

    decision_ns = timestamp_ns(
        decision_time
    )

    signal_idx = int(
        np.searchsorted(
            ts,
            decision_ns,
            side="left"
        )
    )

    if signal_idx >= len(ts):
        continue

    if signal_idx == 0:
        continue

    prev_idx = signal_idx - 1

    if ev.nearer_side == "UPPER":
        direction = "LONG"

        target_price = float(
            ev.upper_band_low
        )

        already_aligned = bool(
            bull_aligned[prev_idx]
        )

        desired_cross = bull_cross

    elif ev.nearer_side == "LOWER":
        direction = "SHORT"

        target_price = float(
            ev.lower_band_high
        )

        already_aligned = bool(
            bear_aligned[prev_idx]
        )

        desired_cross = bear_cross

    else:
        continue

    original_correct = target_correct(
        direction,
        ev.first_touch
    )

    base_record = {
        "event_id": ev.event_id,
        "decision_time": decision_time,
        "year": decision_time.year,
        "direction": direction,
        "target_price": target_price,
        "near_far_ratio": ev.near_far_ratio,
        "h4_2_first_touch": ev.first_touch,
        "original_target_correct": original_correct,
        "already_aligned_before_signal": already_aligned,
    }

    # --------------------------------------------------------
    # Formal H4.6A2 exclusion:
    # if GMMA was already fully aligned in magnet direction on
    # the last completed M1 candle BEFORE H4.2 signal -> no trade.
    # --------------------------------------------------------

    if already_aligned:
        signal_rows.append({
            **base_record,
            "eligibility_status": "EXCLUDED_ALREADY_ALIGNED",
            "gmma_confirmation_time": pd.NaT,
            "minutes_to_confirmation": np.nan,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
        })
        continue

    # --------------------------------------------------------
    # Fresh transition must happen before original H4.2
    # first-magnet resolution.
    #
    # Exclude touch minute itself using side='left'.
    # --------------------------------------------------------

    if pd.notna(ev.touch_time):
        search_end_time = ev.touch_time
    else:
        search_end_time = (
            decision_time
            + pd.Timedelta(
                hours=TRADE_HORIZON_HOURS
            )
        )

    search_end_ns = timestamp_ns(
        search_end_time
    )

    search_end_idx = int(
        np.searchsorted(
            ts,
            search_end_ns,
            side="left"
        )
    )

    search_end_idx = min(
        search_end_idx,
        len(ts)
    )

    if signal_idx >= search_end_idx:
        signal_rows.append({
            **base_record,
            "eligibility_status": "ELIGIBLE_NO_SEARCH_WINDOW",
            "gmma_confirmation_time": pd.NaT,
            "minutes_to_confirmation": np.nan,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
        })
        continue

    local_crosses = desired_cross[
        signal_idx:search_end_idx
    ]

    cross_rel = first_true_index(
        local_crosses
    )

    if cross_rel is None:
        signal_rows.append({
            **base_record,
            "eligibility_status": "ELIGIBLE_NO_FRESH_CROSS",
            "gmma_confirmation_time": pd.NaT,
            "minutes_to_confirmation": np.nan,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
        })
        continue

    cross_idx = (
        signal_idx + cross_rel
    )

    cross_time = pd.Timestamp(
        int(ts[cross_idx]),
        tz="UTC"
    )

    next_idx = cross_idx + 1

    minutes_to_confirmation = float(
        (
            cross_time - decision_time
        )
        / pd.Timedelta(minutes=1)
    )

    if next_idx >= len(ts):
        signal_rows.append({
            **base_record,
            "eligibility_status": "FRESH_CROSS_NO_NEXT_MINUTE",
            "gmma_confirmation_time": cross_time,
            "minutes_to_confirmation": minutes_to_confirmation,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
        })
        continue

    entry_time = pd.Timestamp(
        int(ts[next_idx]),
        tz="UTC"
    )

    # Require exact next minute.
    if (
        entry_time
        != cross_time
        + pd.Timedelta(minutes=1)
    ):
        signal_rows.append({
            **base_record,
            "eligibility_status": "FRESH_CROSS_MISSING_NEXT_MINUTE",
            "gmma_confirmation_time": cross_time,
            "minutes_to_confirmation": minutes_to_confirmation,
            "entry_time": entry_time,
            "entry_price": np.nan,
        })
        continue

    entry_price = float(
        open_arr[next_idx]
    )

    # Target must still be ahead at actual executable entry.
    if (
        direction == "LONG"
        and entry_price >= target_price
    ):
        signal_rows.append({
            **base_record,
            "eligibility_status": "FRESH_CROSS_TARGET_PASSED_BEFORE_ENTRY",
            "gmma_confirmation_time": cross_time,
            "minutes_to_confirmation": minutes_to_confirmation,
            "entry_time": entry_time,
            "entry_price": entry_price,
        })
        continue

    if (
        direction == "SHORT"
        and entry_price <= target_price
    ):
        signal_rows.append({
            **base_record,
            "eligibility_status": "FRESH_CROSS_TARGET_PASSED_BEFORE_ENTRY",
            "gmma_confirmation_time": cross_time,
            "minutes_to_confirmation": minutes_to_confirmation,
            "entry_time": entry_time,
            "entry_price": entry_price,
        })
        continue

    reward_distance = abs(
        target_price - entry_price
    )

    if reward_distance <= 0:
        continue

    # 1:1 symmetric stop around actual GMMA entry.
    if direction == "LONG":
        stop_price = (
            entry_price
            - reward_distance
        )
    else:
        stop_price = (
            entry_price
            + reward_distance
        )

    ribbon_gap_bps = (
        (
            fast_mean[cross_idx]
            - slow_mean[cross_idx]
        )
        /
        close_arr[cross_idx]
        * 10000.0
    )

    signal_rows.append({
        **base_record,
        "eligibility_status": "EXECUTED_FRESH_GMMA",
        "gmma_confirmation_time": cross_time,
        "minutes_to_confirmation": minutes_to_confirmation,
        "entry_time": entry_time,
        "entry_price": entry_price,

        "fast_ribbon_min": fast_min[cross_idx],
        "fast_ribbon_max": fast_max[cross_idx],
        "fast_ribbon_mean": fast_mean[cross_idx],

        "slow_ribbon_min": slow_min[cross_idx],
        "slow_ribbon_max": slow_max[cross_idx],
        "slow_ribbon_mean": slow_mean[cross_idx],

        "ribbon_mean_gap_bps_vs_close": ribbon_gap_bps,
    })

    # --------------------------------------------------------
    # TRADE SIMULATION
    # --------------------------------------------------------

    horizon_time = (
        decision_time
        + pd.Timedelta(
            hours=TRADE_HORIZON_HOURS
        )
    )

    horizon_ns = timestamp_ns(
        horizon_time
    )

    end_idx = int(
        np.searchsorted(
            ts,
            horizon_ns,
            side="left"
        )
    )

    end_idx = min(
        end_idx,
        len(ts)
    )

    if next_idx >= end_idx:
        continue

    highs = high_arr[
        next_idx:end_idx
    ]

    lows = low_arr[
        next_idx:end_idx
    ]

    closes = close_arr[
        next_idx:end_idx
    ]

    trade_ts = ts[
        next_idx:end_idx
    ]

    if highs.size == 0:
        continue

    if direction == "LONG":
        target_mask = (
            highs >= target_price
        )

        stop_mask = (
            lows <= stop_price
        )
    else:
        target_mask = (
            lows <= target_price
        )

        stop_mask = (
            highs >= stop_price
        )

    target_rel = first_true_index(
        target_mask
    )

    stop_rel = first_true_index(
        stop_mask
    )

    if (
        target_rel is None
        and stop_rel is None
    ):
        outcome = "TIME_EXIT"

        exit_rel = (
            len(closes) - 1
        )

        exit_price = float(
            closes[exit_rel]
        )

        exit_time = pd.Timestamp(
            int(trade_ts[exit_rel]),
            tz="UTC"
        )

        if direction == "LONG":
            pnl_price = (
                exit_price
                - entry_price
            )
        else:
            pnl_price = (
                entry_price
                - exit_price
            )

        pnl_R = (
            pnl_price /
            reward_distance
        )

    elif (
        target_rel is not None
        and stop_rel is None
    ):
        outcome = "WIN"
        exit_price = target_price

        exit_time = pd.Timestamp(
            int(trade_ts[target_rel]),
            tz="UTC"
        )

        pnl_R = 1.0

    elif (
        target_rel is None
        and stop_rel is not None
    ):
        outcome = "LOSS"
        exit_price = stop_price

        exit_time = pd.Timestamp(
            int(trade_ts[stop_rel]),
            tz="UTC"
        )

        pnl_R = -1.0

    else:
        if target_rel < stop_rel:
            outcome = "WIN"
            exit_price = target_price

            exit_time = pd.Timestamp(
                int(trade_ts[target_rel]),
                tz="UTC"
            )

            pnl_R = 1.0

        elif stop_rel < target_rel:
            outcome = "LOSS"
            exit_price = stop_price

            exit_time = pd.Timestamp(
                int(trade_ts[stop_rel]),
                tz="UTC"
            )

            pnl_R = -1.0

        else:
            # Same post-entry M1 candle.
            # Conservative primary treatment = stop loss.
            outcome = "AMBIGUOUS_SAME_BAR"
            exit_price = stop_price

            exit_time = pd.Timestamp(
                int(trade_ts[target_rel]),
                tz="UTC"
            )

            pnl_R = -1.0

    if direction == "LONG":
        gross_return = (
            exit_price - entry_price
        ) / entry_price
    else:
        gross_return = (
            entry_price - exit_price
        ) / entry_price

    gross_bps = (
        gross_return *
        10000.0
    )

    risk_bps = (
        reward_distance /
        entry_price *
        10000.0
    )

    minutes_signal_to_entry = float(
        (
            entry_time - decision_time
        )
        / pd.Timedelta(minutes=1)
    )

    trade_rows.append({
        "event_id": ev.event_id,
        "decision_time": decision_time,
        "year": decision_time.year,
        "direction": direction,

        "near_far_ratio": ev.near_far_ratio,
        "h4_2_first_touch": ev.first_touch,
        "original_target_correct": original_correct,

        "gmma_confirmation_time": cross_time,
        "entry_time": entry_time,
        "minutes_signal_to_entry": minutes_signal_to_entry,

        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,

        "risk_price_distance": reward_distance,
        "risk_bps": risk_bps,

        "fast_ribbon_min": fast_min[cross_idx],
        "fast_ribbon_max": fast_max[cross_idx],
        "fast_ribbon_mean": fast_mean[cross_idx],

        "slow_ribbon_min": slow_min[cross_idx],
        "slow_ribbon_max": slow_max[cross_idx],
        "slow_ribbon_mean": slow_mean[cross_idx],

        "ribbon_mean_gap_bps_vs_close": ribbon_gap_bps,

        "outcome": outcome,
        "exit_time": exit_time,
        "exit_price_conservative": exit_price,

        "pnl_R_conservative": pnl_R,
        "gross_return_bps_conservative": gross_bps,
    })

    if (
        counter
        % CHECKPOINT_EVERY_SIGNALS
        == 0
    ):
        safe_to_csv(
            pd.DataFrame(signal_rows),
            "H4_6A2_CHECKPOINT_signal_diagnostics.csv"
        )

        safe_to_csv(
            pd.DataFrame(trade_rows),
            "H4_6A2_CHECKPOINT_trades.csv"
        )

        print(
            f"  Processed "
            f"{counter:,}/{len(strong_events):,} "
            f"signals — checkpoint saved."
        )


signals = pd.DataFrame(
    signal_rows
)

trades = pd.DataFrame(
    trade_rows
)

if signals.empty:
    raise RuntimeError(
        "No H4.6A2 signal diagnostics generated."
    )


# ============================================================
# ELIGIBILITY / SELECTIVITY
# ============================================================

status_summary = (
    signals[
        "eligibility_status"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "eligibility_status"
    )
    .reset_index(
        name="count"
    )
)

status_summary[
    "fraction_of_strong_signals"
] = (
    status_summary["count"]
    /
    len(signals)
)

eligible = signals[
    signals[
        "already_aligned_before_signal"
    ] == False
].copy()

executed_signals = signals[
    signals[
        "eligibility_status"
    ] == "EXECUTED_FRESH_GMMA"
].copy()

resolved_strong = signals[
    signals[
        "original_target_correct"
    ].notna()
].copy()

resolved_eligible = eligible[
    eligible[
        "original_target_correct"
    ].notna()
].copy()

resolved_executed = executed_signals[
    executed_signals[
        "original_target_correct"
    ].notna()
].copy()

selectivity_summary = pd.DataFrame([
    {
        "all_strong_signals": len(signals),

        "resolved_all_strong": len(
            resolved_strong
        ),

        "baseline_target_correct_rate_all_strong": (
            resolved_strong[
                "original_target_correct"
            ].astype(bool).mean()
            if len(resolved_strong)
            else np.nan
        ),

        "eligible_not_already_aligned": len(
            eligible
        ),

        "eligible_fraction_of_strong": (
            len(eligible) /
            len(signals)
            if len(signals)
            else np.nan
        ),

        "resolved_eligible": len(
            resolved_eligible
        ),

        "baseline_target_correct_rate_eligible_before_gmma": (
            resolved_eligible[
                "original_target_correct"
            ].astype(bool).mean()
            if len(resolved_eligible)
            else np.nan
        ),

        "executed_fresh_gmma": len(
            executed_signals
        ),

        "execution_rate_within_eligible": (
            len(executed_signals) /
            len(eligible)
            if len(eligible)
            else np.nan
        ),

        "execution_rate_of_all_strong": (
            len(executed_signals) /
            len(signals)
            if len(signals)
            else np.nan
        ),

        "resolved_executed": len(
            resolved_executed
        ),

        "target_correct_rate_after_fresh_gmma": (
            resolved_executed[
                "original_target_correct"
            ].astype(bool).mean()
            if len(resolved_executed)
            else np.nan
        ),

        "median_minutes_to_fresh_gmma_confirmation": (
            executed_signals[
                "minutes_to_confirmation"
            ].median()
            if len(executed_signals)
            else np.nan
        ),
    }
])


print("\n============================================")
print("H4.6A2 SELECTIVITY")
print("============================================")
print(
    selectivity_summary.to_string(
        index=False
    )
)

print("\nSTATUS COUNTS")
print(
    status_summary.to_string(
        index=False
    )
)


# ============================================================
# OVERALL TRADE SUMMARY
# ============================================================

if not trades.empty:
    overall_summary = pd.DataFrame([
        summarize_trades(trades)
    ])
else:
    overall_summary = pd.DataFrame()

print("\n============================================")
print("H4.6A2 TRADE RESULTS")
print("============================================")

if not overall_summary.empty:
    print(
        overall_summary.to_string(
            index=False
        )
    )
else:
    print("No executed trades.")


# ============================================================
# YEAR-BY-YEAR
# ============================================================

year_rows = []

if not trades.empty:
    for year in sorted(
        trades["year"].unique()
    ):
        g = trades[
            trades["year"] == year
        ]

        if g.empty:
            continue

        s = summarize_trades(g)
        s["year"] = int(year)
        year_rows.append(s)

year_summary = pd.DataFrame(
    year_rows
)

print("\n============================================")
print("H4.6A2 YEAR-BY-YEAR")
print("============================================")

if not year_summary.empty:
    print(
        year_summary.to_string(
            index=False
        )
    )
else:
    print("No executed trades.")


# ============================================================
# LONG / SHORT
# ============================================================

direction_rows = []

if not trades.empty:
    for direction in [
        "LONG",
        "SHORT",
    ]:
        g = trades[
            trades["direction"] == direction
        ]

        if g.empty:
            continue

        s = summarize_trades(g)
        s["direction"] = direction
        direction_rows.append(s)

direction_summary = pd.DataFrame(
    direction_rows
)


# ============================================================
# ENTRY-WAIT BUCKETS
# ============================================================

wait_rows = []

if not trades.empty:
    wait_bins = [
        -np.inf,
        5,
        15,
        30,
        60,
        180,
        np.inf,
    ]

    wait_labels = [
        "<=5m",
        "5-15m",
        "15-30m",
        "30-60m",
        "1-3h",
        ">3h",
    ]

    trades[
        "entry_wait_bucket"
    ] = pd.cut(
        trades[
            "minutes_signal_to_entry"
        ],
        bins=wait_bins,
        labels=wait_labels,
        right=True,
    )

    for bucket in wait_labels:
        g = trades[
            trades[
                "entry_wait_bucket"
            ] == bucket
        ]

        if g.empty:
            continue

        s = summarize_trades(g)
        s["entry_wait_bucket"] = bucket
        wait_rows.append(s)

wait_summary = pd.DataFrame(
    wait_rows
)


# ============================================================
# COST SENSITIVITY
# ============================================================

cost_rows = []

if not trades.empty:
    gross_bps = trades[
        "gross_return_bps_conservative"
    ].astype(float)

    risk_bps = trades[
        "risk_bps"
    ].astype(float)

    gross_R = trades[
        "pnl_R_conservative"
    ].astype(float)

    for cost in ROUND_TRIP_COST_BPS:
        net_bps = (
            gross_bps - cost
        )

        cost_R = (
            cost /
            risk_bps.replace(
                0,
                np.nan
            )
        )

        net_R = (
            gross_R - cost_R
        )

        gp = net_R[
            net_R > 0
        ].sum()

        gl = abs(
            net_R[
                net_R < 0
            ].sum()
        )

        cost_rows.append({
            "round_trip_cost_bps": cost,
            "trades": len(trades),

            "mean_gross_bps": gross_bps.mean(),
            "mean_net_bps": net_bps.mean(),

            "mean_gross_R": gross_R.mean(),
            "mean_net_R": net_R.mean(),

            "net_profit_factor_R": (
                gp / gl
                if gl > 0
                else np.inf
            ),

            "positive_net_trade_rate": (
                net_bps > 0
            ).mean(),
        })

cost_summary = pd.DataFrame(
    cost_rows
)

print("\n============================================")
print("H4.6A2 COST SENSITIVITY")
print("============================================")

if not cost_summary.empty:
    print(
        cost_summary.to_string(
            index=False
        )
    )


# ============================================================
# COST BY YEAR
# ============================================================

cost_year_rows = []

if not trades.empty:
    for year in sorted(
        trades["year"].unique()
    ):
        yg = trades[
            trades["year"] == year
        ].copy()

        gross_bps = yg[
            "gross_return_bps_conservative"
        ].astype(float)

        risk_bps = yg[
            "risk_bps"
        ].astype(float)

        gross_R = yg[
            "pnl_R_conservative"
        ].astype(float)

        for cost in ROUND_TRIP_COST_BPS:
            net_bps = (
                gross_bps - cost
            )

            cost_R = (
                cost /
                risk_bps.replace(
                    0,
                    np.nan
                )
            )

            net_R = (
                gross_R - cost_R
            )

            gp = net_R[
                net_R > 0
            ].sum()

            gl = abs(
                net_R[
                    net_R < 0
                ].sum()
            )

            cost_year_rows.append({
                "year": int(year),
                "round_trip_cost_bps": cost,
                "trades": len(yg),

                "mean_gross_bps": gross_bps.mean(),
                "mean_net_bps": net_bps.mean(),

                "mean_gross_R": gross_R.mean(),
                "mean_net_R": net_R.mean(),

                "net_profit_factor_R": (
                    gp / gl
                    if gl > 0
                    else np.inf
                ),
            })

cost_year_summary = pd.DataFrame(
    cost_year_rows
)


# ============================================================
# SAVE
# ============================================================

safe_to_csv(
    signals,
    "H4_6A2_signal_diagnostics.csv"
)

safe_to_csv(
    status_summary,
    "H4_6A2_status_summary.csv"
)

safe_to_csv(
    selectivity_summary,
    "H4_6A2_selectivity_summary.csv"
)

safe_to_csv(
    trades,
    "H4_6A2_fresh_GMMA_trades.csv"
)

safe_to_csv(
    overall_summary,
    "H4_6A2_overall_summary.csv"
)

safe_to_csv(
    year_summary,
    "H4_6A2_year_summary.csv"
)

safe_to_csv(
    direction_summary,
    "H4_6A2_direction_summary.csv"
)

safe_to_csv(
    wait_summary,
    "H4_6A2_entry_wait_summary.csv"
)

safe_to_csv(
    cost_summary,
    "H4_6A2_cost_sensitivity.csv"
)

safe_to_csv(
    cost_year_summary,
    "H4_6A2_cost_sensitivity_by_year.csv"
)


config_text = f"""H4.6A2 — BTC MAGNET + FRESH GMMA TRANSITION ONLY

MAGNET INPUT:
{EVENT_FILE.name}

PRIMARY MAGNET SAMPLE:
nearer_distance / farther_distance <= {STRONG_CONFIDENCE_MAX_RATIO}

GMMA TIMEFRAME:
1 minute

FAST EMA RIBBON:
{FAST_EMAS}

SLOW EMA RIBBON:
{SLOW_EMAS}

PRE-SIGNAL ALIGNMENT RULE:
- use last completed M1 candle immediately before H4.2 decision
- if already fully aligned in magnet direction, EXCLUDE signal

FRESH BULLISH TRANSITION:
min(fast EMAs) > max(slow EMAs)
AND previous M1 candle was not fully bullish-aligned

FRESH BEARISH TRANSITION:
max(fast EMAs) < min(slow EMAs)
AND previous M1 candle was not fully bearish-aligned

ENTRY:
next M1 open after fresh transition

GMMA MUST OCCUR:
after H4.2 decision and strictly before H4.2 first-magnet touch

TARGET:
original nearer H4.2 magnet band edge

STOP:
symmetric 1:1 stop using actual GMMA entry-to-target distance

HORIZON:
{TRADE_HORIZON_HOURS} hours from original H4.2 signal

ROUND-TRIP COST SENSITIVITY:
{ROUND_TRIP_COST_BPS}

IMPORTANT:
- H4.2 magnets are not recalculated
- re-cross cases after pre-existing alignment are excluded
- same-M1 target+stop ambiguity after entry is conservatively counted as loss
- no parameter optimization is performed
"""

(
    OUTPUT_FOLDER /
    "H4_6A2_config.txt"
).write_text(
    config_text,
    encoding="utf-8"
)

print("\n============================================")
print("FILES SAVED")
print("============================================")

for file in sorted(
    OUTPUT_FOLDER.iterdir()
):
    print(file)

print("\nDone.")

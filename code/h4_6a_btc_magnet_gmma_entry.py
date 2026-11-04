from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.6A — BTC MAGNET-GUIDED GMMA ENTRY
# ============================================================
#
# RESEARCH QUESTION
# -----------------
# We already have evidence that H4.2 magnet geometry contains
# destination information. H4.6A tests whether a proper GMMA
# trend-transition signal can provide the ENTRY timing.
#
# MAGNET LAYER — FROZEN
# ---------------------
# Uses the existing H4_2_magnet_events.csv.
# Only strong-confidence signals are used:
#
#       nearer_distance / farther_distance <= 0.50
#
# The nearer magnet supplies direction + target:
#   nearer upper magnet -> bullish / LONG candidate
#   nearer lower magnet -> bearish / SHORT candidate
#
# GMMA ENTRY LAYER
# ----------------
# Native timeframe: 1 minute.
#
# Fast EMA ribbon:
#   3, 5, 8, 10, 12, 15
#
# Slow EMA ribbon:
#   30, 35, 40, 45, 50, 60
#
# A NEW bullish GMMA confirmation occurs when:
#
#   min(all fast EMAs) > max(all slow EMAs)
#
# on the current COMPLETED M1 candle,
# while this was NOT true on the previous M1 candle.
#
# A NEW bearish confirmation is symmetric:
#
#   max(all fast EMAs) < min(all slow EMAs)
#
# on the current completed candle,
# while this was NOT true on the previous candle.
#
# This deliberately requires complete ribbon separation rather than
# an arbitrary single EMA pair crossover.
#
# ENTRY
# -----
# Enter at the NEXT 1-minute OPEN after the GMMA confirmation candle.
#
# A GMMA confirmation must occur AFTER the H4.2 decision and BEFORE
# either H4.2 magnet resolves.
#
# If a magnet is touched in the same minute as the GMMA confirmation,
# no trade is taken because intraminute ordering is unknowable.
#
# TARGET / STOP
# -------------
# Target = original nearer H4.2 magnet band edge.
#
# Stop = symmetric 1:1 stop around the actual GMMA entry:
#
#       risk distance = entry-to-target distance
#
# This keeps H4.6A focused on ENTRY quality instead of introducing
# another stop-optimization problem.
#
# HORIZON
# -------
# Target/stop or mark-to-market at 24 hours after the ORIGINAL
# H4.2 signal.
#
# COSTS
# -----
# Gross results plus sensitivity to:
#       1, 2, 3, 4, 5, 7.5, 10 bp round trip
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

OUTPUT_FOLDER = SCRIPT_DIR / "H4_6A_results"
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


def original_target_correct(direction, first_touch):
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
            g[
                "minutes_signal_to_entry"
            ].mean()
        ),
        "median_minutes_signal_to_entry": float(
            g[
                "minutes_signal_to_entry"
            ].median()
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
    (events["nearer_side"].isin(
        ["UPPER", "LOWER"]
    ))
    &
    (
        events["near_far_ratio"]
        <= STRONG_CONFIDENCE_MAX_RATIO
    )
].copy()

print("\n============================================")
print("H4.2 MAGNET SIGNALS")
print("============================================")
print(f"All events:          {len(events):,}")
print(
    f"Strong-confidence:  {len(strong_events):,} "
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
print(
    f"Start: {m1['timestamp'].min()}"
)
print(
    f"End:   {m1['timestamp'].max()}"
)


# ============================================================
# BUILD GMMA RIBBON AGGREGATES
# ============================================================
#
# We do not retain 12 full EMA columns.
# We calculate each EMA one at a time and update:
#
#   fast_min / fast_max / fast_mean
#   slow_min / slow_max / slow_mean
#
# This substantially reduces memory use.
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

# The entire slow ribbon is only definitely valid
# once the 60 EMA is available.
gmma_valid = np.arange(n_rows) >= (
    max(SLOW_EMAS) - 1
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

# A NEW full-ribbon crossover/separation.
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
    f"Bullish full-ribbon crosses: "
    f"{bull_cross.sum():,}"
)
print(
    f"Bearish full-ribbon crosses: "
    f"{bear_cross.sum():,}"
)


# ============================================================
# TIMESTAMP PREFLIGHT
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
print(
    f"First signal:   {first_signal_time}"
)
print(
    f"Search index:   {first_idx}"
)

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
# GMMA SIGNAL SELECTION + TRADE SIMULATION
# ============================================================

signal_rows = []
trade_rows = []

print("\n============================================")
print("RUNNING H4.6A")
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

    if ev.nearer_side == "UPPER":
        direction = "LONG"
        target_price = float(
            ev.upper_band_low
        )
        desired_cross = bull_cross
        already_aligned = bool(
            bull_aligned[signal_idx - 1]
            if signal_idx > 0
            else False
        )

    elif ev.nearer_side == "LOWER":
        direction = "SHORT"
        target_price = float(
            ev.lower_band_high
        )
        desired_cross = bear_cross
        already_aligned = bool(
            bear_aligned[signal_idx - 1]
            if signal_idx > 0
            else False
        )

    else:
        continue

    original_correct = (
        original_target_correct(
            direction,
            ev.first_touch
        )
    )

    # --------------------------------------------------------
    # GMMA must confirm before H4.2 event resolution.
    # If H4.2 timed out, search up to original 24h horizon.
    #
    # The touch minute is EXCLUDED:
    # if cross and touch occur in the same M1 bar, no trade.
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
            "event_id": ev.event_id,
            "decision_time": decision_time,
            "year": decision_time.year,
            "direction": direction,
            "target_price": target_price,
            "near_far_ratio": ev.near_far_ratio,
            "h4_2_first_touch": ev.first_touch,
            "original_target_correct": original_correct,
            "already_aligned_before_signal": already_aligned,
            "gmma_status": "NO_SEARCH_WINDOW",
            "gmma_confirmation_time": pd.NaT,
            "minutes_to_confirmation": np.nan,
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
            "event_id": ev.event_id,
            "decision_time": decision_time,
            "year": decision_time.year,
            "direction": direction,
            "target_price": target_price,
            "near_far_ratio": ev.near_far_ratio,
            "h4_2_first_touch": ev.first_touch,
            "original_target_correct": original_correct,
            "already_aligned_before_signal": already_aligned,
            "gmma_status": "NO_NEW_GMMA_CROSS_BEFORE_RESOLUTION",
            "gmma_confirmation_time": pd.NaT,
            "minutes_to_confirmation": np.nan,
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

    minutes_to_confirmation = (
        (
            cross_time - decision_time
        )
        / pd.Timedelta(minutes=1)
    )

    # --------------------------------------------------------
    # Save GMMA state at confirmation for audit.
    # --------------------------------------------------------

    signal_record = {
        "event_id": ev.event_id,
        "decision_time": decision_time,
        "year": decision_time.year,
        "direction": direction,
        "target_price": target_price,
        "near_far_ratio": ev.near_far_ratio,
        "h4_2_first_touch": ev.first_touch,
        "original_target_correct": original_correct,
        "already_aligned_before_signal": already_aligned,
        "gmma_status": "GMMA_CROSS_CONFIRMED",
        "gmma_confirmation_time": cross_time,
        "minutes_to_confirmation": minutes_to_confirmation,

        "fast_ribbon_min": fast_min[cross_idx],
        "fast_ribbon_max": fast_max[cross_idx],
        "fast_ribbon_mean": fast_mean[cross_idx],

        "slow_ribbon_min": slow_min[cross_idx],
        "slow_ribbon_max": slow_max[cross_idx],
        "slow_ribbon_mean": slow_mean[cross_idx],

        "ribbon_mean_gap_bps_vs_close": (
            (
                fast_mean[cross_idx]
                - slow_mean[cross_idx]
            )
            /
            close_arr[cross_idx]
            * 10000.0
        ),
    }

    # --------------------------------------------------------
    # Next-minute executable entry check.
    # --------------------------------------------------------

    if next_idx >= len(ts):
        signal_record[
            "gmma_status"
        ] = "NO_NEXT_MINUTE"
        signal_rows.append(
            signal_record
        )
        continue

    entry_time = pd.Timestamp(
        int(ts[next_idx]),
        tz="UTC"
    )

    if (
        entry_time
        != cross_time
        + pd.Timedelta(minutes=1)
    ):
        signal_record[
            "gmma_status"
        ] = "MISSING_NEXT_MINUTE"
        signal_rows.append(
            signal_record
        )
        continue

    entry_price = float(
        open_arr[next_idx]
    )

    # Target must still be ahead at the actual entry.
    if (
        direction == "LONG"
        and entry_price >= target_price
    ):
        signal_record[
            "gmma_status"
        ] = "TARGET_PASSED_BEFORE_ENTRY"
        signal_rows.append(
            signal_record
        )
        continue

    if (
        direction == "SHORT"
        and entry_price <= target_price
    ):
        signal_record[
            "gmma_status"
        ] = "TARGET_PASSED_BEFORE_ENTRY"
        signal_rows.append(
            signal_record
        )
        continue

    reward_distance = abs(
        target_price - entry_price
    )

    if reward_distance <= 0:
        signal_record[
            "gmma_status"
        ] = "INVALID_TARGET_DISTANCE"
        signal_rows.append(
            signal_record
        )
        continue

    # Symmetric 1:1 stop around actual entry.
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

    signal_record[
        "gmma_status"
    ] = "EXECUTED"

    signal_record[
        "entry_time"
    ] = entry_time

    signal_record[
        "entry_price"
    ] = entry_price

    signal_record[
        "stop_price"
    ] = stop_price

    signal_record[
        "entry_to_target_distance"
    ] = reward_distance

    signal_rows.append(
        signal_record
    )

    # --------------------------------------------------------
    # Trade horizon = 24h from original H4.2 signal.
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

        exit_rel = len(closes) - 1
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
            # Target and stop in same M1 candle.
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
        gross_return * 10000.0
    )

    risk_bps = (
        reward_distance /
        entry_price *
        10000.0
    )

    minutes_signal_to_entry = (
        (
            entry_time
            - decision_time
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

        "ribbon_mean_gap_bps_vs_close": (
            (
                fast_mean[cross_idx]
                - slow_mean[cross_idx]
            )
            /
            close_arr[cross_idx]
            * 10000.0
        ),

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
            "H4_6A_CHECKPOINT_signal_diagnostics.csv"
        )

        safe_to_csv(
            pd.DataFrame(trade_rows),
            "H4_6A_CHECKPOINT_trades.csv"
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
        "No H4.6A signal diagnostics were generated."
    )


# ============================================================
# SIGNAL / SELECTIVITY SUMMARY
# ============================================================

status_summary = (
    signals[
        "gmma_status"
    ]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "gmma_status"
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

executed_signals = signals[
    signals["gmma_status"]
    == "EXECUTED"
].copy()

resolved_signals = signals[
    signals[
        "original_target_correct"
    ].notna()
].copy()

resolved_executed = (
    executed_signals[
        executed_signals[
            "original_target_correct"
        ].notna()
    ]
)

selectivity_summary = pd.DataFrame([
    {
        "strong_signals": len(signals),

        "resolved_strong_signals": len(
            resolved_signals
        ),

        "baseline_target_correct_rate_resolved": (
            resolved_signals[
                "original_target_correct"
            ].astype(bool).mean()
            if len(resolved_signals)
            else np.nan
        ),

        "executed_gmma_signals": len(
            executed_signals
        ),

        "gmma_execution_rate": (
            len(executed_signals)
            / len(signals)
            if len(signals)
            else np.nan
        ),

        "resolved_executed_gmma_signals": len(
            resolved_executed
        ),

        "target_correct_rate_after_gmma_resolved": (
            resolved_executed[
                "original_target_correct"
            ].astype(bool).mean()
            if len(resolved_executed)
            else np.nan
        ),

        "already_aligned_before_signal_rate": (
            signals[
                "already_aligned_before_signal"
            ].mean()
        ),

        "median_minutes_to_gmma_confirmation": (
            executed_signals[
                "minutes_to_confirmation"
            ].median()
            if len(executed_signals)
            else np.nan
        ),
    }
])

print("\n============================================")
print("GMMA SELECTIVITY")
print("============================================")
print(
    selectivity_summary.to_string(
        index=False
    )
)

print("\nGMMA STATUS COUNTS")
print(
    status_summary.to_string(
        index=False
    )
)


# ============================================================
# TRADE SUMMARY
# ============================================================

if not trades.empty:
    overall_summary = pd.DataFrame([
        summarize_trades(trades)
    ])

    print("\n============================================")
    print("H4.6A GMMA TRADE RESULTS")
    print("============================================")
    print(
        overall_summary.to_string(
            index=False
        )
    )
else:
    overall_summary = pd.DataFrame()


# ============================================================
# LONG / SHORT SUMMARY
# ============================================================

direction_rows = []

if not trades.empty:
    for direction in [
        "LONG", "SHORT"
    ]:
        g = trades[
            trades["direction"]
            == direction
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
# YEAR-BY-YEAR
# ============================================================

year_rows = []

if not trades.empty:
    for year in sorted(
        trades["year"].unique()
    ):
        g = trades[
            trades["year"]
            == year
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
print("H4.6A YEAR-BY-YEAR")
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
        net_bps = gross_bps - cost

        # Convert round-trip notional cost into R using
        # each trade's actual 1R price distance.
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
print("H4.6A COST SENSITIVITY")
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
# SAVE OUTPUTS
# ============================================================

safe_to_csv(
    signals,
    "H4_6A_signal_diagnostics.csv"
)

safe_to_csv(
    status_summary,
    "H4_6A_status_summary.csv"
)

safe_to_csv(
    selectivity_summary,
    "H4_6A_selectivity_summary.csv"
)

safe_to_csv(
    trades,
    "H4_6A_GMMA_trades.csv"
)

safe_to_csv(
    overall_summary,
    "H4_6A_overall_summary.csv"
)

safe_to_csv(
    direction_summary,
    "H4_6A_direction_summary.csv"
)

safe_to_csv(
    year_summary,
    "H4_6A_year_summary.csv"
)

safe_to_csv(
    wait_summary,
    "H4_6A_entry_wait_summary.csv"
)

safe_to_csv(
    cost_summary,
    "H4_6A_cost_sensitivity.csv"
)

safe_to_csv(
    cost_year_summary,
    "H4_6A_cost_sensitivity_by_year.csv"
)


config_text = f"""H4.6A — BTC MAGNET-GUIDED GMMA ENTRY

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

BULLISH GMMA CONFIRMATION:
min(fast EMAs) > max(slow EMAs)
AND previous minute was not fully bullish-aligned

BEARISH GMMA CONFIRMATION:
max(fast EMAs) < min(slow EMAs)
AND previous minute was not fully bearish-aligned

ENTRY:
next 1-minute open after confirmation

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
- H4.2 magnets are not recalculated.
- Existing GMMA alignment before the magnet signal is not itself an entry.
- A new post-signal full-ribbon confirmation is required.
- Same-M1 target+stop ambiguity after entry is conservatively counted as loss.
- No parameter optimization is performed.
"""

(
    OUTPUT_FOLDER /
    "H4_6A_config.txt"
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

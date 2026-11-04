from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.6B — BTC MAGNET-GUIDED DIRECTIONAL CHANGE ENTRY
# ============================================================
#
# PURPOSE
# -------
# H4.2 supplies WHERE price is more likely to go.
# Directional Change (DC) supplies WHEN / whether to enter.
#
# The DC detector is carried over from the earlier Algo 2 research:
#
#   downward change if C_t < H_t * (1 - theta_t)
#   upward change   if C_t > L_t * (1 + theta_t)
#
# where:
#
#   theta_t = k * sigma_t
#   k = 1.25  (FROZEN PRIMARY VALUE)
#
# sigma_t is the PREVIOUS COMPLETED UTC DAY'S median of rolling
# 240-minute realized volatility:
#
#   r_t = log(C_t / C_{t-1})
#   RV240_t = sqrt(sum of r^2 over previous 240 one-minute returns)
#
# sigma changes once per UTC day.
#
# WHY k=1.25?
# -----------
# It is NOT selected from the current BTC magnet sample.
# It is carried forward from the earlier independent DC research.
# We therefore avoid searching the BTC H4 sample for the "best" k.
#
# ------------------------------------------------------------
# MAGNET LAYER — FROZEN
# ------------------------------------------------------------
# H4_2_magnet_events.csv
#
# Strong-confidence signals only:
#
#       nearer_distance / farther_distance <= 0.50
#
# nearer upper magnet -> LONG target
# nearer lower magnet -> SHORT target
#
# ------------------------------------------------------------
# STATE-AWARE DC ENTRY RULE
# ------------------------------------------------------------
# At the H4.2 decision, inspect the DC state on the LAST COMPLETED M1 bar.
#
# CASE A — DC already points toward the magnet:
#   Treat the signal as directionally aligned.
#   Enter at the NEXT M1 OPEN after the H4.2 decision.
#
# CASE B — DC points opposite the magnet:
#   Do NOT enter.
#   Wait for a CLOSE-CONFIRMED DC reversal toward the magnet.
#   Enter at the NEXT M1 OPEN after that DC event.
#
# CASE C — DC not yet initialized:
#   Wait for the first close-confirmed DC event toward the magnet.
#
# A waiting DC event must occur STRICTLY BEFORE the H4.2 first-magnet touch.
#
# ------------------------------------------------------------
# TARGET / STOP
# ------------------------------------------------------------
# Target = original nearer H4.2 magnet band edge.
#
# Stop = symmetric 1:1 stop around the ACTUAL DC entry:
#
#       stop distance = entry-to-target distance
#
# This matches the H4.6A2 GMMA comparison and avoids changing both
# entry and risk mechanics at once.
#
# ------------------------------------------------------------
# HORIZON
# ------------------------------------------------------------
# Target / stop / mark-to-market at 24 hours from original H4.2 signal.
#
# ------------------------------------------------------------
# COSTS
# ------------------------------------------------------------
# Gross + round-trip cost sensitivity:
# 1, 2, 3, 4, 5, 7.5, 10 bp
#
# NO THRESHOLD OPTIMIZATION IS PERFORMED.
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

OUTPUT_FOLDER = SCRIPT_DIR / "H4_6B_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

STRONG_CONFIDENCE_MAX_RATIO = 0.50

# Frozen from prior independent Algo 2 DC work.
DC_K = 1.25
RV_WINDOW_MINUTES = 240

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
            raise ValueError(f"No CSV found inside {zip_path}")

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

    missing = [c for c in required if c not in df.columns]

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
        out[c] = pd.to_numeric(df[c], errors="coerce")

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
    pnl_bps = g["gross_return_bps_conservative"].astype(float)

    gp = pnl_r[pnl_r > 0].sum()
    gl = abs(pnl_r[pnl_r < 0].sum())

    return {
        "trades": len(g),
        "wins": int((g["outcome"] == "WIN").sum()),
        "losses": int((g["outcome"] == "LOSS").sum()),
        "time_exits": int((g["outcome"] == "TIME_EXIT").sum()),
        "ambiguous_same_bar": int(
            (g["outcome"] == "AMBIGUOUS_SAME_BAR").sum()
        ),
        "win_rate_raw": float((g["outcome"] == "WIN").mean()),
        "mean_R_conservative": float(pnl_r.mean()),
        "profit_factor_R_conservative": (
            float(gp / gl) if gl > 0 else np.inf
        ),
        "mean_gross_bps": float(pnl_bps.mean()),
        "median_gross_bps": float(pnl_bps.median()),
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
    "upper_band_low",
    "upper_distance",
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
        events["nearer_side"].isin(["UPPER", "LOWER"])
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
print(f"All H4.2 events:     {len(events):,}")
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

    all_data.append(
        clean_dataframe(
            read_zip_csv(path)
        )
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

m1["timestamp"] = m1["timestamp"].dt.as_unit("ns")

ts = m1["timestamp"].astype("int64").to_numpy()
open_arr = m1["open"].to_numpy(dtype=float)
high_arr = m1["high"].to_numpy(dtype=float)
low_arr = m1["low"].to_numpy(dtype=float)
close_arr = m1["close"].to_numpy(dtype=float)

print("\n============================================")
print("BTC M1 DATA")
print("============================================")
print(f"Rows:  {len(m1):,}")
print(f"Start: {m1['timestamp'].min()}")
print(f"End:   {m1['timestamp'].max()}")


# ============================================================
# BUILD CAUSAL VOLATILITY-NORMALIZED DC THRESHOLD
# ============================================================

print("\n============================================")
print("BUILDING DC THRESHOLD")
print("============================================")

close_series = pd.Series(close_arr, copy=False)

log_return = np.log(
    close_series / close_series.shift(1)
)

rv240 = np.sqrt(
    log_return.pow(2)
    .rolling(
        RV_WINDOW_MINUTES,
        min_periods=RV_WINDOW_MINUTES
    )
    .sum()
)

# UTC calendar day for each minute.
utc_day = m1["timestamp"].dt.floor("D")

rv_frame = pd.DataFrame({
    "day": utc_day,
    "rv240": rv240,
})

# Median realized volatility observed WITHIN each completed day.
daily_sigma_observed = (
    rv_frame
    .groupby("day", sort=True)["rv240"]
    .median()
)

# Threshold used on day t is based only on day t-1.
daily_sigma_causal = daily_sigma_observed.shift(1)

sigma_map = utc_day.map(daily_sigma_causal)

sigma_arr = sigma_map.to_numpy(dtype=float)

theta_arr = DC_K * sigma_arr

valid_theta = np.isfinite(theta_arr) & (theta_arr > 0)

print(f"DC multiplier k:          {DC_K}")
print(f"RV window:                {RV_WINDOW_MINUTES} minutes")
print(f"Minutes with valid theta: {valid_theta.sum():,}")
print(
    f"Median theta:             "
    f"{np.nanmedian(theta_arr):.6f} "
    f"({np.nanmedian(theta_arr) * 100:.4f}%)"
)


# ============================================================
# RUN ONE INDEPENDENT DC STATE MACHINE
# ============================================================
#
# State coding:
#    0 = not initialized
#   +1 = UP state
#   -1 = DOWN state
#
# Bar high/low updates the extreme.
# Bar CLOSE confirms a reversal.
# New leg begins from the confirmation CLOSE.
# First initialization creates no DC event.
# ============================================================

print("\n============================================")
print("RUNNING DC STATE MACHINE")
print("============================================")

n = len(m1)

dc_state = np.zeros(n, dtype=np.int8)
up_event = np.zeros(n, dtype=bool)
down_event = np.zeros(n, dtype=bool)

state = 0
running_high = np.nan
running_low = np.nan
init_high = np.nan
init_low = np.nan
threshold_started = False

for i in range(n):

    theta = theta_arr[i]

    if not np.isfinite(theta) or theta <= 0:
        dc_state[i] = state
        continue

    # Start fresh at first causally valid threshold.
    if not threshold_started:
        threshold_started = True
        state = 0
        init_high = high_arr[i]
        init_low = low_arr[i]
        dc_state[i] = 0
        continue

    if state == 0:
        init_high = max(init_high, high_arr[i])
        init_low = min(init_low, low_arr[i])

        # First state initialization: no event.
        if close_arr[i] > init_low * (1.0 + theta):
            state = 1
            running_high = close_arr[i]

        elif close_arr[i] < init_high * (1.0 - theta):
            state = -1
            running_low = close_arr[i]

        dc_state[i] = state
        continue

    if state == 1:
        # Update extreme using bar high first.
        running_high = max(running_high, high_arr[i])

        if close_arr[i] < running_high * (1.0 - theta):
            down_event[i] = True
            state = -1

            # New leg starts at confirmation close.
            running_low = close_arr[i]

        dc_state[i] = state
        continue

    if state == -1:
        # Update extreme using bar low first.
        running_low = min(running_low, low_arr[i])

        if close_arr[i] > running_low * (1.0 + theta):
            up_event[i] = True
            state = 1

            # New leg starts at confirmation close.
            running_high = close_arr[i]

        dc_state[i] = state
        continue


print(f"Upward DC events:   {up_event.sum():,}")
print(f"Downward DC events: {down_event.sum():,}")
print(
    "State coverage: "
    f"UP={(dc_state == 1).mean():.2%}, "
    f"DOWN={(dc_state == -1).mean():.2%}, "
    f"UNINIT={(dc_state == 0).mean():.2%}"
)


# ============================================================
# PREFLIGHT
# ============================================================

first_signal_time = strong_events["decision_time"].iloc[0]
first_signal_ns = timestamp_ns(first_signal_time)

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
# STATE-AWARE MAGNET + DC ENTRY
# ============================================================

signal_rows = []
trade_rows = []

print("\n============================================")
print("RUNNING H4.6B")
print("============================================")

for counter, ev in enumerate(
    strong_events.itertuples(index=False),
    start=1
):

    decision_time = ev.decision_time
    decision_ns = timestamp_ns(decision_time)

    signal_idx = int(
        np.searchsorted(
            ts,
            decision_ns,
            side="left"
        )
    )

    if signal_idx <= 0 or signal_idx >= len(ts):
        continue

    prev_idx = signal_idx - 1
    state_at_signal = int(dc_state[prev_idx])

    if ev.nearer_side == "UPPER":
        direction = "LONG"
        target_price = float(ev.upper_band_low)
        desired_state = 1
        desired_event = up_event

    elif ev.nearer_side == "LOWER":
        direction = "SHORT"
        target_price = float(ev.lower_band_high)
        desired_state = -1
        desired_event = down_event

    else:
        continue

    original_correct = target_correct(
        direction,
        ev.first_touch
    )

    base = {
        "event_id": ev.event_id,
        "decision_time": decision_time,
        "year": decision_time.year,
        "direction": direction,
        "target_price": target_price,
        "near_far_ratio": ev.near_far_ratio,
        "h4_2_first_touch": ev.first_touch,
        "original_target_correct": original_correct,

        "dc_k": DC_K,
        "sigma_at_signal": sigma_arr[prev_idx],
        "theta_at_signal": theta_arr[prev_idx],

        "dc_state_at_signal": (
            "UP" if state_at_signal == 1
            else "DOWN" if state_at_signal == -1
            else "UNINITIALIZED"
        ),
    }

    if not np.isfinite(theta_arr[prev_idx]):
        signal_rows.append({
            **base,
            "entry_source": "NONE",
            "dc_status": "NO_CAUSAL_THRESHOLD_AT_SIGNAL",
            "dc_confirmation_time": pd.NaT,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
        })
        continue

    # --------------------------------------------------------
    # CASE A: already directionally aligned.
    # Entry at next M1 open after H4.2 decision.
    # --------------------------------------------------------

    if state_at_signal == desired_state:
        entry_source = "ALIGNED_AT_SIGNAL"
        dc_confirmation_time = pd.NaT

        entry_idx = signal_idx + 1

        if entry_idx >= len(ts):
            signal_rows.append({
                **base,
                "entry_source": entry_source,
                "dc_status": "NO_NEXT_MINUTE",
                "dc_confirmation_time": pd.NaT,
                "entry_time": pd.NaT,
                "entry_price": np.nan,
            })
            continue

        entry_time = pd.Timestamp(
            int(ts[entry_idx]),
            tz="UTC"
        )

        # Exact next-minute continuity.
        expected_entry_time = (
            decision_time + pd.Timedelta(minutes=1)
        )

        if entry_time != expected_entry_time:
            signal_rows.append({
                **base,
                "entry_source": entry_source,
                "dc_status": "MISSING_NEXT_MINUTE",
                "dc_confirmation_time": pd.NaT,
                "entry_time": entry_time,
                "entry_price": np.nan,
            })
            continue

        # If H4.2 already resolved in the decision minute, entry is too late.
        if (
            pd.notna(ev.touch_time)
            and ev.touch_time < entry_time
        ):
            signal_rows.append({
                **base,
                "entry_source": entry_source,
                "dc_status": "MAGNET_RESOLVED_BEFORE_ENTRY",
                "dc_confirmation_time": pd.NaT,
                "entry_time": entry_time,
                "entry_price": np.nan,
            })
            continue

        dc_status = "EXECUTED_ALIGNED"

    # --------------------------------------------------------
    # CASE B/C: opposite or uninitialized.
    # Wait for first close-confirmed DC event toward magnet.
    # --------------------------------------------------------

    else:
        entry_source = (
            "REVERSAL_TO_MAGNET"
            if state_at_signal in [-1, 1]
            else "INITIALIZED_THEN_EVENT_TO_MAGNET"
        )

        if pd.notna(ev.touch_time):
            search_end_time = ev.touch_time
        else:
            search_end_time = (
                decision_time
                + pd.Timedelta(hours=TRADE_HORIZON_HOURS)
            )

        search_end_ns = timestamp_ns(search_end_time)

        # Exclude the H4.2 touch minute itself.
        search_end_idx = int(
            np.searchsorted(
                ts,
                search_end_ns,
                side="left"
            )
        )

        search_end_idx = min(search_end_idx, len(ts))

        if signal_idx >= search_end_idx:
            signal_rows.append({
                **base,
                "entry_source": entry_source,
                "dc_status": "NO_SEARCH_WINDOW",
                "dc_confirmation_time": pd.NaT,
                "entry_time": pd.NaT,
                "entry_price": np.nan,
            })
            continue

        local_events = desired_event[
            signal_idx:search_end_idx
        ]

        event_rel = first_true_index(local_events)

        if event_rel is None:
            signal_rows.append({
                **base,
                "entry_source": entry_source,
                "dc_status": "NO_DC_TURN_TOWARD_MAGNET_BEFORE_RESOLUTION",
                "dc_confirmation_time": pd.NaT,
                "entry_time": pd.NaT,
                "entry_price": np.nan,
            })
            continue

        event_idx = signal_idx + event_rel

        dc_confirmation_time = pd.Timestamp(
            int(ts[event_idx]),
            tz="UTC"
        )

        entry_idx = event_idx + 1

        if entry_idx >= len(ts):
            continue

        entry_time = pd.Timestamp(
            int(ts[entry_idx]),
            tz="UTC"
        )

        if (
            entry_time
            != dc_confirmation_time
            + pd.Timedelta(minutes=1)
        ):
            signal_rows.append({
                **base,
                "entry_source": entry_source,
                "dc_status": "MISSING_NEXT_MINUTE_AFTER_DC",
                "dc_confirmation_time": dc_confirmation_time,
                "entry_time": entry_time,
                "entry_price": np.nan,
            })
            continue

        dc_status = "EXECUTED_AFTER_DC_TURN"

    # --------------------------------------------------------
    # EXECUTABLE ENTRY GEOMETRY
    # --------------------------------------------------------

    entry_price = float(open_arr[entry_idx])

    if direction == "LONG" and entry_price >= target_price:
        signal_rows.append({
            **base,
            "entry_source": entry_source,
            "dc_status": "TARGET_PASSED_BEFORE_ENTRY",
            "dc_confirmation_time": dc_confirmation_time,
            "entry_time": entry_time,
            "entry_price": entry_price,
        })
        continue

    if direction == "SHORT" and entry_price <= target_price:
        signal_rows.append({
            **base,
            "entry_source": entry_source,
            "dc_status": "TARGET_PASSED_BEFORE_ENTRY",
            "dc_confirmation_time": dc_confirmation_time,
            "entry_time": entry_time,
            "entry_price": entry_price,
        })
        continue

    reward_distance = abs(
        target_price - entry_price
    )

    if reward_distance <= 0:
        continue

    # 1:1 stop for apples-to-apples comparison with GMMA.
    if direction == "LONG":
        stop_price = entry_price - reward_distance
    else:
        stop_price = entry_price + reward_distance

    minutes_signal_to_entry = float(
        (
            entry_time - decision_time
        )
        / pd.Timedelta(minutes=1)
    )

    signal_rows.append({
        **base,
        "entry_source": entry_source,
        "dc_status": dc_status,
        "dc_confirmation_time": dc_confirmation_time,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "minutes_signal_to_entry": minutes_signal_to_entry,
    })

    # --------------------------------------------------------
    # TRADE SIMULATION
    # --------------------------------------------------------

    horizon_time = (
        decision_time
        + pd.Timedelta(hours=TRADE_HORIZON_HOURS)
    )

    horizon_ns = timestamp_ns(horizon_time)

    end_idx = int(
        np.searchsorted(
            ts,
            horizon_ns,
            side="left"
        )
    )

    end_idx = min(end_idx, len(ts))

    if entry_idx >= end_idx:
        continue

    highs = high_arr[entry_idx:end_idx]
    lows = low_arr[entry_idx:end_idx]
    closes = close_arr[entry_idx:end_idx]
    trade_ts = ts[entry_idx:end_idx]

    if highs.size == 0:
        continue

    if direction == "LONG":
        target_mask = highs >= target_price
        stop_mask = lows <= stop_price
    else:
        target_mask = lows <= target_price
        stop_mask = highs >= stop_price

    target_rel = first_true_index(target_mask)
    stop_rel = first_true_index(stop_mask)

    if target_rel is None and stop_rel is None:
        outcome = "TIME_EXIT"
        exit_rel = len(closes) - 1
        exit_price = float(closes[exit_rel])

        exit_time = pd.Timestamp(
            int(trade_ts[exit_rel]),
            tz="UTC"
        )

        if direction == "LONG":
            pnl_price = exit_price - entry_price
        else:
            pnl_price = entry_price - exit_price

        pnl_R = pnl_price / reward_distance

    elif target_rel is not None and stop_rel is None:
        outcome = "WIN"
        exit_price = target_price
        exit_time = pd.Timestamp(
            int(trade_ts[target_rel]),
            tz="UTC"
        )
        pnl_R = 1.0

    elif target_rel is None and stop_rel is not None:
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

    gross_bps = gross_return * 10000.0

    risk_bps = (
        reward_distance
        / entry_price
        * 10000.0
    )

    trade_rows.append({
        "event_id": ev.event_id,
        "decision_time": decision_time,
        "year": decision_time.year,
        "direction": direction,

        "near_far_ratio": ev.near_far_ratio,
        "h4_2_first_touch": ev.first_touch,
        "original_target_correct": original_correct,

        "dc_k": DC_K,
        "sigma_at_signal": sigma_arr[prev_idx],
        "theta_at_signal": theta_arr[prev_idx],
        "dc_state_at_signal": base["dc_state_at_signal"],

        "entry_source": entry_source,
        "dc_confirmation_time": dc_confirmation_time,

        "entry_time": entry_time,
        "minutes_signal_to_entry": minutes_signal_to_entry,

        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,

        "risk_price_distance": reward_distance,
        "risk_bps": risk_bps,

        "outcome": outcome,
        "exit_time": exit_time,
        "exit_price_conservative": exit_price,

        "pnl_R_conservative": pnl_R,
        "gross_return_bps_conservative": gross_bps,
    })

    if counter % CHECKPOINT_EVERY_SIGNALS == 0:
        safe_to_csv(
            pd.DataFrame(signal_rows),
            "H4_6B_CHECKPOINT_signal_diagnostics.csv"
        )

        safe_to_csv(
            pd.DataFrame(trade_rows),
            "H4_6B_CHECKPOINT_trades.csv"
        )

        print(
            f"  Processed "
            f"{counter:,}/{len(strong_events):,} "
            f"signals — checkpoint saved."
        )


signals = pd.DataFrame(signal_rows)
trades = pd.DataFrame(trade_rows)

if signals.empty:
    raise RuntimeError(
        "No H4.6B signal diagnostics generated."
    )


# ============================================================
# STATUS / SELECTIVITY
# ============================================================

status_summary = (
    signals["dc_status"]
    .value_counts(dropna=False)
    .rename_axis("dc_status")
    .reset_index(name="count")
)

status_summary[
    "fraction_of_strong_signals"
] = (
    status_summary["count"]
    / len(signals)
)

executed_signals = signals[
    signals["dc_status"].isin(
        [
            "EXECUTED_ALIGNED",
            "EXECUTED_AFTER_DC_TURN",
        ]
    )
].copy()

resolved_all = signals[
    signals["original_target_correct"].notna()
].copy()

resolved_executed = executed_signals[
    executed_signals[
        "original_target_correct"
    ].notna()
].copy()

selectivity_summary = pd.DataFrame([
    {
        "strong_signals": len(signals),

        "resolved_strong_signals": len(resolved_all),

        "baseline_target_correct_rate_resolved": (
            resolved_all[
                "original_target_correct"
            ].astype(bool).mean()
            if len(resolved_all)
            else np.nan
        ),

        "executed_dc_signals": len(executed_signals),

        "dc_execution_rate": (
            len(executed_signals) / len(signals)
            if len(signals)
            else np.nan
        ),

        "resolved_executed_dc": len(resolved_executed),

        "target_correct_rate_after_dc_resolved": (
            resolved_executed[
                "original_target_correct"
            ].astype(bool).mean()
            if len(resolved_executed)
            else np.nan
        ),

        "aligned_at_signal_count": int(
            (
                executed_signals["entry_source"]
                == "ALIGNED_AT_SIGNAL"
            ).sum()
        ),

        "reversal_to_magnet_count": int(
            (
                executed_signals["entry_source"]
                == "REVERSAL_TO_MAGNET"
            ).sum()
        ),

        "median_minutes_signal_to_entry": (
            executed_signals[
                "minutes_signal_to_entry"
            ].median()
            if len(executed_signals)
            else np.nan
        ),
    }
])

print("\n============================================")
print("H4.6B SELECTIVITY")
print("============================================")
print(selectivity_summary.to_string(index=False))

print("\nSTATUS COUNTS")
print(status_summary.to_string(index=False))


# ============================================================
# OVERALL TRADE RESULTS
# ============================================================

if not trades.empty:
    overall_summary = pd.DataFrame([
        summarize_trades(trades)
    ])
else:
    overall_summary = pd.DataFrame()

print("\n============================================")
print("H4.6B OVERALL TRADE RESULTS")
print("============================================")

if not overall_summary.empty:
    print(overall_summary.to_string(index=False))
else:
    print("No executed trades.")


# ============================================================
# ENTRY SOURCE BREAKDOWN
# ============================================================

source_rows = []

if not trades.empty:
    for source in sorted(
        trades["entry_source"].dropna().unique()
    ):
        g = trades[
            trades["entry_source"] == source
        ]

        if g.empty:
            continue

        s = summarize_trades(g)
        s["entry_source"] = source
        source_rows.append(s)

source_summary = pd.DataFrame(source_rows)

print("\n============================================")
print("H4.6B ENTRY-SOURCE BREAKDOWN")
print("============================================")

if not source_summary.empty:
    print(source_summary.to_string(index=False))


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

year_summary = pd.DataFrame(year_rows)

print("\n============================================")
print("H4.6B YEAR-BY-YEAR")
print("============================================")

if not year_summary.empty:
    print(year_summary.to_string(index=False))


# ============================================================
# ENTRY SOURCE BY YEAR
# ============================================================

source_year_rows = []

if not trades.empty:
    for year in sorted(
        trades["year"].unique()
    ):
        for source in sorted(
            trades["entry_source"].dropna().unique()
        ):
            g = trades[
                (trades["year"] == year)
                &
                (trades["entry_source"] == source)
            ]

            if g.empty:
                continue

            s = summarize_trades(g)
            s["year"] = int(year)
            s["entry_source"] = source
            source_year_rows.append(s)

source_year_summary = pd.DataFrame(
    source_year_rows
)


# ============================================================
# LONG / SHORT
# ============================================================

direction_rows = []

if not trades.empty:
    for direction in ["LONG", "SHORT"]:
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

        cost_R = (
            cost
            /
            risk_bps.replace(0, np.nan)
        )

        net_R = gross_R - cost_R

        gp = net_R[net_R > 0].sum()
        gl = abs(net_R[net_R < 0].sum())

        cost_rows.append({
            "round_trip_cost_bps": cost,
            "trades": len(trades),

            "mean_gross_bps": gross_bps.mean(),
            "mean_net_bps": net_bps.mean(),

            "mean_gross_R": gross_R.mean(),
            "mean_net_R": net_R.mean(),

            "net_profit_factor_R": (
                gp / gl if gl > 0 else np.inf
            ),

            "positive_net_trade_rate": (
                net_bps > 0
            ).mean(),
        })

cost_summary = pd.DataFrame(cost_rows)

print("\n============================================")
print("H4.6B COST SENSITIVITY")
print("============================================")

if not cost_summary.empty:
    print(cost_summary.to_string(index=False))


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
            net_bps = gross_bps - cost

            cost_R = (
                cost
                /
                risk_bps.replace(0, np.nan)
            )

            net_R = gross_R - cost_R

            gp = net_R[net_R > 0].sum()
            gl = abs(net_R[net_R < 0].sum())

            cost_year_rows.append({
                "year": int(year),
                "round_trip_cost_bps": cost,
                "trades": len(yg),

                "mean_gross_bps": gross_bps.mean(),
                "mean_net_bps": net_bps.mean(),

                "mean_gross_R": gross_R.mean(),
                "mean_net_R": net_R.mean(),

                "net_profit_factor_R": (
                    gp / gl if gl > 0 else np.inf
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
    "H4_6B_signal_diagnostics.csv"
)

safe_to_csv(
    trades,
    "H4_6B_DC_trades.csv"
)

safe_to_csv(
    status_summary,
    "H4_6B_status_summary.csv"
)

safe_to_csv(
    selectivity_summary,
    "H4_6B_selectivity_summary.csv"
)

safe_to_csv(
    overall_summary,
    "H4_6B_overall_summary.csv"
)

safe_to_csv(
    source_summary,
    "H4_6B_entry_source_summary.csv"
)

safe_to_csv(
    year_summary,
    "H4_6B_year_summary.csv"
)

safe_to_csv(
    source_year_summary,
    "H4_6B_entry_source_by_year.csv"
)

safe_to_csv(
    direction_summary,
    "H4_6B_direction_summary.csv"
)

safe_to_csv(
    cost_summary,
    "H4_6B_cost_sensitivity.csv"
)

safe_to_csv(
    cost_year_summary,
    "H4_6B_cost_sensitivity_by_year.csv"
)

daily_threshold_output = pd.DataFrame({
    "day": daily_sigma_causal.index,
    "sigma_previous_day_median_rv240": (
        daily_sigma_causal.values
    ),
    "theta_k_1_25": (
        daily_sigma_causal.values * DC_K
    ),
})

safe_to_csv(
    daily_threshold_output,
    "H4_6B_daily_thresholds.csv"
)


config_text = f"""H4.6B — BTC MAGNET-GUIDED DIRECTIONAL CHANGE ENTRY

MAGNET INPUT:
{EVENT_FILE.name}

PRIMARY MAGNET SAMPLE:
nearer_distance / farther_distance <= {STRONG_CONFIDENCE_MAX_RATIO}

DC DETECTOR:
downward change if close < running_high * (1 - theta)
upward change if close > running_low * (1 + theta)

DC THRESHOLD:
theta_t = k * sigma_t

FIXED K:
{DC_K}

WHY K IS FIXED:
Carried forward from prior independent Algo 2 DC research.
It is NOT selected by searching the current BTC magnet results.

SIGMA:
Previous completed UTC day's median of rolling
{RV_WINDOW_MINUTES}-minute realized volatility.

REALIZED VOLATILITY:
sqrt(sum of squared one-minute log returns over
{RV_WINDOW_MINUTES} minutes)

STATE-AWARE ENTRY:

1. If DC state on last completed M1 candle already points toward magnet:
   enter at next M1 open after H4.2 decision.

2. If DC state points opposite magnet:
   wait for close-confirmed DC reversal toward magnet,
   then enter at next M1 open.

3. If DC is uninitialized:
   wait for first confirmed DC event toward magnet.

WAITING EVENT CUTOFF:
DC confirmation must occur strictly before H4.2 first-magnet touch.

TARGET:
Original nearer H4.2 magnet band edge.

STOP:
Symmetric 1:1 stop using actual DC entry-to-target distance.

HORIZON:
{TRADE_HORIZON_HOURS} hours from original H4.2 signal.

ROUND-TRIP COST SENSITIVITY:
{ROUND_TRIP_COST_BPS}

IMPORTANT:
- H4.2 magnets are not recalculated.
- No DC threshold grid is searched in this experiment.
- Same-M1 target+stop ambiguity after entry is conservatively counted as loss.
- Entry-source strata are reported separately:
  ALIGNED_AT_SIGNAL versus REVERSAL_TO_MAGNET.
"""

(
    OUTPUT_FOLDER /
    "H4_6B_config.txt"
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

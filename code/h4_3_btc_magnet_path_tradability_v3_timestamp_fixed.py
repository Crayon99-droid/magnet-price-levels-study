from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.3 V3 — BTC MAGNET PATH QUALITY & TRADABILITY
# ============================================================
#
# This is the corrected/audited version.
# V3 explicitly forces nanosecond timestamps for pandas-version compatibility.
#
# WHAT IT TESTS
# -------------
# Uses the SAVED H4.2 magnet events. It does NOT rebuild or optimize
# the magnet detector.
#
# For each event:
#   - nearer H4.2 magnet = target
#   - entry = H4.2 decision-time price
#   - fixed stop distances = 0.25x, 0.50x, 0.75x, 1.00x target distance
#   - trade horizon = 24 hours from entry
#
# It measures:
#   1) destination correctness
#   2) MAE/MFE before the H4.2 first-magnet event terminates
#   3) tradability with fixed stop fractions
#   4) year-by-year stability
#   5) high-confidence subset where near/far distance <= 0.50
#
# IMPORTANT CORRECTIONS VS V1
# ---------------------------
# 1) NO_EXIT trades are NOT silently dropped.
#    If neither target nor stop is reached in 24h, the trade is closed
#    at the last available 1-minute close and included in expectancy.
#
# 2) MAE/MFE are reported in TWO forms:
#    - strict_pre_terminal: excludes the terminal minute bar, avoiding
#      false ordering assumptions inside the final 1-minute candle.
#    - inclusive_terminal: includes it as an upper-bound diagnostic.
#
# 3) Same-minute target+stop hits remain AMBIGUOUS.
#    Primary "conservative" PnL counts ambiguous trades as full stop losses.
#    A separate unambiguous-only summary is also saved.
#
# 4) Trade simulation runs for the full 24h trading horizon, rather than
#    stopping merely because the opposite H4.2 magnet was touched.
#
# 5) Uses NumPy/searchsorted instead of slow row-by-row minute iteration.
#
# RESULTS ARE GROSS BEFORE FEES / SLIPPAGE / FUNDING.
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

OUTPUT_FOLDER = SCRIPT_DIR / "H4_3_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

STOP_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
TRADE_HORIZON_HOURS = 24
CHECKPOINT_EVERY_EVENTS = 500


# ============================================================
# FILE HELPERS
# ============================================================

def safe_to_csv(frame, filename):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_FOLDER / filename
    frame.to_csv(path, index=False)
    return path


def read_zip_csv(zip_path):
    print(f"\nLoading: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]

        if not csv_files:
            raise ValueError(f"No CSV found inside {zip_path}")

        csv_name = csv_files[0]
        print(f"  CSV inside ZIP: {csv_name}")

        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    print(f"  Rows loaded: {len(df):,}")
    return df


def clean_dataframe(df):
    required = ["timestamp_utc", "open", "high", "low", "close"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
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


def first_true_index(mask):
    """
    Return first True array position, or None.
    """
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def timestamp_ns(value):
    """
    Return an explicit nanosecond integer timestamp.
    This avoids pandas-version-dependent datetime resolution issues.
    """
    if pd.isna(value):
        return None
    return int(pd.Timestamp(value).as_unit("ns").value)


def confidence_bucket(ratio):
    if pd.isna(ratio):
        return "NA"
    if ratio <= 0.50:
        return "<=0.50"
    if ratio <= 0.75:
        return "0.50-0.75"
    if ratio <= 1.00:
        return "0.75-1.00"
    return ">1.00"


# ============================================================
# LOAD H4.2 EVENTS
# ============================================================

if not EVENT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{EVENT_FILE}\n\n"
        "Put H4_2_magnet_events.csv in the same folder as this script."
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

missing_event_cols = [
    c for c in required_event_columns
    if c not in events.columns
]

if missing_event_cols:
    raise ValueError(
        f"H4.2 event file is missing columns: {missing_event_cols}"
    )

events["decision_time"] = pd.to_datetime(
    events["decision_time"],
    utc=True,
    errors="coerce"
)

events["touch_time"] = pd.to_datetime(
    events["touch_time"],
    utc=True,
    errors="coerce"
)

events = (
    events
    .dropna(subset=["decision_time", "entry_price"])
    .sort_values("decision_time")
    .reset_index(drop=True)
)

print("\n============================================")
print("H4.2 EVENTS LOADED")
print("============================================")
print(f"Events: {len(events):,}")
print(f"Start:  {events['decision_time'].min()}")
print(f"End:    {events['decision_time'].max()}")


# ============================================================
# LOAD BTC 1-MINUTE DATA
# ============================================================

all_data = []

for filename in FILES:
    path = SCRIPT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing BTC file:\n{path}\n"
            "Put the four BTC ZIP files in the same folder as this script."
        )

    raw = read_zip_csv(path)
    all_data.append(clean_dataframe(raw))

m1 = pd.concat(all_data, ignore_index=True)

m1 = (
    m1.sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

print("\n============================================")
print("FULL BTC DATASET")
print("============================================")
print(f"Rows:  {len(m1):,}")
print(f"Start: {m1['timestamp'].min()}")
print(f"End:   {m1['timestamp'].max()}")

# Fast NumPy arrays.
# IMPORTANT: force nanosecond resolution explicitly.
# Newer pandas versions may internally store timezone-aware datetimes
# at microsecond resolution. Timestamp.value is nanoseconds, so mixing
# the two units can make every searchsorted lookup miss the dataset.
m1["timestamp"] = m1["timestamp"].dt.as_unit("ns")
events["decision_time"] = events["decision_time"].dt.as_unit("ns")
events["touch_time"] = events["touch_time"].dt.as_unit("ns")

ts = m1["timestamp"].astype("int64").to_numpy()
high_arr = m1["high"].to_numpy(dtype=float)
low_arr = m1["low"].to_numpy(dtype=float)
close_arr = m1["close"].to_numpy(dtype=float)

# ============================================================
# PREFLIGHT TIMESTAMP SANITY CHECK
# ============================================================

first_event_time = events["decision_time"].iloc[0]
first_event_ns = timestamp_ns(first_event_time)
first_idx = int(np.searchsorted(ts, first_event_ns, side="left"))

print("\n============================================")
print("TIMESTAMP SANITY CHECK")
print("============================================")
print(f"Minute dtype:      {m1['timestamp'].dtype}")
print(f"First event time:  {first_event_time}")
print(f"First event ns:    {first_event_ns}")
print(f"First minute ns:   {int(ts[0])}")
print(f"Search index:      {first_idx}")

if first_idx >= len(ts):
    raise RuntimeError(
        "Timestamp unit mismatch: first H4.2 event falls beyond the BTC "
        "minute array. This should not happen in V3."
    )

matched_time = m1["timestamp"].iloc[first_idx]
print(f"Matched minute:    {matched_time}")
print("Timestamp alignment: OK")


# ============================================================
# EVENT ANALYSIS
# ============================================================

path_rows = []
trade_rows = []

print("\n============================================")
print("ANALYZING PATH QUALITY AND TRADABILITY")
print("============================================")

for n, ev in enumerate(events.itertuples(index=False), start=1):

    if ev.nearer_side not in ["UPPER", "LOWER"]:
        continue

    decision_time = ev.decision_time
    decision_ns = timestamp_ns(decision_time)

    entry = float(ev.entry_price)
    upper_distance = float(ev.upper_distance)
    lower_distance = float(ev.lower_distance)
    atr15 = float(ev.atr15)

    if upper_distance <= 0 or lower_distance <= 0:
        continue

    if ev.nearer_side == "UPPER":
        direction = "LONG"
        target_price = float(ev.upper_band_low)
        opposite_price = float(ev.lower_band_high)
        target_distance = target_price - entry
        farther_distance = lower_distance
    else:
        direction = "SHORT"
        target_price = float(ev.lower_band_high)
        opposite_price = float(ev.upper_band_low)
        target_distance = entry - target_price
        farther_distance = upper_distance

    if target_distance <= 0 or farther_distance <= 0:
        continue

    near_far_ratio = target_distance / farther_distance

    # --------------------------------------------------------
    # Locate event start in the minute arrays.
    # --------------------------------------------------------

    start_idx = int(np.searchsorted(ts, decision_ns, side="left"))

    if start_idx >= len(ts):
        continue

    # --------------------------------------------------------
    # H4.2 path endpoint for MAE/MFE.
    # If H4.2 resolved, terminal minute = first magnet touch.
    # Otherwise use 24h.
    # --------------------------------------------------------

    if pd.notna(ev.touch_time):
        terminal_time = ev.touch_time
    else:
        terminal_time = decision_time + pd.Timedelta(
            hours=TRADE_HORIZON_HOURS
        )

    terminal_ns = timestamp_ns(terminal_time)

    terminal_idx = int(
        np.searchsorted(ts, terminal_ns, side="left")
    )

    # Strict-pre-terminal segment excludes the terminal minute.
    strict_end_idx = min(terminal_idx, len(ts))

    # Inclusive segment includes terminal minute if available.
    inclusive_end_idx = min(terminal_idx + 1, len(ts))

    # At least one bar is needed.
    strict_highs = high_arr[start_idx:strict_end_idx]
    strict_lows = low_arr[start_idx:strict_end_idx]

    inclusive_highs = high_arr[start_idx:inclusive_end_idx]
    inclusive_lows = low_arr[start_idx:inclusive_end_idx]

    if inclusive_highs.size == 0:
        continue

    def excursions(highs, lows):
        if highs.size == 0:
            return 0.0, 0.0

        if direction == "LONG":
            mfe = max(0.0, float(np.max(highs)) - entry)
            mae = max(0.0, entry - float(np.min(lows)))
        else:
            mfe = max(0.0, entry - float(np.min(lows)))
            mae = max(0.0, float(np.max(highs)) - entry)

        return mfe, mae

    strict_mfe, strict_mae = excursions(strict_highs, strict_lows)
    incl_mfe, incl_mae = excursions(inclusive_highs, inclusive_lows)

    target_was_first = (
        (ev.nearer_side == "UPPER" and ev.first_touch == "UPPER") or
        (ev.nearer_side == "LOWER" and ev.first_touch == "LOWER")
    )

    h42_resolved = ev.first_touch in ["UPPER", "LOWER"]

    path_rows.append({
        "event_id": ev.event_id,
        "decision_time": decision_time,
        "year": decision_time.year,
        "direction": direction,
        "entry_price": entry,
        "target_price": target_price,
        "opposite_magnet_price": opposite_price,
        "target_distance": target_distance,
        "farther_distance": farther_distance,
        "near_far_ratio": near_far_ratio,
        "confidence_bucket": confidence_bucket(near_far_ratio),
        "atr15": atr15,
        "target_distance_atr": (
            target_distance / atr15 if atr15 > 0 else np.nan
        ),
        "h4_2_first_touch": ev.first_touch,
        "h4_2_resolved": h42_resolved,
        "target_was_first": target_was_first,
        "h4_2_touch_time": ev.touch_time,

        "mfe_strict_pre_terminal": strict_mfe,
        "mae_strict_pre_terminal": strict_mae,
        "mfe_target_units_strict": strict_mfe / target_distance,
        "mae_target_units_strict": strict_mae / target_distance,

        "mfe_including_terminal": incl_mfe,
        "mae_including_terminal": incl_mae,
        "mfe_target_units_inclusive": incl_mfe / target_distance,
        "mae_target_units_inclusive": incl_mae / target_distance,
    })

    # --------------------------------------------------------
    # Full 24h trade window
    # --------------------------------------------------------

    horizon_time = decision_time + pd.Timedelta(
        hours=TRADE_HORIZON_HOURS
    )
    horizon_ns = timestamp_ns(horizon_time)

    trade_end_idx = int(
        np.searchsorted(ts, horizon_ns, side="left")
    )

    trade_end_idx = min(trade_end_idx, len(ts))

    highs = high_arr[start_idx:trade_end_idx]
    lows = low_arr[start_idx:trade_end_idx]
    closes = close_arr[start_idx:trade_end_idx]
    trade_ts = ts[start_idx:trade_end_idx]

    if highs.size == 0:
        continue

    # Target first index is common across all stop fractions.
    if direction == "LONG":
        target_mask = highs >= target_price
    else:
        target_mask = lows <= target_price

    target_hit_idx = first_true_index(target_mask)

    for stop_fraction in STOP_FRACTIONS:

        if direction == "LONG":
            stop_price = entry - stop_fraction * target_distance
            stop_mask = lows <= stop_price
        else:
            stop_price = entry + stop_fraction * target_distance
            stop_mask = highs >= stop_price

        stop_hit_idx = first_true_index(stop_mask)

        outcome = None
        exit_rel_idx = None
        ambiguous = False

        if target_hit_idx is None and stop_hit_idx is None:
            outcome = "TIME_EXIT"
            exit_rel_idx = len(closes) - 1

        elif target_hit_idx is not None and stop_hit_idx is None:
            outcome = "WIN"
            exit_rel_idx = target_hit_idx

        elif target_hit_idx is None and stop_hit_idx is not None:
            outcome = "LOSS"
            exit_rel_idx = stop_hit_idx

        else:
            # Both occurred sometime in horizon.
            if target_hit_idx < stop_hit_idx:
                outcome = "WIN"
                exit_rel_idx = target_hit_idx
            elif stop_hit_idx < target_hit_idx:
                outcome = "LOSS"
                exit_rel_idx = stop_hit_idx
            else:
                outcome = "AMBIGUOUS_SAME_BAR"
                exit_rel_idx = target_hit_idx
                ambiguous = True

        exit_ns = int(trade_ts[exit_rel_idx])
        exit_time = pd.Timestamp(exit_ns, tz="UTC")

        # ----------------------------------------------------
        # PnL in target-distance units
        # ----------------------------------------------------

        if outcome == "WIN":
            pnl_target_units = 1.0

        elif outcome == "LOSS":
            pnl_target_units = -stop_fraction

        elif outcome == "AMBIGUOUS_SAME_BAR":
            # Primary conservative convention:
            # assume stop occurred first.
            pnl_target_units = -stop_fraction

        else:
            # Forced time exit at final available close.
            exit_price = float(closes[exit_rel_idx])

            if direction == "LONG":
                pnl_target_units = (
                    exit_price - entry
                ) / target_distance
            else:
                pnl_target_units = (
                    entry - exit_price
                ) / target_distance

        # Convert target-unit PnL to risk-unit R.
        pnl_risk_units = pnl_target_units / stop_fraction

        trade_rows.append({
            "event_id": ev.event_id,
            "decision_time": decision_time,
            "year": decision_time.year,
            "direction": direction,
            "entry_price": entry,
            "target_price": target_price,
            "target_distance": target_distance,
            "farther_distance": farther_distance,
            "near_far_ratio": near_far_ratio,
            "confidence_bucket": confidence_bucket(near_far_ratio),
            "target_distance_atr": (
                target_distance / atr15 if atr15 > 0 else np.nan
            ),
            "stop_fraction": stop_fraction,
            "stop_price": stop_price,
            "outcome": outcome,
            "exit_time": exit_time,
            "ambiguous_same_bar": ambiguous,
            "pnl_target_units_conservative": pnl_target_units,
            "pnl_risk_units_conservative": pnl_risk_units,
        })

    if n % CHECKPOINT_EVERY_EVENTS == 0:
        p1 = safe_to_csv(
            pd.DataFrame(path_rows),
            "H4_3_CHECKPOINT_path_metrics.csv"
        )
        p2 = safe_to_csv(
            pd.DataFrame(trade_rows),
            "H4_3_CHECKPOINT_trade_tests.csv"
        )

        print(
            f"  Processed {n:,}/{len(events):,} events | "
            f"checkpoint saved"
        )


paths = pd.DataFrame(path_rows)
trades = pd.DataFrame(trade_rows)

if paths.empty or trades.empty:
    raise RuntimeError("No H4.3 observations were generated.")


# ============================================================
# SUMMARY FUNCTIONS
# ============================================================

def summarize_trades(g):
    if g.empty:
        return None

    pnl = g["pnl_target_units_conservative"].astype(float)

    gp = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())

    return {
        "trades": len(g),
        "target_wins": int((g["outcome"] == "WIN").sum()),
        "stop_losses": int((g["outcome"] == "LOSS").sum()),
        "time_exits": int((g["outcome"] == "TIME_EXIT").sum()),
        "ambiguous_same_bar": int(
            (g["outcome"] == "AMBIGUOUS_SAME_BAR").sum()
        ),
        "target_hit_rate": float((g["outcome"] == "WIN").mean()),
        "mean_pnl_target_units_conservative": float(pnl.mean()),
        "mean_R_conservative": float(
            g["pnl_risk_units_conservative"].mean()
        ),
        "profit_factor_conservative": (
            float(gp / gl) if gl > 0 else np.inf
        ),
    }


# ============================================================
# PATH QUALITY SUMMARY
# ============================================================

resolved_paths = paths[paths["h4_2_resolved"] == True].copy()
correct_paths = resolved_paths[
    resolved_paths["target_was_first"] == True
].copy()

print("\n============================================")
print("H4.3 PATH QUALITY")
print("============================================")
print(f"Usable path events:       {len(paths):,}")
print(f"H4.2 resolved events:     {len(resolved_paths):,}")
print(
    f"Nearest target first (resolved only): "
    f"{correct_paths.shape[0] / len(resolved_paths):.2%}"
)

mae_thresholds = [0.10, 0.25, 0.50, 0.75, 1.00]
mae_rows = []

for threshold in mae_thresholds:
    mae_rows.append({
        "mae_threshold_x_target": threshold,
        "correct_destination_events": len(correct_paths),
        "strict_fraction_within_threshold": (
            correct_paths["mae_target_units_strict"] <= threshold
        ).mean(),
        "inclusive_fraction_within_threshold": (
            correct_paths["mae_target_units_inclusive"] <= threshold
        ).mean(),
    })

mae_summary = pd.DataFrame(mae_rows)

print("\nCORRECT DESTINATIONS — MAE BEFORE TARGET")
print(mae_summary.to_string(index=False))


# ============================================================
# OVERALL STOP SUMMARY
# ============================================================

overall_rows = []

for sf in STOP_FRACTIONS:
    g = trades[trades["stop_fraction"] == sf]
    s = summarize_trades(g)

    s["stop_fraction"] = sf
    s["break_even_target_hit_rate_if_no_time_exits"] = (
        sf / (1.0 + sf)
    )

    overall_rows.append(s)

overall_summary = pd.DataFrame(overall_rows)

print("\n============================================")
print("FIXED STOP TESTS — ALL EVENTS")
print("PRIMARY METRIC: CONSERVATIVE, INCLUDING TIME EXITS")
print("============================================")
print(overall_summary.to_string(index=False))


# ============================================================
# UNAMBIGUOUS-ONLY DIAGNOSTIC
# ============================================================

unambiguous = trades[
    trades["outcome"] != "AMBIGUOUS_SAME_BAR"
].copy()

unamb_rows = []

for sf in STOP_FRACTIONS:
    g = unambiguous[unambiguous["stop_fraction"] == sf]
    s = summarize_trades(g)
    s["stop_fraction"] = sf
    unamb_rows.append(s)

unambiguous_summary = pd.DataFrame(unamb_rows)


# ============================================================
# CONFIDENCE-BUCKET SUMMARY
# ============================================================

confidence_order = ["<=0.50", "0.50-0.75", "0.75-1.00", ">1.00"]
confidence_rows = []

for sf in STOP_FRACTIONS:
    for bucket in confidence_order:
        g = trades[
            (trades["stop_fraction"] == sf) &
            (trades["confidence_bucket"] == bucket)
        ]

        if g.empty:
            continue

        s = summarize_trades(g)
        s["stop_fraction"] = sf
        s["confidence_bucket"] = bucket
        confidence_rows.append(s)

confidence_summary = pd.DataFrame(confidence_rows)

print("\n============================================")
print("STOP TESTS BY NEAR/FAR DISTANCE RATIO")
print("============================================")
print(confidence_summary.to_string(index=False))


# ============================================================
# YEAR-BY-YEAR SUMMARY
# ============================================================

year_rows = []

for year in sorted(trades["year"].unique()):
    for sf in STOP_FRACTIONS:
        g = trades[
            (trades["year"] == year) &
            (trades["stop_fraction"] == sf)
        ]

        if g.empty:
            continue

        s = summarize_trades(g)
        s["year"] = int(year)
        s["stop_fraction"] = sf
        year_rows.append(s)

year_summary = pd.DataFrame(year_rows)

print("\n============================================")
print("YEAR-BY-YEAR — ALL EVENTS")
print("============================================")
print(year_summary.to_string(index=False))


# ============================================================
# STRONG-CONFIDENCE SUBSET: NEAR/FAR <= 0.50
# ============================================================

strong = trades[trades["near_far_ratio"] <= 0.50].copy()

strong_overall_rows = []

for sf in STOP_FRACTIONS:
    g = strong[strong["stop_fraction"] == sf]

    if g.empty:
        continue

    s = summarize_trades(g)
    s["stop_fraction"] = sf
    strong_overall_rows.append(s)

strong_overall_summary = pd.DataFrame(strong_overall_rows)

print("\n============================================")
print("STRONG CONFIDENCE: NEAR/FAR <= 0.50")
print("============================================")
print(strong_overall_summary.to_string(index=False))

strong_year_rows = []

for year in sorted(strong["year"].unique()):
    for sf in STOP_FRACTIONS:
        g = strong[
            (strong["year"] == year) &
            (strong["stop_fraction"] == sf)
        ]

        if g.empty:
            continue

        s = summarize_trades(g)
        s["year"] = int(year)
        s["stop_fraction"] = sf
        strong_year_rows.append(s)

strong_year_summary = pd.DataFrame(strong_year_rows)

print("\n============================================")
print("STRONG CONFIDENCE BY YEAR")
print("============================================")
print(strong_year_summary.to_string(index=False))


# ============================================================
# PATH QUALITY BY CONFIDENCE
# ============================================================

path_conf_rows = []

for bucket in confidence_order:
    g = resolved_paths[
        resolved_paths["confidence_bucket"] == bucket
    ]

    if g.empty:
        continue

    correct = g[g["target_was_first"] == True]

    path_conf_rows.append({
        "confidence_bucket": bucket,
        "resolved_events": len(g),
        "target_first_rate": (
            len(correct) / len(g) if len(g) else np.nan
        ),
        "median_mae_target_units_strict_correct_only": (
            correct["mae_target_units_strict"].median()
            if len(correct) else np.nan
        ),
        "median_mae_target_units_inclusive_correct_only": (
            correct["mae_target_units_inclusive"].median()
            if len(correct) else np.nan
        ),
    })

path_conf_summary = pd.DataFrame(path_conf_rows)


# ============================================================
# SAVE OUTPUTS
# ============================================================

safe_to_csv(paths, "H4_3_path_metrics.csv")
safe_to_csv(trades, "H4_3_trade_tests.csv")
safe_to_csv(mae_summary, "H4_3_mae_summary.csv")
safe_to_csv(overall_summary, "H4_3_overall_stop_summary.csv")
safe_to_csv(
    unambiguous_summary,
    "H4_3_unambiguous_stop_summary.csv"
)
safe_to_csv(
    confidence_summary,
    "H4_3_confidence_stop_summary.csv"
)
safe_to_csv(year_summary, "H4_3_year_stop_summary.csv")
safe_to_csv(
    strong_overall_summary,
    "H4_3_strong_confidence_overall_summary.csv"
)
safe_to_csv(
    strong_year_summary,
    "H4_3_strong_confidence_year_summary.csv"
)
safe_to_csv(
    path_conf_summary,
    "H4_3_path_quality_by_confidence.csv"
)

config_text = f"""H4.3 V3 — BTC MAGNET PATH QUALITY & TRADABILITY

INPUT EVENT FILE:
{EVENT_FILE.name}

STOP_FRACTIONS:
{STOP_FRACTIONS}

TRADE_HORIZON_HOURS:
{TRADE_HORIZON_HOURS}

PRIMARY PNL CONVENTION:
- WIN = +1.0 target-distance unit
- LOSS = -stop_fraction target-distance units
- AMBIGUOUS_SAME_BAR = conservatively treated as full stop loss
- TIME_EXIT = directional mark-to-market at final available 1m close

PATH-METRIC CONVENTION:
- strict_pre_terminal excludes the H4.2 terminal minute
- inclusive_terminal includes it
- this avoids pretending we know intraminute ordering

IMPORTANT:
- H4.2 magnets are NOT recalculated or optimized.
- Results are GROSS before fees, slippage, and funding.
- Checkpoints are saved every {CHECKPOINT_EVERY_EVENTS} events.
"""

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

(OUTPUT_FOLDER / "H4_3_config.txt").write_text(
    config_text,
    encoding="utf-8"
)

print("\n============================================")
print("FILES SAVED")
print("============================================")

for file in sorted(OUTPUT_FOLDER.iterdir()):
    print(file)

print("\nDone.")

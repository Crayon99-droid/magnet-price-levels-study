from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.4 — BTC PULLBACK ENTRY INTO A PREDICTED MAGNET
# ============================================================
#
# PURPOSE
# -------
# H4.3 showed:
#   - the nearer magnet is often the correct destination
#   - correct destination paths often make a meaningful adverse
#     excursion before reaching the magnet
#
# H4.4 tests whether WAITING for that adverse excursion can improve
# trade economics.
#
# This script does NOT rebuild or optimize H4.2 magnets.
# It uses the saved H4_2_magnet_events.csv exactly as generated.
#
# PRIMARY SAMPLE
# --------------
# Only H4.2 "strong-confidence" signals are tested:
#
#       nearer_distance / farther_distance <= 0.50
#
# PREDECLARED PULLBACK ENTRIES
# ----------------------------
# Pullback entry is placed against the predicted direction by:
#
#       0.25x, 0.50x, 0.75x
#
# of the ORIGINAL target distance D.
#
# LONG example:
#   signal price = P0
#   target       = P0 + D
#   pullback     = P0 - xD
#
# SHORT is symmetric.
#
# FIXED INVALIDATION / STOP
# -------------------------
# To avoid introducing another optimization grid, the stop is frozen at
# 1.00x ORIGINAL target distance on the adverse side of the signal:
#
# LONG stop  = P0 - 1.00D
# SHORT stop = P0 + 1.00D
#
# Therefore, after pullback fill:
#
# x=0.25 -> reward/risk = 1.25 / 0.75 = 1.667
# x=0.50 -> reward/risk = 1.50 / 0.50 = 3.000
# x=0.75 -> reward/risk = 1.75 / 0.25 = 7.000
#
# ENTRY VALIDITY
# --------------
# The pullback entry must occur BEFORE the original H4.2 first-magnet
# event terminates. If the original target or opposite magnet is touched
# first, there is no pullback trade.
#
# EXIT HORIZON
# ------------
# Once filled, the trade can run until target/stop or until 24 hours
# after the ORIGINAL H4.2 decision time. A remaining open trade is
# marked-to-market at the final available 1-minute close.
#
# INTRAMINUTE AMBIGUITY
# ---------------------
# With 1-minute OHLC we cannot know exact within-bar ordering.
# - If pullback entry and target are both touched in the fill minute,
#   the trade is marked AMBIGUOUS_FILL_BAR.
# - Primary conservative PnL treats this as a full stop loss.
# - Unambiguous-only diagnostics are saved separately.
#
# COSTS
# -----
# Results are first calculated GROSS, then cost sensitivity is reported
# at round-trip costs of:
#
#       1, 2, 3, 4, 5, 7.5, 10 bp
#
# No parameter optimization is performed in this script.
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

OUTPUT_FOLDER = SCRIPT_DIR / "H4_4_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

STRONG_CONFIDENCE_MAX_RATIO = 0.50

PULLBACK_FRACTIONS = [0.25, 0.50, 0.75]

# Frozen from the H4.3 family of tests.
STOP_FROM_SIGNAL_FRACTION = 1.00

TRADE_HORIZON_HOURS = 24

ROUND_TRIP_COST_BPS = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

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
    required = ["timestamp_utc", "open", "high", "low", "close"]

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
    return int(pd.Timestamp(value).as_unit("ns").value)


def first_true_index(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def summarize_executed(g):
    """
    Summary using conservative PnL, including time exits and treating
    ambiguous fill bars as full stop losses.
    """

    if g.empty:
        return None

    pnl_bps = g["gross_return_bps_conservative"].astype(float)
    pnl_r = g["pnl_R_conservative"].astype(float)

    gross_profit_r = pnl_r[pnl_r > 0].sum()
    gross_loss_r = abs(pnl_r[pnl_r < 0].sum())

    return {
        "executed_trades": len(g),
        "wins": int((g["outcome"] == "WIN").sum()),
        "losses": int((g["outcome"] == "LOSS").sum()),
        "time_exits": int((g["outcome"] == "TIME_EXIT").sum()),
        "ambiguous_fill_bar": int(
            (g["outcome"] == "AMBIGUOUS_FILL_BAR").sum()
        ),
        "win_rate_raw": float((g["outcome"] == "WIN").mean()),
        "mean_R_conservative": float(pnl_r.mean()),
        "profit_factor_R_conservative": (
            float(gross_profit_r / gross_loss_r)
            if gross_loss_r > 0 else np.inf
        ),
        "mean_gross_bps_per_executed_trade": float(pnl_bps.mean()),
        "median_gross_bps_per_executed_trade": float(pnl_bps.median()),
    }


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

missing = [c for c in required_event_columns if c not in events.columns]

if missing:
    raise ValueError(
        f"H4.2 event file is missing columns: {missing}"
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
    .dropna(subset=["decision_time", "entry_price"])
    .sort_values("decision_time")
    .reset_index(drop=True)
)

# Recompute ratio from saved distances.
events["near_far_ratio"] = np.minimum(
    events["upper_distance"],
    events["lower_distance"],
) / np.maximum(
    events["upper_distance"],
    events["lower_distance"],
)

strong_events = events[
    (events["nearer_side"].isin(["UPPER", "LOWER"])) &
    (events["near_far_ratio"] <= STRONG_CONFIDENCE_MAX_RATIO)
].copy()

print("\n============================================")
print("H4.2 SIGNALS")
print("============================================")
print(f"All H4.2 events:          {len(events):,}")
print(
    f"Strong-confidence events: {len(strong_events):,} "
    f"(near/far <= {STRONG_CONFIDENCE_MAX_RATIO:.2f})"
)

if strong_events.empty:
    raise RuntimeError("No strong-confidence H4.2 events found.")


# ============================================================
# LOAD BTC M1
# ============================================================

all_data = []

for filename in FILES:
    path = SCRIPT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing BTC file:\n{path}\n"
            "Put all four BTC ZIP files in the same folder as this script."
        )

    all_data.append(clean_dataframe(read_zip_csv(path)))

m1 = pd.concat(all_data, ignore_index=True)

m1 = (
    m1
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

m1["timestamp"] = m1["timestamp"].dt.as_unit("ns")

ts = m1["timestamp"].astype("int64").to_numpy()
high_arr = m1["high"].to_numpy(dtype=float)
low_arr = m1["low"].to_numpy(dtype=float)
close_arr = m1["close"].to_numpy(dtype=float)

print("\n============================================")
print("BTC DATA")
print("============================================")
print(f"Rows:  {len(m1):,}")
print(f"Start: {m1['timestamp'].min()}")
print(f"End:   {m1['timestamp'].max()}")


# ============================================================
# TIMESTAMP PREFLIGHT
# ============================================================

first_signal_time = strong_events["decision_time"].iloc[0]
first_signal_ns = timestamp_ns(first_signal_time)
first_idx = int(np.searchsorted(ts, first_signal_ns, side="left"))

print("\n============================================")
print("TIMESTAMP SANITY CHECK")
print("============================================")
print(f"First signal:  {first_signal_time}")
print(f"Search index:  {first_idx}")

if first_idx >= len(ts):
    raise RuntimeError(
        "Timestamp alignment failed: H4.2 signals fall outside BTC M1 data."
    )

print(f"Matched minute: {m1['timestamp'].iloc[first_idx]}")
print("Timestamp alignment: OK")


# ============================================================
# H4.4 SIMULATION
# ============================================================

signal_rows = []
trade_rows = []

print("\n============================================")
print("RUNNING H4.4 PULLBACK TEST")
print("============================================")

for signal_counter, ev in enumerate(
    strong_events.itertuples(index=False),
    start=1
):

    decision_time = ev.decision_time
    signal_price = float(ev.entry_price)
    atr15 = float(ev.atr15)

    upper_distance = float(ev.upper_distance)
    lower_distance = float(ev.lower_distance)

    if upper_distance <= 0 or lower_distance <= 0:
        continue

    if ev.nearer_side == "UPPER":
        direction = "LONG"
        target_price = float(ev.upper_band_low)
        opposite_magnet_price = float(ev.lower_band_high)
        target_distance = target_price - signal_price
        farther_distance = lower_distance

    else:
        direction = "SHORT"
        target_price = float(ev.lower_band_high)
        opposite_magnet_price = float(ev.upper_band_low)
        target_distance = signal_price - target_price
        farther_distance = upper_distance

    if target_distance <= 0 or farther_distance <= 0:
        continue

    ratio = target_distance / farther_distance

    # --------------------------------------------------------
    # Entry-validity window:
    # pullback must fill before H4.2 event terminates.
    # --------------------------------------------------------

    decision_ns = timestamp_ns(decision_time)
    start_idx = int(np.searchsorted(ts, decision_ns, side="left"))

    if start_idx >= len(ts):
        continue

    if pd.notna(ev.touch_time):
        validity_end_time = ev.touch_time
    else:
        validity_end_time = (
            decision_time + pd.Timedelta(hours=TRADE_HORIZON_HOURS)
        )

    validity_end_ns = timestamp_ns(validity_end_time)

    # Include the terminal minute so that same-bar pullback vs magnet
    # can be detected and explicitly marked ambiguous.
    validity_end_idx = int(
        np.searchsorted(ts, validity_end_ns, side="right")
    )
    validity_end_idx = min(validity_end_idx, len(ts))

    valid_highs = high_arr[start_idx:validity_end_idx]
    valid_lows = low_arr[start_idx:validity_end_idx]
    valid_ts = ts[start_idx:validity_end_idx]

    if valid_highs.size == 0:
        continue

    signal_record = {
        "event_id": ev.event_id,
        "decision_time": decision_time,
        "year": decision_time.year,
        "direction": direction,
        "signal_price": signal_price,
        "target_price": target_price,
        "opposite_magnet_price": opposite_magnet_price,
        "target_distance": target_distance,
        "farther_distance": farther_distance,
        "near_far_ratio": ratio,
        "target_distance_atr": (
            target_distance / atr15 if atr15 > 0 else np.nan
        ),
        "h4_2_first_touch": ev.first_touch,
        "h4_2_touch_time": ev.touch_time,
    }

    signal_rows.append(signal_record)

    # --------------------------------------------------------
    # Fixed stop from ORIGINAL signal price.
    # --------------------------------------------------------

    if direction == "LONG":
        fixed_stop_price = (
            signal_price -
            STOP_FROM_SIGNAL_FRACTION * target_distance
        )
    else:
        fixed_stop_price = (
            signal_price +
            STOP_FROM_SIGNAL_FRACTION * target_distance
        )

    # --------------------------------------------------------
    # Test each pullback depth independently.
    # --------------------------------------------------------

    for pullback_fraction in PULLBACK_FRACTIONS:

        if direction == "LONG":
            pullback_price = (
                signal_price -
                pullback_fraction * target_distance
            )

            fill_mask = valid_lows <= pullback_price
            target_in_validity_mask = valid_highs >= target_price

        else:
            pullback_price = (
                signal_price +
                pullback_fraction * target_distance
            )

            fill_mask = valid_highs >= pullback_price
            target_in_validity_mask = valid_lows <= target_price

        fill_rel_idx = first_true_index(fill_mask)
        target_validity_rel_idx = first_true_index(
            target_in_validity_mask
        )

        # No pullback fill before H4.2 event terminates.
        if fill_rel_idx is None:
            trade_rows.append({
                **signal_record,
                "pullback_fraction": pullback_fraction,
                "pullback_price": pullback_price,
                "stop_price": fixed_stop_price,
                "filled": False,
                "fill_time": pd.NaT,
                "pre_entry_status": "NOT_FILLED",
                "reward_distance": np.nan,
                "risk_distance": np.nan,
                "planned_reward_risk": np.nan,
                "outcome": "NO_TRADE",
                "exit_time": pd.NaT,
                "exit_price_conservative": np.nan,
                "pnl_R_conservative": np.nan,
                "gross_return_bps_conservative": np.nan,
            })
            continue

        # If target and pullback first occur on the same minute,
        # we cannot know if target happened before the limit fill.
        pre_entry_ambiguous = (
            target_validity_rel_idx is not None and
            target_validity_rel_idx == fill_rel_idx
        )

        fill_abs_idx = start_idx + fill_rel_idx
        fill_ns = int(ts[fill_abs_idx])
        fill_time = pd.Timestamp(fill_ns, tz="UTC")

        reward_distance = abs(target_price - pullback_price)
        risk_distance = abs(pullback_price - fixed_stop_price)

        if reward_distance <= 0 or risk_distance <= 0:
            continue

        planned_rr = reward_distance / risk_distance

        # ----------------------------------------------------
        # Full trade horizon ends 24h after original signal.
        # --------------------------------------------------------

        horizon_time = (
            decision_time + pd.Timedelta(hours=TRADE_HORIZON_HOURS)
        )
        horizon_ns = timestamp_ns(horizon_time)

        horizon_end_idx = int(
            np.searchsorted(ts, horizon_ns, side="left")
        )
        horizon_end_idx = min(horizon_end_idx, len(ts))

        trade_highs = high_arr[fill_abs_idx:horizon_end_idx]
        trade_lows = low_arr[fill_abs_idx:horizon_end_idx]
        trade_closes = close_arr[fill_abs_idx:horizon_end_idx]
        trade_ts = ts[fill_abs_idx:horizon_end_idx]

        if trade_highs.size == 0:
            continue

        # ----------------------------------------------------
        # Conservative treatment of fill-bar ambiguity.
        # ----------------------------------------------------

        if pre_entry_ambiguous:
            outcome = "AMBIGUOUS_FILL_BAR"
            exit_time = fill_time
            exit_price = fixed_stop_price
            pnl_R = -1.0

        else:
            if direction == "LONG":
                target_mask = trade_highs >= target_price
                stop_mask = trade_lows <= fixed_stop_price
            else:
                target_mask = trade_lows <= target_price
                stop_mask = trade_highs >= fixed_stop_price

            target_rel = first_true_index(target_mask)
            stop_rel = first_true_index(stop_mask)

            if target_rel is None and stop_rel is None:
                outcome = "TIME_EXIT"
                exit_rel = len(trade_closes) - 1
                exit_price = float(trade_closes[exit_rel])
                exit_time = pd.Timestamp(
                    int(trade_ts[exit_rel]),
                    tz="UTC"
                )

                if direction == "LONG":
                    pnl_price = exit_price - pullback_price
                else:
                    pnl_price = pullback_price - exit_price

                pnl_R = pnl_price / risk_distance

            elif target_rel is not None and stop_rel is None:
                outcome = "WIN"
                exit_rel = target_rel
                exit_price = target_price
                exit_time = pd.Timestamp(
                    int(trade_ts[exit_rel]),
                    tz="UTC"
                )
                pnl_R = planned_rr

            elif target_rel is None and stop_rel is not None:
                outcome = "LOSS"
                exit_rel = stop_rel
                exit_price = fixed_stop_price
                exit_time = pd.Timestamp(
                    int(trade_ts[exit_rel]),
                    tz="UTC"
                )
                pnl_R = -1.0

            else:
                if target_rel < stop_rel:
                    outcome = "WIN"
                    exit_rel = target_rel
                    exit_price = target_price
                    exit_time = pd.Timestamp(
                        int(trade_ts[exit_rel]),
                        tz="UTC"
                    )
                    pnl_R = planned_rr

                elif stop_rel < target_rel:
                    outcome = "LOSS"
                    exit_rel = stop_rel
                    exit_price = fixed_stop_price
                    exit_time = pd.Timestamp(
                        int(trade_ts[exit_rel]),
                        tz="UTC"
                    )
                    pnl_R = -1.0

                else:
                    # Same minute target and stop after fill.
                    outcome = "AMBIGUOUS_FILL_BAR"
                    exit_rel = target_rel
                    exit_price = fixed_stop_price
                    exit_time = pd.Timestamp(
                        int(trade_ts[exit_rel]),
                        tz="UTC"
                    )
                    pnl_R = -1.0

        # Gross return in actual notional basis points.
        if direction == "LONG":
            gross_return = (
                exit_price - pullback_price
            ) / pullback_price
        else:
            gross_return = (
                pullback_price - exit_price
            ) / pullback_price

        gross_return_bps = gross_return * 10000.0

        trade_rows.append({
            **signal_record,
            "pullback_fraction": pullback_fraction,
            "pullback_price": pullback_price,
            "stop_price": fixed_stop_price,
            "filled": True,
            "fill_time": fill_time,
            "pre_entry_status": (
                "AMBIGUOUS_TARGET_AND_FILL_SAME_BAR"
                if pre_entry_ambiguous
                else "VALID_FILL"
            ),
            "reward_distance": reward_distance,
            "risk_distance": risk_distance,
            "planned_reward_risk": planned_rr,
            "outcome": outcome,
            "exit_time": exit_time,
            "exit_price_conservative": exit_price,
            "pnl_R_conservative": pnl_R,
            "gross_return_bps_conservative": gross_return_bps,
        })

    if signal_counter % CHECKPOINT_EVERY_SIGNALS == 0:
        p1 = safe_to_csv(
            pd.DataFrame(signal_rows),
            "H4_4_CHECKPOINT_signals.csv"
        )
        p2 = safe_to_csv(
            pd.DataFrame(trade_rows),
            "H4_4_CHECKPOINT_trades.csv"
        )

        print(
            f"  Processed {signal_counter:,}/{len(strong_events):,} "
            f"strong-confidence signals — checkpoint saved."
        )


signals = pd.DataFrame(signal_rows)
trades = pd.DataFrame(trade_rows)

if signals.empty or trades.empty:
    raise RuntimeError("No H4.4 observations were generated.")


# ============================================================
# FILL-RATE SUMMARY
# ============================================================

fill_rows = []

for pf in PULLBACK_FRACTIONS:
    g = trades[trades["pullback_fraction"] == pf]

    signals_tested = len(g)
    fills = int(g["filled"].sum())

    fill_rows.append({
        "pullback_fraction": pf,
        "signals_tested": signals_tested,
        "fills": fills,
        "fill_rate": fills / signals_tested if signals_tested else np.nan,
        "no_trade_count": int((g["outcome"] == "NO_TRADE").sum()),
    })

fill_summary = pd.DataFrame(fill_rows)

print("\n============================================")
print("H4.4 PULLBACK FILL RATES")
print("============================================")
print(fill_summary.to_string(index=False))


# ============================================================
# EXECUTED TRADE SUMMARY
# ============================================================

executed = trades[trades["filled"] == True].copy()

overall_rows = []

for pf in PULLBACK_FRACTIONS:
    g = executed[executed["pullback_fraction"] == pf]

    if g.empty:
        continue

    s = summarize_executed(g)

    s["pullback_fraction"] = pf
    s["signals_available"] = int(
        (trades["pullback_fraction"] == pf).sum()
    )
    s["fill_rate"] = (
        s["executed_trades"] / s["signals_available"]
    )
    s["mean_gross_bps_per_original_signal"] = (
        g["gross_return_bps_conservative"].sum() /
        s["signals_available"]
    )
    s["planned_reward_risk"] = float(
        g["planned_reward_risk"].iloc[0]
    )

    overall_rows.append(s)

overall_summary = pd.DataFrame(overall_rows)

print("\n============================================")
print("H4.4 EXECUTED TRADE RESULTS")
print("CONSERVATIVE AMBIGUITY TREATMENT")
print("============================================")
print(overall_summary.to_string(index=False))


# ============================================================
# UNAMBIGUOUS-ONLY DIAGNOSTIC
# ============================================================

unambiguous = executed[
    executed["outcome"] != "AMBIGUOUS_FILL_BAR"
].copy()

unamb_rows = []

for pf in PULLBACK_FRACTIONS:
    g = unambiguous[
        unambiguous["pullback_fraction"] == pf
    ]

    if g.empty:
        continue

    s = summarize_executed(g)
    s["pullback_fraction"] = pf
    unamb_rows.append(s)

unambiguous_summary = pd.DataFrame(unamb_rows)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

year_rows = []

for year in sorted(executed["year"].unique()):
    for pf in PULLBACK_FRACTIONS:

        g = executed[
            (executed["year"] == year) &
            (executed["pullback_fraction"] == pf)
        ]

        if g.empty:
            continue

        s = summarize_executed(g)
        s["year"] = int(year)
        s["pullback_fraction"] = pf

        yearly_signals = int(
            (
                (trades["year"] == year) &
                (trades["pullback_fraction"] == pf)
            ).sum()
        )

        s["signals_available"] = yearly_signals
        s["fill_rate"] = (
            s["executed_trades"] / yearly_signals
            if yearly_signals else np.nan
        )

        year_rows.append(s)

year_summary = pd.DataFrame(year_rows)

print("\n============================================")
print("H4.4 YEAR-BY-YEAR")
print("============================================")
print(year_summary.to_string(index=False))


# ============================================================
# COST SENSITIVITY
# ============================================================

cost_rows = []

for pf in PULLBACK_FRACTIONS:

    g = executed[
        executed["pullback_fraction"] == pf
    ].copy()

    if g.empty:
        continue

    gross_bps = g["gross_return_bps_conservative"].astype(float)

    for cost_bps in ROUND_TRIP_COST_BPS:
        net = gross_bps - cost_bps

        cost_rows.append({
            "pullback_fraction": pf,
            "round_trip_cost_bps": cost_bps,
            "executed_trades": len(g),
            "mean_gross_bps": gross_bps.mean(),
            "mean_net_bps": net.mean(),
            "median_net_bps": net.median(),
            "net_positive_trade_rate": (net > 0).mean(),
            "total_net_bps": net.sum(),
        })

cost_summary = pd.DataFrame(cost_rows)

print("\n============================================")
print("H4.4 COST SENSITIVITY")
print("============================================")
print(cost_summary.to_string(index=False))


# ============================================================
# COST SENSITIVITY BY YEAR
# ============================================================

cost_year_rows = []

for year in sorted(executed["year"].unique()):
    for pf in PULLBACK_FRACTIONS:

        g = executed[
            (executed["year"] == year) &
            (executed["pullback_fraction"] == pf)
        ].copy()

        if g.empty:
            continue

        gross_bps = g["gross_return_bps_conservative"].astype(float)

        for cost_bps in ROUND_TRIP_COST_BPS:
            net = gross_bps - cost_bps

            cost_year_rows.append({
                "year": int(year),
                "pullback_fraction": pf,
                "round_trip_cost_bps": cost_bps,
                "executed_trades": len(g),
                "mean_gross_bps": gross_bps.mean(),
                "mean_net_bps": net.mean(),
                "total_net_bps": net.sum(),
            })

cost_year_summary = pd.DataFrame(cost_year_rows)


# ============================================================
# SAVE EVERYTHING
# ============================================================

safe_to_csv(signals, "H4_4_signals.csv")
safe_to_csv(trades, "H4_4_pullback_trades.csv")
safe_to_csv(fill_summary, "H4_4_fill_summary.csv")
safe_to_csv(overall_summary, "H4_4_overall_summary.csv")
safe_to_csv(
    unambiguous_summary,
    "H4_4_unambiguous_summary.csv"
)
safe_to_csv(year_summary, "H4_4_year_summary.csv")
safe_to_csv(cost_summary, "H4_4_cost_sensitivity.csv")
safe_to_csv(
    cost_year_summary,
    "H4_4_cost_sensitivity_by_year.csv"
)

config_text = f"""H4.4 — BTC PULLBACK ENTRY INTO A PREDICTED MAGNET

INPUT:
{EVENT_FILE.name}

PRIMARY SAMPLE:
nearer_distance / farther_distance <= {STRONG_CONFIDENCE_MAX_RATIO}

PULLBACK_FRACTIONS:
{PULLBACK_FRACTIONS}

STOP_FROM_SIGNAL_FRACTION:
{STOP_FROM_SIGNAL_FRACTION}

TRADE_HORIZON_HOURS:
{TRADE_HORIZON_HOURS}

ROUND_TRIP_COST_BPS:
{ROUND_TRIP_COST_BPS}

ENTRY RULE:
- nearer H4.2 magnet defines direction/target
- pullback is placed against target direction by x * original target distance
- pullback must fill before original H4.2 first-magnet event terminates

STOP RULE:
- stop is frozen at 1.00 * original target distance on adverse side of
  the ORIGINAL signal price
- no stop optimization grid is used

AMBIGUITY:
- same-minute fill + target/stop ambiguity is treated conservatively
  as a full stop loss in primary results
- unambiguous-only diagnostics are saved separately

TIME EXIT:
- remaining open trades are marked-to-market at final available 1m close
  at 24 hours after original H4.2 signal

COSTS:
- gross results plus sensitivity to predeclared round-trip costs
- no funding model included yet

NO PARAMETER OPTIMIZATION IS PERFORMED.
"""

(OUTPUT_FOLDER / "H4_4_config.txt").write_text(
    config_text,
    encoding="utf-8"
)

print("\n============================================")
print("FILES SAVED")
print("============================================")

for file in sorted(OUTPUT_FOLDER.iterdir()):
    print(file)

print("\nDone.")

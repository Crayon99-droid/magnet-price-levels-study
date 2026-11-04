from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.5B — BTC CONFIRMED MAGNET FAILURE REVERSAL
# ============================================================
#
# PURPOSE
# -------
# H4.5 reversed immediately when an adverse threshold was touched.
# That created heavy same-minute ambiguity.
#
# H4.5B changes ONLY the entry timing:
#
#   1) A strong-confidence H4.2 signal exists.
#   2) Price touches an adverse threshold:
#        0.25D / 0.50D / 0.75D
#   3) Wait for that 1-minute candle to CLOSE.
#   4) If neither H4.2 magnet has resolved on that same minute,
#      enter the REVERSAL at the NEXT 1-minute OPEN.
#
# Everything else is deliberately frozen:
#
#   - reversal direction = opposite original magnet direction
#   - reversal target = original H4.2 opposite magnet
#   - reversal stop = original H4.2 signal price
#   - horizon = 24h from original H4.2 signal
#
# This is intended to answer one clean question:
#
#   Does confirmation at the end of the adverse-trigger minute
#   make the failure reversal tradable?
#
# NO PARAMETER OPTIMIZATION.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

H44_FILE = SCRIPT_DIR / "H4_4_pullback_trades.csv"

FILES = [
    "BTCUSDT_1m_2021.csv.zip",
    "BTCUSDT_1m_2022.csv.zip",
    "BTCUSDT_1m_2023.csv.zip",
    "BTCUSDT_1m_2024.csv.zip",
]

OUTPUT_FOLDER = SCRIPT_DIR / "H4_5B_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

PULLBACK_FRACTIONS = [0.25, 0.50, 0.75]

TRADE_HORIZON_HOURS = 24

ROUND_TRIP_COST_BPS = [
    1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0
]

CHECKPOINT_EVERY_CONFIRMED_TRIGGERS = 500


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
    return int(pd.Timestamp(value).as_unit("ns").value)


def first_true_index(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def classify_original_destination(row):
    """
    Separate correct / wrong / unresolved rather than calling every
    non-correct observation 'wrong'.
    """
    ft = str(row["h4_2_first_touch"])
    direction = str(row["direction"])

    if ft not in ["UPPER", "LOWER"]:
        return "UNRESOLVED_OR_AMBIGUOUS"

    if (
        (direction == "LONG" and ft == "UPPER") or
        (direction == "SHORT" and ft == "LOWER")
    ):
        return "CORRECT"

    return "WRONG"


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
        "ambiguous_post_entry_bar": int(
            (g["outcome"] == "AMBIGUOUS_POST_ENTRY_BAR").sum()
        ),
        "win_rate_raw": float(
            (g["outcome"] == "WIN").mean()
        ),
        "mean_R_conservative": float(pnl_r.mean()),
        "profit_factor_R_conservative": (
            float(gp / gl) if gl > 0 else np.inf
        ),
        "mean_gross_bps": float(pnl_bps.mean()),
        "median_gross_bps": float(pnl_bps.median()),
    }


# ============================================================
# LOAD H4.4 TRIGGER POPULATION
# ============================================================

if not H44_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{H44_FILE}\n\n"
        "Put H4_4_pullback_trades.csv in the same folder as this script."
    )

h44 = pd.read_csv(H44_FILE)

required = [
    "event_id",
    "decision_time",
    "year",
    "direction",
    "signal_price",
    "target_price",
    "opposite_magnet_price",
    "target_distance",
    "farther_distance",
    "near_far_ratio",
    "pullback_fraction",
    "pullback_price",
    "filled",
    "fill_time",
    "h4_2_first_touch",
    "h4_2_touch_time",
]

missing = [c for c in required if c not in h44.columns]

if missing:
    raise ValueError(
        f"H4.4 file is missing required columns: {missing}"
    )

h44["decision_time"] = pd.to_datetime(
    h44["decision_time"],
    utc=True,
    errors="coerce"
).dt.as_unit("ns")

h44["fill_time"] = pd.to_datetime(
    h44["fill_time"],
    utc=True,
    errors="coerce"
).dt.as_unit("ns")

h44["h4_2_touch_time"] = pd.to_datetime(
    h44["h4_2_touch_time"],
    utc=True,
    errors="coerce"
).dt.as_unit("ns")

if h44["filled"].dtype != bool:
    h44["filled"] = (
        h44["filled"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

h44["original_destination_class"] = h44.apply(
    classify_original_destination,
    axis=1
)

raw_triggers = h44[
    (h44["filled"] == True) &
    (h44["pullback_fraction"].isin(PULLBACK_FRACTIONS))
].copy()

raw_triggers = (
    raw_triggers
    .dropna(
        subset=[
            "decision_time",
            "fill_time",
            "signal_price",
            "pullback_price",
        ]
    )
    .sort_values(["decision_time", "pullback_fraction"])
    .reset_index(drop=True)
)

print("\n============================================")
print("H4.5B RAW ADVERSE TRIGGERS")
print("============================================")
print(f"Rows: {len(raw_triggers):,}")

for pf in PULLBACK_FRACTIONS:
    g = raw_triggers[
        raw_triggers["pullback_fraction"] == pf
    ]
    print(f"  {pf:.2f}D: {len(g):,}")


# ============================================================
# CONFIRMATION FILTER
# ============================================================
#
# A trigger is confirmed only if the original H4.2 event has NOT
# resolved on or before the adverse trigger minute.
#
# If h4_2_touch_time == fill_time, then a magnet was touched in the
# same minute as the adverse trigger. We cannot know ordering within
# that minute, so we DO NOT trade it.
#
# If touch_time is NaT (timeout), the adverse trigger can still confirm.
# ============================================================

def is_confirmed_trigger(row):
    fill_time = row["fill_time"]
    touch_time = row["h4_2_touch_time"]

    if pd.isna(fill_time):
        return False

    if pd.isna(touch_time):
        return True

    return touch_time > fill_time


raw_triggers["confirmed_trigger"] = raw_triggers.apply(
    is_confirmed_trigger,
    axis=1
)

confirmed = raw_triggers[
    raw_triggers["confirmed_trigger"] == True
].copy()

print("\n============================================")
print("CONFIRMED TRIGGERS")
print("============================================")
print(f"Confirmed: {len(confirmed):,}")

confirmation_rows = []

for pf in PULLBACK_FRACTIONS:
    raw = raw_triggers[
        raw_triggers["pullback_fraction"] == pf
    ]
    conf = confirmed[
        confirmed["pullback_fraction"] == pf
    ]

    row = {
        "pullback_fraction": pf,
        "raw_triggers": len(raw),
        "confirmed_triggers": len(conf),
        "confirmation_rate": (
            len(conf) / len(raw) if len(raw) else np.nan
        ),
    }

    for cls in [
        "CORRECT",
        "WRONG",
        "UNRESOLVED_OR_AMBIGUOUS",
    ]:
        raw_cls = raw[
            raw["original_destination_class"] == cls
        ]
        conf_cls = conf[
            conf["original_destination_class"] == cls
        ]

        row[f"raw_{cls.lower()}"] = len(raw_cls)
        row[f"confirmed_{cls.lower()}"] = len(conf_cls)

        row[f"P_confirmed_given_{cls.lower()}"] = (
            len(conf_cls) / len(raw_cls)
            if len(raw_cls) else np.nan
        )

    if len(conf):
        row["P_wrong_given_confirmed"] = (
            (
                conf["original_destination_class"]
                == "WRONG"
            ).mean()
        )
        row["P_correct_given_confirmed"] = (
            (
                conf["original_destination_class"]
                == "CORRECT"
            ).mean()
        )
    else:
        row["P_wrong_given_confirmed"] = np.nan
        row["P_correct_given_confirmed"] = np.nan

    confirmation_rows.append(row)

confirmation_summary = pd.DataFrame(
    confirmation_rows
)

print(confirmation_summary.to_string(index=False))


# ============================================================
# LOAD BTC M1
# ============================================================

all_data = []

for filename in FILES:
    path = SCRIPT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing BTC file:\n{path}\n"
            "Put all four BTC ZIP files beside this script."
        )

    raw = read_zip_csv(path)
    all_data.append(clean_dataframe(raw))

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
print("BTC DATA")
print("============================================")
print(f"Rows:  {len(m1):,}")
print(f"Start: {m1['timestamp'].min()}")
print(f"End:   {m1['timestamp'].max()}")


# ============================================================
# PREFLIGHT
# ============================================================

if confirmed.empty:
    raise RuntimeError(
        "No confirmed H4.5B triggers survived the close-confirmation rule."
    )

first_fill = confirmed["fill_time"].iloc[0]
first_fill_ns = timestamp_ns(first_fill)

first_idx = int(
    np.searchsorted(ts, first_fill_ns, side="left")
)

print("\n============================================")
print("TIMESTAMP SANITY CHECK")
print("============================================")
print(f"First confirmed trigger: {first_fill}")
print(f"Search index:            {first_idx}")

if first_idx >= len(ts):
    raise RuntimeError(
        "Timestamp alignment failed."
    )

print(
    f"Matched trigger minute:  "
    f"{m1['timestamp'].iloc[first_idx]}"
)
print("Timestamp alignment: OK")


# ============================================================
# SIMULATE NEXT-MINUTE-OPEN REVERSAL
# ============================================================

rows = []

print("\n============================================")
print("RUNNING H4.5B REVERSALS")
print("============================================")

for counter, tr in enumerate(
    confirmed.itertuples(index=False),
    start=1
):

    original_direction = tr.direction

    if original_direction == "LONG":
        reversal_direction = "SHORT"
    elif original_direction == "SHORT":
        reversal_direction = "LONG"
    else:
        continue

    signal_price = float(tr.signal_price)
    opposite_magnet = float(tr.opposite_magnet_price)
    fill_time = tr.fill_time
    decision_time = tr.decision_time
    pf = float(tr.pullback_fraction)

    # --------------------------------------------------------
    # Entry = next 1-minute OPEN after trigger minute.
    # --------------------------------------------------------

    fill_ns = timestamp_ns(fill_time)
    fill_idx = int(
        np.searchsorted(ts, fill_ns, side="left")
    )

    next_idx = fill_idx + 1

    if next_idx >= len(ts):
        continue

    next_time = pd.Timestamp(
        int(ts[next_idx]),
        tz="UTC"
    )

    # Require the next bar to be exactly one minute later.
    # If data are missing, skip rather than invent an entry.
    if (
        next_time
        != fill_time + pd.Timedelta(minutes=1)
    ):
        rows.append({
            "event_id": tr.event_id,
            "decision_time": decision_time,
            "year": int(tr.year),
            "pullback_fraction": pf,
            "original_direction": original_direction,
            "reversal_direction": reversal_direction,
            "trigger_time": fill_time,
            "entry_time": next_time,
            "signal_price": signal_price,
            "entry_price": np.nan,
            "reversal_stop": signal_price,
            "reversal_target": opposite_magnet,
            "original_destination_class": tr.original_destination_class,
            "trade_status": "SKIPPED_MISSING_NEXT_MINUTE",
            "outcome": "NO_TRADE",
            "planned_reward_risk": np.nan,
            "exit_time": pd.NaT,
            "exit_price_conservative": np.nan,
            "pnl_R_conservative": np.nan,
            "gross_return_bps_conservative": np.nan,
        })
        continue

    entry_price = float(open_arr[next_idx])
    reversal_stop = signal_price
    reversal_target = opposite_magnet

    # --------------------------------------------------------
    # Geometry check at actual next-minute open.
    # If the market has already crossed stop or target before we
    # can enter, there is no executable trade.
    # --------------------------------------------------------

    if reversal_direction == "SHORT":
        if entry_price >= reversal_stop:
            trade_status = "NO_TRADE_STOP_PASSED_BEFORE_ENTRY"
        elif entry_price <= reversal_target:
            trade_status = "NO_TRADE_TARGET_PASSED_BEFORE_ENTRY"
        else:
            trade_status = "EXECUTED"
    else:
        if entry_price <= reversal_stop:
            trade_status = "NO_TRADE_STOP_PASSED_BEFORE_ENTRY"
        elif entry_price >= reversal_target:
            trade_status = "NO_TRADE_TARGET_PASSED_BEFORE_ENTRY"
        else:
            trade_status = "EXECUTED"

    if trade_status != "EXECUTED":
        rows.append({
            "event_id": tr.event_id,
            "decision_time": decision_time,
            "year": int(tr.year),
            "pullback_fraction": pf,
            "original_direction": original_direction,
            "reversal_direction": reversal_direction,
            "trigger_time": fill_time,
            "entry_time": next_time,
            "signal_price": signal_price,
            "entry_price": entry_price,
            "reversal_stop": reversal_stop,
            "reversal_target": reversal_target,
            "original_destination_class": tr.original_destination_class,
            "trade_status": trade_status,
            "outcome": "NO_TRADE",
            "planned_reward_risk": np.nan,
            "exit_time": pd.NaT,
            "exit_price_conservative": np.nan,
            "pnl_R_conservative": np.nan,
            "gross_return_bps_conservative": np.nan,
        })
        continue

    risk_distance = abs(
        entry_price - reversal_stop
    )
    reward_distance = abs(
        reversal_target - entry_price
    )

    if risk_distance <= 0 or reward_distance <= 0:
        continue

    planned_rr = reward_distance / risk_distance

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

    if next_idx >= end_idx:
        continue

    highs = high_arr[next_idx:end_idx]
    lows = low_arr[next_idx:end_idx]
    closes = close_arr[next_idx:end_idx]
    trade_ts = ts[next_idx:end_idx]

    if highs.size == 0:
        continue

    if reversal_direction == "SHORT":
        target_mask = lows <= reversal_target
        stop_mask = highs >= reversal_stop
    else:
        target_mask = highs >= reversal_target
        stop_mask = lows <= reversal_stop

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

        if reversal_direction == "SHORT":
            pnl_price = entry_price - exit_price
        else:
            pnl_price = exit_price - entry_price

        pnl_R = pnl_price / risk_distance

    elif target_rel is not None and stop_rel is None:
        outcome = "WIN"
        exit_rel = target_rel
        exit_price = reversal_target

        exit_time = pd.Timestamp(
            int(trade_ts[exit_rel]),
            tz="UTC"
        )

        pnl_R = planned_rr

    elif target_rel is None and stop_rel is not None:
        outcome = "LOSS"
        exit_rel = stop_rel
        exit_price = reversal_stop

        exit_time = pd.Timestamp(
            int(trade_ts[exit_rel]),
            tz="UTC"
        )

        pnl_R = -1.0

    else:
        if target_rel < stop_rel:
            outcome = "WIN"
            exit_rel = target_rel
            exit_price = reversal_target

            exit_time = pd.Timestamp(
                int(trade_ts[exit_rel]),
                tz="UTC"
            )

            pnl_R = planned_rr

        elif stop_rel < target_rel:
            outcome = "LOSS"
            exit_rel = stop_rel
            exit_price = reversal_stop

            exit_time = pd.Timestamp(
                int(trade_ts[exit_rel]),
                tz="UTC"
            )

            pnl_R = -1.0

        else:
            # Both target and stop in same post-entry minute.
            # Conservative primary treatment = stop loss.
            outcome = "AMBIGUOUS_POST_ENTRY_BAR"
            exit_rel = target_rel
            exit_price = reversal_stop

            exit_time = pd.Timestamp(
                int(trade_ts[exit_rel]),
                tz="UTC"
            )

            pnl_R = -1.0

    if reversal_direction == "SHORT":
        gross_return = (
            entry_price - exit_price
        ) / entry_price
    else:
        gross_return = (
            exit_price - entry_price
        ) / entry_price

    gross_bps = gross_return * 10000.0

    rows.append({
        "event_id": tr.event_id,
        "decision_time": decision_time,
        "year": int(tr.year),
        "pullback_fraction": pf,
        "original_direction": original_direction,
        "reversal_direction": reversal_direction,
        "trigger_time": fill_time,
        "entry_time": next_time,
        "signal_price": signal_price,
        "entry_price": entry_price,
        "reversal_stop": reversal_stop,
        "reversal_target": reversal_target,
        "risk_distance": risk_distance,
        "reward_distance": reward_distance,
        "planned_reward_risk": planned_rr,
        "original_destination_class": tr.original_destination_class,
        "trade_status": "EXECUTED",
        "outcome": outcome,
        "exit_time": exit_time,
        "exit_price_conservative": exit_price,
        "pnl_R_conservative": pnl_R,
        "gross_return_bps_conservative": gross_bps,
    })

    if (
        counter
        % CHECKPOINT_EVERY_CONFIRMED_TRIGGERS
        == 0
    ):
        safe_to_csv(
            pd.DataFrame(rows),
            "H4_5B_CHECKPOINT_confirmed_reversal_trades.csv"
        )

        print(
            f"  Processed "
            f"{counter:,}/{len(confirmed):,} "
            f"confirmed triggers — checkpoint saved."
        )


results = pd.DataFrame(rows)

if results.empty:
    raise RuntimeError(
        "No H4.5B observations were generated."
    )

executed = results[
    results["trade_status"] == "EXECUTED"
].copy()

if executed.empty:
    raise RuntimeError(
        "No executable H4.5B trades survived next-minute-open checks."
    )


# ============================================================
# EXECUTION SUMMARY
# ============================================================

execution_rows = []

for pf in PULLBACK_FRACTIONS:
    all_pf = results[
        results["pullback_fraction"] == pf
    ]
    ex_pf = executed[
        executed["pullback_fraction"] == pf
    ]

    execution_rows.append({
        "pullback_fraction": pf,
        "confirmed_triggers": len(all_pf),
        "executed_trades": len(ex_pf),
        "execution_rate_after_confirmation": (
            len(ex_pf) / len(all_pf)
            if len(all_pf) else np.nan
        ),
        "no_trade_stop_passed": int(
            (
                all_pf["trade_status"]
                == "NO_TRADE_STOP_PASSED_BEFORE_ENTRY"
            ).sum()
        ),
        "no_trade_target_passed": int(
            (
                all_pf["trade_status"]
                == "NO_TRADE_TARGET_PASSED_BEFORE_ENTRY"
            ).sum()
        ),
        "missing_next_minute": int(
            (
                all_pf["trade_status"]
                == "SKIPPED_MISSING_NEXT_MINUTE"
            ).sum()
        ),
    })

execution_summary = pd.DataFrame(
    execution_rows
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall_rows = []

for pf in PULLBACK_FRACTIONS:
    g = executed[
        executed["pullback_fraction"] == pf
    ]

    if g.empty:
        continue

    s = summarize_trades(g)
    s["pullback_fraction"] = pf
    s["mean_planned_RR"] = float(
        g["planned_reward_risk"].mean()
    )
    s["median_planned_RR"] = float(
        g["planned_reward_risk"].median()
    )

    overall_rows.append(s)

overall_summary = pd.DataFrame(
    overall_rows
)

print("\n============================================")
print("H4.5B OVERALL RESULTS")
print("============================================")
print(overall_summary.to_string(index=False))


# ============================================================
# UNAMBIGUOUS-ONLY DIAGNOSTIC
# ============================================================

unambiguous = executed[
    executed["outcome"]
    != "AMBIGUOUS_POST_ENTRY_BAR"
].copy()

unamb_rows = []

for pf in PULLBACK_FRACTIONS:
    g = unambiguous[
        unambiguous["pullback_fraction"] == pf
    ]

    if g.empty:
        continue

    s = summarize_trades(g)
    s["pullback_fraction"] = pf
    unamb_rows.append(s)

unambiguous_summary = pd.DataFrame(
    unamb_rows
)


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

        s = summarize_trades(g)
        s["year"] = int(year)
        s["pullback_fraction"] = pf
        s["mean_planned_RR"] = float(
            g["planned_reward_risk"].mean()
        )

        year_rows.append(s)

year_summary = pd.DataFrame(
    year_rows
)

print("\n============================================")
print("H4.5B YEAR-BY-YEAR")
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

    gross = (
        g["gross_return_bps_conservative"]
        .astype(float)
    )

    for cost in ROUND_TRIP_COST_BPS:
        net = gross - cost

        cost_rows.append({
            "pullback_fraction": pf,
            "round_trip_cost_bps": cost,
            "trades": len(g),
            "mean_gross_bps": gross.mean(),
            "mean_net_bps": net.mean(),
            "median_net_bps": net.median(),
            "total_net_bps": net.sum(),
        })

cost_summary = pd.DataFrame(
    cost_rows
)

print("\n============================================")
print("H4.5B COST SENSITIVITY")
print("============================================")
print(cost_summary.to_string(index=False))


# ============================================================
# COST BY YEAR
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

        gross = (
            g["gross_return_bps_conservative"]
            .astype(float)
        )

        for cost in ROUND_TRIP_COST_BPS:
            net = gross - cost

            cost_year_rows.append({
                "year": int(year),
                "pullback_fraction": pf,
                "round_trip_cost_bps": cost,
                "trades": len(g),
                "mean_gross_bps": gross.mean(),
                "mean_net_bps": net.mean(),
                "total_net_bps": net.sum(),
            })

cost_year_summary = pd.DataFrame(
    cost_year_rows
)


# ============================================================
# SAVE
# ============================================================

safe_to_csv(
    confirmation_summary,
    "H4_5B_confirmation_summary.csv"
)

safe_to_csv(
    execution_summary,
    "H4_5B_execution_summary.csv"
)

safe_to_csv(
    results,
    "H4_5B_confirmed_reversal_trades.csv"
)

safe_to_csv(
    overall_summary,
    "H4_5B_overall_summary.csv"
)

safe_to_csv(
    unambiguous_summary,
    "H4_5B_unambiguous_summary.csv"
)

safe_to_csv(
    year_summary,
    "H4_5B_year_summary.csv"
)

safe_to_csv(
    cost_summary,
    "H4_5B_cost_sensitivity.csv"
)

safe_to_csv(
    cost_year_summary,
    "H4_5B_cost_sensitivity_by_year.csv"
)

config_text = f"""H4.5B — BTC CONFIRMED MAGNET FAILURE REVERSAL

INPUT:
{H44_FILE.name}

ADVERSE TRIGGERS:
{PULLBACK_FRACTIONS}

CONFIRMATION RULE:
- adverse threshold must be touched
- wait until that 1-minute candle completes
- if H4.2 first-magnet touch occurred on or before that same minute,
  do not trade
- otherwise enter reversal at next 1-minute open

REVERSAL TARGET:
Original H4.2 opposite magnet.

REVERSAL STOP:
Original H4.2 signal price.

TRADE HORIZON:
{TRADE_HORIZON_HOURS} hours from original H4.2 signal.

ROUND-TRIP COST SENSITIVITY:
{ROUND_TRIP_COST_BPS}

AMBIGUITY AFTER ENTRY:
If reversal target and stop are both touched in the same post-entry
1-minute bar, primary result counts a full -1R loss.

NO PARAMETER OPTIMIZATION.
"""

(
    OUTPUT_FOLDER /
    "H4_5B_config.txt"
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

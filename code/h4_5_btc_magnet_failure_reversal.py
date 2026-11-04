from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.5 — BTC MAGNET FAILURE REVERSAL
# ============================================================
#
# PURPOSE
# -------
# H4.4 showed that adverse pullback thresholds are reached by almost
# every ultimately wrong-destination signal, but they are ALSO reached
# by many ultimately correct-destination signals.
#
# H4.5 asks:
#
#   Once a strong-confidence magnet signal moves adversely by
#   0.25D / 0.50D / 0.75D before its original magnet resolves,
#   does REVERSING direction produce an edge?
#
# This script reuses H4_4_pullback_trades.csv so the trigger population
# is EXACTLY the same as H4.4. It does not rebuild H4.2 magnets.
#
# TWO PREDECLARED REVERSAL TARGETS
# --------------------------------
#
# A) FAILURE_LEVEL
#    Reversal target = original H4.4 fixed invalidation level:
#      original LONG  -> reversal SHORT target = P0 - 1.00D
#      original SHORT -> reversal LONG  target = P0 + 1.00D
#
#    Reversal stop = original signal price P0.
#
#    Planned reward/risk:
#      trigger 0.25D -> 3.00R
#      trigger 0.50D -> 1.00R
#      trigger 0.75D -> 0.333R
#
# B) OPPOSITE_MAGNET
#    Reversal target = the original H4.2 opposite magnet.
#    Reversal stop   = original signal price P0.
#
#    This directly tests the "wrong destination" hypothesis.
#
# SAMPLE
# ------
# H4.4 already contains only H4.2 strong-confidence signals:
# near/far <= 0.50.
#
# TRIGGER
# -------
# A reversal exists only where H4.4 says filled == True.
# That means the adverse threshold was reached before the original
# H4.2 first-magnet event terminated.
#
# INTRAMINUTE AMBIGUITY
# ---------------------
# We cannot know within a 1-minute candle whether:
#   - reversal entry happened before stop/target, or
#   - stop/target happened before reversal entry.
#
# Therefore if the FILL MINUTE also touches either reversal stop or
# reversal target, the primary result marks it AMBIGUOUS_FILL_BAR and
# conservatively counts it as a full -1R loss.
#
# A separate unambiguous-only summary is also saved.
#
# HORIZON
# -------
# Exit target/stop, otherwise mark-to-market at 24 hours after the
# ORIGINAL H4.2 signal time.
#
# COSTS
# -----
# Gross results and cost sensitivity at 1,2,3,4,5,7.5,10 bp round trip.
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

OUTPUT_FOLDER = SCRIPT_DIR / "H4_5_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

PULLBACK_FRACTIONS = [0.25, 0.50, 0.75]

TARGET_MODELS = [
    "FAILURE_LEVEL",
    "OPPOSITE_MAGNET",
]

TRADE_HORIZON_HOURS = 24

ROUND_TRIP_COST_BPS = [
    1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0
]

CHECKPOINT_EVERY_TRIGGERS = 500


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
            n for n in z.namelist()
            if n.lower().endswith(".csv")
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


def summarize(g):
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
        "time_exits": int(
            (g["outcome"] == "TIME_EXIT").sum()
        ),
        "ambiguous_fill_bar": int(
            (g["outcome"] == "AMBIGUOUS_FILL_BAR").sum()
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
# LOAD H4.4 TRIGGER FILE
# ============================================================

if not H44_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{H44_FILE}\n\n"
        "Put H4_4_pullback_trades.csv in the same folder as this script."
    )

h44 = pd.read_csv(H44_FILE)

required_h44 = [
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
    "stop_price",
    "filled",
    "fill_time",
    "pre_entry_status",
    "h4_2_first_touch",
]

missing = [c for c in required_h44 if c not in h44.columns]

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

# CSV booleans may be read as bool or strings.
if h44["filled"].dtype != bool:
    h44["filled"] = (
        h44["filled"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

triggers = h44[
    (h44["filled"] == True) &
    (h44["pullback_fraction"].isin(PULLBACK_FRACTIONS))
].copy()

triggers = (
    triggers
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
print("H4.5 TRIGGERS LOADED")
print("============================================")
print(f"H4.4 rows:          {len(h44):,}")
print(f"Filled triggers:    {len(triggers):,}")

for pf in PULLBACK_FRACTIONS:
    n = int(
        (triggers["pullback_fraction"] == pf).sum()
    )
    print(f"  {pf:.2f}D trigger: {n:,}")


# ============================================================
# DIAGNOSTIC: 99% IS NOT P(WRONG | TRIGGER)
# ============================================================

def original_destination_correct(row):
    return (
        (
            row["direction"] == "LONG" and
            row["h4_2_first_touch"] == "UPPER"
        )
        or
        (
            row["direction"] == "SHORT" and
            row["h4_2_first_touch"] == "LOWER"
        )
    )


h44["original_destination_correct"] = h44.apply(
    original_destination_correct,
    axis=1
)

diagnostic_rows = []

for pf in PULLBACK_FRACTIONS:
    g = h44[h44["pullback_fraction"] == pf].copy()

    wrong = g[g["original_destination_correct"] == False]
    correct = g[g["original_destination_correct"] == True]
    filled = g[g["filled"] == True]

    diagnostic_rows.append({
        "pullback_fraction": pf,
        "all_signals": len(g),
        "wrong_destination_signals": len(wrong),
        "correct_destination_signals": len(correct),

        "P_trigger_given_wrong": (
            wrong["filled"].mean()
            if len(wrong) else np.nan
        ),

        "P_trigger_given_correct": (
            correct["filled"].mean()
            if len(correct) else np.nan
        ),

        "P_wrong_given_trigger": (
            (
                filled["original_destination_correct"]
                == False
            ).mean()
            if len(filled) else np.nan
        ),

        "P_correct_given_trigger": (
            (
                filled["original_destination_correct"]
                == True
            ).mean()
            if len(filled) else np.nan
        ),
    })

trigger_diagnostic = pd.DataFrame(
    diagnostic_rows
)

print("\n============================================")
print("TRIGGER DIAGNOSTIC")
print("============================================")
print(
    "P(trigger | wrong) is NOT the same as "
    "P(wrong | trigger)."
)
print(trigger_diagnostic.to_string(index=False))


# ============================================================
# LOAD BTC M1 DATA
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

first_fill = triggers["fill_time"].iloc[0]
first_fill_ns = timestamp_ns(first_fill)

first_idx = int(
    np.searchsorted(
        ts,
        first_fill_ns,
        side="left"
    )
)

print("\n============================================")
print("TIMESTAMP SANITY CHECK")
print("============================================")
print(f"First reversal trigger: {first_fill}")
print(f"Search index:           {first_idx}")

if first_idx >= len(ts):
    raise RuntimeError(
        "Timestamp alignment failed."
    )

print(
    f"Matched minute:         "
    f"{m1['timestamp'].iloc[first_idx]}"
)
print("Timestamp alignment: OK")


# ============================================================
# REVERSAL SIMULATION
# ============================================================

rows = []

print("\n============================================")
print("RUNNING H4.5 REVERSALS")
print("============================================")

for trigger_counter, tr in enumerate(
    triggers.itertuples(index=False),
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
    reversal_entry = float(tr.pullback_price)
    original_target = float(tr.target_price)
    opposite_magnet = float(tr.opposite_magnet_price)

    # H4.4 stop_price is exactly the original 1.0D invalidation.
    failure_level = float(tr.stop_price)

    pullback_fraction = float(tr.pullback_fraction)

    fill_time = tr.fill_time
    decision_time = tr.decision_time

    # Reversal stop is original signal price.
    reversal_stop = signal_price

    if reversal_direction == "SHORT":
        # Targets must be below reversal entry.
        target_map = {
            "FAILURE_LEVEL": failure_level,
            "OPPOSITE_MAGNET": opposite_magnet,
        }
    else:
        # Targets must be above reversal entry.
        target_map = {
            "FAILURE_LEVEL": failure_level,
            "OPPOSITE_MAGNET": opposite_magnet,
        }

    # Full horizon from original H4.2 decision.
    horizon_time = (
        decision_time +
        pd.Timedelta(hours=TRADE_HORIZON_HOURS)
    )

    fill_ns = timestamp_ns(fill_time)
    horizon_ns = timestamp_ns(horizon_time)

    fill_idx = int(
        np.searchsorted(ts, fill_ns, side="left")
    )

    end_idx = int(
        np.searchsorted(
            ts,
            horizon_ns,
            side="left"
        )
    )

    end_idx = min(end_idx, len(ts))

    if fill_idx >= end_idx:
        continue

    highs = high_arr[fill_idx:end_idx]
    lows = low_arr[fill_idx:end_idx]
    closes = close_arr[fill_idx:end_idx]
    trade_ts = ts[fill_idx:end_idx]

    if highs.size == 0:
        continue

    # Was the H4.4 trigger itself already marked ambiguous?
    pre_entry_ambiguous = (
        str(tr.pre_entry_status)
        ==
        "AMBIGUOUS_TARGET_AND_FILL_SAME_BAR"
    )

    for target_model in TARGET_MODELS:

        reversal_target = float(
            target_map[target_model]
        )

        # Geometry validation.
        if reversal_direction == "SHORT":
            if not (
                reversal_target < reversal_entry <
                reversal_stop
            ):
                continue
        else:
            if not (
                reversal_stop < reversal_entry <
                reversal_target
            ):
                continue

        risk_distance = abs(
            reversal_entry - reversal_stop
        )

        reward_distance = abs(
            reversal_target - reversal_entry
        )

        if risk_distance <= 0 or reward_distance <= 0:
            continue

        planned_rr = (
            reward_distance / risk_distance
        )

        # ----------------------------------------------------
        # Fill-bar ambiguity check.
        # ----------------------------------------------------

        fill_high = float(highs[0])
        fill_low = float(lows[0])

        if reversal_direction == "SHORT":
            fillbar_stop_touch = (
                fill_high >= reversal_stop
            )
            fillbar_target_touch = (
                fill_low <= reversal_target
            )
        else:
            fillbar_stop_touch = (
                fill_low <= reversal_stop
            )
            fillbar_target_touch = (
                fill_high >= reversal_target
            )

        fillbar_ambiguous = (
            pre_entry_ambiguous
            or fillbar_stop_touch
            or fillbar_target_touch
        )

        if fillbar_ambiguous:
            outcome = "AMBIGUOUS_FILL_BAR"
            exit_time = fill_time
            exit_price = reversal_stop
            pnl_R = -1.0

        else:
            # Start exit evaluation from minute AFTER fill.
            h = highs[1:]
            l = lows[1:]
            c = closes[1:]
            t = trade_ts[1:]

            if h.size == 0:
                continue

            if reversal_direction == "SHORT":
                target_mask = l <= reversal_target
                stop_mask = h >= reversal_stop
            else:
                target_mask = h >= reversal_target
                stop_mask = l <= reversal_stop

            target_rel = first_true_index(
                target_mask
            )

            stop_rel = first_true_index(
                stop_mask
            )

            if target_rel is None and stop_rel is None:
                outcome = "TIME_EXIT"

                exit_rel = len(c) - 1
                exit_price = float(c[exit_rel])

                exit_time = pd.Timestamp(
                    int(t[exit_rel]),
                    tz="UTC"
                )

                if reversal_direction == "SHORT":
                    pnl_price = (
                        reversal_entry - exit_price
                    )
                else:
                    pnl_price = (
                        exit_price - reversal_entry
                    )

                pnl_R = (
                    pnl_price / risk_distance
                )

            elif target_rel is not None and stop_rel is None:
                outcome = "WIN"
                exit_price = reversal_target

                exit_time = pd.Timestamp(
                    int(t[target_rel]),
                    tz="UTC"
                )

                pnl_R = planned_rr

            elif target_rel is None and stop_rel is not None:
                outcome = "LOSS"
                exit_price = reversal_stop

                exit_time = pd.Timestamp(
                    int(t[stop_rel]),
                    tz="UTC"
                )

                pnl_R = -1.0

            else:
                if target_rel < stop_rel:
                    outcome = "WIN"
                    exit_price = reversal_target

                    exit_time = pd.Timestamp(
                        int(t[target_rel]),
                        tz="UTC"
                    )

                    pnl_R = planned_rr

                elif stop_rel < target_rel:
                    outcome = "LOSS"
                    exit_price = reversal_stop

                    exit_time = pd.Timestamp(
                        int(t[stop_rel]),
                        tz="UTC"
                    )

                    pnl_R = -1.0

                else:
                    # Target and stop in same post-fill minute.
                    outcome = "AMBIGUOUS_FILL_BAR"
                    exit_price = reversal_stop

                    exit_time = pd.Timestamp(
                        int(t[target_rel]),
                        tz="UTC"
                    )

                    pnl_R = -1.0

        if reversal_direction == "SHORT":
            gross_return = (
                reversal_entry - exit_price
            ) / reversal_entry
        else:
            gross_return = (
                exit_price - reversal_entry
            ) / reversal_entry

        gross_bps = gross_return * 10000.0

        original_correct = (
            (
                original_direction == "LONG"
                and tr.h4_2_first_touch == "UPPER"
            )
            or
            (
                original_direction == "SHORT"
                and tr.h4_2_first_touch == "LOWER"
            )
        )

        rows.append({
            "event_id": tr.event_id,
            "decision_time": decision_time,
            "year": int(tr.year),
            "pullback_fraction": pullback_fraction,

            "original_direction": original_direction,
            "reversal_direction": reversal_direction,

            "signal_price": signal_price,
            "reversal_entry": reversal_entry,
            "reversal_stop": reversal_stop,
            "target_model": target_model,
            "reversal_target": reversal_target,

            "original_target": original_target,
            "opposite_magnet": opposite_magnet,
            "failure_level": failure_level,

            "risk_distance": risk_distance,
            "reward_distance": reward_distance,
            "planned_reward_risk": planned_rr,

            "fill_time": fill_time,
            "pre_entry_status": tr.pre_entry_status,

            "h4_2_first_touch": tr.h4_2_first_touch,
            "original_destination_correct": original_correct,

            "outcome": outcome,
            "exit_time": exit_time,
            "exit_price_conservative": exit_price,
            "pnl_R_conservative": pnl_R,
            "gross_return_bps_conservative": gross_bps,
        })

    if (
        trigger_counter
        % CHECKPOINT_EVERY_TRIGGERS
        == 0
    ):
        safe_to_csv(
            pd.DataFrame(rows),
            "H4_5_CHECKPOINT_reversal_trades.csv"
        )

        print(
            f"  Processed "
            f"{trigger_counter:,}/{len(triggers):,} "
            f"triggers — checkpoint saved."
        )


results = pd.DataFrame(rows)

if results.empty:
    raise RuntimeError(
        "No H4.5 reversal trades were generated."
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall_rows = []

for tm in TARGET_MODELS:
    for pf in PULLBACK_FRACTIONS:
        g = results[
            (results["target_model"] == tm) &
            (results["pullback_fraction"] == pf)
        ]

        if g.empty:
            continue

        s = summarize(g)

        s["target_model"] = tm
        s["pullback_fraction"] = pf
        s["mean_planned_RR"] = float(
            g["planned_reward_risk"].mean()
        )

        overall_rows.append(s)

overall = pd.DataFrame(overall_rows)

print("\n============================================")
print("H4.5 OVERALL RESULTS")
print("CONSERVATIVE AMBIGUITY TREATMENT")
print("============================================")
print(overall.to_string(index=False))


# ============================================================
# UNAMBIGUOUS ONLY
# ============================================================

clean = results[
    results["outcome"]
    != "AMBIGUOUS_FILL_BAR"
].copy()

clean_rows = []

for tm in TARGET_MODELS:
    for pf in PULLBACK_FRACTIONS:
        g = clean[
            (clean["target_model"] == tm) &
            (clean["pullback_fraction"] == pf)
        ]

        if g.empty:
            continue

        s = summarize(g)
        s["target_model"] = tm
        s["pullback_fraction"] = pf
        clean_rows.append(s)

unambiguous_summary = pd.DataFrame(clean_rows)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

year_rows = []

for year in sorted(results["year"].unique()):
    for tm in TARGET_MODELS:
        for pf in PULLBACK_FRACTIONS:

            g = results[
                (results["year"] == year) &
                (results["target_model"] == tm) &
                (results["pullback_fraction"] == pf)
            ]

            if g.empty:
                continue

            s = summarize(g)
            s["year"] = int(year)
            s["target_model"] = tm
            s["pullback_fraction"] = pf

            year_rows.append(s)

year_summary = pd.DataFrame(year_rows)

print("\n============================================")
print("H4.5 YEAR-BY-YEAR")
print("============================================")
print(year_summary.to_string(index=False))


# ============================================================
# COST SENSITIVITY
# ============================================================

cost_rows = []

for tm in TARGET_MODELS:
    for pf in PULLBACK_FRACTIONS:

        g = results[
            (results["target_model"] == tm) &
            (results["pullback_fraction"] == pf)
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
                "target_model": tm,
                "pullback_fraction": pf,
                "round_trip_cost_bps": cost,
                "trades": len(g),
                "mean_gross_bps": gross.mean(),
                "mean_net_bps": net.mean(),
                "median_net_bps": net.median(),
                "total_net_bps": net.sum(),
            })

cost_summary = pd.DataFrame(cost_rows)

print("\n============================================")
print("H4.5 COST SENSITIVITY")
print("============================================")
print(cost_summary.to_string(index=False))


# ============================================================
# COST BY YEAR
# ============================================================

cost_year_rows = []

for year in sorted(results["year"].unique()):
    for tm in TARGET_MODELS:
        for pf in PULLBACK_FRACTIONS:

            g = results[
                (results["year"] == year) &
                (results["target_model"] == tm) &
                (results["pullback_fraction"] == pf)
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
                    "target_model": tm,
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
    trigger_diagnostic,
    "H4_5_trigger_diagnostic.csv"
)

safe_to_csv(
    results,
    "H4_5_reversal_trades.csv"
)

safe_to_csv(
    overall,
    "H4_5_overall_summary.csv"
)

safe_to_csv(
    unambiguous_summary,
    "H4_5_unambiguous_summary.csv"
)

safe_to_csv(
    year_summary,
    "H4_5_year_summary.csv"
)

safe_to_csv(
    cost_summary,
    "H4_5_cost_sensitivity.csv"
)

safe_to_csv(
    cost_year_summary,
    "H4_5_cost_sensitivity_by_year.csv"
)

config_text = f"""H4.5 — BTC MAGNET FAILURE REVERSAL

INPUT:
{H44_FILE.name}

PULLBACK TRIGGERS:
{PULLBACK_FRACTIONS}

TARGET MODELS:
{TARGET_MODELS}

REVERSAL STOP:
Original H4.2/H4.4 signal price.

FAILURE_LEVEL TARGET:
Original H4.4 fixed invalidation level at 1.00D adverse
from the original signal.

OPPOSITE_MAGNET TARGET:
Original H4.2 opposite magnet.

TRADE HORIZON:
{TRADE_HORIZON_HOURS} hours from original H4.2 signal.

ROUND-TRIP COST SENSITIVITY:
{ROUND_TRIP_COST_BPS}

PRIMARY AMBIGUITY RULE:
If fill minute touches reversal stop or reversal target, or if H4.4
already marked target-and-fill on the same minute, count the trade
conservatively as a full -1R loss.

NO PARAMETER OPTIMIZATION.
"""

(
    OUTPUT_FOLDER /
    "H4_5_config.txt"
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

from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.G1B — DOES FRESH GMMA ADD EDGE BEYOND SIMPLE PRICE/GEOMETRY CONFIRMATION?
# ============================================================
#
# PURPOSE
# -------
# Fresh GMMA looked promising, but it may simply be waiting until price
# has already moved toward the intended target.
#
# This experiment compares Fresh GMMA against two deliberately simple,
# predeclared confirmation rules on the SAME H4.6A2-eligible signal pool:
#
#   A) FRESH_GMMA
#      Original frozen H4.6A2 entry.
#
#   B) PROGRESS_25
#      Enter after a completed M1 close has progressed 25% of the way
#      from the original H4.2 decision price toward the original target.
#
#   C) GEOM_PLUS_5PP
#      Recompute exact first-passage probability between the two frozen
#      H4.2 barriers after every completed M1 close. Enter after the
#      intended near-target probability has improved by +5 percentage
#      points versus its value at the H4.2 decision.
#
# These are NOT optimized thresholds. They are placebo / mechanism tests.
#
# All strategies:
#   - use the same H4.6A2 eligible (not already-aligned) population;
#   - confirmation must occur strictly before H4.2 first touch;
#   - enter at next M1 open;
#   - use the original intended near barrier as target;
#   - test 0.75D and 1.00D stops from ACTUAL entry;
#   - use the original 24h H4.2 horizon;
#   - treat same-M1 target+stop as STOP first (conservative).
#
# PRIMARY QUESTION
# ----------------
# Does Fresh GMMA outperform simple progress / dynamic-geometry
# confirmation on:
#   - excess TP probability over Brownian break-even,
#   - gross bp/trade,
#   - PF,
#   - year stability,
#   - costs,
#   - matched-event paired PnL?
#
# IMPORTANT
# ---------
# This remains BTC 2021-2024 development-sample research.
# A winner here is still a hypothesis requiring external/locked validation.
# ============================================================

BASE = Path(__file__).resolve().parent
OUT = BASE / "H4_G1B_results"
OUT.mkdir(exist_ok=True)

STOP_FRACTIONS = [0.75, 1.00]
ROUND_TRIP_COST_BPS = [1.0, 2.0, 3.0, 4.0, 5.0]
PROGRESS_FRACTION = 0.25
GEOMETRY_PROBABILITY_INCREMENT = 0.05
HORIZON_HOURS = 24

BOOT_REPS = 10000
BOOT_SEED = 20260908

BTC_FILES = [
    BASE / f"BTCUSDT_1m_{year}.csv.zip"
    for year in (2021, 2022, 2023, 2024)
]


# ============================================================
# FILE DISCOVERY
# ============================================================

def first_existing(candidates, description):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {description}.\nTried:\n"
        + "\n".join(str(p) for p in candidates)
    )


A2_DIAGNOSTICS = first_existing(
    [
        BASE / "H4_6A2_signal_diagnostics.csv",
        BASE / "H4_6A2_results" / "H4_6A2_signal_diagnostics.csv",
    ],
    "H4_6A2_signal_diagnostics.csv",
)

H4_2_EVENTS = first_existing(
    [
        BASE / "H4_2_magnet_events.csv",
        BASE / "H4_2_results" / "H4_2_magnet_events.csv",
    ],
    "H4_2_magnet_events.csv",
)


# ============================================================
# HELPERS
# ============================================================

def save(df, filename):
    path = OUT / filename
    df.to_csv(path, index=False)
    return path


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

    required = ["timestamp_utc", "open", "high", "low", "close"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns {missing}")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce"),
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "high": pd.to_numeric(raw["high"], errors="coerce"),
        "low": pd.to_numeric(raw["low"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
    }).dropna()

    return out.sort_values("timestamp").drop_duplicates("timestamp")


def ns(value):
    return int(pd.Timestamp(value).as_unit("ns").value)


def first_hit(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def profit_factor(values):
    v = np.asarray(values, dtype=float)
    gp = v[v > 0].sum()
    gl = -v[v < 0].sum()
    return float(gp / gl) if gl > 0 else np.inf


def target_geometry_probability(price, target, opposite, direction):
    """
    Exact zero-drift first-passage probability of hitting TARGET before
    OPPOSITE while price is strictly between the two frozen barriers.
    """
    if direction == "LONG":
        lower = opposite
        upper = target
        if not (lower < price < upper):
            return np.nan
        return (price - lower) / (upper - lower)

    lower = target
    upper = opposite
    if not (lower < price < upper):
        return np.nan
    return (upper - price) / (upper - lower)


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
    blocks = x.groupby("month")[value_col].agg(["sum", "count"])

    sums = blocks["sum"].to_numpy(float)
    counts = blocks["count"].to_numpy(float)

    rng = np.random.default_rng(BOOT_SEED)
    vals = np.empty(BOOT_REPS)

    for i in range(BOOT_REPS):
        j = rng.integers(0, len(blocks), len(blocks))
        vals[i] = sums[j].sum() / counts[j].sum()

    lo, hi = np.quantile(vals, [0.025, 0.975])

    return {
        "observations": len(x),
        "month_blocks": len(blocks),
        "mean": float(x[value_col].mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "fraction_le_0": float((vals <= 0).mean()),
    }


def paired_month_bootstrap(frame, value_col, label):
    """
    frame must already contain one row per matched event and a difference
    column where positive means GMMA better than comparator.
    """
    x = frame[["decision_time", value_col]].dropna().copy()

    if x.empty:
        return {
            "comparison": label,
            "matched_events": 0,
        }

    x["month"] = x["decision_time"].dt.strftime("%Y-%m")
    blocks = x.groupby("month")[value_col].agg(["sum", "count"])

    sums = blocks["sum"].to_numpy(float)
    counts = blocks["count"].to_numpy(float)

    rng = np.random.default_rng(BOOT_SEED)
    vals = np.empty(BOOT_REPS)

    for i in range(BOOT_REPS):
        j = rng.integers(0, len(blocks), len(blocks))
        vals[i] = sums[j].sum() / counts[j].sum()

    lo, hi = np.quantile(vals, [0.025, 0.975])

    return {
        "comparison": label,
        "matched_events": len(x),
        "month_blocks": len(blocks),
        "observed_mean_difference": float(x[value_col].mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "fraction_difference_le_0": float((vals <= 0).mean()),
    }


# ============================================================
# LOAD SIGNAL / EVENT DATA
# ============================================================

diag = pd.read_csv(A2_DIAGNOSTICS)
h42 = pd.read_csv(H4_2_EVENTS)

diag["decision_time"] = pd.to_datetime(
    diag["decision_time"], utc=True, errors="coerce"
)
diag["gmma_confirmation_time"] = pd.to_datetime(
    diag["gmma_confirmation_time"], utc=True, errors="coerce"
)
diag["entry_time"] = pd.to_datetime(
    diag["entry_time"], utc=True, errors="coerce"
)

h42["decision_time"] = pd.to_datetime(
    h42["decision_time"], utc=True, errors="coerce"
)
h42["touch_time"] = pd.to_datetime(
    h42["touch_time"], utc=True, errors="coerce"
)

# Fair comparison pool = H4.6A2 signals that were NOT already aligned.
eligible = diag[
    diag["already_aligned_before_signal"].astype(str).str.lower().eq("false")
].copy()

needed_h42 = [
    "event_id",
    "entry_price",
    "upper_band_low",
    "lower_band_high",
    "upper_distance",
    "lower_distance",
    "nearer_side",
    "first_touch",
    "touch_time",
]

eligible = eligible.merge(
    h42[needed_h42],
    on="event_id",
    how="left",
    validate="one_to_one",
)

eligible = eligible.dropna(
    subset=[
        "decision_time",
        "entry_price_y",
        "upper_band_low",
        "lower_band_high",
        "nearer_side",
    ]
).reset_index(drop=True)

print("\n============================================")
print("H4.G1B INPUT")
print("============================================")
print("A2 diagnostics:", A2_DIAGNOSTICS)
print("H4.2 events:", H4_2_EVENTS)
print(f"Eligible not-already-aligned signals: {len(eligible):,}")


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
open_arr = m1["open"].to_numpy(float)
high_arr = m1["high"].to_numpy(float)
low_arr = m1["low"].to_numpy(float)
close_arr = m1["close"].to_numpy(float)

print(f"BTC M1 rows: {len(m1):,}")


# ============================================================
# GENERATE ENTRY CANDIDATES
# ============================================================

entry_rows = []

for n, ev in enumerate(eligible.itertuples(index=False), 1):

    decision_time = pd.Timestamp(ev.decision_time)
    signal_price = float(ev.entry_price_y)

    if ev.nearer_side == "UPPER":
        direction = "LONG"
        target = float(ev.upper_band_low)
        opposite = float(ev.lower_band_high)
        side = 1.0

    elif ev.nearer_side == "LOWER":
        direction = "SHORT"
        target = float(ev.lower_band_high)
        opposite = float(ev.upper_band_low)
        side = -1.0

    else:
        continue

    signal_target_distance = side * (target - signal_price)

    if signal_target_distance <= 0:
        continue

    p0 = target_geometry_probability(
        signal_price, target, opposite, direction
    )

    if not np.isfinite(p0):
        continue

    # Search window ends strictly before original H4.2 first-touch minute.
    start_idx = int(np.searchsorted(tns, ns(decision_time), side="left"))

    horizon_time = decision_time + pd.Timedelta(hours=HORIZON_HOURS)
    horizon_idx = int(np.searchsorted(tns, ns(horizon_time), side="left"))

    if pd.notna(ev.touch_time):
        touch_idx = int(
            np.searchsorted(tns, ns(ev.touch_time), side="left")
        )
        search_end = min(touch_idx, horizon_idx, len(tns))
    else:
        search_end = min(horizon_idx, len(tns))

    if start_idx >= search_end:
        continue

    # --------------------------------------------------------
    # A) FRESH GMMA — frozen recorded H4.6A2 entry
    # --------------------------------------------------------
    if (
        str(ev.eligibility_status) == "EXECUTED_FRESH_GMMA"
        and pd.notna(ev.entry_time)
        and pd.notna(ev.entry_price_x)
    ):
        gm_entry_time = pd.Timestamp(ev.entry_time)
        gm_entry_price = float(ev.entry_price_x)

        gm_idx = int(np.searchsorted(tns, ns(gm_entry_time), side="left"))

        if gm_idx < len(tns):
            p_entry = target_geometry_probability(
                gm_entry_price, target, opposite, direction
            )

            progress = (
                side * (gm_entry_price - signal_price)
                / signal_target_distance
            )

            entry_rows.append({
                "event_id": ev.event_id,
                "decision_time": decision_time,
                "year": decision_time.year,
                "strategy": "FRESH_GMMA",
                "direction": direction,
                "signal_price": signal_price,
                "target_price": target,
                "opposite_barrier": opposite,
                "initial_target_geometry_probability": p0,
                "entry_time": gm_entry_time,
                "entry_price": gm_entry_price,
                "entry_geometry_probability": p_entry,
                "geometry_probability_increase": (
                    p_entry - p0 if np.isfinite(p_entry) else np.nan
                ),
                "progress_fraction_at_entry": progress,
                "minutes_signal_to_entry": (
                    (gm_entry_time - decision_time)
                    / pd.Timedelta(minutes=1)
                ),
            })

    # Search confirmation closes.
    closes = close_arr[start_idx:search_end]

    # --------------------------------------------------------
    # B) 25% simple progress placebo
    # --------------------------------------------------------
    progress_threshold_price = (
        signal_price
        + side * PROGRESS_FRACTION * signal_target_distance
    )

    if direction == "LONG":
        progress_mask = closes >= progress_threshold_price
    else:
        progress_mask = closes <= progress_threshold_price

    rel = first_hit(progress_mask)

    if rel is not None:
        conf_idx = start_idx + rel
        entry_idx = conf_idx + 1

        if entry_idx < len(tns):
            conf_time = pd.Timestamp(int(tns[conf_idx]), tz="UTC")
            entry_time = pd.Timestamp(int(tns[entry_idx]), tz="UTC")

            # Require exact next M1.
            if entry_time == conf_time + pd.Timedelta(minutes=1):
                entry_price = float(open_arr[entry_idx])

                # Target must still be ahead.
                if side * (target - entry_price) > 0:
                    p_entry = target_geometry_probability(
                        entry_price, target, opposite, direction
                    )

                    progress = (
                        side * (entry_price - signal_price)
                        / signal_target_distance
                    )

                    entry_rows.append({
                        "event_id": ev.event_id,
                        "decision_time": decision_time,
                        "year": decision_time.year,
                        "strategy": "PROGRESS_25",
                        "direction": direction,
                        "signal_price": signal_price,
                        "target_price": target,
                        "opposite_barrier": opposite,
                        "initial_target_geometry_probability": p0,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "entry_geometry_probability": p_entry,
                        "geometry_probability_increase": (
                            p_entry - p0 if np.isfinite(p_entry) else np.nan
                        ),
                        "progress_fraction_at_entry": progress,
                        "minutes_signal_to_entry": (
                            (entry_time - decision_time)
                            / pd.Timedelta(minutes=1)
                        ),
                    })

    # --------------------------------------------------------
    # C) Dynamic geometry confidence: +5 percentage points
    # --------------------------------------------------------
    needed_p = min(
        p0 + GEOMETRY_PROBABILITY_INCREMENT,
        0.999999,
    )

    geom_mask = np.zeros(len(closes), dtype=bool)

    for j, price in enumerate(closes):
        p_now = target_geometry_probability(
            float(price), target, opposite, direction
        )
        geom_mask[j] = (
            np.isfinite(p_now)
            and p_now >= needed_p
        )

    rel = first_hit(geom_mask)

    if rel is not None:
        conf_idx = start_idx + rel
        entry_idx = conf_idx + 1

        if entry_idx < len(tns):
            conf_time = pd.Timestamp(int(tns[conf_idx]), tz="UTC")
            entry_time = pd.Timestamp(int(tns[entry_idx]), tz="UTC")

            if entry_time == conf_time + pd.Timedelta(minutes=1):
                entry_price = float(open_arr[entry_idx])

                if side * (target - entry_price) > 0:
                    p_entry = target_geometry_probability(
                        entry_price, target, opposite, direction
                    )

                    progress = (
                        side * (entry_price - signal_price)
                        / signal_target_distance
                    )

                    entry_rows.append({
                        "event_id": ev.event_id,
                        "decision_time": decision_time,
                        "year": decision_time.year,
                        "strategy": "GEOM_PLUS_5PP",
                        "direction": direction,
                        "signal_price": signal_price,
                        "target_price": target,
                        "opposite_barrier": opposite,
                        "initial_target_geometry_probability": p0,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "entry_geometry_probability": p_entry,
                        "geometry_probability_increase": (
                            p_entry - p0 if np.isfinite(p_entry) else np.nan
                        ),
                        "progress_fraction_at_entry": progress,
                        "minutes_signal_to_entry": (
                            (entry_time - decision_time)
                            / pd.Timedelta(minutes=1)
                        ),
                    })

    if n % 100 == 0:
        print(f"Entry search: {n}/{len(eligible)}")


entries = pd.DataFrame(entry_rows)

if entries.empty:
    raise RuntimeError("No G1B entries generated.")

entries["entry_time"] = pd.to_datetime(
    entries["entry_time"], utc=True
)
entries["decision_time"] = pd.to_datetime(
    entries["decision_time"], utc=True
)


# ============================================================
# SIMULATE SAME TARGET WITH 0.75D / 1.00D STOPS
# ============================================================

trade_rows = []

for n, en in enumerate(entries.itertuples(index=False), 1):

    direction = en.direction
    side = 1.0 if direction == "LONG" else -1.0

    entry = float(en.entry_price)
    target = float(en.target_price)

    D = side * (target - entry)

    if D <= 0:
        continue

    entry_idx = int(np.searchsorted(tns, ns(en.entry_time), side="left"))
    horizon_time = pd.Timestamp(en.decision_time) + pd.Timedelta(hours=HORIZON_HOURS)
    end_idx = int(np.searchsorted(tns, ns(horizon_time), side="left"))
    end_idx = min(end_idx, len(tns))

    if entry_idx >= end_idx:
        continue

    highs = high_arr[entry_idx:end_idx]
    lows = low_arr[entry_idx:end_idx]
    closes = close_arr[entry_idx:end_idx]
    times = tns[entry_idx:end_idx]

    if direction == "LONG":
        target_rel = first_hit(highs >= target)
    else:
        target_rel = first_hit(lows <= target)

    for s in STOP_FRACTIONS:

        stop = entry - side * s * D

        if direction == "LONG":
            stop_rel = first_hit(lows <= stop)
        else:
            stop_rel = first_hit(highs >= stop)

        # Conservative stop-first tie.
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
            exit_rel = len(closes) - 1
            exit_price = float(closes[-1])
            pnl_D = side * (exit_price - entry) / D

        gross_bps = side * (exit_price / entry - 1.0) * 10000.0
        risk_bps = s * D / entry * 10000.0

        trade_rows.append({
            **en._asdict(),
            "stop_fraction_D": s,
            "stop_price": stop,
            "brownian_break_even_tp_rate": s / (1.0 + s),
            "outcome": outcome,
            "exit_minutes": exit_rel + 1,
            "exit_price": exit_price,
            "pnl_D_units": pnl_D,
            "pnl_R_units": pnl_D / s,
            "gross_return_bps": gross_bps,
            "risk_bps": risk_bps,
        })

trades = pd.DataFrame(trade_rows)


# ============================================================
# EXECUTION / ENTRY CHARACTERISTICS
# ============================================================

eligible_count = len(eligible)

execution_rows = []

for strategy in ["FRESH_GMMA", "PROGRESS_25", "GEOM_PLUS_5PP"]:
    g = entries[entries["strategy"] == strategy]

    execution_rows.append({
        "strategy": strategy,
        "eligible_signal_pool": eligible_count,
        "executed_entries": len(g),
        "execution_rate": len(g) / eligible_count,
        "median_minutes_signal_to_entry": g["minutes_signal_to_entry"].median(),
        "median_progress_fraction_at_entry": g["progress_fraction_at_entry"].median(),
        "median_initial_geometry_probability": g["initial_target_geometry_probability"].median(),
        "median_entry_geometry_probability": g["entry_geometry_probability"].median(),
        "median_geometry_probability_increase": g["geometry_probability_increase"].median(),
    })

execution_summary = pd.DataFrame(execution_rows)


# ============================================================
# STRATEGY SUMMARY
# ============================================================

summary_rows = []

for strategy in ["FRESH_GMMA", "PROGRESS_25", "GEOM_PLUS_5PP"]:
    for s in STOP_FRACTIONS:

        g = trades[
            (trades["strategy"] == strategy)
            &
            (trades["stop_fraction_D"] == s)
        ].copy()

        if g.empty:
            continue

        tp_rate = (g["outcome"] == "TP").mean()
        be = s / (1.0 + s)

        boot = month_block_bootstrap_mean(
            g,
            "gross_return_bps",
        )

        summary_rows.append({
            "strategy": strategy,
            "stop_fraction_D": s,
            "trades": len(g),

            "tp_rate": tp_rate,
            "brownian_break_even_tp_rate": be,
            "tp_rate_minus_brownian_break_even_pp": tp_rate - be,

            "mean_pnl_D_units": g["pnl_D_units"].mean(),
            "mean_pnl_R_units": g["pnl_R_units"].mean(),

            "gross_profit_factor_bps": profit_factor(
                g["gross_return_bps"]
            ),
            "mean_gross_bps": g["gross_return_bps"].mean(),

            "gross_bps_ci95_low": boot["ci95_low"],
            "gross_bps_ci95_high": boot["ci95_high"],
            "gross_bps_bootstrap_fraction_le_0": boot["fraction_le_0"],

            "median_risk_bps": g["risk_bps"].median(),
        })

strategy_summary = pd.DataFrame(summary_rows)


# ============================================================
# COST SENSITIVITY
# ============================================================

cost_rows = []

for strategy in ["FRESH_GMMA", "PROGRESS_25", "GEOM_PLUS_5PP"]:
    for s in STOP_FRACTIONS:

        g = trades[
            (trades["strategy"] == strategy)
            &
            (trades["stop_fraction_D"] == s)
        ].copy()

        if g.empty:
            continue

        for cost in ROUND_TRIP_COST_BPS:

            net_bps = g["gross_return_bps"] - cost

            net_R = (
                g["pnl_R_units"]
                -
                cost / g["risk_bps"].replace(0, np.nan)
            )

            cost_rows.append({
                "strategy": strategy,
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
# YEAR-BY-YEAR
# ============================================================

year_rows = []

for strategy in ["FRESH_GMMA", "PROGRESS_25", "GEOM_PLUS_5PP"]:
    for s in STOP_FRACTIONS:

        sg = trades[
            (trades["strategy"] == strategy)
            &
            (trades["stop_fraction_D"] == s)
        ]

        for year in sorted(sg["year"].unique()):
            g = sg[sg["year"] == year]

            tp_rate = (g["outcome"] == "TP").mean()
            be = s / (1.0 + s)

            year_rows.append({
                "year": int(year),
                "strategy": strategy,
                "stop_fraction_D": s,
                "trades": len(g),
                "tp_rate": tp_rate,
                "brownian_break_even_tp_rate": be,
                "tp_rate_minus_brownian_break_even_pp": tp_rate - be,
                "mean_gross_bps": g["gross_return_bps"].mean(),
                "gross_profit_factor_bps": profit_factor(
                    g["gross_return_bps"]
                ),
                "mean_pnl_R_units": g["pnl_R_units"].mean(),
            })

year_summary = pd.DataFrame(year_rows)


# ============================================================
# MATCHED-EVENT PAIRWISE COMPARISONS
# ============================================================

paired_rows = []
paired_boot_rows = []

for s in STOP_FRACTIONS:

    st = trades[
        trades["stop_fraction_D"] == s
    ][
        [
            "event_id",
            "decision_time",
            "strategy",
            "pnl_R_units",
            "gross_return_bps",
        ]
    ].copy()

    wide_R = st.pivot(
        index=["event_id", "decision_time"],
        columns="strategy",
        values="pnl_R_units",
    )

    wide_bp = st.pivot(
        index=["event_id", "decision_time"],
        columns="strategy",
        values="gross_return_bps",
    )

    for comparator in ["PROGRESS_25", "GEOM_PLUS_5PP"]:

        if (
            "FRESH_GMMA" not in wide_R.columns
            or comparator not in wide_R.columns
        ):
            continue

        matched_idx = (
            wide_R[
                ["FRESH_GMMA", comparator]
            ]
            .dropna()
            .index
        )

        if len(matched_idx) == 0:
            continue

        pr = wide_R.loc[matched_idx].reset_index()
        pb = wide_bp.loc[matched_idx].reset_index()

        pr["diff_R_GMMA_minus_comparator"] = (
            pr["FRESH_GMMA"] - pr[comparator]
        )

        pb["diff_bps_GMMA_minus_comparator"] = (
            pb["FRESH_GMMA"] - pb[comparator]
        )

        merged = pr[
            [
                "event_id",
                "decision_time",
                "diff_R_GMMA_minus_comparator",
            ]
        ].merge(
            pb[
                [
                    "event_id",
                    "decision_time",
                    "diff_bps_GMMA_minus_comparator",
                ]
            ],
            on=["event_id", "decision_time"],
            how="inner",
        )

        label = f"FRESH_GMMA_minus_{comparator}_stop_{s:.2f}D"

        paired_rows.append({
            "comparison": label,
            "stop_fraction_D": s,
            "comparator": comparator,
            "matched_events": len(merged),
            "mean_R_difference": merged[
                "diff_R_GMMA_minus_comparator"
            ].mean(),
            "mean_bps_difference": merged[
                "diff_bps_GMMA_minus_comparator"
            ].mean(),
            "gmma_better_R_fraction": (
                merged["diff_R_GMMA_minus_comparator"] > 0
            ).mean(),
        })

        boot_R = paired_month_bootstrap(
            merged,
            "diff_R_GMMA_minus_comparator",
            label + "_R",
        )

        boot_bp = paired_month_bootstrap(
            merged,
            "diff_bps_GMMA_minus_comparator",
            label + "_BPS",
        )

        paired_boot_rows.append({
            **boot_R,
            "metric": "R",
        })

        paired_boot_rows.append({
            **boot_bp,
            "metric": "BPS",
        })

paired_summary = pd.DataFrame(paired_rows)
paired_bootstrap = pd.DataFrame(paired_boot_rows)


# ============================================================
# SAVE
# ============================================================

save(entries, "H4_G1B_entries.csv")
save(trades, "H4_G1B_trade_event_level.csv")
save(execution_summary, "H4_G1B_execution_summary.csv")
save(strategy_summary, "H4_G1B_strategy_summary.csv")
save(cost_summary, "H4_G1B_cost_sensitivity.csv")
save(year_summary, "H4_G1B_year_summary.csv")
save(paired_summary, "H4_G1B_matched_pair_summary.csv")
save(paired_bootstrap, "H4_G1B_matched_pair_bootstrap.csv")

config = f"""H4.G1B — FRESH GMMA VS SIMPLE CONFIRMATION PLACEBOS

FAIR SIGNAL POOL:
H4.6A2 strong signals that were NOT already fully GMMA-aligned
before the H4.2 decision.

STRATEGY A:
FRESH_GMMA
Frozen H4.6A2 entry.

STRATEGY B:
PROGRESS_25
First completed M1 close reaching {PROGRESS_FRACTION:.0%} progress
from H4.2 decision price toward original near target.
Entry next M1 open.

STRATEGY C:
GEOM_PLUS_5PP
First completed M1 close where exact target first-passage probability
between the two frozen H4.2 barriers improves by
{GEOMETRY_PROBABILITY_INCREMENT:.0%} absolute versus the decision-time
geometry probability.
Entry next M1 open.

CONFIRMATION CUTOFF:
Strictly before original H4.2 first-touch minute.

TARGET:
Original H4.2 nearer barrier.

STOP FRACTIONS:
{STOP_FRACTIONS}

BROWNIAN BREAK-EVEN FOR TARGET +1D / STOP -sD:
P_BE = s/(1+s)

COST SENSITIVITY:
{ROUND_TRIP_COST_BPS} round-trip bp.

HORIZON:
Original H4.2 decision + {HORIZON_HOURS}h.

SAME-M1 TP/SL:
Stop first.

PRIMARY QUESTION:
Does Fresh GMMA outperform simple price progress and dynamic geometry
confidence on both unmatched strategy summaries and matched-event
paired comparisons?

IMPORTANT:
This is still development-sample research.
Do not optimize 25% progress, +5pp geometry, or stop fractions after
seeing these results merely to rescue a hypothesis.
"""

(OUT / "H4_G1B_config.txt").write_text(
    config,
    encoding="utf-8",
)

print("\n============================================")
print("EXECUTION SUMMARY")
print("============================================")
print(execution_summary.to_string(index=False))

print("\n============================================")
print("STRATEGY SUMMARY")
print("============================================")
print(strategy_summary.to_string(index=False))

print("\n============================================")
print("MATCHED PAIR SUMMARY")
print("============================================")
print(paired_summary.to_string(index=False))

print("\nSaved to:", OUT)
print("Done.")

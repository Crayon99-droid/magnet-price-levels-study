from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.NULL2 — EXACT-DISTANCE MIRROR PLACEBO ON REAL BTC PATHS
# ============================================================
#
# PURPOSE
# -------
# H4.NULL1 showed that most of the "nearer barrier wins" effect is
# explained by first-passage geometry.
#
# H4.NULL2 now asks a stronger empirical question using the ACTUAL BTC
# price paths:
#
#   Do the REAL structural barriers outperform arbitrary barriers
#   with EXACTLY THE SAME pair of distances from current price?
#
# ------------------------------------------------------------
# PLACEBO DESIGN
# ------------------------------------------------------------
#
# For an H4.2 event:
#
#   current price = P0
#   real upper distance = Du
#   real lower distance = Dl
#
# Real barriers:
#
#   upper_real = P0 + Du
#   lower_real = P0 - Dl
#
# Mirror-placebo barriers:
#
#   upper_placebo = P0 + Dl
#   lower_placebo = P0 - Du
#
# So the placebo uses THE SAME two distances, but swaps which side
# receives each distance.
#
# This preserves exactly:
#   - decision time
#   - actual subsequent BTC path
#   - volatility environment
#   - target-distance pair
#   - near/far ratio
#   - first-passage geometry
#
# What it destroys is the actual structural placement:
# PDH/PDL, swings, weekly levels, etc. no longer determine which side
# receives the near versus far distance.
#
# Therefore, if the real structural barriers are genuinely informative,
# the real orientation should outperform the mirrored orientation.
#
# ------------------------------------------------------------
# PRIMARY STATISTICS
# ------------------------------------------------------------
#
# 1) Real nearer-first rate
# 2) Mirror nearer-first rate
# 3) Real minus mirror paired advantage
# 4) Residual above exact geometry for both
# 5) Calendar-month block bootstrap of the paired advantage
#
# We report:
#   - all events
#   - strong ratio <= 0.50
#   - near barrier contains PDH/PDL
#   - strong ratio <= 0.50 AND near barrier contains PDH/PDL
#
# Other structural families are also reported descriptively.
#
# IMPORTANT:
# This is a null-control study, NOT a trading backtest.
# No parameter optimization is performed.
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

OUTPUT_FOLDER = SCRIPT_DIR / "H4_NULL2_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

HORIZON_HOURS = 24

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260908


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
        "open",
        "high",
        "low",
        "close",
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
        errors="coerce",
    )

    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    return (
        out
        .dropna()
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


def first_touch_from_path(
    highs,
    lows,
    upper_price,
    lower_price,
):
    """
    Return:
      first_touch: UPPER / LOWER / TIMEOUT / AMBIGUOUS
      rel_index: integer minute offset or None
    """

    upper_hits = np.flatnonzero(
        highs >= upper_price
    )

    lower_hits = np.flatnonzero(
        lows <= lower_price
    )

    iu = (
        int(upper_hits[0])
        if upper_hits.size
        else None
    )

    il = (
        int(lower_hits[0])
        if lower_hits.size
        else None
    )

    if iu is None and il is None:
        return "TIMEOUT", None

    if iu is not None and il is None:
        return "UPPER", iu

    if iu is None and il is not None:
        return "LOWER", il

    if iu < il:
        return "UPPER", iu

    if il < iu:
        return "LOWER", il

    return "AMBIGUOUS", iu


def geometry_p_nearer(du, dl):
    near = min(du, dl)
    far = max(du, dl)

    return far / (near + far)


def paired_summary(frame, sample_name):
    """
    Main paired comparison.

    We report all-resolution rates, then compare first-touch outcomes
    only where BOTH real and mirror resolve unambiguously.
    """

    g = frame.copy()

    n = len(g)

    real_resolved = g[
        "real_first_touch"
    ].isin(["UPPER", "LOWER"])

    mirror_resolved = g[
        "mirror_first_touch"
    ].isin(["UPPER", "LOWER"])

    both = g[
        real_resolved
        &
        mirror_resolved
    ].copy()

    if both.empty:
        return {
            "sample": sample_name,
            "events": n,
            "real_resolution_rate": real_resolved.mean(),
            "mirror_resolution_rate": mirror_resolved.mean(),
            "paired_resolved_events": 0,
            "real_nearer_first": np.nan,
            "mirror_nearer_first": np.nan,
            "geometry_expected_nearer_first": np.nan,
            "real_residual_above_geometry": np.nan,
            "mirror_residual_above_geometry": np.nan,
            "real_minus_mirror_pp": np.nan,
            "actual_advantage_over_random_orientation_pp": np.nan,
        }

    return {
        "sample": sample_name,
        "events": n,

        "real_resolution_rate": real_resolved.mean(),
        "mirror_resolution_rate": mirror_resolved.mean(),

        "paired_resolved_events": len(both),

        "real_nearer_first": (
            both["real_nearer_won"].mean()
        ),

        "mirror_nearer_first": (
            both["mirror_nearer_won"].mean()
        ),

        "geometry_expected_nearer_first": (
            both["p_nearer_geometry"].mean()
        ),

        "real_residual_above_geometry": (
            (
                both["real_nearer_won"]
                -
                both["p_nearer_geometry"]
            ).mean()
        ),

        "mirror_residual_above_geometry": (
            (
                both["mirror_nearer_won"]
                -
                both["p_nearer_geometry"]
            ).mean()
        ),

        "real_minus_mirror_pp": (
            (
                both["real_nearer_won"]
                -
                both["mirror_nearer_won"]
            ).mean()
        ),

        # If orientation were random 50/50 between real and mirror,
        # expected performance is their average.
        "actual_advantage_over_random_orientation_pp": (
            0.5
            *
            (
                both["real_nearer_won"]
                -
                both["mirror_nearer_won"]
            ).mean()
        ),
    }


def month_block_bootstrap_paired(
    frame,
    sample_name,
    reps,
    seed,
):
    """
    Calendar-month block bootstrap for REAL - MIRROR nearer-win
    difference on events where both orientations resolve.
    """

    g = frame[
        frame["real_first_touch"].isin(
            ["UPPER", "LOWER"]
        )
        &
        frame["mirror_first_touch"].isin(
            ["UPPER", "LOWER"]
        )
    ].copy()

    if g.empty:
        return {
            "sample": sample_name,
            "paired_resolved_events": 0,
            "month_blocks": 0,
            "observed_real_minus_mirror": np.nan,
            "bootstrap_95pct_low": np.nan,
            "bootstrap_95pct_high": np.nan,
            "fraction_bootstrap_le_0": np.nan,
        }

    g["paired_advantage"] = (
        g["real_nearer_won"]
        -
        g["mirror_nearer_won"]
    )

    g["month_block"] = (
        g["decision_time"]
        .dt.strftime("%Y-%m")
    )

    block_stats = (
        g.groupby(
            "month_block"
        )["paired_advantage"]
        .agg(["sum", "count"])
        .reset_index(drop=True)
    )

    sums = block_stats[
        "sum"
    ].to_numpy(dtype=float)

    counts = block_stats[
        "count"
    ].to_numpy(dtype=float)

    n_blocks = len(block_stats)

    rng = np.random.default_rng(seed)

    boot = np.empty(
        reps,
        dtype=float
    )

    for i in range(reps):
        draw = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        boot[i] = (
            sums[draw].sum()
            /
            counts[draw].sum()
        )

    low, high = np.quantile(
        boot,
        [0.025, 0.975]
    )

    return {
        "sample": sample_name,
        "paired_resolved_events": len(g),
        "month_blocks": n_blocks,
        "observed_real_minus_mirror": (
            g["paired_advantage"].mean()
        ),
        "bootstrap_95pct_low": low,
        "bootstrap_95pct_high": high,
        "fraction_bootstrap_le_0": (
            boot <= 0
        ).mean(),
    }


# ============================================================
# LOAD H4.2 EVENTS
# ============================================================

if not EVENT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{EVENT_FILE}\n\n"
        "Put H4_2_magnet_events.csv beside this script."
    )

events = pd.read_csv(
    EVENT_FILE
)

required_event_cols = [
    "event_id",
    "decision_time",
    "year",
    "entry_price",

    "upper_band_low",
    "lower_band_high",

    "upper_distance",
    "lower_distance",

    "nearer_side",
    "first_touch",
    "touch_time",

    "upper_types",
    "lower_types",
]

missing = [
    c for c in required_event_cols
    if c not in events.columns
]

if missing:
    raise ValueError(
        f"H4.2 file missing columns: {missing}"
    )

events["decision_time"] = pd.to_datetime(
    events["decision_time"],
    utc=True,
    errors="coerce",
).dt.as_unit("ns")

events["touch_time"] = pd.to_datetime(
    events["touch_time"],
    utc=True,
    errors="coerce",
).dt.as_unit("ns")

events = (
    events
    .dropna(
        subset=[
            "decision_time",
            "entry_price",
            "upper_distance",
            "lower_distance",
            "upper_band_low",
            "lower_band_high",
        ]
    )
    .sort_values(
        "decision_time"
    )
    .reset_index(drop=True)
)

events = events[
    (events["upper_distance"] > 0)
    &
    (events["lower_distance"] > 0)
].copy()

events["near_distance"] = np.minimum(
    events["upper_distance"],
    events["lower_distance"],
)

events["far_distance"] = np.maximum(
    events["upper_distance"],
    events["lower_distance"],
)

events["near_far_ratio"] = (
    events["near_distance"]
    /
    events["far_distance"]
)

events["p_nearer_geometry"] = (
    events.apply(
        lambda r: geometry_p_nearer(
            float(r["upper_distance"]),
            float(r["lower_distance"]),
        ),
        axis=1,
    )
)

events["near_types"] = np.where(
    events["nearer_side"] == "UPPER",
    events["upper_types"].fillna(""),
    events["lower_types"].fillna(""),
)

events["near_has_pdh_pdl"] = (
    events["near_types"]
    .astype(str)
    .str.contains(
        r"PDH|PDL",
        regex=True,
        na=False,
    )
)

events["near_has_pwh_pwl"] = (
    events["near_types"]
    .astype(str)
    .str.contains(
        r"PWH|PWL",
        regex=True,
        na=False,
    )
)

events["near_has_prev8h"] = (
    events["near_types"]
    .astype(str)
    .str.contains(
        r"PREV_8H_HIGH|PREV_8H_LOW",
        regex=True,
        na=False,
    )
)

events["near_has_swing15"] = (
    events["near_types"]
    .astype(str)
    .str.contains(
        r"SWING_HIGH_15M|SWING_LOW_15M",
        regex=True,
        na=False,
    )
)

print("\n============================================")
print("H4.NULL2 EVENTS")
print("============================================")
print(f"Events loaded: {len(events):,}")


# ============================================================
# LOAD BTC M1
# ============================================================

all_data = []

for filename in FILES:
    path = (
        SCRIPT_DIR /
        filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing BTC file:\n{path}\n"
            "Put the four BTC ZIP files beside this script."
        )

    raw = read_zip_csv(
        path
    )

    all_data.append(
        clean_dataframe(raw)
    )

m1 = pd.concat(
    all_data,
    ignore_index=True,
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

high_arr = (
    m1["high"]
    .to_numpy(dtype=float)
)

low_arr = (
    m1["low"]
    .to_numpy(dtype=float)
)

print("\n============================================")
print("BTC M1 DATA")
print("============================================")
print(f"Rows:  {len(m1):,}")
print(f"Start: {m1['timestamp'].min()}")
print(f"End:   {m1['timestamp'].max()}")


# ============================================================
# PREFLIGHT
# ============================================================

first_time = (
    events[
        "decision_time"
    ].iloc[0]
)

first_ns = timestamp_ns(
    first_time
)

first_idx = int(
    np.searchsorted(
        ts,
        first_ns,
        side="left",
    )
)

print("\n============================================")
print("TIMESTAMP SANITY CHECK")
print("============================================")
print(f"First event:    {first_time}")
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
# REAL VS MIRROR FIRST-TOUCH RECOMPUTATION
# ============================================================

rows = []

print("\n============================================")
print("RUNNING REAL VS MIRROR PLACEBO")
print("============================================")

for counter, ev in enumerate(
    events.itertuples(index=False),
    start=1,
):

    decision_time = ev.decision_time
    entry_price = float(
        ev.entry_price
    )

    du = float(
        ev.upper_distance
    )

    dl = float(
        ev.lower_distance
    )

    # Real H4.2 barrier edges.
    real_upper = float(
        ev.upper_band_low
    )

    real_lower = float(
        ev.lower_band_high
    )

    # Exact-distance mirror placebo.
    mirror_upper = (
        entry_price + dl
    )

    mirror_lower = (
        entry_price - du
    )

    start_ns = timestamp_ns(
        decision_time
    )

    end_time = (
        decision_time
        + pd.Timedelta(
            hours=HORIZON_HOURS
        )
    )

    end_ns = timestamp_ns(
        end_time
    )

    start_idx = int(
        np.searchsorted(
            ts,
            start_ns,
            side="left",
        )
    )

    end_idx = int(
        np.searchsorted(
            ts,
            end_ns,
            side="left",
        )
    )

    end_idx = min(
        end_idx,
        len(ts),
    )

    if (
        start_idx >= len(ts)
        or start_idx >= end_idx
    ):
        continue

    highs = high_arr[
        start_idx:end_idx
    ]

    lows = low_arr[
        start_idx:end_idx
    ]

    real_touch, real_rel = (
        first_touch_from_path(
            highs,
            lows,
            real_upper,
            real_lower,
        )
    )

    mirror_touch, mirror_rel = (
        first_touch_from_path(
            highs,
            lows,
            mirror_upper,
            mirror_lower,
        )
    )

    # Real nearer side.
    real_nearer_side = (
        "UPPER"
        if du < dl
        else "LOWER"
    )

    # Mirror nearer side reverses because distances swap.
    mirror_nearer_side = (
        "UPPER"
        if dl < du
        else "LOWER"
    )

    real_nearer_won = (
        int(
            real_touch
            == real_nearer_side
        )
        if real_touch
        in ["UPPER", "LOWER"]
        else np.nan
    )

    mirror_nearer_won = (
        int(
            mirror_touch
            == mirror_nearer_side
        )
        if mirror_touch
        in ["UPPER", "LOWER"]
        else np.nan
    )

    # Audit against saved H4.2 outcome.
    saved_touch = str(
        ev.first_touch
    )

    audit_match = (
        real_touch == saved_touch
    )

    rows.append({
        "event_id": ev.event_id,
        "decision_time": decision_time,
        "year": int(ev.year),
        "entry_price": entry_price,

        "upper_distance": du,
        "lower_distance": dl,
        "near_distance": min(du, dl),
        "far_distance": max(du, dl),
        "near_far_ratio": (
            min(du, dl)
            / max(du, dl)
        ),

        "p_nearer_geometry": ev.p_nearer_geometry,

        "real_upper": real_upper,
        "real_lower": real_lower,

        "mirror_upper": mirror_upper,
        "mirror_lower": mirror_lower,

        "real_nearer_side": real_nearer_side,
        "mirror_nearer_side": mirror_nearer_side,

        "real_first_touch": real_touch,
        "mirror_first_touch": mirror_touch,

        "real_nearer_won": real_nearer_won,
        "mirror_nearer_won": mirror_nearer_won,

        "real_touch_minutes": (
            float(real_rel)
            if real_rel is not None
            else np.nan
        ),

        "mirror_touch_minutes": (
            float(mirror_rel)
            if mirror_rel is not None
            else np.nan
        ),

        "saved_h42_first_touch": saved_touch,
        "real_recompute_matches_saved_h42": audit_match,

        "near_types": ev.near_types,
        "near_has_pdh_pdl": bool(
            ev.near_has_pdh_pdl
        ),
        "near_has_pwh_pwl": bool(
            ev.near_has_pwh_pwl
        ),
        "near_has_prev8h": bool(
            ev.near_has_prev8h
        ),
        "near_has_swing15": bool(
            ev.near_has_swing15
        ),
    })

    if counter % 500 == 0:
        safe_to_csv(
            pd.DataFrame(rows),
            "H4_NULL2_CHECKPOINT_event_level.csv",
        )

        print(
            f"  Processed "
            f"{counter:,}/{len(events):,} events "
            f"— checkpoint saved."
        )


result = pd.DataFrame(
    rows
)

if result.empty:
    raise RuntimeError(
        "No H4.NULL2 observations generated."
    )


# ============================================================
# AUDIT
# ============================================================

audit_summary = pd.DataFrame([
    {
        "events_recomputed": len(result),

        "saved_h42_match_rate": (
            result[
                "real_recompute_matches_saved_h42"
            ].mean()
        ),

        "saved_h42_mismatches": int(
            (
                ~result[
                    "real_recompute_matches_saved_h42"
                ]
            ).sum()
        ),

        "real_timeout_rate": (
            result[
                "real_first_touch"
            ].eq("TIMEOUT").mean()
        ),

        "mirror_timeout_rate": (
            result[
                "mirror_first_touch"
            ].eq("TIMEOUT").mean()
        ),

        "real_ambiguous_rate": (
            result[
                "real_first_touch"
            ].eq("AMBIGUOUS").mean()
        ),

        "mirror_ambiguous_rate": (
            result[
                "mirror_first_touch"
            ].eq("AMBIGUOUS").mean()
        ),
    }
])

print("\n============================================")
print("H4.2 RECOMPUTATION AUDIT")
print("============================================")
print(
    audit_summary.to_string(
        index=False
    )
)


# ============================================================
# PRIMARY SAMPLES
# ============================================================

sample_defs = [
    (
        "ALL",
        result
    ),
    (
        "STRONG_RATIO_<=0.50",
        result[
            result[
                "near_far_ratio"
            ] <= 0.50
        ]
    ),
    (
        "NEAR_HAS_PDH_PDL",
        result[
            result[
                "near_has_pdh_pdl"
            ] == True
        ]
    ),
    (
        "STRONG_AND_NEAR_HAS_PDH_PDL",
        result[
            (
                result[
                    "near_far_ratio"
                ] <= 0.50
            )
            &
            (
                result[
                    "near_has_pdh_pdl"
                ] == True
            )
        ]
    ),
]

summary_rows = []
bootstrap_rows = []

for name, g in sample_defs:
    summary_rows.append(
        paired_summary(
            g,
            name,
        )
    )

    bootstrap_rows.append(
        month_block_bootstrap_paired(
            g,
            name,
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
        )
    )

primary_summary = pd.DataFrame(
    summary_rows
)

bootstrap_summary = pd.DataFrame(
    bootstrap_rows
)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

year_rows = []

for year in sorted(
    result["year"].unique()
):
    yg = result[
        result["year"] == year
    ]

    for name, mask_fn in [
        (
            "ALL",
            lambda x: x
        ),
        (
            "STRONG_RATIO_<=0.50",
            lambda x: x[
                x[
                    "near_far_ratio"
                ] <= 0.50
            ]
        ),
        (
            "NEAR_HAS_PDH_PDL",
            lambda x: x[
                x[
                    "near_has_pdh_pdl"
                ] == True
            ]
        ),
        (
            "STRONG_AND_NEAR_HAS_PDH_PDL",
            lambda x: x[
                (
                    x[
                        "near_far_ratio"
                    ] <= 0.50
                )
                &
                (
                    x[
                        "near_has_pdh_pdl"
                    ] == True
                )
            ]
        ),
    ]:
        g = mask_fn(
            yg
        )

        row = paired_summary(
            g,
            f"{year}_{name}",
        )

        row["year"] = int(
            year
        )

        row["subset"] = name

        year_rows.append(
            row
        )

year_summary = pd.DataFrame(
    year_rows
)


# ============================================================
# STRUCTURAL FAMILY DESCRIPTIVE TESTS
# ============================================================

family_rows = []

family_cols = {
    "PDH_PDL": "near_has_pdh_pdl",
    "PWH_PWL": "near_has_pwh_pwl",
    "PREVIOUS_8H": "near_has_prev8h",
    "SWING_15M": "near_has_swing15",
}

for family, col in family_cols.items():

    for status_value, status_label in [
        (True, "PRESENT"),
        (False, "ABSENT"),
    ]:
        g = result[
            result[col] == status_value
        ]

        row = paired_summary(
            g,
            f"{family}_{status_label}",
        )

        row["family"] = family
        row["status"] = status_label
        row["subset"] = "ALL"

        family_rows.append(
            row
        )

        gs = g[
            g["near_far_ratio"]
            <= 0.50
        ]

        row_s = paired_summary(
            gs,
            f"{family}_{status_label}_STRONG",
        )

        row_s["family"] = family
        row_s["status"] = status_label
        row_s["subset"] = "RATIO_<=0.50"

        family_rows.append(
            row_s
        )

family_summary = pd.DataFrame(
    family_rows
)


# ============================================================
# TOUCH-TIME COMPARISON
# ============================================================

touch_time_rows = []

for name, g in sample_defs:
    both = g[
        g[
            "real_first_touch"
        ].isin(["UPPER", "LOWER"])
        &
        g[
            "mirror_first_touch"
        ].isin(["UPPER", "LOWER"])
    ].copy()

    if both.empty:
        continue

    touch_time_rows.append({
        "sample": name,
        "paired_resolved_events": len(both),

        "median_real_touch_minutes": (
            both[
                "real_touch_minutes"
            ].median()
        ),

        "median_mirror_touch_minutes": (
            both[
                "mirror_touch_minutes"
            ].median()
        ),

        "mean_real_minus_mirror_touch_minutes": (
            (
                both[
                    "real_touch_minutes"
                ]
                -
                both[
                    "mirror_touch_minutes"
                ]
            ).mean()
        ),
    })

touch_time_summary = pd.DataFrame(
    touch_time_rows
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

safe_to_csv(
    result,
    "H4_NULL2_event_level_mirror_placebo.csv",
)

safe_to_csv(
    audit_summary,
    "H4_NULL2_recomputation_audit.csv",
)

safe_to_csv(
    primary_summary,
    "H4_NULL2_primary_summary.csv",
)

safe_to_csv(
    bootstrap_summary,
    "H4_NULL2_month_block_bootstrap.csv",
)

safe_to_csv(
    year_summary,
    "H4_NULL2_year_summary.csv",
)

safe_to_csv(
    family_summary,
    "H4_NULL2_family_summary.csv",
)

safe_to_csv(
    touch_time_summary,
    "H4_NULL2_touch_time_summary.csv",
)


# ============================================================
# HUMAN-READABLE VERDICT TEMPLATE
# ============================================================

verdict_lines = [
    "H4.NULL2 — EXACT-DISTANCE MIRROR PLACEBO",
    "",
    "Interpretation:",
    "",
    "The mirror placebo uses the SAME two barrier distances and the",
    "SAME future BTC path, but swaps which side receives each distance.",
    "Therefore geometry is exactly preserved while the real structural",
    "placement is broken.",
    "",
]

for _, r in primary_summary.iterrows():
    verdict_lines.extend([
        f"SAMPLE: {r['sample']}",
        f"Events: {int(r['events'])}",
        f"Paired resolved events: {int(r['paired_resolved_events'])}",
        (
            f"Real nearer-first: "
            f"{r['real_nearer_first']:.6f}"
            if pd.notna(r["real_nearer_first"])
            else "Real nearer-first: NA"
        ),
        (
            f"Mirror nearer-first: "
            f"{r['mirror_nearer_first']:.6f}"
            if pd.notna(r["mirror_nearer_first"])
            else "Mirror nearer-first: NA"
        ),
        (
            f"Real - mirror: "
            f"{r['real_minus_mirror_pp']:.6f} "
            f"({r['real_minus_mirror_pp'] * 100:.3f} pp)"
            if pd.notna(r["real_minus_mirror_pp"])
            else "Real - mirror: NA"
        ),
        "",
    ])

verdict_lines.extend([
    "Decision rule:",
    "",
    "1. If REAL does not outperform MIRROR, the structural-level story",
    "   is not supported by this placebo.",
    "",
    "2. If REAL consistently outperforms MIRROR across years and the",
    "   month-block bootstrap interval stays above zero, structural",
    "   placement may contain information beyond geometry and generic",
    "   BTC path dynamics.",
    "",
    "3. PDH/PDL receives special attention because NULL1 showed the",
    "   largest exploratory residual there. It must now survive this",
    "   exact-distance placebo before being promoted as a candidate",
    "   structural feature.",
])

(
    OUTPUT_FOLDER /
    "H4_NULL2_verdict.txt"
).write_text(
    "\n".join(verdict_lines),
    encoding="utf-8",
)


config_text = f"""H4.NULL2 — EXACT-DISTANCE MIRROR PLACEBO

INPUT:
{EVENT_FILE.name}

BTC DATA:
{FILES}

HORIZON:
{HORIZON_HOURS} hours from H4.2 decision

REAL BARRIERS:
upper = original H4.2 upper band near edge
lower = original H4.2 lower band near edge

MIRROR PLACEBO:
upper_placebo = entry_price + original lower distance
lower_placebo = entry_price - original upper distance

WHAT IS PRESERVED EXACTLY:
- event time
- real future BTC path
- upper/lower distance pair
- near/far ratio
- geometry probability
- volatility/regime environment

WHAT IS BROKEN:
- the actual side/location of PDH/PDL, PWH/PWL, 8h extremes,
  and 15m swing-derived barriers

PRIMARY SAMPLES:
- all
- near/far <= 0.50
- nearer barrier contains PDH/PDL
- near/far <= 0.50 AND nearer barrier contains PDH/PDL

BOOTSTRAP:
calendar-month block bootstrap
repetitions = {BOOTSTRAP_REPS}
seed = {BOOTSTRAP_SEED}

IMPORTANT:
- This is a null-control study, not a trading backtest.
- No parameter optimization is performed.
- Same-minute upper+lower touches are marked ambiguous, not ordered.
"""

(
    OUTPUT_FOLDER /
    "H4_NULL2_config.txt"
).write_text(
    config_text,
    encoding="utf-8",
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n============================================")
print("H4.NULL2 PRIMARY SUMMARY")
print("============================================")
print(
    primary_summary.to_string(
        index=False
    )
)

print("\n============================================")
print("H4.NULL2 MONTH-BLOCK BOOTSTRAP")
print("============================================")
print(
    bootstrap_summary.to_string(
        index=False
    )
)

print("\n============================================")
print("FILES SAVED")
print("============================================")

for file in sorted(
    OUTPUT_FOLDER.iterdir()
):
    print(file)

print("\nDone.")

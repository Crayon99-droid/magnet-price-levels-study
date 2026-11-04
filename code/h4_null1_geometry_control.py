from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# H4.NULL1 — GEOMETRY CONTROL FOR STRUCTURAL BARRIER HYPOTHESIS
# ============================================================
#
# PURPOSE
# -------
# Earlier H4 work described unresolved PDH/PDL, PWH/PWL,
# previous 8h highs/lows, and confirmed 15m swings as "magnets".
#
# A reviewer correctly pointed out that a nearer barrier should be
# hit first more often EVEN IN A ZERO-DRIFT RANDOM WALK, purely due
# to first-passage geometry.
#
# For upper distance a and lower distance b:
#
#       P(UPPER first) = b / (a + b)
#
# Therefore:
#
#       P(NEARER first) = far / (near + far)
#
# H4.NULL1 asks:
#
#   1) How much of H4.2's observed first-touch accuracy is explained
#      by geometry alone?
#
#   2) Is there a residual above the geometry null?
#
#   3) Do our structural features (level count, source diversity,
#      PD/PW/8h/swing composition) add anything AFTER distance is
#      accounted for?
#
# IMPORTANT
# ---------
# This is NOT a trading backtest.
# This is a foundational null-control study.
#
# No thresholds are optimized here.
# No new "magnet" parameters are created.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

EVENT_FILE = SCRIPT_DIR / "H4_2_magnet_events.csv"

OUTPUT_FOLDER = SCRIPT_DIR / "H4_NULL1_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260908

# Predeclared distance-ratio buckets from prior H4 analysis.
RATIO_BINS = [-np.inf, 0.50, 0.75, 1.00]
RATIO_LABELS = ["<=0.50", "0.50-0.75", "0.75-1.00"]


# ============================================================
# HELPERS
# ============================================================

def safe_to_csv(frame, filename):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_FOLDER / filename
    frame.to_csv(path, index=False)
    return path


def bootstrap_month_blocks(frame, residual_col, reps, seed):
    """
    Resample calendar-month blocks with replacement.
    Returns bootstrap distribution of the event-weighted mean residual.

    This preserves within-month event dependence better than treating
    all events as independent Bernoulli observations.
    """
    x = frame.copy()

    if x.empty:
        return np.array([], dtype=float)

    x["month_block"] = (
        pd.to_datetime(x["decision_time"], utc=True)
        .dt.strftime("%Y-%m")
    )

    block_stats = (
        x.groupby("month_block")[residual_col]
        .agg(["sum", "count"])
        .reset_index(drop=True)
    )

    n_blocks = len(block_stats)

    if n_blocks == 0:
        return np.array([], dtype=float)

    sums = block_stats["sum"].to_numpy(dtype=float)
    counts = block_stats["count"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)

    out = np.empty(reps, dtype=float)

    for i in range(reps):
        draw = rng.integers(
            0,
            n_blocks,
            size=n_blocks
        )

        out[i] = (
            sums[draw].sum()
            /
            counts[draw].sum()
        )

    return out


def residual_summary(frame, label):
    """
    Summarize observed nearer-first rate versus event-specific
    zero-drift geometry expectation.
    """
    if frame.empty:
        return {
            "sample": label,
            "observations": 0,
            "observed_nearer_first": np.nan,
            "geometry_expected_nearer_first": np.nan,
            "residual_probability_points": np.nan,
            "naive_independence_z": np.nan,
        }

    y = frame["nearer_won"].to_numpy(dtype=float)
    p = frame["p_nearer_geometry"].to_numpy(dtype=float)

    residual = y - p

    denom = np.sqrt(
        np.sum(
            p * (1.0 - p)
        )
    )

    z = (
        residual.sum() / denom
        if denom > 0
        else np.nan
    )

    return {
        "sample": label,
        "observations": len(frame),
        "observed_nearer_first": y.mean(),
        "geometry_expected_nearer_first": p.mean(),
        "residual_probability_points": residual.mean(),
        "naive_independence_z": z,
    }


def side_probability_geometry(row, side):
    """
    Geometry probability that requested side is touched first.
    """
    p_upper = row["p_upper_geometry"]

    if side == "UPPER":
        return p_upper

    if side == "LOWER":
        return 1.0 - p_upper

    return np.nan


def summarize_feature_side(frame, side_col, label):
    """
    For a structural rule that selects UPPER or LOWER:
    compare its observed first-touch rate with the geometry probability
    assigned to the SAME selected side.

    A positive residual would suggest the feature adds information
    beyond distance.
    """
    g = frame[
        frame[side_col].isin(["UPPER", "LOWER"])
    ].copy()

    if g.empty:
        return {
            "feature_test": label,
            "observations": 0,
            "observed_selected_side_first": np.nan,
            "geometry_expected_selected_side_first": np.nan,
            "residual_probability_points": np.nan,
        }

    g["selected_side_won"] = (
        g["first_touch"] == g[side_col]
    ).astype(int)

    g["p_selected_geometry"] = g.apply(
        lambda r: side_probability_geometry(
            r,
            r[side_col]
        ),
        axis=1
    )

    return {
        "feature_test": label,
        "observations": len(g),
        "observed_selected_side_first": (
            g["selected_side_won"].mean()
        ),
        "geometry_expected_selected_side_first": (
            g["p_selected_geometry"].mean()
        ),
        "residual_probability_points": (
            (
                g["selected_side_won"]
                -
                g["p_selected_geometry"]
            ).mean()
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

required = [
    "event_id",
    "decision_time",
    "year",
    "entry_price",

    "upper_distance",
    "lower_distance",

    "upper_level_count",
    "lower_level_count",

    "upper_source_diversity",
    "lower_source_diversity",

    "upper_types",
    "lower_types",

    "nearer_side",
    "stronger_side",
    "first_touch",
    "ambiguous_same_bar",
]

missing = [
    c for c in required
    if c not in events.columns
]

if missing:
    raise ValueError(
        f"H4.2 event file is missing columns: {missing}"
    )

events["decision_time"] = pd.to_datetime(
    events["decision_time"],
    utc=True,
    errors="coerce"
)

events["ambiguous_same_bar"] = (
    events["ambiguous_same_bar"]
    .fillna(False)
    .astype(bool)
)

resolved = events[
    events["first_touch"].isin(
        ["UPPER", "LOWER"]
    )
    &
    (~events["ambiguous_same_bar"])
].copy()

resolved = resolved[
    (resolved["upper_distance"] > 0)
    &
    (resolved["lower_distance"] > 0)
].copy()

if resolved.empty:
    raise RuntimeError(
        "No resolved, unambiguous H4.2 events found."
    )


# ============================================================
# EXACT GEOMETRY NULL
# ============================================================

resolved["p_upper_geometry"] = (
    resolved["lower_distance"]
    /
    (
        resolved["upper_distance"]
        +
        resolved["lower_distance"]
    )
)

resolved["near_distance"] = np.minimum(
    resolved["upper_distance"],
    resolved["lower_distance"],
)

resolved["far_distance"] = np.maximum(
    resolved["upper_distance"],
    resolved["lower_distance"],
)

resolved["near_far_ratio"] = (
    resolved["near_distance"]
    /
    resolved["far_distance"]
)

resolved["p_nearer_geometry"] = (
    resolved["far_distance"]
    /
    (
        resolved["near_distance"]
        +
        resolved["far_distance"]
    )
)

resolved["nearer_won"] = (
    resolved["first_touch"]
    ==
    resolved["nearer_side"]
).astype(int)

resolved["geometry_residual_nearer"] = (
    resolved["nearer_won"]
    -
    resolved["p_nearer_geometry"]
)

resolved["ratio_bucket"] = pd.cut(
    resolved["near_far_ratio"],
    bins=RATIO_BINS,
    labels=RATIO_LABELS,
    right=True,
)

print("\n============================================")
print("H4.NULL1 — EXACT GEOMETRY CONTROL")
print("============================================")
print(f"Resolved events: {len(resolved):,}")


# ============================================================
# OVERALL / STRONG SUBSET
# ============================================================

overall_rows = [
    residual_summary(
        resolved,
        "ALL_RESOLVED"
    ),
    residual_summary(
        resolved[
            resolved["near_far_ratio"]
            <= 0.50
        ],
        "STRONG_RATIO_<=0.50"
    ),
]

overall_summary = pd.DataFrame(
    overall_rows
)


# ============================================================
# MONTH-BLOCK BOOTSTRAP
# ============================================================

bootstrap_rows = []

for sample_name, g in [
    (
        "ALL_RESOLVED",
        resolved
    ),
    (
        "STRONG_RATIO_<=0.50",
        resolved[
            resolved["near_far_ratio"]
            <= 0.50
        ]
    ),
]:
    boot = bootstrap_month_blocks(
        g,
        residual_col="geometry_residual_nearer",
        reps=BOOTSTRAP_REPS,
        seed=BOOTSTRAP_SEED,
    )

    if boot.size:
        ci_low, ci_high = np.quantile(
            boot,
            [0.025, 0.975]
        )

        bootstrap_rows.append({
            "sample": sample_name,
            "observed_mean_residual": (
                g["geometry_residual_nearer"].mean()
            ),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "month_blocks": (
                g["decision_time"]
                .dt.strftime("%Y-%m")
                .nunique()
            ),
            "bootstrap_mean": boot.mean(),
            "bootstrap_95pct_low": ci_low,
            "bootstrap_95pct_high": ci_high,
            "fraction_bootstrap_residual_le_0": (
                boot <= 0
            ).mean(),
        })

bootstrap_summary = pd.DataFrame(
    bootstrap_rows
)


# ============================================================
# DISTANCE-RATIO BUCKETS
# ============================================================

ratio_rows = []

for bucket in RATIO_LABELS:
    g = resolved[
        resolved["ratio_bucket"]
        == bucket
    ]

    row = residual_summary(
        g,
        bucket
    )

    row["mean_near_far_ratio"] = (
        g["near_far_ratio"].mean()
        if len(g)
        else np.nan
    )

    ratio_rows.append(row)

ratio_summary = pd.DataFrame(
    ratio_rows
)


# ============================================================
# YEAR-BY-YEAR
# ============================================================

year_rows = []

for year in sorted(
    resolved["year"].unique()
):
    g = resolved[
        resolved["year"] == year
    ]

    row = residual_summary(
        g,
        f"{int(year)}_ALL"
    )

    row["year"] = int(year)
    row["subset"] = "ALL"
    year_rows.append(row)

    gs = g[
        g["near_far_ratio"] <= 0.50
    ]

    row_s = residual_summary(
        gs,
        f"{int(year)}_STRONG"
    )

    row_s["year"] = int(year)
    row_s["subset"] = "RATIO_<=0.50"
    year_rows.append(row_s)

year_summary = pd.DataFrame(
    year_rows
)


# ============================================================
# GEOMETRY CALIBRATION
# ============================================================
#
# If geometry alone explains the first-touch behavior well,
# observed near-first rates should roughly match event-specific
# p_nearer_geometry across calibration bins.
# ============================================================

resolved["geometry_probability_bin"] = pd.qcut(
    resolved["p_nearer_geometry"],
    q=10,
    duplicates="drop"
)

calibration = (
    resolved
    .groupby(
        "geometry_probability_bin",
        observed=True
    )
    .agg(
        observations=(
            "nearer_won",
            "size"
        ),
        observed_nearer_first=(
            "nearer_won",
            "mean"
        ),
        geometry_expected_nearer_first=(
            "p_nearer_geometry",
            "mean"
        ),
        mean_near_far_ratio=(
            "near_far_ratio",
            "mean"
        ),
    )
    .reset_index()
)

calibration[
    "residual_probability_points"
] = (
    calibration[
        "observed_nearer_first"
    ]
    -
    calibration[
        "geometry_expected_nearer_first"
    ]
)

calibration[
    "geometry_probability_bin"
] = calibration[
    "geometry_probability_bin"
].astype(str)


# ============================================================
# STRUCTURAL SIDE TESTS AFTER GEOMETRY
# ============================================================

# 1) Original H4.2 "stronger" side.
feature_rows = []

feature_rows.append(
    summarize_feature_side(
        resolved,
        "stronger_side",
        "ORIGINAL_H4_2_STRONGER_SIDE"
    )
)

# 2) Raw level-count winner.
resolved["higher_level_count_side"] = np.where(
    resolved["upper_level_count"]
    > resolved["lower_level_count"],
    "UPPER",
    np.where(
        resolved["lower_level_count"]
        > resolved["upper_level_count"],
        "LOWER",
        "EQUAL",
    )
)

feature_rows.append(
    summarize_feature_side(
        resolved,
        "higher_level_count_side",
        "HIGHER_LEVEL_COUNT_SIDE"
    )
)

# 3) Source-diversity winner.
resolved["higher_source_diversity_side"] = np.where(
    resolved["upper_source_diversity"]
    > resolved["lower_source_diversity"],
    "UPPER",
    np.where(
        resolved["lower_source_diversity"]
        > resolved["upper_source_diversity"],
        "LOWER",
        "EQUAL",
    )
)

feature_rows.append(
    summarize_feature_side(
        resolved,
        "higher_source_diversity_side",
        "HIGHER_SOURCE_DIVERSITY_SIDE"
    )
)

feature_side_summary = pd.DataFrame(
    feature_rows
)


# ============================================================
# NEAR-SIDE STRUCTURAL FAMILY RESIDUALS
# ============================================================
#
# These are descriptive tests:
# when the NEARER barrier contains a given structural family,
# does its first-touch rate exceed its own geometry expectation?
#
# Positive residual does NOT prove causality; it tells us where a
# structural residual may be worth testing out-of-sample later.
# ============================================================

resolved["near_types"] = np.where(
    resolved["nearer_side"] == "UPPER",
    resolved["upper_types"].fillna(""),
    resolved["lower_types"].fillna(""),
)

family_patterns = {
    "PREVIOUS_DAY_EXTREME": r"PDH|PDL",
    "PREVIOUS_WEEK_EXTREME": r"PWH|PWL",
    "PREVIOUS_8H_EXTREME": r"PREV_8H_HIGH|PREV_8H_LOW",
    "CONFIRMED_15M_SWING": r"SWING_HIGH_15M|SWING_LOW_15M",
}

family_rows = []

for family, pattern in family_patterns.items():
    has_family = (
        resolved["near_types"]
        .astype(str)
        .str.contains(
            pattern,
            regex=True,
            na=False
        )
    )

    for present_value, present_label in [
        (True, "PRESENT"),
        (False, "ABSENT"),
    ]:
        g = resolved[
            has_family == present_value
        ]

        family_rows.append({
            "family": family,
            "near_side_family_status": present_label,
            "observations": len(g),
            "observed_nearer_first": (
                g["nearer_won"].mean()
                if len(g)
                else np.nan
            ),
            "geometry_expected_nearer_first": (
                g["p_nearer_geometry"].mean()
                if len(g)
                else np.nan
            ),
            "residual_probability_points": (
                g["geometry_residual_nearer"].mean()
                if len(g)
                else np.nan
            ),
            "mean_near_far_ratio": (
                g["near_far_ratio"].mean()
                if len(g)
                else np.nan
            ),
        })

family_summary = pd.DataFrame(
    family_rows
)


# ============================================================
# STRONG-SUBSET STRUCTURAL FAMILY RESIDUALS
# ============================================================

strong = resolved[
    resolved["near_far_ratio"] <= 0.50
].copy()

strong["near_types"] = np.where(
    strong["nearer_side"] == "UPPER",
    strong["upper_types"].fillna(""),
    strong["lower_types"].fillna(""),
)

strong_family_rows = []

for family, pattern in family_patterns.items():
    has_family = (
        strong["near_types"]
        .astype(str)
        .str.contains(
            pattern,
            regex=True,
            na=False
        )
    )

    for present_value, present_label in [
        (True, "PRESENT"),
        (False, "ABSENT"),
    ]:
        g = strong[
            has_family == present_value
        ]

        strong_family_rows.append({
            "family": family,
            "near_side_family_status": present_label,
            "observations": len(g),
            "observed_nearer_first": (
                g["nearer_won"].mean()
                if len(g)
                else np.nan
            ),
            "geometry_expected_nearer_first": (
                g["p_nearer_geometry"].mean()
                if len(g)
                else np.nan
            ),
            "residual_probability_points": (
                g["geometry_residual_nearer"].mean()
                if len(g)
                else np.nan
            ),
            "mean_near_far_ratio": (
                g["near_far_ratio"].mean()
                if len(g)
                else np.nan
            ),
        })

strong_family_summary = pd.DataFrame(
    strong_family_rows
)


# ============================================================
# BRIER SCORE / LOG LOSS DIAGNOSTICS
# ============================================================
#
# Geometry predicts P(UPPER first), not simply a binary class.
# These metrics assess probability quality.
# ============================================================

resolved["upper_won"] = (
    resolved["first_touch"] == "UPPER"
).astype(int)

p_upper = np.clip(
    resolved["p_upper_geometry"].to_numpy(dtype=float),
    1e-12,
    1.0 - 1e-12,
)

y_upper = resolved[
    "upper_won"
].to_numpy(dtype=float)

geometry_brier = np.mean(
    (y_upper - p_upper) ** 2
)

geometry_logloss = -np.mean(
    y_upper * np.log(p_upper)
    +
    (1.0 - y_upper)
    * np.log(1.0 - p_upper)
)

constant_p = y_upper.mean()

constant_brier = np.mean(
    (y_upper - constant_p) ** 2
)

constant_logloss = -np.mean(
    y_upper * np.log(constant_p)
    +
    (1.0 - y_upper)
    * np.log(1.0 - constant_p)
)

probability_metrics = pd.DataFrame([
    {
        "model": "ZERO_DRIFT_GEOMETRY",
        "brier_score": geometry_brier,
        "log_loss": geometry_logloss,
    },
    {
        "model": "CONSTANT_UPPER_FIRST_RATE",
        "brier_score": constant_brier,
        "log_loss": constant_logloss,
    },
])


# ============================================================
# SAVE EVENT-LEVEL NULL DATA
# ============================================================

event_output_columns = [
    "event_id",
    "decision_time",
    "year",
    "entry_price",

    "upper_distance",
    "lower_distance",

    "near_distance",
    "far_distance",
    "near_far_ratio",

    "nearer_side",
    "first_touch",
    "nearer_won",

    "p_upper_geometry",
    "p_nearer_geometry",
    "geometry_residual_nearer",

    "upper_level_count",
    "lower_level_count",
    "upper_source_diversity",
    "lower_source_diversity",

    "upper_types",
    "lower_types",
    "near_types",

    "stronger_side",
    "higher_level_count_side",
    "higher_source_diversity_side",
]

event_null = resolved[
    event_output_columns
].copy()


# ============================================================
# SAVE OUTPUTS
# ============================================================

safe_to_csv(
    event_null,
    "H4_NULL1_event_level_geometry.csv"
)

safe_to_csv(
    overall_summary,
    "H4_NULL1_overall_geometry_summary.csv"
)

safe_to_csv(
    bootstrap_summary,
    "H4_NULL1_month_block_bootstrap.csv"
)

safe_to_csv(
    ratio_summary,
    "H4_NULL1_ratio_bucket_summary.csv"
)

safe_to_csv(
    year_summary,
    "H4_NULL1_year_summary.csv"
)

safe_to_csv(
    calibration,
    "H4_NULL1_geometry_calibration.csv"
)

safe_to_csv(
    feature_side_summary,
    "H4_NULL1_structural_side_tests.csv"
)

safe_to_csv(
    family_summary,
    "H4_NULL1_near_family_residuals.csv"
)

safe_to_csv(
    strong_family_summary,
    "H4_NULL1_strong_near_family_residuals.csv"
)

safe_to_csv(
    probability_metrics,
    "H4_NULL1_probability_metrics.csv"
)


# ============================================================
# HUMAN-READABLE VERDICT
# ============================================================

all_row = overall_summary[
    overall_summary["sample"]
    == "ALL_RESOLVED"
].iloc[0]

strong_row = overall_summary[
    overall_summary["sample"]
    == "STRONG_RATIO_<=0.50"
].iloc[0]

verdict = f"""H4.NULL1 — GEOMETRY CONTROL

This study deliberately replaces the previous informal comparison
against 50% with the exact zero-drift first-passage geometry null.

ALL RESOLVED EVENTS
-------------------
N:
{int(all_row['observations'])}

Observed nearer-first rate:
{all_row['observed_nearer_first']:.6f}

Geometry-only expected nearer-first rate:
{all_row['geometry_expected_nearer_first']:.6f}

Residual above geometry:
{all_row['residual_probability_points']:.6f}
({all_row['residual_probability_points'] * 100:.3f} percentage points)


STRONG DISTANCE-ASYMMETRY SUBSET (near/far <= 0.50)
---------------------------------------------------
N:
{int(strong_row['observations'])}

Observed nearer-first rate:
{strong_row['observed_nearer_first']:.6f}

Geometry-only expected nearer-first rate:
{strong_row['geometry_expected_nearer_first']:.6f}

Residual above geometry:
{strong_row['residual_probability_points']:.6f}
({strong_row['residual_probability_points'] * 100:.3f} percentage points)


INTERPRETATION RULE
-------------------
Do NOT interpret the observed nearer-first rate itself as evidence of
a liquidity magnet.

Only the RESIDUAL above a credible geometry / market-dynamics null can
support a structural-level claim.

If structural features such as level count, source diversity, or
specific level families fail to add positive residual after geometry,
the "liquidity magnet" mechanism should be rejected or reframed.

If a residual remains, it must next survive:
1) empirical matched-placebo barriers on the actual BTC paths,
2) out-of-sample market tests,
3) separate entry-model placebo controls.
"""

(
    OUTPUT_FOLDER /
    "H4_NULL1_verdict.txt"
).write_text(
    verdict,
    encoding="utf-8"
)


config_text = f"""H4.NULL1 — GEOMETRY CONTROL FOR STRUCTURAL BARRIERS

INPUT:
{EVENT_FILE.name}

PRIMARY ZERO-DRIFT FIRST-PASSAGE NULL:
P(UPPER first) = lower_distance / (upper_distance + lower_distance)

P(NEARER first):
far_distance / (near_distance + far_distance)

STRONG SUBSET:
near_distance / far_distance <= 0.50

BOOTSTRAP:
calendar-month block bootstrap
repetitions = {BOOTSTRAP_REPS}
seed = {BOOTSTRAP_SEED}

STRUCTURAL TESTS AFTER GEOMETRY:
- original H4.2 stronger side
- higher raw level-count side
- higher source-diversity side
- near-side structural family:
  PDH/PDL
  PWH/PWL
  previous 8h high/low
  confirmed 15m swing

IMPORTANT:
- This is a null-control study, not a trading backtest.
- No parameters are optimized.
- A positive raw first-touch percentage is NOT treated as evidence
  of magnetism unless it exceeds the relevant geometry null.
"""

(
    OUTPUT_FOLDER /
    "H4_NULL1_config.txt"
).write_text(
    config_text,
    encoding="utf-8"
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n============================================")
print("OVERALL GEOMETRY RESULT")
print("============================================")
print(
    overall_summary.to_string(
        index=False
    )
)

print("\n============================================")
print("MONTH-BLOCK BOOTSTRAP")
print("============================================")
print(
    bootstrap_summary.to_string(
        index=False
    )
)

print("\n============================================")
print("STRUCTURAL SIDE TESTS AFTER GEOMETRY")
print("============================================")
print(
    feature_side_summary.to_string(
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

from pathlib import Path
import pandas as pd
import numpy as np

try:
    import statsmodels.api as sm
except ImportError as exc:
    raise ImportError(
        "This test requires statsmodels. Install it once with:\n"
        "pip install statsmodels"
    ) from exc

# ============================================================
# H4.MODEL1 — GEOMETRY + TREND + STRUCTURE INCREMENTAL OOS TEST
# ============================================================
#
# PURPOSE
# -------
# We now know:
#   - raw first-touch accuracy is largely explained by geometry;
#   - BTC real barriers beat a mirrored placebo;
#   - 24h trend alone did NOT explain that NULL2 advantage;
#   - EURUSD did NOT replicate the BTC PDH/PDL effect.
#
# This test replaces further slicing with an expanding-window,
# year-out-of-sample probability model.
#
# Exact geometry is used as a FIXED logistic offset:
#
#   P(UPPER first) = Dl / (Du + Dl)
#
# therefore:
#
#   logit(P_UPPER) = log(Dl / Du)
#
# We then ask whether adding TREND and STRUCTURAL IDENTITY improves
# probability forecasts OUT OF SAMPLE.
#
# ------------------------------------------------------------
# PRIMARY MODELS — PREDECLARED
# ------------------------------------------------------------
#
# M0_GEOMETRY
#   exact geometry only
#
# M1_TREND24
#   geometry offset + learned intercept + z_24h
#
# M2_PDH_PDL
#   M1 + signed near-side PDH/PDL indicator
#
# Signed PDH/PDL:
#   +1 if nearer side is UPPER and contains PDH/PDL
#   -1 if nearer side is LOWER and contains PDH/PDL
#    0 otherwise
#
# This directly asks whether PDH/PDL shifts the probability toward
# the structurally identified side after geometry + 24h trend.
#
# ------------------------------------------------------------
# SECONDARY / EXPLORATORY
# ------------------------------------------------------------
#
# M3_ALL_STRUCTURE
#   M1 + signed PDH/PDL + PWH/PWL + previous-8h + swing15
#
# M4_ALL_STRUCTURE_PLUS_4H
#   M3 + z_4h
#
# 24h remains the PRIMARY trend variable because it was predeclared
# in NULL2T. 4h is secondary and cannot replace it after results.
#
# ------------------------------------------------------------
# WALK-FORWARD OOS
# ------------------------------------------------------------
#
# Train 2021       -> Test 2022
# Train 2021-2022  -> Test 2023
# Train 2021-2023  -> Test 2024
#
# No coefficient is fitted using the test year.
# No feature threshold is optimized.
#
# ------------------------------------------------------------
# PRIMARY DECISION QUESTION
# ------------------------------------------------------------
#
# Does M2_PDH_PDL beat M1_TREND24 OUT OF SAMPLE?
#
# If not:
#   PDH/PDL has not demonstrated incremental forecasting value
#   after geometry + trend.
#
# If yes, consistently:
#   structural identity has stronger evidence than raw NULL2 slices.
#
# We also report M3/M4, but they are secondary/exploratory.
# ============================================================

BASE = Path(__file__).resolve().parent
INPUT_FILE = BASE / "H4_NULL2T_event_level.csv"
OUT = BASE / "H4_MODEL1_results"
OUT.mkdir(exist_ok=True)

BOOT_REPS = 10000
BOOT_SEED = 20260908
EPS = 1e-12

PRIMARY_MODELS = {
    "M0_GEOMETRY": [],
    "M1_TREND24": ["const", "z_24h"],
    "M2_PDH_PDL": ["const", "z_24h", "signed_pdh_pdl"],
    "M3_ALL_STRUCTURE": [
        "const", "z_24h",
        "signed_pdh_pdl",
        "signed_pwh_pwl",
        "signed_prev8h",
        "signed_swing15",
    ],
    "M4_ALL_STRUCTURE_PLUS_4H": [
        "const", "z_24h", "z_4h",
        "signed_pdh_pdl",
        "signed_pwh_pwl",
        "signed_prev8h",
        "signed_swing15",
    ],
}

FOLDS = [
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
]


# ============================================================
# HELPERS
# ============================================================

def save(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    return p


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)

    return out


def logloss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    return float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y) * np.log(1.0 - p)
        )
    )


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    return float(
        np.mean(
            (y - p) ** 2
        )
    )


def model_metrics(y, p):
    return {
        "n": len(y),
        "brier": brier(y, p),
        "log_loss": logloss(y, p),
        "accuracy_0_5": float(
            (
                (np.asarray(p) >= 0.5)
                ==
                np.asarray(y).astype(bool)
            ).mean()
        ),
        "mean_pred_upper": float(
            np.mean(p)
        ),
        "observed_upper_rate": float(
            np.mean(y)
        ),
    }


def fit_offset_logit(train, features):
    """
    Fit:
        logit(P_upper) = geometry_offset + X beta

    Geometry coefficient is fixed at 1 via statsmodels offset.
    """
    if not features:
        return None

    X = train[features].astype(float)
    y = train["upper_won"].astype(float)
    offset = train["geometry_logit"].astype(float)

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        offset=offset,
    )

    result = model.fit(
        maxiter=200,
        disp=0,
    )

    return result


def predict_offset_logit(result, test, features):
    if not features:
        return test[
            "p_upper_geometry"
        ].to_numpy(dtype=float)

    X = test[features].astype(float)

    linear = (
        test["geometry_logit"].to_numpy(dtype=float)
        +
        X.to_numpy(dtype=float)
        @ result.params.to_numpy(dtype=float)
    )

    return sigmoid(linear)


def month_block_bootstrap_improvement(
    predictions,
    model_a,
    model_b,
    reps,
    seed,
):
    """
    Positive values mean MODEL B improves upon MODEL A.

    Log-loss improvement:
        loss_A - loss_B

    Brier improvement:
        brier_A - brier_B
    """
    d = predictions.copy()

    pa = np.clip(
        d[f"p_{model_a}"].to_numpy(dtype=float),
        EPS,
        1.0 - EPS,
    )

    pb = np.clip(
        d[f"p_{model_b}"].to_numpy(dtype=float),
        EPS,
        1.0 - EPS,
    )

    y = d[
        "upper_won"
    ].to_numpy(dtype=float)

    ll_a = -(
        y * np.log(pa)
        +
        (1.0 - y) * np.log(1.0 - pa)
    )

    ll_b = -(
        y * np.log(pb)
        +
        (1.0 - y) * np.log(1.0 - pb)
    )

    br_a = (
        y - pa
    ) ** 2

    br_b = (
        y - pb
    ) ** 2

    d = d[
        [
            "decision_time"
        ]
    ].copy()

    d["ll_improvement"] = (
        ll_a - ll_b
    )

    d["brier_improvement"] = (
        br_a - br_b
    )

    d["month"] = (
        d["decision_time"]
        .dt.strftime("%Y-%m")
    )

    blocks = (
        d.groupby("month")
        .agg(
            ll_sum=(
                "ll_improvement",
                "sum",
            ),
            brier_sum=(
                "brier_improvement",
                "sum",
            ),
            count=(
                "ll_improvement",
                "size",
            ),
        )
        .reset_index(drop=True)
    )

    ll_sum = blocks[
        "ll_sum"
    ].to_numpy(dtype=float)

    br_sum = blocks[
        "brier_sum"
    ].to_numpy(dtype=float)

    count = blocks[
        "count"
    ].to_numpy(dtype=float)

    rng = np.random.default_rng(
        seed
    )

    boot_ll = np.empty(
        reps,
        dtype=float,
    )

    boot_br = np.empty(
        reps,
        dtype=float,
    )

    for i in range(reps):
        idx = rng.integers(
            0,
            len(blocks),
            size=len(blocks),
        )

        denom = count[
            idx
        ].sum()

        boot_ll[i] = (
            ll_sum[idx].sum()
            /
            denom
        )

        boot_br[i] = (
            br_sum[idx].sum()
            /
            denom
        )

    ll_lo, ll_hi = np.quantile(
        boot_ll,
        [0.025, 0.975],
    )

    br_lo, br_hi = np.quantile(
        boot_br,
        [0.025, 0.975],
    )

    return {
        "baseline_model": model_a,
        "candidate_model": model_b,
        "oos_events": len(predictions),
        "month_blocks": len(blocks),

        "mean_logloss_improvement": float(
            d["ll_improvement"].mean()
        ),
        "logloss_improvement_ci95_low": float(
            ll_lo
        ),
        "logloss_improvement_ci95_high": float(
            ll_hi
        ),
        "fraction_bootstrap_logloss_improvement_le_0": float(
            (
                boot_ll <= 0
            ).mean()
        ),

        "mean_brier_improvement": float(
            d["brier_improvement"].mean()
        ),
        "brier_improvement_ci95_low": float(
            br_lo
        ),
        "brier_improvement_ci95_high": float(
            br_hi
        ),
        "fraction_bootstrap_brier_improvement_le_0": float(
            (
                boot_br <= 0
            ).mean()
        ),
    }


# ============================================================
# LOAD / PREPARE
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing {INPUT_FILE.name}. "
        "Put H4_NULL2T_event_level.csv beside this script."
    )

df = pd.read_csv(
    INPUT_FILE
)

required = [
    "decision_time",
    "year",
    "upper_distance",
    "lower_distance",
    "real_nearer_side",
    "real_first_touch",
    "z_24h",
    "z_4h",
    "near_has_pdh_pdl",
    "near_has_pwh_pwl",
    "near_has_prev8h",
    "near_has_swing15",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Input file missing columns: {missing}"
    )

df["decision_time"] = pd.to_datetime(
    df["decision_time"],
    utc=True,
    errors="coerce",
)

# Probability model is for first-touch direction among
# resolved, unambiguous events.
df = df[
    df[
        "real_first_touch"
    ].isin(
        ["UPPER", "LOWER"]
    )
].copy()

df = df.dropna(
    subset=[
        "decision_time",
        "year",
        "upper_distance",
        "lower_distance",
        "real_nearer_side",
        "z_24h",
        "z_4h",
    ]
)

df = df[
    (df["upper_distance"] > 0)
    &
    (df["lower_distance"] > 0)
].copy()

df["year"] = (
    df["year"]
    .astype(int)
)

df["upper_won"] = (
    df[
        "real_first_touch"
    ] == "UPPER"
).astype(int)

# Exact first-passage geometry.
df["p_upper_geometry"] = (
    df["lower_distance"]
    /
    (
        df["upper_distance"]
        +
        df["lower_distance"]
    )
)

df["p_upper_geometry"] = np.clip(
    df["p_upper_geometry"],
    EPS,
    1.0 - EPS,
)

df["geometry_logit"] = np.log(
    df["lower_distance"]
    /
    df["upper_distance"]
)

df["const"] = 1.0

near_sign = np.where(
    df["real_nearer_side"]
    == "UPPER",
    1.0,
    -1.0,
)

def bool_series(col):
    return (
        df[col]
        .astype(str)
        .str.lower()
        .eq("true")
        .astype(float)
    )

df["signed_pdh_pdl"] = (
    near_sign
    *
    bool_series(
        "near_has_pdh_pdl"
    )
)

df["signed_pwh_pwl"] = (
    near_sign
    *
    bool_series(
        "near_has_pwh_pwl"
    )
)

df["signed_prev8h"] = (
    near_sign
    *
    bool_series(
        "near_has_prev8h"
    )
)

df["signed_swing15"] = (
    near_sign
    *
    bool_series(
        "near_has_swing15"
    )
)

print("\n============================================")
print("H4.MODEL1 INPUT")
print("============================================")
print(f"Resolved observations: {len(df):,}")
print(
    df.groupby("year")[
        "upper_won"
    ].agg(
        ["count", "mean"]
    )
)


# ============================================================
# EXPANDING-WINDOW OOS
# ============================================================

prediction_rows = []
coefficient_rows = []
fold_metric_rows = []

for train_end, test_year in FOLDS:

    train = df[
        df["year"]
        <= train_end
    ].copy()

    test = df[
        df["year"]
        == test_year
    ].copy()

    if train.empty or test.empty:
        continue

    print(
        f"\nTrain <= {train_end}: {len(train):,} "
        f"| Test {test_year}: {len(test):,}"
    )

    fold_predictions = test[
        [
            "decision_time",
            "year",
            "upper_won",
            "p_upper_geometry",
            "upper_distance",
            "lower_distance",
            "real_nearer_side",
            "z_24h",
            "z_4h",
            "signed_pdh_pdl",
            "signed_pwh_pwl",
            "signed_prev8h",
            "signed_swing15",
        ]
    ].copy()

    for model_name, features in PRIMARY_MODELS.items():

        if model_name == "M0_GEOMETRY":
            pred = test[
                "p_upper_geometry"
            ].to_numpy(dtype=float)

        else:
            result = fit_offset_logit(
                train,
                features,
            )

            pred = predict_offset_logit(
                result,
                test,
                features,
            )

            for feature, coef in result.params.items():
                coefficient_rows.append({
                    "train_end_year": train_end,
                    "test_year": test_year,
                    "model": model_name,
                    "feature": feature,
                    "coefficient": float(
                        coef
                    ),
                })

        fold_predictions[
            f"p_{model_name}"
        ] = pred

        metrics = model_metrics(
            test["upper_won"],
            pred,
        )

        metrics.update({
            "train_end_year": train_end,
            "test_year": test_year,
            "model": model_name,
        })

        fold_metric_rows.append(
            metrics
        )

    prediction_rows.append(
        fold_predictions
    )


oos = pd.concat(
    prediction_rows,
    ignore_index=True,
)

fold_metrics = pd.DataFrame(
    fold_metric_rows
)

coefficients = pd.DataFrame(
    coefficient_rows
)


# ============================================================
# AGGREGATE OOS METRICS
# ============================================================

aggregate_rows = []

for model_name in PRIMARY_MODELS:
    p = oos[
        f"p_{model_name}"
    ].to_numpy(dtype=float)

    metrics = model_metrics(
        oos["upper_won"],
        p,
    )

    metrics["model"] = model_name

    # Improvement signs are made intuitive:
    # positive = candidate better than baseline.
    m0_brier = brier(
        oos["upper_won"],
        oos[
            "p_M0_GEOMETRY"
        ],
    )

    m0_ll = logloss(
        oos["upper_won"],
        oos[
            "p_M0_GEOMETRY"
        ],
    )

    metrics[
        "brier_improvement_vs_geometry"
    ] = (
        m0_brier
        -
        metrics["brier"]
    )

    metrics[
        "logloss_improvement_vs_geometry"
    ] = (
        m0_ll
        -
        metrics["log_loss"]
    )

    if model_name not in [
        "M0_GEOMETRY",
        "M1_TREND24",
    ]:
        m1_brier = brier(
            oos["upper_won"],
            oos[
                "p_M1_TREND24"
            ],
        )

        m1_ll = logloss(
            oos["upper_won"],
            oos[
                "p_M1_TREND24"
            ],
        )

        metrics[
            "brier_improvement_vs_M1_TREND24"
        ] = (
            m1_brier
            -
            metrics["brier"]
        )

        metrics[
            "logloss_improvement_vs_M1_TREND24"
        ] = (
            m1_ll
            -
            metrics["log_loss"]
        )

    aggregate_rows.append(
        metrics
    )

aggregate_metrics = pd.DataFrame(
    aggregate_rows
)


# ============================================================
# BLOCK-BOOTSTRAP INCREMENTAL OOS VALUE
# ============================================================

comparisons = [
    (
        "M0_GEOMETRY",
        "M1_TREND24",
    ),
    (
        "M1_TREND24",
        "M2_PDH_PDL",
    ),
    (
        "M1_TREND24",
        "M3_ALL_STRUCTURE",
    ),
    (
        "M3_ALL_STRUCTURE",
        "M4_ALL_STRUCTURE_PLUS_4H",
    ),
]

bootstrap_rows = []

for baseline, candidate in comparisons:
    bootstrap_rows.append(
        month_block_bootstrap_improvement(
            oos,
            baseline,
            candidate,
            reps=BOOT_REPS,
            seed=BOOT_SEED,
        )
    )

bootstrap_summary = pd.DataFrame(
    bootstrap_rows
)


# ============================================================
# YEAR-SPECIFIC INCREMENTAL COMPARISON
# ============================================================

year_comparison_rows = []

for year in sorted(
    oos["year"].unique()
):

    yg = oos[
        oos["year"] == year
    ]

    for baseline, candidate in comparisons:

        y = yg[
            "upper_won"
        ]

        pb = yg[
            f"p_{baseline}"
        ]

        pc = yg[
            f"p_{candidate}"
        ]

        year_comparison_rows.append({
            "year": int(year),
            "baseline_model": baseline,
            "candidate_model": candidate,
            "events": len(yg),

            "brier_improvement": (
                brier(y, pb)
                -
                brier(y, pc)
            ),

            "logloss_improvement": (
                logloss(y, pb)
                -
                logloss(y, pc)
            ),
        })

year_comparisons = pd.DataFrame(
    year_comparison_rows
)


# ============================================================
# OOS CALIBRATION — PRIMARY M0/M1/M2
# ============================================================

calibration_rows = []

for model_name in [
    "M0_GEOMETRY",
    "M1_TREND24",
    "M2_PDH_PDL",
]:

    temp = oos[
        [
            "upper_won",
            f"p_{model_name}",
        ]
    ].copy()

    temp["bin"] = pd.qcut(
        temp[
            f"p_{model_name}"
        ],
        q=10,
        duplicates="drop",
    )

    c = (
        temp
        .groupby(
            "bin",
            observed=True,
        )
        .agg(
            observations=(
                "upper_won",
                "size",
            ),
            observed_upper_rate=(
                "upper_won",
                "mean",
            ),
            mean_predicted_upper=(
                f"p_{model_name}",
                "mean",
            ),
        )
        .reset_index()
    )

    c["model"] = model_name
    c["bin"] = (
        c["bin"]
        .astype(str)
    )

    calibration_rows.append(
        c
    )

calibration = pd.concat(
    calibration_rows,
    ignore_index=True,
)


# ============================================================
# SAVE
# ============================================================

save(
    oos,
    "H4_MODEL1_oos_predictions.csv",
)

save(
    fold_metrics,
    "H4_MODEL1_fold_metrics.csv",
)

save(
    aggregate_metrics,
    "H4_MODEL1_aggregate_oos_metrics.csv",
)

save(
    bootstrap_summary,
    "H4_MODEL1_incremental_block_bootstrap.csv",
)

save(
    year_comparisons,
    "H4_MODEL1_year_incremental_comparison.csv",
)

save(
    coefficients,
    "H4_MODEL1_fold_coefficients.csv",
)

save(
    calibration,
    "H4_MODEL1_oos_calibration.csv",
)


# ============================================================
# VERDICT / CONFIG
# ============================================================

config = f"""H4.MODEL1 — GEOMETRY + TREND + STRUCTURE INCREMENTAL OOS TEST

INPUT:
{INPUT_FILE.name}

OUTCOME:
UPPER first vs LOWER first among resolved, unambiguous H4.NULL2T events.

FIXED GEOMETRY OFFSET:
logit(P_upper) = log(lower_distance / upper_distance)

PRIMARY MODELS:
M0_GEOMETRY:
  exact geometry only

M1_TREND24:
  geometry offset + intercept + z_24h

M2_PDH_PDL:
  M1 + signed near-side PDH/PDL indicator

SECONDARY / EXPLORATORY:
M3_ALL_STRUCTURE:
  M1 + signed PDH/PDL + PWH/PWL + previous-8h + swing15

M4_ALL_STRUCTURE_PLUS_4H:
  M3 + z_4h

WALK-FORWARD:
Train 2021 -> Test 2022
Train 2021-2022 -> Test 2023
Train 2021-2023 -> Test 2024

PRIMARY DECISION:
Does M2_PDH_PDL improve OOS Brier score and log loss relative to
M1_TREND24, preferably in multiple years and with month-block
bootstrap improvement above zero?

BOOTSTRAP:
calendar-month blocks
repetitions = {BOOT_REPS}
seed = {BOOT_SEED}

INTERPRETATION:
- Positive loss improvement means candidate is better.
- If M2 fails to beat M1, PDH/PDL has not demonstrated incremental
  predictive value after geometry + trend.
- M3/M4 are exploratory and must not rescue a failed primary M2.
- No feature thresholds or model hyperparameters are optimized.
"""

(
    OUT / "H4_MODEL1_config.txt"
).write_text(
    config,
    encoding="utf-8",
)

print("\n============================================")
print("AGGREGATE OOS METRICS")
print("============================================")
print(
    aggregate_metrics.to_string(
        index=False
    )
)

print("\n============================================")
print("INCREMENTAL BLOCK BOOTSTRAP")
print("============================================")
print(
    bootstrap_summary.to_string(
        index=False
    )
)

print("\n============================================")
print("YEAR INCREMENTAL COMPARISONS")
print("============================================")
print(
    year_comparisons.to_string(
        index=False
    )
)

print("\nSaved to:", OUT)
print("\nDone.")

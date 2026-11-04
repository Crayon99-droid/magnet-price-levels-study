from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# H4.NULL2T — TREND-CONDITIONED MIRROR PLACEBO
#
# Goal:
# Test whether the BTC NULL2 real-vs-mirror advantage is actually
# explained by pre-existing trend persistence rather than structural
# level identity.
#
# PRIMARY trend variable (fixed before results):
#   R24 = log(C[t-1m] / C[t-24h-1m])
#   RV24 = sqrt(sum of squared M1 log returns over same completed 24h)
#   Z24 = R24 / RV24
#
# Regimes:
#   |Z24| < 0.50      = NEUTRAL
#   0.50 <= |Z24| <1 = MODERATE
#   |Z24| >= 1        = STRONG
#
# Alignment:
#   Z24 > 0 + real nearer side UPPER = ALIGNED
#   Z24 < 0 + real nearer side LOWER = ALIGNED
#   opposite = OPPOSED
#
# PRIMARY outcome on events where real and mirror both resolve:
#   paired_advantage = real_nearer_won - mirror_nearer_won
#
# Secondary robustness: same calculation using 4h normalized momentum.
# No optimization. 24h remains primary regardless of results.

BASE = Path(__file__).resolve().parent
EVENT_FILE = BASE / "H4_NULL2_event_level_mirror_placebo.csv"
BTC_FILES = [f"BTCUSDT_1m_{y}.csv.zip" for y in (2021, 2022, 2023, 2024)]
OUT = BASE / "H4_NULL2T_results"
OUT.mkdir(exist_ok=True)

NEUTRAL = 0.50
STRONG = 1.00
BOOT_REPS = 10000
SEED = 20260908

def save(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    return p

def load_btc_zip(path):
    with zipfile.ZipFile(path, "r") as z:
        csvs = [x for x in z.namelist() if x.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"No CSV inside {path}")
        with z.open(csvs[0]) as f:
            d = pd.read_csv(f)
    req = ["timestamp_utc", "close"]
    miss = [c for c in req if c not in d.columns]
    if miss:
        raise ValueError(f"{path.name} missing {miss}")
    x = pd.DataFrame({
        "timestamp": pd.to_datetime(d["timestamp_utc"], utc=True, errors="coerce"),
        "close": pd.to_numeric(d["close"], errors="coerce"),
    }).dropna()
    return x.sort_values("timestamp").drop_duplicates("timestamp")

def ns(ts):
    return int(pd.Timestamp(ts).as_unit("ns").value)

def trailing_momentum(decision_time, hours, tns, closes, r2prefix):
    # last completed M1 bar before decision
    end = np.searchsorted(tns, ns(decision_time), side="left") - 1
    start_boundary = pd.Timestamp(decision_time) - pd.Timedelta(hours=hours)
    start = np.searchsorted(tns, ns(start_boundary), side="left") - 1
    if start < 0 or end <= start:
        return None
    p0, p1 = closes[start], closes[end]
    if p0 <= 0 or p1 <= 0:
        return None
    ret = float(np.log(p1 / p0))
    rv2 = float(r2prefix[end + 1] - r2prefix[start + 1])
    if rv2 <= 0:
        return None
    rv = float(np.sqrt(rv2))
    return ret, rv, ret / rv

def bucket(z):
    a = abs(z)
    if a < NEUTRAL:
        return "NEUTRAL"
    if a < STRONG:
        return "MODERATE"
    return "STRONG"

def alignment(z, near_side):
    if abs(z) < NEUTRAL:
        return "NEUTRAL"
    trend_side = "UPPER" if z > 0 else "LOWER"
    return "ALIGNED" if trend_side == near_side else "OPPOSED"

def align_score(z, near_side):
    # positive means trend points toward real nearer side
    return z if near_side == "UPPER" else -z

def group_summary(g, label, zcol):
    real_res = g["real_first_touch"].isin(["UPPER", "LOWER"])
    mirror_res = g["mirror_first_touch"].isin(["UPPER", "LOWER"])
    paired = g[real_res & mirror_res]
    rr = g[real_res]
    row = {
        "sample": label,
        "events": len(g),
        "real_resolved": len(rr),
        "paired_resolved": len(paired),
        "mean_abs_z": g[zcol].abs().mean() if len(g) else np.nan,
    }
    if len(rr):
        row["real_nearer_first"] = rr["real_nearer_won"].mean()
        row["geometry_expected"] = rr["p_nearer_geometry"].mean()
        row["real_minus_geometry"] = (
            rr["real_nearer_won"] - rr["p_nearer_geometry"]
        ).mean()
    if len(paired):
        row["paired_real_nearer_first"] = paired["real_nearer_won"].mean()
        row["paired_mirror_nearer_first"] = paired["mirror_nearer_won"].mean()
        row["real_minus_mirror"] = (
            paired["real_nearer_won"] - paired["mirror_nearer_won"]
        ).mean()
    return row

def block_boot(g, label):
    p = g[
        g["real_first_touch"].isin(["UPPER", "LOWER"]) &
        g["mirror_first_touch"].isin(["UPPER", "LOWER"])
    ].copy()
    if p.empty:
        return {"sample": label, "paired_resolved": 0}
    p["adv"] = p["real_nearer_won"] - p["mirror_nearer_won"]
    p["month"] = p["decision_time"].dt.strftime("%Y-%m")
    b = p.groupby("month")["adv"].agg(["sum", "count"])
    sums, counts = b["sum"].to_numpy(float), b["count"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    vals = np.empty(BOOT_REPS)
    for i in range(BOOT_REPS):
        j = rng.integers(0, len(b), len(b))
        vals[i] = sums[j].sum() / counts[j].sum()
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {
        "sample": label,
        "paired_resolved": len(p),
        "month_blocks": len(b),
        "observed_real_minus_mirror": p["adv"].mean(),
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_fraction_le_0": (vals <= 0).mean(),
    }

def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan
    xm, ym = x.mean(), y.mean()
    den = ((x - xm) ** 2).sum()
    if den <= 0:
        return np.nan, np.nan
    beta = ((x - xm) * (y - ym)).sum() / den
    alpha = ym - beta * xm
    return float(alpha), float(beta)

def bootstrap_regression(g, score_col, outcome_col, label):
    x = g[["decision_time", score_col, outcome_col]].dropna().copy()
    x["month"] = x["decision_time"].dt.strftime("%Y-%m")
    groups = {k: v for k, v in x.groupby("month")}
    names = list(groups)
    a0, b0 = ols(x[score_col], x[outcome_col])
    rng = np.random.default_rng(SEED)
    A, B = np.empty(BOOT_REPS), np.empty(BOOT_REPS)
    for i in range(BOOT_REPS):
        draw = rng.integers(0, len(names), len(names))
        s = pd.concat([groups[names[j]] for j in draw], ignore_index=True)
        A[i], B[i] = ols(s[score_col], s[outcome_col])
    return {
        "test": label,
        "observations": len(x),
        "month_blocks": len(names),
        "intercept": a0,
        "slope": b0,
        "intercept_ci95_low": np.nanquantile(A, .025),
        "intercept_ci95_high": np.nanquantile(A, .975),
        "slope_ci95_low": np.nanquantile(B, .025),
        "slope_ci95_high": np.nanquantile(B, .975),
        "fraction_slope_le_0": np.nanmean(B <= 0),
    }

# ---------- load NULL2 ----------
if not EVENT_FILE.exists():
    raise FileNotFoundError(
        f"Missing {EVENT_FILE.name}. Put it beside this script."
    )

e = pd.read_csv(EVENT_FILE)
req = [
    "event_id","decision_time","year","near_far_ratio","p_nearer_geometry",
    "real_nearer_side","real_first_touch","mirror_first_touch",
    "real_nearer_won","mirror_nearer_won","near_has_pdh_pdl"
]
miss = [c for c in req if c not in e.columns]
if miss:
    raise ValueError(f"NULL2 event file missing columns: {miss}")

e["decision_time"] = pd.to_datetime(e["decision_time"], utc=True, errors="coerce")
e = e.dropna(subset=["decision_time","real_nearer_side"]).sort_values("decision_time")
e["near_has_pdh_pdl"] = (
    e["near_has_pdh_pdl"].astype(str).str.lower().eq("true")
)

# ---------- load BTC M1 ----------
parts = []
for fn in BTC_FILES:
    p = BASE / fn
    if not p.exists():
        raise FileNotFoundError(f"Missing {fn}")
    print("Loading", fn)
    parts.append(load_btc_zip(p))
m1 = pd.concat(parts, ignore_index=True).sort_values("timestamp")
m1 = m1.drop_duplicates("timestamp").reset_index(drop=True)
m1["timestamp"] = m1["timestamp"].dt.as_unit("ns")

tns = m1["timestamp"].astype("int64").to_numpy()
cl = m1["close"].to_numpy(float)
lr = np.full(len(cl), np.nan)
lr[1:] = np.log(cl[1:] / cl[:-1])
r2prefix = np.concatenate([[0.0], np.cumsum(np.nan_to_num(lr**2))])

# ---------- attach causal trend ----------
rows = []
for i, r in enumerate(e.itertuples(index=False), 1):
    m24 = trailing_momentum(r.decision_time, 24, tns, cl, r2prefix)
    m4 = trailing_momentum(r.decision_time, 4, tns, cl, r2prefix)
    if m24 is None or m4 is None:
        continue
    d = r._asdict()
    d.update({
        "ret_24h": m24[0], "rv_24h": m24[1], "z_24h": m24[2],
        "trend_bucket_24h": bucket(m24[2]),
        "trend_alignment_24h": alignment(m24[2], r.real_nearer_side),
        "alignment_score_24h": align_score(m24[2], r.real_nearer_side),
        "ret_4h": m4[0], "rv_4h": m4[1], "z_4h": m4[2],
        "trend_bucket_4h": bucket(m4[2]),
        "trend_alignment_4h": alignment(m4[2], r.real_nearer_side),
        "alignment_score_4h": align_score(m4[2], r.real_nearer_side),
    })
    if r.real_first_touch in ("UPPER","LOWER") and r.mirror_first_touch in ("UPPER","LOWER"):
        d["paired_advantage"] = float(r.real_nearer_won) - float(r.mirror_nearer_won)
    else:
        d["paired_advantage"] = np.nan
    if r.real_first_touch in ("UPPER","LOWER"):
        d["real_geometry_residual"] = float(r.real_nearer_won) - float(r.p_nearer_geometry)
    else:
        d["real_geometry_residual"] = np.nan
    rows.append(d)
    if i % 1000 == 0:
        print(f"Trend attached to {i:,}/{len(e):,}")

x = pd.DataFrame(rows)
x["decision_time"] = pd.to_datetime(x["decision_time"], utc=True)

# ---------- primary 24h ----------
groups = [
    ("ALL", x),
    ("24H_NEUTRAL", x[x.trend_alignment_24h == "NEUTRAL"]),
    ("24H_ALIGNED", x[x.trend_alignment_24h == "ALIGNED"]),
    ("24H_OPPOSED", x[x.trend_alignment_24h == "OPPOSED"]),
    ("24H_MODERATE_ALIGNED", x[(x.trend_bucket_24h=="MODERATE") & (x.trend_alignment_24h=="ALIGNED")]),
    ("24H_MODERATE_OPPOSED", x[(x.trend_bucket_24h=="MODERATE") & (x.trend_alignment_24h=="OPPOSED")]),
    ("24H_STRONG_ALIGNED", x[(x.trend_bucket_24h=="STRONG") & (x.trend_alignment_24h=="ALIGNED")]),
    ("24H_STRONG_OPPOSED", x[(x.trend_bucket_24h=="STRONG") & (x.trend_alignment_24h=="OPPOSED")]),
]
primary = pd.DataFrame([group_summary(g,n,"z_24h") for n,g in groups])
primary_boot = pd.DataFrame([block_boot(g,n) for n,g in groups])

# ---------- frozen H4 strong subset ----------
s = x[x.near_far_ratio <= 0.50]
sgroups = [
    ("H4_STRONG_ALL", s),
    ("H4_STRONG_24H_NEUTRAL", s[s.trend_alignment_24h=="NEUTRAL"]),
    ("H4_STRONG_24H_ALIGNED", s[s.trend_alignment_24h=="ALIGNED"]),
    ("H4_STRONG_24H_OPPOSED", s[s.trend_alignment_24h=="OPPOSED"]),
]
strong_summary = pd.DataFrame([group_summary(g,n,"z_24h") for n,g in sgroups])
strong_boot = pd.DataFrame([block_boot(g,n) for n,g in sgroups])

# ---------- PDH/PDL exploratory ----------
pdh = x[x.near_has_pdh_pdl]
pgroups = [
    ("PDH_PDL_ALL", pdh),
    ("PDH_PDL_24H_NEUTRAL", pdh[pdh.trend_alignment_24h=="NEUTRAL"]),
    ("PDH_PDL_24H_ALIGNED", pdh[pdh.trend_alignment_24h=="ALIGNED"]),
    ("PDH_PDL_24H_OPPOSED", pdh[pdh.trend_alignment_24h=="OPPOSED"]),
]
pdh_summary = pd.DataFrame([group_summary(g,n,"z_24h") for n,g in pgroups])
pdh_boot = pd.DataFrame([block_boot(g,n) for n,g in pgroups])

# ---------- secondary 4h ----------
g4 = [
    ("4H_NEUTRAL", x[x.trend_alignment_4h=="NEUTRAL"]),
    ("4H_ALIGNED", x[x.trend_alignment_4h=="ALIGNED"]),
    ("4H_OPPOSED", x[x.trend_alignment_4h=="OPPOSED"]),
]
secondary = pd.DataFrame([group_summary(g,n,"z_4h") for n,g in g4])
secondary_boot = pd.DataFrame([block_boot(g,n) for n,g in g4])

# ---------- continuous tests ----------
paired = x[x.paired_advantage.notna()]
realres = x[x.real_geometry_residual.notna()]
reg = pd.DataFrame([
    bootstrap_regression(
        paired, "alignment_score_24h", "paired_advantage",
        "PAIRED_ADVANTAGE_vs_24H_ALIGNMENT"
    ),
    bootstrap_regression(
        realres, "alignment_score_24h", "real_geometry_residual",
        "GEOMETRY_RESIDUAL_vs_24H_ALIGNMENT"
    ),
    bootstrap_regression(
        paired, "alignment_score_4h", "paired_advantage",
        "PAIRED_ADVANTAGE_vs_4H_ALIGNMENT"
    ),
])

# ---------- year x state ----------
yr = []
for year in sorted(x.year.dropna().unique()):
    y = x[x.year == year]
    for st in ("NEUTRAL","ALIGNED","OPPOSED"):
        row = group_summary(y[y.trend_alignment_24h==st], f"{int(year)}_{st}", "z_24h")
        row["year"] = int(year)
        row["state"] = st
        yr.append(row)
year_summary = pd.DataFrame(yr)

# ---------- save ----------
save(x, "H4_NULL2T_event_level.csv")
save(primary, "H4_NULL2T_primary_24h_summary.csv")
save(primary_boot, "H4_NULL2T_primary_24h_bootstrap.csv")
save(strong_summary, "H4_NULL2T_strong_H4_summary.csv")
save(strong_boot, "H4_NULL2T_strong_H4_bootstrap.csv")
save(pdh_summary, "H4_NULL2T_PDH_PDL_summary.csv")
save(pdh_boot, "H4_NULL2T_PDH_PDL_bootstrap.csv")
save(secondary, "H4_NULL2T_secondary_4h_summary.csv")
save(secondary_boot, "H4_NULL2T_secondary_4h_bootstrap.csv")
save(reg, "H4_NULL2T_continuous_trend_regression.csv")
save(year_summary, "H4_NULL2T_year_alignment_summary.csv")

config = f"""H4.NULL2T — TREND-CONDITIONED MIRROR PLACEBO

PRIMARY TREND:
24h volatility-normalized trailing return.
Z24 = log(C[t-1]/C[t-24h-1]) / sqrt(sum M1 log-return^2 over same completed window)

FIXED REGIMES:
NEUTRAL: |Z24| < {NEUTRAL}
MODERATE: {NEUTRAL} <= |Z24| < {STRONG}
STRONG: |Z24| >= {STRONG}

ALIGNMENT:
Trend direction compared with REAL nearer barrier side.

PRIMARY OUTCOME:
real_nearer_won - mirror_nearer_won

PRIMARY QUESTION:
Does the NULL2 real-vs-mirror advantage survive after conditioning
on pre-existing trend direction and strength?

SECONDARY:
4h normalized momentum. 24h remains primary regardless of results.

BOOTSTRAP:
Calendar-month block bootstrap, {BOOT_REPS} reps, seed {SEED}.

TREND-CONFOUND SUPPORTED IF:
- advantage ~0 in neutral,
- advantage positive when aligned,
- advantage <=0 when opposed,
- continuous alignment slope >0.

STRUCTURAL RESIDUAL SUPPORTED IF:
- real still beats mirror in neutral,
- advantage survives opposed states,
- meaningful positive residual remains after trend conditioning.

No trading backtest. No GMMA/DC. No threshold optimization.
"""
(OUT / "H4_NULL2T_config.txt").write_text(config, encoding="utf-8")

print("\nPRIMARY 24H SUMMARY")
print(primary.to_string(index=False))
print("\nPRIMARY 24H BOOTSTRAP")
print(primary_boot.to_string(index=False))
print("\nCONTINUOUS TREND TEST")
print(reg.to_string(index=False))
print("\nSaved to:", OUT)

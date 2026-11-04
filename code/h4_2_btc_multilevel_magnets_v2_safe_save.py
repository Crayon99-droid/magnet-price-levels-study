from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.2 — BTC MULTI-LEVEL LIQUIDITY MAGNET BANDS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FOLDER = SCRIPT_DIR

FILES = [
    "BTCUSDT_1m_2021.csv.zip",
    "BTCUSDT_1m_2022.csv.zip",
    "BTCUSDT_1m_2023.csv.zip",
    "BTCUSDT_1m_2024.csv.zip",
]

OUTPUT_FOLDER = SCRIPT_DIR / "H4_2_results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

DECISION_TF = "15min"
ATR_LENGTH = 14
CLUSTER_ATR_FRACTION = 0.15
SWING_LEFT = 2
SWING_RIGHT = 2
SWING_LOOKBACK_DAYS = 7
UTC_BLOCK_HOURS = 8
MAX_EVENT_HOURS = 24
MIN_LEVELS_PER_SIDE = 1

# Save a rolling checkpoint so a long run is never lost again.
CHECKPOINT_EVERY_EVENTS = 250


def ensure_output_folder():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def safe_to_csv(frame, filename, index=False):
    """
    Save into an absolute folder next to this script.
    Re-create the directory immediately before each save.
    """
    ensure_output_folder()
    path = OUTPUT_FOLDER / filename
    frame.to_csv(path, index=index)
    return path


def read_zip_csv(zip_path):
    print(f"\nLoading: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV found inside {zip_path}")
        csv_name = csv_files[0]
        print(f"  CSV inside ZIP: {csv_name}")
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    print(f"  Rows loaded: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
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


def previous_week_start(ts):
    day = ts.floor("D")
    return day - pd.Timedelta(days=day.weekday())


def current_block_start(ts):
    day = ts.floor("D")
    block_hour = (ts.hour // UTC_BLOCK_HOURS) * UTC_BLOCK_HOURS
    return day + pd.Timedelta(hours=block_hour)


def make_level(level_type, side, price, active_from, origin_time):
    return {
        "level_type": level_type,
        "side": side,
        "price": float(price),
        "active_from": active_from,
        "origin_time": origin_time,
    }


def was_swept(level, decision_time, minute_indexed):
    start = level["active_from"]

    if pd.isna(start) or start >= decision_time:
        return False

    hist = minute_indexed.loc[
        (minute_indexed.index >= start) &
        (minute_indexed.index < decision_time)
    ]

    if hist.empty:
        return False

    if level["side"] == "UPPER":
        return bool((hist["high"] >= level["price"]).any())
    return bool((hist["low"] <= level["price"]).any())


def cluster_levels(levels, tolerance):
    if not levels:
        return []

    ordered = sorted(levels, key=lambda x: x["price"])
    groups = []
    current = [ordered[0]]

    for level in ordered[1:]:
        if level["price"] - current[-1]["price"] <= tolerance:
            current.append(level)
        else:
            groups.append(current)
            current = [level]

    groups.append(current)

    clusters = []
    for members in groups:
        prices = [m["price"] for m in members]
        types = sorted(set(m["level_type"] for m in members))
        clusters.append({
            "band_low": float(min(prices)),
            "band_high": float(max(prices)),
            "center": float(np.mean(prices)),
            "level_count": len(members),
            "source_diversity": len(types),
            "types": types,
            "members": members,
        })

    return clusters


def serialize_members(members):
    parts = []
    for m in sorted(members, key=lambda x: x["price"]):
        parts.append(
            f"{m['level_type']}@{m['price']:.2f}"
            f"[origin={m['origin_time']}]"
        )
    return " | ".join(parts)


all_data = []

for filename in FILES:
    path = DATA_FOLDER / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing file:\n{path.resolve()}\n"
            "Put this script in the same folder as the four BTC ZIP files."
        )

    raw = read_zip_csv(path)
    clean = clean_dataframe(raw)
    print(
        f"  Clean period: {clean['timestamp'].min()} -> "
        f"{clean['timestamp'].max()}"
    )
    all_data.append(clean)

df = pd.concat(all_data, ignore_index=True)

df = (
    df.sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

print("\n============================================")
print("PATHS")
print("============================================")
print(f"Script folder: {SCRIPT_DIR}")
print(f"Output folder: {OUTPUT_FOLDER}")
ensure_output_folder()

print("\n============================================")
print("FULL 1-MINUTE DATASET")
print("============================================")
print(f"Rows:  {len(df):,}")
print(f"Start: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")

m1 = df.set_index("timestamp")[["open", "high", "low", "close"]].copy()

bars15 = (
    m1.resample(
        DECISION_TF,
        label="left",
        closed="left",
    )
    .agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })
    .dropna()
)

prev_close = bars15["close"].shift(1)

tr = pd.concat(
    [
        bars15["high"] - bars15["low"],
        (bars15["high"] - prev_close).abs(),
        (bars15["low"] - prev_close).abs(),
    ],
    axis=1,
).max(axis=1)

bars15["ATR"] = tr.rolling(
    ATR_LENGTH,
    min_periods=ATR_LENGTH
).mean()

highs = bars15["high"]
lows = bars15["low"]

swing_high = pd.Series(True, index=bars15.index)
swing_low = pd.Series(True, index=bars15.index)

for k in range(1, SWING_LEFT + 1):
    swing_high &= highs > highs.shift(k)
    swing_low &= lows < lows.shift(k)

for k in range(1, SWING_RIGHT + 1):
    swing_high &= highs > highs.shift(-k)
    swing_low &= lows < lows.shift(-k)

swing_records = []
swing_indices = np.where((swing_high | swing_low).fillna(False).values)[0]

for idx in swing_indices:
    ts = bars15.index[idx]
    confirm_idx = idx + SWING_RIGHT + 1

    if confirm_idx >= len(bars15):
        continue

    active_from = bars15.index[confirm_idx]

    if bool(swing_high.iloc[idx]):
        swing_records.append({
            "origin_time": ts,
            "active_from": active_from,
            "price": float(bars15["high"].iloc[idx]),
            "side": "UPPER",
            "level_type": "SWING_HIGH_15M",
        })

    if bool(swing_low.iloc[idx]):
        swing_records.append({
            "origin_time": ts,
            "active_from": active_from,
            "price": float(bars15["low"].iloc[idx]),
            "side": "LOWER",
            "level_type": "SWING_LOW_15M",
        })

swings = pd.DataFrame(swing_records)

if not swings.empty:
    swings = swings.sort_values("active_from").reset_index(drop=True)

print("\n15m bars:", f"{len(bars15):,}")
print("Confirmed swing levels:", f"{len(swings):,}")

daily = (
    m1.resample("1D", label="left", closed="left")
    .agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })
    .dropna()
)

weekly = (
    m1.resample(
        "W-MON",
        label="left",
        closed="left",
    )
    .agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })
    .dropna()
)

blocks8 = (
    m1.resample(
        f"{UTC_BLOCK_HOURS}h",
        label="left",
        closed="left",
        origin="start_day",
    )
    .agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })
    .dropna()
)


def candidate_levels_at(decision_time, current_price):
    levels = []

    day_start = decision_time.floor("D")
    prev_day_start = day_start - pd.Timedelta(days=1)

    if prev_day_start in daily.index:
        r = daily.loc[prev_day_start]
        levels.append(make_level("PDH", "UPPER", r["high"], day_start, prev_day_start))
        levels.append(make_level("PDL", "LOWER", r["low"], day_start, prev_day_start))

    week_start = previous_week_start(decision_time)
    prev_week = week_start - pd.Timedelta(days=7)

    if prev_week in weekly.index:
        r = weekly.loc[prev_week]
        levels.append(make_level("PWH", "UPPER", r["high"], week_start, prev_week))
        levels.append(make_level("PWL", "LOWER", r["low"], week_start, prev_week))

    block_start = current_block_start(decision_time)
    prev_block_start = block_start - pd.Timedelta(hours=UTC_BLOCK_HOURS)

    if prev_block_start in blocks8.index:
        r = blocks8.loc[prev_block_start]
        levels.append(
            make_level(
                "PREV_8H_HIGH",
                "UPPER",
                r["high"],
                block_start,
                prev_block_start,
            )
        )
        levels.append(
            make_level(
                "PREV_8H_LOW",
                "LOWER",
                r["low"],
                block_start,
                prev_block_start,
            )
        )

    if not swings.empty:
        earliest = decision_time - pd.Timedelta(days=SWING_LOOKBACK_DAYS)

        recent_swings = swings[
            (swings["active_from"] <= decision_time) &
            (swings["origin_time"] >= earliest)
        ]

        for s in recent_swings.itertuples(index=False):
            levels.append({
                "level_type": s.level_type,
                "side": s.side,
                "price": float(s.price),
                "active_from": s.active_from,
                "origin_time": s.origin_time,
            })

    active = []

    for level in levels:
        if level["side"] == "UPPER" and level["price"] <= current_price:
            continue
        if level["side"] == "LOWER" and level["price"] >= current_price:
            continue
        if was_swept(level, decision_time, m1):
            continue
        active.append(level)

    return active


events = []
components = []

decision_times = bars15.index
start_i = max(ATR_LENGTH + 5, 20)
next_allowed_time = decision_times[start_i]

print("\n============================================")
print("BUILDING NON-OVERLAPPING MAGNET EVENTS")
print("============================================")

event_id = 0

for i in range(start_i, len(decision_times)):
    decision_time = decision_times[i]

    if decision_time < next_allowed_time:
        continue

    atr = bars15["ATR"].iloc[i - 1]

    if pd.isna(atr) or atr <= 0:
        continue

    current_price = bars15["open"].iloc[i]
    tolerance = float(CLUSTER_ATR_FRACTION * atr)

    levels = candidate_levels_at(decision_time, current_price)

    upper_levels = [x for x in levels if x["side"] == "UPPER"]
    lower_levels = [x for x in levels if x["side"] == "LOWER"]

    if (
        len(upper_levels) < MIN_LEVELS_PER_SIDE or
        len(lower_levels) < MIN_LEVELS_PER_SIDE
    ):
        continue

    upper_clusters = cluster_levels(upper_levels, tolerance)
    lower_clusters = cluster_levels(lower_levels, tolerance)

    if not upper_clusters or not lower_clusters:
        continue

    upper = min(
        upper_clusters,
        key=lambda c: c["band_low"] - current_price
    )

    lower = min(
        lower_clusters,
        key=lambda c: current_price - c["band_high"]
    )

    upper_distance = upper["band_low"] - current_price
    lower_distance = current_price - lower["band_high"]

    if upper_distance <= 0 or lower_distance <= 0:
        continue

    if upper_distance < lower_distance:
        nearer_side = "UPPER"
    elif lower_distance < upper_distance:
        nearer_side = "LOWER"
    else:
        nearer_side = "EQUAL"

    upper_strength_key = (
        upper["level_count"],
        upper["source_diversity"],
    )
    lower_strength_key = (
        lower["level_count"],
        lower["source_diversity"],
    )

    if upper_strength_key > lower_strength_key:
        stronger_side = "UPPER"
    elif lower_strength_key > upper_strength_key:
        stronger_side = "LOWER"
    else:
        stronger_side = "EQUAL"

    end_time = decision_time + pd.Timedelta(hours=MAX_EVENT_HOURS)

    future = m1.loc[
        (m1.index >= decision_time) &
        (m1.index < end_time)
    ]

    outcome = "TIMEOUT"
    touch_time = pd.NaT
    ambiguous = False

    for ts, bar in future.iterrows():
        hit_upper = bar["high"] >= upper["band_low"]
        hit_lower = bar["low"] <= lower["band_high"]

        if hit_upper and hit_lower:
            outcome = "BOTH_SAME_BAR"
            touch_time = ts
            ambiguous = True
            break
        elif hit_upper:
            outcome = "UPPER"
            touch_time = ts
            break
        elif hit_lower:
            outcome = "LOWER"
            touch_time = ts
            break

    if pd.isna(touch_time):
        event_end = end_time
    else:
        event_end = touch_time + pd.Timedelta(minutes=1)

    next_allowed_time = event_end.ceil(DECISION_TF)

    event_id += 1

    farther_side = (
        "LOWER" if nearer_side == "UPPER"
        else "UPPER" if nearer_side == "LOWER"
        else "EQUAL"
    )

    farther_is_stronger = (
        stronger_side != "EQUAL" and
        farther_side == stronger_side
    )

    nearer_correct = (
        nearer_side == outcome
        if outcome in ["UPPER", "LOWER"] and nearer_side != "EQUAL"
        else np.nan
    )

    stronger_correct = (
        stronger_side == outcome
        if outcome in ["UPPER", "LOWER"] and stronger_side != "EQUAL"
        else np.nan
    )

    events.append({
        "event_id": event_id,
        "decision_time": decision_time,
        "year": decision_time.year,
        "entry_price": current_price,
        "atr15": atr,
        "cluster_tolerance": tolerance,

        "upper_band_low": upper["band_low"],
        "upper_band_high": upper["band_high"],
        "upper_center": upper["center"],
        "upper_distance": upper_distance,
        "upper_distance_atr": upper_distance / atr,
        "upper_level_count": upper["level_count"],
        "upper_source_diversity": upper["source_diversity"],
        "upper_types": ",".join(upper["types"]),
        "upper_members": serialize_members(upper["members"]),

        "lower_band_low": lower["band_low"],
        "lower_band_high": lower["band_high"],
        "lower_center": lower["center"],
        "lower_distance": lower_distance,
        "lower_distance_atr": lower_distance / atr,
        "lower_level_count": lower["level_count"],
        "lower_source_diversity": lower["source_diversity"],
        "lower_types": ",".join(lower["types"]),
        "lower_members": serialize_members(lower["members"]),

        "nearer_side": nearer_side,
        "stronger_side": stronger_side,
        "farther_side": farther_side,
        "farther_is_stronger": farther_is_stronger,

        "first_touch": outcome,
        "touch_time": touch_time,
        "hours_to_touch": (
            (touch_time - decision_time).total_seconds() / 3600
            if pd.notna(touch_time)
            else np.nan
        ),

        "nearer_correct": nearer_correct,
        "stronger_correct": stronger_correct,
        "ambiguous_same_bar": ambiguous,
    })

    for side_name, cluster in [("UPPER", upper), ("LOWER", lower)]:
        for member in cluster["members"]:
            components.append({
                "event_id": event_id,
                "decision_time": decision_time,
                "side": side_name,
                "band_low": cluster["band_low"],
                "band_high": cluster["band_high"],
                "level_type": member["level_type"],
                "level_price": member["price"],
                "origin_time": member["origin_time"],
                "active_from": member["active_from"],
            })

    if event_id % CHECKPOINT_EVERY_EVENTS == 0:
        checkpoint_events = pd.DataFrame(events)
        checkpoint_components = pd.DataFrame(components)

        p1 = safe_to_csv(
            checkpoint_events,
            "H4_2_CHECKPOINT_magnet_events.csv",
            index=False
        )

        p2 = safe_to_csv(
            checkpoint_components,
            "H4_2_CHECKPOINT_magnet_components.csv",
            index=False
        )

        print(
            f"  Events built: {event_id:,} | latest: {decision_time}\n"
            f"  Checkpoint saved: {p1}\n"
            f"  Components saved: {p2}"
        )

events = pd.DataFrame(events)
components = pd.DataFrame(components)

if events.empty:
    raise RuntimeError("No H4.2 events were generated.")

resolved = events[
    events["first_touch"].isin(["UPPER", "LOWER"])
].copy()

print("\n============================================")
print("H4.2 MULTI-LEVEL MAGNET RESULTS")
print("============================================")
print(f"Total events:          {len(events):,}")
print(f"Resolved:              {len(resolved):,}")
print(f"Timeouts:              {(events['first_touch'] == 'TIMEOUT').sum():,}")
print(f"Ambiguous same 1m bar: {(events['first_touch'] == 'BOTH_SAME_BAR').sum():,}")

nearer_test = resolved[
    resolved["nearer_side"].isin(["UPPER", "LOWER"])
].copy()

nearer_accuracy = nearer_test["nearer_correct"].mean()

print("\n--------------------------------------------")
print("BASELINE: NEARER MAGNET")
print("--------------------------------------------")
print(f"Observations: {len(nearer_test):,}")
print(f"Nearer magnet touched first: {nearer_accuracy:.2%}")

strength_test = resolved[
    resolved["stronger_side"].isin(["UPPER", "LOWER"])
].copy()

stronger_accuracy = strength_test["stronger_correct"].mean()

print("\n--------------------------------------------")
print("RAW STRENGTH TEST")
print("--------------------------------------------")
print("Strength = higher level count, then source diversity.")
print(f"Observations with unequal strength: {len(strength_test):,}")
print(f"Stronger magnet touched first:      {stronger_accuracy:.2%}")

counter_distance = resolved[
    resolved["farther_is_stronger"] == True
].copy()

if len(counter_distance) > 0:
    farther_stronger_win = (
        counter_distance["first_touch"] ==
        counter_distance["stronger_side"]
    ).mean()
else:
    farther_stronger_win = np.nan

print("\n--------------------------------------------")
print("CRUCIAL TEST: FARTHER BUT STRONGER MAGNET")
print("--------------------------------------------")
print(f"Observations: {len(counter_distance):,}")

if len(counter_distance) > 0:
    print(
        "Farther-but-stronger magnet touched first: "
        f"{farther_stronger_win:.2%}"
    )
    print(
        "Nearer magnet touched first in same subset: "
        f"{1 - farther_stronger_win:.2%}"
    )
else:
    print("No observations.")


def cluster_class(n):
    if n <= 1:
        return "1"
    elif n == 2:
        return "2"
    return "3+"


resolved["upper_count_class"] = resolved["upper_level_count"].map(cluster_class)
resolved["lower_count_class"] = resolved["lower_level_count"].map(cluster_class)

combo_summary = (
    resolved
    .groupby(
        ["upper_count_class", "lower_count_class"],
        observed=True
    )
    .agg(
        observations=("event_id", "size"),
        upper_first_rate=(
            "first_touch",
            lambda x: (x == "UPPER").mean()
        ),
        nearer_accuracy=("nearer_correct", "mean"),
    )
    .reset_index()
)

print("\n--------------------------------------------")
print("CLUSTER COUNT COMBINATIONS")
print("--------------------------------------------")
print(combo_summary.to_string(index=False))

resolved["count_diff_upper_minus_lower"] = (
    resolved["upper_level_count"] -
    resolved["lower_level_count"]
)

strength_diff_summary = (
    resolved
    .groupby("count_diff_upper_minus_lower")
    .agg(
        observations=("event_id", "size"),
        upper_first_rate=(
            "first_touch",
            lambda x: (x == "UPPER").mean()
        ),
        mean_upper_distance_atr=("upper_distance_atr", "mean"),
        mean_lower_distance_atr=("lower_distance_atr", "mean"),
    )
    .reset_index()
    .sort_values("count_diff_upper_minus_lower")
)

resolved["upper_to_lower_distance_ratio"] = (
    resolved["upper_distance"] /
    resolved["lower_distance"]
)

distance_bins = [0, 0.50, 0.75, 1.00, 1.25, 2.00, np.inf]

resolved["distance_ratio_bucket"] = pd.cut(
    resolved["upper_to_lower_distance_ratio"],
    bins=distance_bins,
    include_lowest=True,
)

distance_summary = (
    resolved
    .groupby("distance_ratio_bucket", observed=True)
    .agg(
        observations=("event_id", "size"),
        upper_first_rate=(
            "first_touch",
            lambda x: (x == "UPPER").mean()
        ),
        mean_upper_count=("upper_level_count", "mean"),
        mean_lower_count=("lower_level_count", "mean"),
    )
    .reset_index()
)

year_summary = (
    resolved
    .groupby("year")
    .agg(
        observations=("event_id", "size"),
        upper_first_rate=(
            "first_touch",
            lambda x: (x == "UPPER").mean()
        ),
        nearer_accuracy=("nearer_correct", "mean"),
        stronger_accuracy=("stronger_correct", "mean"),
        mean_hours_to_touch=("hours_to_touch", "mean"),
    )
    .reset_index()
)

counter_year = (
    counter_distance
    .groupby("year")
    .agg(
        observations=("event_id", "size"),
        farther_stronger_first_rate=("stronger_correct", "mean"),
    )
    .reset_index()
)

safe_to_csv(
    events,
    "H4_2_magnet_events.csv",
    index=False,
)

safe_to_csv(
    components,
    "H4_2_magnet_components.csv",
    index=False,
)

safe_to_csv(
    combo_summary,
    "H4_2_cluster_count_summary.csv",
    index=False,
)

safe_to_csv(
    strength_diff_summary,
    "H4_2_strength_difference_summary.csv",
    index=False,
)

safe_to_csv(
    distance_summary,
    "H4_2_distance_ratio_summary.csv",
    index=False,
)

safe_to_csv(
    year_summary,
    "H4_2_year_summary.csv",
    index=False,
)

safe_to_csv(
    counter_distance,
    "H4_2_farther_but_stronger_events.csv",
    index=False,
)

safe_to_csv(
    counter_year,
    "H4_2_farther_but_stronger_year_summary.csv",
    index=False,
)

config_text = f"""H4.2 BTC MULTI-LEVEL LIQUIDITY MAGNET CONFIGURATION

DECISION_TF={DECISION_TF}
ATR_LENGTH={ATR_LENGTH}
CLUSTER_ATR_FRACTION={CLUSTER_ATR_FRACTION}
SWING_LEFT={SWING_LEFT}
SWING_RIGHT={SWING_RIGHT}
SWING_LOOKBACK_DAYS={SWING_LOOKBACK_DAYS}
UTC_BLOCK_HOURS={UTC_BLOCK_HOURS}
MAX_EVENT_HOURS={MAX_EVENT_HOURS}
MIN_LEVELS_PER_SIDE={MIN_LEVELS_PER_SIDE}

Candidate level families:
- PDH / PDL
- PWH / PWL
- Previous completed {UTC_BLOCK_HOURS}h UTC block high / low
- Confirmed {DECISION_TF} swing highs / lows

Events are non-overlapping.
No optimization is performed in this script.
"""

ensure_output_folder()
(OUTPUT_FOLDER / "H4_2_config.txt").write_text(
    config_text,
    encoding="utf-8"
)

print("\n============================================")
print("FILES SAVED")
print("============================================")
for file in sorted(OUTPUT_FOLDER.iterdir()):
    print(file)

print("\nDone.")

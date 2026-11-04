from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# H4.1 — BTC PREVIOUS-DAY HIGH / LOW LIQUIDITY MAGNET TEST
# V2 — explicit Bybit timestamp_utc handling
# ============================================================

DATA_FOLDER = Path(".")

FILES = [
    "BTCUSDT_1m_2021.csv.zip",
    "BTCUSDT_1m_2022.csv.zip",
    "BTCUSDT_1m_2023.csv.zip",
    "BTCUSDT_1m_2024.csv.zip",
]

OUTPUT_FOLDER = Path("H4_1_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD CSV FROM ZIP
# ============================================================

def read_zip_csv(zip_path):
    print(f"\nLoading: {zip_path}")

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
    print(f"  Columns: {list(df.columns)}")

    return df


# ============================================================
# 2. CLEAN BYBIT DATA
# ============================================================

def clean_dataframe(df):
    """
    Expected Bybit columns:
    timestamp_utc, timestamp_ms, open, high, low, close,
    volume_base, turnover_usdt

    We explicitly use timestamp_utc.
    """

    required_columns = [
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing expected Bybit columns: "
            f"{missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    result = pd.DataFrame()

    result["timestamp"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    for name in ["open", "high", "low", "close"]:
        result[name] = pd.to_numeric(
            df[name],
            errors="coerce"
        )

    result = (
        result
        .dropna()
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# 3. LOAD AND COMBINE ALL YEARS
# ============================================================

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
        f"  Clean period: "
        f"{clean['timestamp'].min()} -> "
        f"{clean['timestamp'].max()}"
    )

    all_data.append(clean)


df = pd.concat(
    all_data,
    ignore_index=True
)

df = (
    df
    .sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

print("\n============================================")
print("FULL DATASET")
print("============================================")
print(f"Rows:  {len(df):,}")
print(f"Start: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")


# ============================================================
# 4. CREATE DAILY LEVELS
# ============================================================

# BTC is 24/7. We use UTC calendar days.
df["date"] = df["timestamp"].dt.floor("D")

daily = (
    df.groupby("date")
      .agg(
          day_open=("open", "first"),
          day_high=("high", "max"),
          day_low=("low", "min"),
          day_close=("close", "last"),
          bars=("timestamp", "count"),
      )
      .reset_index()
)

daily["PDH"] = daily["day_high"].shift(1)
daily["PDL"] = daily["day_low"].shift(1)
daily["prev_close"] = daily["day_close"].shift(1)
daily["prev_range"] = daily["PDH"] - daily["PDL"]

daily["inside_prev_range"] = (
    (daily["day_open"] < daily["PDH"]) &
    (daily["day_open"] > daily["PDL"])
)

print("\n============================================")
print("DAILY DATA")
print("============================================")
print(f"Trading days: {len(daily):,}")
print(
    "Days opening inside previous day's range: "
    f"{daily['inside_prev_range'].sum():,}"
)


# ============================================================
# 5. FIRST-TOUCH TEST
# ============================================================

events = []

valid_days = daily[
    daily["inside_prev_range"] &
    daily["PDH"].notna() &
    daily["PDL"].notna()
].copy()


for row in valid_days.itertuples():

    date = row.date
    entry = row.day_open
    pdh = row.PDH
    pdl = row.PDL

    day_bars = df[df["date"] == date]

    upper_distance = pdh - entry
    lower_distance = entry - pdl

    if upper_distance < lower_distance:
        predicted_side = "PDH"
    elif lower_distance < upper_distance:
        predicted_side = "PDL"
    else:
        predicted_side = "EQUAL"

    actual_side = "NEITHER"
    touch_time = pd.NaT
    ambiguous = False

    for bar in day_bars.itertuples():

        hit_pdh = bar.high >= pdh
        hit_pdl = bar.low <= pdl

        if hit_pdh and hit_pdl:
            actual_side = "BOTH_SAME_BAR"
            touch_time = bar.timestamp
            ambiguous = True
            break

        elif hit_pdh:
            actual_side = "PDH"
            touch_time = bar.timestamp
            break

        elif hit_pdl:
            actual_side = "PDL"
            touch_time = bar.timestamp
            break

    correct_prediction = (
        predicted_side == actual_side
        if actual_side in ["PDH", "PDL"]
        else np.nan
    )

    events.append({
        "date": date,
        "entry": entry,
        "PDH": pdh,
        "PDL": pdl,
        "previous_range": row.prev_range,
        "distance_to_PDH": upper_distance,
        "distance_to_PDL": lower_distance,
        "normalized_position": (
            (entry - pdl) /
            (pdh - pdl)
        ),
        "predicted_nearest_magnet": predicted_side,
        "first_touch": actual_side,
        "touch_time": touch_time,
        "ambiguous_same_bar": ambiguous,
        "prediction_correct": correct_prediction,
    })


events = pd.DataFrame(events)


# ============================================================
# 6. SIMPLE 1:1 TRADE TOWARD NEAREST MAGNET
# ============================================================

trade_results = []

for row in events.itertuples():

    if row.predicted_nearest_magnet not in ["PDH", "PDL"]:
        continue

    date = row.date
    entry = row.entry

    day_bars = df[df["date"] == date]

    if row.predicted_nearest_magnet == "PDH":

        side = "LONG"
        target = row.PDH
        risk = target - entry
        stop = entry - risk

    else:

        side = "SHORT"
        target = row.PDL
        risk = entry - target
        stop = entry + risk

    if risk <= 0:
        continue

    outcome = "NO_EXIT"
    exit_time = pd.NaT
    ambiguous = False

    for bar in day_bars.itertuples():

        if side == "LONG":
            target_hit = bar.high >= target
            stop_hit = bar.low <= stop
        else:
            target_hit = bar.low <= target
            stop_hit = bar.high >= stop

        if target_hit and stop_hit:
            outcome = "BOTH_SAME_BAR"
            exit_time = bar.timestamp
            ambiguous = True
            break

        elif target_hit:
            outcome = "WIN"
            exit_time = bar.timestamp
            break

        elif stop_hit:
            outcome = "LOSS"
            exit_time = bar.timestamp
            break

    if outcome == "WIN":
        R = 1.0
    elif outcome == "LOSS":
        R = -1.0
    else:
        R = np.nan

    trade_results.append({
        "date": date,
        "side": side,
        "entry": entry,
        "target": target,
        "stop": stop,
        "risk_distance": risk,
        "outcome": outcome,
        "exit_time": exit_time,
        "ambiguous_same_bar": ambiguous,
        "R": R,
    })


trades = pd.DataFrame(trade_results)


# ============================================================
# 7. SUMMARY STATISTICS
# ============================================================

resolved = events[
    events["first_touch"].isin(["PDH", "PDL"])
].copy()

upper_first = (
    resolved["first_touch"] == "PDH"
).mean()

lower_first = (
    resolved["first_touch"] == "PDL"
).mean()

nearest_accuracy = (
    resolved["prediction_correct"]
    .astype(float)
    .mean()
)


print("\n\n============================================")
print("H4.1 — PDH / PDL MAGNET TEST")
print("============================================")

print(f"Valid observations: {len(events):,}")
print(f"Resolved same day:  {len(resolved):,}")

print(
    f"Neither level touched: "
    f"{(events['first_touch'] == 'NEITHER').sum():,}"
)

print(
    f"Ambiguous same candle: "
    f"{(events['first_touch'] == 'BOTH_SAME_BAR').sum():,}"
)

print("\nFIRST TOUCH DISTRIBUTION")
print(f"PDH first: {upper_first:.2%}")
print(f"PDL first: {lower_first:.2%}")

print("\nNAIVE DISTANCE MODEL")
print(
    "Accuracy of simply choosing the closer level: "
    f"{nearest_accuracy:.2%}"
)


# ============================================================
# 8. 1:1 TRADE RESULTS
# ============================================================

clean_trades = trades[
    trades["outcome"].isin(["WIN", "LOSS"])
].copy()

wins = (
    clean_trades["outcome"] == "WIN"
).sum()

losses = (
    clean_trades["outcome"] == "LOSS"
).sum()

total = wins + losses

if total > 0:

    win_rate = wins / total

    expectancy_R = clean_trades["R"].mean()

    gross_profit = clean_trades.loc[
        clean_trades["R"] > 0,
        "R"
    ].sum()

    gross_loss = abs(
        clean_trades.loc[
            clean_trades["R"] < 0,
            "R"
        ].sum()
    )

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf

else:

    win_rate = np.nan
    expectancy_R = np.nan
    profit_factor = np.nan


print("\n============================================")
print("SIMPLE 1:1 TRADE TOWARD NEAREST MAGNET")
print("============================================")

print(f"Resolved trades: {total:,}")
print(f"Wins:            {wins:,}")
print(f"Losses:          {losses:,}")
print(f"Win rate:        {win_rate:.2%}")
print(f"Expectancy:      {expectancy_R:.4f} R/trade")
print(f"Profit factor:   {profit_factor:.4f}")

print(
    "Ambiguous same candle: "
    f"{(trades['outcome'] == 'BOTH_SAME_BAR').sum():,}"
)

print(
    "No exit during day: "
    f"{(trades['outcome'] == 'NO_EXIT').sum():,}"
)


# ============================================================
# 9. POSITION-IN-RANGE ANALYSIS
# ============================================================

resolved["position_bucket"] = pd.cut(
    resolved["normalized_position"],
    bins=[
        0.00,
        0.20,
        0.40,
        0.60,
        0.80,
        1.00,
    ],
    include_lowest=True
)

bucket_summary = (
    resolved
    .groupby(
        "position_bucket",
        observed=True
    )
    .agg(
        observations=("first_touch", "size"),

        PDH_first_rate=(
            "first_touch",
            lambda x: (x == "PDH").mean()
        ),

        nearest_accuracy=(
            "prediction_correct",
            "mean"
        ),
    )
    .reset_index()
)

print("\n============================================")
print("WHERE DID THE DAY OPEN INSIDE YESTERDAY'S RANGE?")
print("============================================")
print(bucket_summary.to_string(index=False))


# ============================================================
# 10. YEAR-BY-YEAR STABILITY
# ============================================================

resolved["year"] = resolved["date"].dt.year

year_summary = (
    resolved
    .groupby("year")
    .agg(
        observations=("first_touch", "size"),

        PDH_first_rate=(
            "first_touch",
            lambda x: (x == "PDH").mean()
        ),

        nearest_accuracy=(
            "prediction_correct",
            "mean"
        ),
    )
    .reset_index()
)

print("\n============================================")
print("YEAR-BY-YEAR RESULTS")
print("============================================")
print(year_summary.to_string(index=False))


# ============================================================
# 11. SAVE EVERYTHING
# ============================================================

events.to_csv(
    OUTPUT_FOLDER / "H4_1_daily_magnet_events.csv",
    index=False
)

trades.to_csv(
    OUTPUT_FOLDER / "H4_1_nearest_magnet_1R_trades.csv",
    index=False
)

bucket_summary.to_csv(
    OUTPUT_FOLDER / "H4_1_position_bucket_summary.csv",
    index=False
)

year_summary.to_csv(
    OUTPUT_FOLDER / "H4_1_year_summary.csv",
    index=False
)

daily.to_csv(
    OUTPUT_FOLDER / "H4_1_daily_levels.csv",
    index=False
)


print("\n============================================")
print("FILES SAVED")
print("============================================")

for file in sorted(OUTPUT_FOLDER.iterdir()):
    print(file)

print("\nDone.")

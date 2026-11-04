# BTCUSDT Magnetic Levels + Directional Change — First Crypto Experiment

**Research date:** 2026-10-12  
**Status:** Completed engineering and falsification pilot; **no demonstrated trading edge**.  
**Asset:** BTCUSDT Bybit USDT linear perpetual, 1-minute klines, 2021–2024.

## Executive conclusion

The four supplied ZIP archives exactly matched the known 2026-09-03 Bybit source manifest hashes. All 2,103,840 minute candles were continuous and internally consistent. The original H4.6B directional-change threshold was reconstructed from the historical research specification, but the original **H4.2 magnetic-event table was not supplied**. Magnetic candidates were therefore reconstructed causally in a **new related study**, not presented as an exact reproduction of the older H4.2 backtest.

Directional accuracy is insufficient: among 6,354 resolved next-destination events, the nearer magnetic level was reached first **73.07%** of the time versus **73.04%** predicted by a simple distance-based, driftless first-passage heuristic. Weekly block-bootstrap 95% interval for the difference was **−1.03 to +1.13 percentage points**. The heuristic is not a formally calibrated market null, especially after conditioning on resolution within 24 hours.

All tested *tradable* rules had **negative average R after modeled fees and slippage**. Earlier timeouts and a causal observed-excursion profit lock did not generate consistent positive expectancy. No strategy is ready for deployment.

## 1. Input-source audit

| Year | Minutes | SHA-256 prefix | UTC chronology and OHLCV |
|---|---:|---|---|
| 2021 | 525,600 | `620a6f5f7814db6c` | PASS |
| 2022 | 525,600 | `65604ac53fcb89f9` | PASS |
| 2023 | 525,600 | `778e89c855d26559` | PASS |
| 2024 | 527,040 | `710ae80850facbd6` | PASS |
| **Total** | **2,103,840** | | **PASS** |

The audit validated each ZIP CRC, exact archived member name, full archive SHA-256, timestamp-text/millisecond agreement, OHLC consistency, numeric integrity, and uninterrupted 60,000-millisecond chronology including year boundaries. No interpolation was performed.

## 2. Causal signal construction

- **DC event:** Directional reversal confirmed only on a completed 1-minute close.
- **Volatility threshold:** `theta[t] = 1.25 × median(yesterday's rolling 240m realized volatility)` where rolling volatility is the square root of the 240-minute sum of squared 1-minute log returns; the entire preceding UTC day must have ended.
- **Magnetic levels:** Previous completed UTC-day highs/lows, previous completed Monday-aligned week highs/lows, completed 8h high/low, and M15 pivots confirmed only after two subsequent 15-minute bars finish.
- **Level expiration:** Day 7 days; week 28 days; 8h 3 days; confirmed M15 pivot 5 days, or earlier upon first OHLC touch.
- **Entry:** Following M1 candle's **open**, not same-minute close. One position at a time per policy; same-minute stop and target crossings resolved conservatively against the trade.
- **Risk:** Initial stop distance = signal-time `theta × entry price` (1R). Tested gross take-profits +0.25R, +0.5R, +1R and 30- or 120-minute deadlines. A separate 24-hour horizon was used only to study which **magnetic destination is touched first**, not to keep trades open for 24 hours.
- **Trading-cost assumption:** 5.5bp taker fee per fill (11bp round trip) plus 1bp assumed execution friction per fill (2bp round trip), total **13bp/notional**. These are modeling assumptions, **not reconstructed historical bid/ask quotes or exact historical account-tier fees**.
- **Funding:** Historic funding rates were **not supplied and are not charged**. Trades spanning a nominal 8-hour settlement boundary are flagged. Therefore “net” means **net of modeled transaction fees/slippage, before unknown funding**.
- **Universe:** 7,096 raw DC events; 50,456 constructed historical level records; 6,729 DC events with two valid magnetic sides. One final event lacked a complete destination-label horizon, leaving 6,728 labeled events, including 6,354 resolved hits, 371 censored, and three same-minute ambiguous hits.

### Not the original H4.6B exact reproduction

Original H4.6B required `H4_2_magnet_events.csv`, a frozen H4.2 event definition, and a wait for a subsequent DC reversal when the magnet and current DC direction disagree. This successor pilot reconstructs levels and tests **immediate next-minute entries** on the event opportunity set. The H4.6B deferred-entry arm has **not yet been reproduced**. Results must not be labeled H4.6B replication.

## 3. The destination question

| Year | Resolved events | Near magnet first | Distance-only heuristic |
|---|---:|---:|---:|
| 2021 | 1,489 | 73.27% | 73.99% |
| 2022 | 1,467 | 73.35% | 72.81% |
| 2023 | 1,768 | 72.96% | 72.12% |
| 2024 | 1,630 | 72.76% | 73.41% |
| **Combined** | **6,354** | **73.07%** | **73.04%** |

With near/far distance ratio ≤0.5, near-first rises to 82.22%, but the geometry heuristic also rises to 81.59%. **High hit rate is not evidence of a tradeable informational edge.**

## 4. Actual trades: after assumed 13bp costs

Each policy enforces no overlapping positions and is calculated on its own eligible event stream. Profit factor uses net **R**, not a proxy hit rate. A zero or censored destination label does not suppress a valid entry test. The 2024 year is a chronological checkpoint only: 2021–2024 BTC is already part of historical Find Algo research and should **not** be characterized as never-touched holdout data.

| Policy | 2021–23 trades | Average net R | 2024 trades | Average net R | 2024 PF |
|---|---:|---:|---:|---:|---:|
| DC direction, TP +0.5R, ≤120min | 4,656 | **−0.1927R** | 1,546 | **−0.1510R** | 0.527 |
| Nearest magnet, TP +0.5R, ≤120min | 4,688 | **−0.1627R** | 1,574 | **−0.1409R** | 0.549 |
| DC aligned with nearer magnet and distance ratio ≤0.5, TP +0.5R, ≤120min | 2,683 | **−0.1536R** | 938 | **−0.1209R** | 0.592 |
| Same alignment plus >=0.5R magnet room, TP +0.5R, ≤30min | 572 | **−0.2230R** | 185 | **−0.2450R** | 0.329 |

Weekly block-bootstrap intervals for the *aligned & ratio≤0.5* strategy: **2021–23 [−0.1860, −0.1209] R**, and **2024 [−0.1605, −0.0841] R**. These are firmly on the negative side under the modeled assumptions. Adding just 5bp of transaction friction makes the results worse.

## 5. Profit-taking and overshoot-excursion management

The management comparison uses the **same exact already-selected trade entries**, to prevent a shorter duration from silently adding more trades and biasing comparisons. A causal favorable-excursion proxy is evaluated only after each **completed minute close**; a changed stop takes effect no earlier than the **following minute**. It is not yet a fitted intrinsic-time overshoot classifier.

| Matched-entry exit rule | Mean net R 2021–23 | Mean net R 2024 |
|---|---:|---:|
| Fixed TP +0.5R, initial stop −1R, max 120 min | −0.1536 | −0.1209 |
| Protect profits on completed-bar progression, allow +1R target | −0.1385 | −0.1302 |
| Exit after a confirmed −0.25R adverse close (after 10 min), +0.5R TP | −0.1722 | −0.1368 |
| Profit protection **and** early-failure exit, +1R TP | −0.1591 | −0.1300 |

The profit-protection improvement over fixed +0.5R was **+0.0151R** in 2021–23 (paired weekly bootstrap interval **[+0.0017,+0.0286] R**) but **−0.0092R** in 2024 (interval **[−0.0249,+0.0079] R**). This difference is not robust across years. Early adverse exits made the results worse in both periods.

## 6. What failed and why

1. The apparent magnet-first ~73% accuracy nearly matches geometry alone. The geometry number is a modeling heuristic, not proof of causal liquidity attraction.
2. The financial trade is not the same as the destination first-passage label. Trades may hit the stop or time out even when a magnet is eventually reached.
3. Fee friction is large relative to **net** expected movement. The *magnet-only* 120-minute policy averaged approximately **+3.23bp gross in 2021**, **+1.88bp in 2022**, **+0.59bp in 2023**, and **+0.08bp in 2024**; each is lower than the 13bp modeled round trip.
4. Shorter holding windows increase event capacity but do not automatically produce a favorable gross movement.
5. Favorable excursions can be protected once observed, but there is insufficient evidence that this rule increases overall expectancy.

## 7. What remains untested, not disproven

- **Original H4.6B deferred alignment**: do not enter against an established DC state; wait for a later close-confirmed reversal toward the anchored magnet before the level sweeps.
- **Causal overshoot state forecasting**: features on intrinsic DC-event sequences, including *past* completed OS lengths and contemporaneously observed partial OS, evaluated for *future* continuation after every decision point. Do not use final OS size in the entry feature set.
- **Dynamic economic action threshold**: forecast **gross available movement after entry** conditional on spread, fees, volatility and expected holding time. A destination-hit prediction is insufficient.
- **Fresh markets and years**: BTC 2025–26 if supplied, plus ETH/XRP 2021–26, with complete instrument specifications and historical funding files for the relevant periods. Do not silently reuse EURUSD 15-pip optimization thresholds.

## 8. Reproduce these results on Windows CMD

Unzip the accompanying research bundle. Keep the four *original* BTC ZIPs in a common directory with these scripts. Uploaded file suffixes such as `(2)` are accepted if there is only one archive per year.

```bat
py -m pip install numpy pandas numba
py test_btc_magnet_dc_v2.py
py btc_magnet_dc_v2.py --data-dir . --out btc_magnet_dc_v2
py btc_overshoot_management.py --data-dir . --out btc_magnet_dc_v2
py btc_magnet_dc_stat_audit.py btc_magnet_dc_v2
```

The package includes the full source, unit tests, event and trade CSVs, statistical audit, frozen assumptions, and execution log. Source ZIPs are **not duplicated**.

**Research decision:** REJECT these exact immediate-entry and early-exit rules as profitability candidates. Preserve the tested negative results. Continue hypothesis testing with explicit DC-state transitions, an economic positive-expected-value filter, and genuinely later market data before risking capital.

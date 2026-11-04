# H5 — Multi-scale directional change + magnetic destination prediction

**Research date:** 12 October 2026  
**Decision:** **Do not deploy.** No cross-market, later-year, cost-surviving statistical edge established.  
**Markets:** Bybit BTCUSDT (2021–2024), ETHUSDT and XRPUSDT (2021–22 August 2026) USDT linear perpetual one-minute candles.

## 1. Frozen working hypothesis

At each *completed slow directional-change* event, the preceding short-DC event history, prior overshoot measurements, unresolved magnetic levels, volatility and momentum may allow a model to estimate the **net expected R** of entering either long or short. A candidate is entered only when its predicted net return is positive; otherwise it is skipped. The question is not whether the signal predicts the next candle's sign, but whether it selects trading outcomes surviving execution costs.

**Important limitation of originality:** The original H4.2 magnet event file remains unavailable. The magnetic levels here are a newly reconstructed causal approximation from completed previous day/week/8-hour high/lows and confirmed M15 two-left/two-right pivots, not an exact reproduction of H4.2's clustered bands. This is a new H5 experiment.

## 2. Data audit

16 uploaded yearly archives, **7,740,300** one-minute bars; their hashes agree with the previous locally generated audit (see `source_audit.json`). BTC also agrees with the earlier independent production Bybit manifest. **ETH and XRP do not have independent original source manifests**: hashes certify internal consistency and repeatability, not exchange provenance on their own. ZIP CRC, exact 1-minute chronology, OHLC ordering, and timestamp agreement were checked during the raw-data processing run.

| Asset | Date coverage | Candles | Slow DCs | Eligible dual-magnet slow events |
|---|---|---:|---:|---:|
| BTCUSDT | 2021-01-01–2024-12-31 | 2,103,840 | 7,096 | 6,729 |
| ETHUSDT | 2021-03-15–2026-08-22 19:16 UTC | 2,860,997 | 8,919 | 8,566 |
| XRPUSDT | 2021-05-13 09:34 UTC–2026-08-22 19:16 UTC | 2,775,463 | 8,819 | 8,412 |

The event universe has **23,707** eligible dual-magnet slow DC events, with hypothetical long and short returns recorded separately (**47,414 action labels**). These action labels are *overlapping counterfactuals*, not independent realized trading returns; portfolio simulations select at most one action per event and prohibit simultaneous positions **per symbol**.

## 3. Causal event and execution specification

- **Slow DC:** close-confirmed threshold `1.25 × previous completed UTC day's median rolling 240-minute log-return volatility`.
- **Fast DC:** `0.50 ×` that same base volatility, recording completed fast-event sequences and past overshoots.
- **Magnetic candidates:** previous completed day/week/8-hour extremes plus M15 pivots only when two right-hand bars have closed. Any prior levels already crossed by minute OHLC prices are retired.
- **Features:** 32 variables including fast DC counts in last 120 minutes, 3 previous fast overshoot measurements, previous slow DC overshoot, recent historic price returns, volatility, target distance/risk, magnet confluence, and time of day. No final overshoot length for an open leg is used.
- **Decision:** score both sides at the slow-DC bar close; enter selected direction at **next bar open**, or remain flat.
- **Payoff:** 1R stop, +0.5R take-profit, 120-minute maximum. When TP and stop are touched in the same bar, record stop first.
- **Costs:** round-trip **13 basis points of notional** (11 bp modeled market-taker fees plus 2 bp execution slippage). The data do **not** contain actual historical bid/ask spreads; funding, borrow/financing and market impact are not modeled. This is a modeled, not fully executable, result.
- **Baselines:** unconditional slow DC direction, inverse slow DC, nearest magnet, and DC/magnet alignment with room >=0.5R.
- **Statistical models:** pre-specified ridge regression using DC-only, magnet-only, combined features; and histogram gradient-boosted regression using combined features. Positive predicted **net R** is the trading hurdle; no retrospective cut-off search was applied.
- **Chronological checks:** models trained 2021–2022 were assessed on 2023; models refitted using 2021–2023 were assessed on 2024–2026. Several periods were previously consulted in other studies, so none should be called *pristine independently untouched* confirmation data. No BTC 2025/2026 available.

## 4. Fundamental problem found in two-scale DC confirmations

At a completed slow-DC event, the contemporaneous fast DC already points in the same direction:

| Asset | Fraction of slow events where fast DC already agrees |
|---|---:|
| BTC | 99.69% |
| ETH | 99.80% |
| XRP | 99.76% |

Thus a rule requiring simultaneous fast/slow direction agreement contains **virtually no filtering information** at this decision point. This is a structural relationship between thresholds, not independent evidence of profitable continuation. The *history* of prior smaller DC events still contains variation and was included in ML features.

## 5. Pre-cost versus net economics: unconditional slow DC baseline

The unconditional slow DC model has negligible or negative average *gross* expectancy on the later years. Costs make it clearly negative:

| Asset / year | Gross R/trade | Net R/trade | Non-overlapping trades |
|---|---:|---:|---:|
| BTC 2024 | −0.0103 | −0.1559 | 1,456 |
| ETH 2024 | −0.0112 | −0.1267 | 1,413 |
| ETH 2025 | +0.0023 | −0.0974 | 1,355 |
| XRP 2024 | +0.0019 | −0.0969 | 1,370 |
| XRP 2025 | −0.0010 | −0.0973 | 1,468 |

Costs are especially important when volatility-adjusted 1R stop distances are small; actual cost/R varies by event.

## 6. Nonlinear ML signal-selection test — later-year results

Model: **fixed-parameter** `HistGradientBoostingRegressor`, trained with BTC+ETH+XRP 2021–2023. At each slow-DC event, select the side with larger model-predicted net R only if that prediction is **positive**. For these selected trades, no overlaps per symbol. The 2023 development diagnostic used a separately trained 2021–2022 version.

| Market | Year | Trades | Net R/trade | Profit factor |
|---|---:|---:|---:|---:|
| BTC | 2024 | **6** | **+0.4241** | no losses in this tiny sample |
| ETH | 2024 | 12 | −0.1470 | 0.621 |
| ETH | 2025 | 12 | −0.4225 | 0.297 |
| ETH | 2026 partial | 3 | −0.3641 | 0.285 |
| XRP | 2024 | 43 | +0.0711 | 1.282 |
| XRP | 2025 | 46 | −0.0439 | 0.855 |
| XRP | 2026 partial | 6 | −0.2037 | 0.420 |

The most attractive-looking examples are statistically insufficient: BTC's six trades have **no loss**, making profit factor undefined rather than infinite; XRP 2024's +0.0711R (43 trades) has a week-block-bootstrap 95% interval **−0.1498R to +0.3033R**, which includes zero. By 2025/26, the same fixed model was negative in both available assets. **It fails the cross-market temporal robustness test.**

## 7. Simpler combined ridge model — later-year results

| Market | Year | Trades | Net R/trade | Profit factor |
|---|---:|---:|---:|---:|
| BTC | 2024 | 5 | +0.0723 | 1.344 |
| ETH | 2024 | 18 | −0.2666 | 0.419 |
| ETH | 2025 | 30 | +0.0330 | 1.159 |
| ETH | 2026 partial | 8 | +0.1864 | 3.716 |
| XRP | 2024 | 162 | −0.0500 | 0.810 |
| XRP | 2025 | 109 | −0.1178 | 0.634 |
| XRP | 2026 partial | 16 | −0.0359 | 0.853 |

The small positive ETH 2025 result has a week-block-bootstrap 95% interval **−0.1504R to +0.3063R**; the 2026 gain uses **only eight trades**. No stable cross-market edge. Other tested model families and controls are reported in `yearly_results.csv` and `development_2023_results.csv` to avoid winner-only disclosure.

A single ETH 2025 magnet-only ridge variant had +0.1045R across 21 trades with a positive naive weekly bootstrap interval, but it also produced very few trades and losses in adjacent years. This is **hypothesis-generating only**; there is no independent validation, and multiple strategies/years were examined. Do not promote it from one isolated positive year.

## 8. Scientific interpretation

1. **The double-DC alignment rule should be retired.** It almost never discriminates.
2. **Weak gross advantage precedes the cost problem.** A lower fee alone will not create a reliable directional effect; replacing taker fills with hypothetical maker fills is not valid without fill probabilities and adverse selection.
3. **Model abstention is economically rational**, but positive fitted scores rarely survived later-year distribution changes. Filtering to six wins in a year is not enough evidence for deployment.
4. **Magnet semantics remain unproven beyond distance.** The new magnetic features do not give stable out-of-period improvement versus simpler directional-only models. Historical highs/lows are levels, not actual measured resting orders.
5. **Neural nets may help model event sequences**, but cannot create predictive information from redundant or noncausal features. A CNN/GRU should be a challenger to cost-adjusted linear and boosted controls, not an assumed route to profit.

## 9. Recommended next hypothesis H6 — sweep, reaction, and intrinsic-time transition

Rather than trying to **trade toward** an often-nearby price magnet after a completed slow DC event, test **whether behavior when price first touches/sweeps an objectively defined prior high/low is predictively asymmetric**:

- Event: a first sweep of an unswept completed PDH/PDL or 8-hour high/low, timestamped only once the M1 bar closes.
- Observed reaction: close back across the swept level versus continued closing beyond it (rejection vs acceptance); these are **two competing hypotheses**, not a preselected winning side.
- DC context: preceding 3–8 *completed* fast DC/overshoot segments and their compression/extension, and the current slow DC regime.
- Payoff labels: tradable move after **next** M1 open to cost-covered exits in 15/30/60/120 minutes, stop first on ambiguity; choose TP/SL by ex-ante risk and validate no-trade decision.
- Controls: identical timestamps with levels shifted to same-distance synthetic nonmagnetic levels; volatility- and time-matched random entries; no sweep/acceptance context; cross-market and chronological splits.
- Start with probability calibration / simple hazard model and only then a small temporal CNN or GRU if enough event count and demonstrated incremental predictive information.
- Enforce historical fee/slippage sensitivity, actual contract specifications, funding events, quantized position sizes and **new future data** for final proof.

This new hypothesis supplies a possible causal mechanism (liquidity interaction and post-sweep response), rather than assuming *all* magnetic levels attract prices. It has **not yet been tested**.

## 10. Reproducibility and limitations

Code: `magnet_dc_h5.py`, `btc_magnet_dc_v2.py` (causal DC and magnet routines), `h5_statistics.py`, `test_magnet_dc_h5.py`; tests pass. Recompute using the 16 original uploaded ZIP archives and `crypto_source_dc_audit.json`; no original large archives are redistributed in the research pack. `yearly_results.csv` contains all baselines/models, not merely favorable rows. The H5 output and training library may vary with software versions if not pinned; core inputs, model hyperparameters and random seed are specified in source.

**Main reference:** Robert Kissell, *Algorithmic Trading Methods*, 2nd ed., chapters 3 (transaction costs), 7 (probability models), 9 (machine learning), 10 (hypothesis tests), 13 (algorithmic adaptation), 18 (backtesting) and 19 (ML for execution optimization). Note that Kissell's Chapter 19 focuses on **execution schedules**, which does not independently support profitability of our DC+magnet entry hypothesis.

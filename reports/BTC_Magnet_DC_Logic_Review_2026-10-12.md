# BTCUSDT magnetic-target + DC logic review

**Research status:** New diagnostic; no demonstrated trading edge. October 12, 2026.

## Research context and source

Dataset: the four identical audited BTCUSDT Bybit linear-perpetual 1-minute candle archives for 2021–2024. Preexisting causal magnetic/DC event table: `btc_magnet_dc_v2/dc_magnet_events_all.csv.gz` (not the unavailable original H4.2 event table). Earlier research had already examined BTC 2021–2024: **no claim of fresh holdout**.

Main methodological reference: Robert Kissell, *Algorithmic Trading Methods*, second edition (2021). Specific relevant sections: Ch. 3, pp. 57–97 (transaction costs; performance evaluation); Ch. 7, pp. 175–195 (probability models); Ch. 9, pp. 221–232 (classification and machine learning); Ch. 10, pp. 235–236 (scientific method; independent tests); Ch. 13, pp. 323–348 (trading goals and adaptation tactics); Ch. 18, pp. 487–490 (backtesting and implementation costs); Ch. 19, pp. 519–542 (neural nets for *execution schedule optimization*, which is not itself a directional alpha model).

## Root-cause diagnostic of the older aligned rule

Old rule entered after a close-confirmed DC aligned with the nearer unresolved magnet when near/far distance ratio <=0.5. It then used a **fixed +0.5R take-profit** and a **1R stop** based on the DC threshold, rather than using the magnet as the exit target. In the archived non-overlapping entry logs:

| Year | Median magnet distance, / DC risk | % with magnet closer than +0.5R TP | % with magnet room < 13 bp + 0.2R |
|---|---:|---:|---:|
| 2021 | 0.230R | 84.6% | 59.9% |
| 2022 | 0.261R | 77.7% | 60.7% |
| 2023 | 0.301R | 69.5% | 63.0% |
| 2024 | 0.249R | 80.9% | 63.6% |

Note: comparison is to the *signal-close* magnet distance stored by the old engine; the exact next-open entry can shift distance slightly. The fixed 13bp all-in fee/friction assumption is **not** actual historical bid/ask or funding.

Because the magnetic target commonly lies before the 0.5R TP, the conceptually 'correct' destination forecast can occur without reaching the intended profit. This is a structural incoherence, **not** proof the magnetic level must reverse price.

Further, 2023 had median risk ~65.7bp for the aligned rule, implying round-trip cost at the median about 0.198R. The same rule's average gross price return was ~0.03bp/trade versus 13bp modeled round-trip cost.

## New matched-entry experimental diagnostic

New executable script: `btc_target_aware_diagnostic.py`. Uses exactly the stored causal DC/magnet signals on 2021–2024 and the original BTC candles. For a fair comparison, entry events are kept >=121 minutes apart under a **common 120-minute time embargo** independent of chosen exit, thereby ensuring paired trade entries across exit variants. The entry occurs at the next minute open, a 1R stop based on the past-only DC theta, 120-minute horizon, cost of 13bp of notional (5.5bp fee/side + 1bp slippage/side as a *model assumption*), no funding, worst-case stop priority for an ambiguous candle.

Two exit definitions: (A) fixed +0.5R gross TP, (B) fixed target at the nearest magnetic level. Two cohorts: (i) aligned DC and near/far magnet ratio<=0.5 (all economically reachable targets); (ii) same conditions plus the actual next-open magnet target return >=13bp + 0.2R (a *post-open observable* feasibility filter). The eligibility filter does not use future OHLC values. A live implementation would need to place the order after observing or controlling the next-open price; a next-open instantaneous decision is a modeling simplification and can require an execution-delay sensitivity test.

### Per-year net R (average per trade)

| Year | All matched: +0.5R TP | All matched: magnet TP | Feasible cohort: +0.5R TP | Feasible cohort: magnet TP |
|---|---:|---:|---:|---:|
| 2021 | -0.0937 | -0.0963 | -0.0668 | -0.0389 |
| 2022 | -0.0919 | -0.1315 | -0.1308 | -0.1380 |
| 2023 | -0.2294 | -0.2386 | -0.3190 | -0.3302 |
| 2024 | -0.1217 | -0.1690 | -0.1264 | -0.1348 |

All are negative. Causally tying the TP to the magnetic destination is insufficient. The alternative exits use *paired identical entries*; the filtered cohort is different from the unfiltered cohort, so do not interpret the rows as a fully paired three-strategy horse race. These results are **not** out-of-sample. The new test is diagnostic, not a deployable strategy. It omits funding and actual bid/ask data and may be optimistic for a limit-order fill at a point magnet.

## Interpretation

1. The 24-hour *first destination* statistic (~73%) nearly matched a distance-only null; it did not establish liquidity attraction independent of geometry.
2. Even an eventual correct direction cannot pay a TP that is beyond the supposed target when price does not overshoot far enough.
3. Stopping at the magnet can close earlier but has too little room in most events to overcome transaction costs and adverse paths.
4. DC confirmation occurs only after a meaningful counter-move already happened; its value as an *entry time* must be independently tested, not inferred from its value as a *market segmentation method*.
5. Lower realized volatility increases cost in R units for θ-sized stops. In 2023, the median aligned risk was ~65.7bp and the fixed 13bp modeled cost ~0.198R.
6. The 'magnet' level definition is a proxy for historical price features, not a direct measurement of resting exchange limit orders.

## Recommendation: a new, explicitly testable hypothesis family

**Hypothesis R1 — Dual-scale intrinsic-time entry.** Use a slower DC threshold to classify regime and a faster, independently specified threshold to monitor current micro-overshoot/enter. Do not fit fast threshold on untouched future data. Compare with the single-scale close-confirmed DC entry.

**Hypothesis R2 — Target-conditional expected monetary value.** For the same event evaluate (LONG, SHORT, FLAT) and candidate exits. Use known target room and fees to disallow trades with nonpositive *maximum potential net payout*. Learn/probabilistically estimate the distribution of stop-hit, target-hit, and time-out payoffs; select the action with positive estimated net EV, not highest raw hit rate.

**Hypothesis R3 — Overshoot continuation forecast.** Features at decision time: last 8–16 **completed** DC/OS event magnitudes and durations, current *partial* OS observed up to now, cross-scale state, θ/volatility, magnet-distance asymmetry and source, session, and recent signed return/volatility. Train a simple calibrated logistic/gradient-boosted model first; compare a small GRU/TCN with equal information and chronological folds. Avoid using final overshoot size or future extrema in features.

**Hypothesis R4 — Exit-versus-hold decision.** When an existing trade reaches positive P&L, compare the conditional *expected additional* payoff of HOLD with SELL-NOW accounting for remaining fees, attainable price, and drawdown; evaluate out of sample against fixed exits. Never use future overshoot termination in current decisions.

### Research gates

- Intrinsic feature generation and action labels must be causal; test an extra one-minute decision/entry delay.
- Cost schedule: 13bp base; at least 18bp adverse, also use actual Bybit instrument minimum orders, fees and historical funding when supplied. Fees may differ by account tier.
- Source events should be chronological 2021–2023 development; BTC 2024 is already historically researched and not pristine. Seek fresh BTC 2025–2026 for genuine prospective out-of-time assessment. For ETH/XRP, use 2021–2023 discovery, 2024 checkpoint and a frozen 2025–2026 confirmation (while acknowledging past research touching those assets).
- Report gross and net bp, net R, weekly/block-bootstrap interval, annual/monthly stability, PF, max drawdown, turn-over, volume/fee sensitivity and number of attempted parameterizations. Keep *do nothing* as the benchmark.
- A model that simply increases accuracy but has negative net EV has failed the trading goal.

## Method warning

Kissell's book is primarily about algorithmic *execution and implementation* cost modeling. Its Ch. 19 neural-network example addresses trade-schedule optimization and does not establish a profitable predictive neural-network trading strategy. We are applying its statistical framework to a *separate* profit-seeking strategy hypothesis, not claiming that the book proves magnetic liquidity targets or DC overshoot profitability.

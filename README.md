# Liquidity-Magnet Price Levels in Bitcoin Perpetuals

Do prior highs and lows "pull" price toward them, and can a trader profit from
knowing which one price will reach next? This repository tests both questions on
2.1 million one-minute Bybit BTCUSDT candles (2021-2024), with follow-up work on
ETH and XRP.

**Working paper:** [Where Price Goes Is Not How to Profit: Testing "Liquidity
Magnet" Price Levels in Bitcoin Perpetual Futures](paper/Magnet_Levels_BTC_Null_Result_Tariq_2026.pdf)
(Rayan Tariq, 2026)

## Findings

| Question | Result |
|---|---|
| How often is the nearer level reached first? | 71.9% of 7,085 events |
| How much of that does distance alone explain? | 70.5% (driftless first-passage geometry) |
| Residual above geometry | +1.45 pp, month-block 95% CI [0.32, 2.61] |
| Real levels vs exact-distance mirror placebo | +2.16 pp [0.07, 4.34]; +8.79 pp when the near level is a previous-day high/low (512 events) |
| Stable over time and trend states? | No: concentrated in 2024 and in trendless periods |
| Better out-of-sample forecasts than distance alone? | No: geometry-only model had the best Brier score and log loss in walk-forward tests 2022-2024 |
| Best entry design, gross | Fresh GMMA alignment: +8.27 bp per trade, below a ~13 bp round-trip cost |
| Better than non-magnet confirmations? | No: simple progress/geometry confirmations earned as much on the same signals |
| Independent reconstruction, after 13 bp costs | Every policy negative; intervals entirely below zero |
| ML selection on BTC/ETH/XRP (H5) | Small, unstable positives; failed later-year cross-market tests |

Where price goes next is mostly a matter of distance, and knowing it did not pay.

## Repository layout

```
paper/               working paper (PDF) and the script that builds it from results/
code/                H4 study scripts: level construction, entry tests, placebos, OOS models
results/             summary outputs of every H4 stage (CSV) and frozen configs (TXT)
reports/             original research reports (Phase 1 findings, logic review, H5)
successor_studies/   independent reconstruction (v2), target-aware diagnostic, H5 ML
                     extension: code, tests, methods JSON, summary results
```

Stage-to-file map: Appendix B of the paper.

## Reproduce

```bash
pip install -r requirements.txt
cd paper && python build_paper.py      # rebuilds the PDF from results/ and successor_studies/
```

Re-running the studies themselves needs Bybit one-minute BTCUSDT candles (and ETH/XRP for
H5), which are not redistributed; they can be downloaded from Bybit's public endpoints,
for example with the tools in
[crypto-strategy-research](https://github.com/Crayon99-droid/crypto-strategy-research).
Large event- and trade-level files are available on request.

Not investment advice.

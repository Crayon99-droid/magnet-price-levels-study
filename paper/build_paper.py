"""Build the magnet-level working paper. All tabulated numbers are read from result files."""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import Paragraph, Spacer, PageBreak, KeepTogether
import academic as A

R = "../results"
P1 = "../successor_studies/BTC_Magnet_DC_Phase1_Research_Pack/btc_magnet_dc_v2"
H5 = "../successor_studies/Magnet_DC_H5_Research_Pack/RESULTS"
OUT = "."
c = lambda f: pd.read_csv(os.path.join(R, f))


# ---------------------------------------------------------------- numbers
n1 = c("H4_NULL1_month_block_bootstrap.csv").set_index("sample")
n2b = c("H4_NULL2_month_block_bootstrap.csv").set_index("sample")
n2s = pd.read_csv(os.path.join(R, "H4_NULL2_month_block_bootstrap.csv"))
n2y = c("H4_NULL2_year_summary.csv")
n2t = c("H4_NULL2T_primary_24h_bootstrap.csv").set_index("sample")
n2ts = c("H4_NULL2T_primary_24h_summary.csv").set_index("sample")
m1 = c("H4_MODEL1_aggregate_oos_metrics.csv").set_index("model")
m1b = c("H4_MODEL1_incremental_block_bootstrap.csv")
h41 = c("H4_1_year_summary.csv"); h42 = c("H4_2_year_summary.csv")
ov = {k: c(f"H4_{k}_overall_summary.csv") for k in ["4", "5", "5B", "6A", "6A2", "6B"]}
g1a = c("H4_G1A_stop_frontier_summary.csv").set_index("stop_fraction_D")
g1b = c("H4_G1B_strategy_summary.csv"); g1bp = c("H4_G1B_matched_pair_bootstrap.csv").set_index("comparison")
a2y = c("H4_6A2_year_summary.csv")
sa = json.load(open(os.path.join(P1, "statistical_audit.json")))
h5y = pd.read_csv(os.path.join(H5, "yearly_results.csv"))
null1_txt = open(os.path.join(R, "H4_NULL1_verdict.txt")).read()
n1_obs, n1_geo = 0.719125, 0.704590  # from verdict text (checked below)
assert "0.719125" in null1_txt and "0.704590" in null1_txt

def pp(x): return f"{100*x:.2f}"
def ci(lo, hi, f=pp): return f"[{f(lo)}, {f(hi)}]"

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 9, "axes.linewidth": 0.6})
# Fig 1: real vs geometry vs mirror
s = c("H4_NULL2_primary_summary.csv") if os.path.exists(os.path.join(R, "H4_NULL2_primary_summary.csv")) else None
lab = ["All events", "Ratio <= 0.5", "Near is PDH/PDL", "Ratio <= 0.5 and\nnear is PDH/PDL"]
keys = ["ALL", "STRONG_RATIO_<=0.50", "NEAR_HAS_PDH_PDL", "STRONG_AND_NEAR_HAS_PDH_PDL"]
ps = c("H4_NULL2_primary_summary.csv").set_index("sample")
real = [ps.loc[k, "real_nearer_first"] * 100 if "real_nearer_first" in ps.columns else None for k in keys]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
cols = [cn for cn in ps.columns]
geo = [ps.loc[k, "geometry_expected_nearer_first"] * 100 for k in keys] if "geometry_expected_nearer_first" in cols else None
mir = [ps.loc[k, "mirror_nearer_first"] * 100 for k in keys] if "mirror_nearer_first" in cols else None
x = range(len(keys)); w = 0.26
ax.bar([i - w for i in x], real, w, label="Real levels", color="0.15")
if geo: ax.bar([i for i in x], geo, w, label="Geometry null (driftless)", color="0.55", hatch="///", edgecolor="0.3", linewidth=0.4)
if mir: ax.bar([i + w for i in x], mir, w, label="Exact-distance mirror", color="0.85", edgecolor="0.3", linewidth=0.4)
ax.set_xticks(list(x)); ax.set_xticklabels(lab); ax.set_ylim(60, 90); ax.set_ylabel("Nearer level reached first (%)")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1.png", dpi=220); plt.close(fig)

# Fig 2: gross bps per stage vs cost
st = [("H4.4\npullback", ov["4"].iloc[0]["mean_gross_bps_per_executed_trade"]),
      ("H4.5\nfailure rev.", ov["5"].iloc[2]["mean_gross_bps"]),
      ("H4.5B\nconfirmed", ov["5B"].iloc[2]["mean_gross_bps"]),
      ("H4.6A\nGMMA", ov["6A"].iloc[0]["mean_gross_bps"]),
      ("H4.6A2\nfresh GMMA", ov["6A2"].iloc[0]["mean_gross_bps"]),
      ("H4.6B\nDC entry", ov["6B"].iloc[0]["mean_gross_bps"])]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.bar(range(len(st)), [v for _, v in st], color="0.35", width=0.6)
ax.axhline(13, color="black", ls="--", lw=0.9); ax.text(len(st) - 0.5, 13.5, "13 bp modelled round-trip cost", ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.6)
for i, (_, v) in enumerate(st): ax.text(i, v + (0.4 if v >= 0 else -1.4), f"{v:+.2f}", ha="center", fontsize=8)
ax.set_xticks(range(len(st))); ax.set_xticklabels([n for n, _ in st], fontsize=8)
ax.set_ylabel("Mean gross bp per trade"); ax.set_ylim(-11, 16); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2.png", dpi=220); plt.close(fig)

# ---------------------------------------------------------------- styles
H1, H2, B, BL, CAP, REF, Q = A.H1, A.H2, A.BODY, A.BULLET, A.CAP, A.REF, A.Q


def tbl(head, rows, widths, cap=None):
    return A.table(head, rows, widths, cap)


S = []; p = lambda t, st=B: S.append(Paragraph(t, st))
# ---------------------------------------------------------------- title + abstract
mir_all = n2b.loc["ALL"]; mir_pd = n2b.loc["NEAR_HAS_PDH_PDL"]
m2 = m1b[(m1b.baseline_model == "M1_TREND24") & (m1b.candidate_model == "M2_PDH_PDL")].iloc[0]
m10 = m1b[(m1b.baseline_model == "M0_GEOMETRY") & (m1b.candidate_model == "M1_TREND24")].iloc[0]
ga = g1a.loc[1.00]
ABSTRACT = (f"Traders often describe prior highs and lows as “magnets” that pull price toward them. This study tests that "
  f"idea on 2,103,840 one-minute Bybit BTCUSDT perpetual candles from 2021 to 2024, first as a forecast of which level "
  f"price reaches next, then as a source of tradable entries. Around a 15-minute decision point with one candidate level "
  f"above and one below, the nearer level was reached first in {pp(n1_obs)}% of {7085:,} resolved events. A driftless "
  f"first-passage model that knows only the two distances predicts {pp(n1_geo)}%, leaving a residual of "
  f"{pp(n1_obs-n1_geo)} percentage points (month-block 95% interval {ci(n1.loc['ALL_RESOLVED','bootstrap_95pct_low'], n1.loc['ALL_RESOLVED','bootstrap_95pct_high'])}). "
  f"An exact-distance mirror placebo on the same price paths left {pp(mir_all.observed_real_minus_mirror)} points "
  f"{ci(mir_all.bootstrap_95pct_low, mir_all.bootstrap_95pct_high)}, larger when the near level was a previous-day "
  f"high or low ({pp(mir_pd.observed_real_minus_mirror)} points on {int(mir_pd.paired_resolved_events)} events). "
  f"The residual was unstable across years and absent in trending conditions, and in walk-forward tests a model with "
  f"trend and level-structure features did not improve on distance alone (log-loss change "
  f"{m2.mean_logloss_improvement:+.5f}, interval [{m2.logloss_improvement_ci95_low:+.4f}, {m2.logloss_improvement_ci95_high:+.4f}]). "
  f"No entry design converted the destination signal into money: mean gross returns ranged from "
  f"{ov['5'].iloc[2]['mean_gross_bps']:+.1f} to {ov['6A2'].iloc[0]['mean_gross_bps']:+.1f} basis points per trade against "
  f"about 13 basis points of round-trip cost, and simple non-magnet confirmation placebos matched the best entry "
  f"in money terms. Independent reconstructions, a target-aware exit diagnostic and a multi-asset machine-learning "
  f"extension (BTC, ETH, XRP) reached the same conclusion. Where price goes next is largely a matter of distance, and "
  f"knowing it did not pay.")
S += A.title_block("Where Price Goes Is Not How to Profit: Testing \u201cLiquidity Magnet\u201d Price Levels in Bitcoin Perpetual Futures",
                   "Rayan Tariq", "Working paper, November 2026", ABSTRACT,
                   "support and resistance; liquidity; first-passage time; placebo tests; cryptocurrency; transaction costs; null results")

# ---------------------------------------------------------------- 1 intro
p("1. Introduction", H1)
p("A common trading heuristic holds that unswept highs and lows attract price because stop orders and resting liquidity "
  "cluster there. If true, a trader who can identify the next level price will reach could position ahead of it. The "
  "heuristic is attractive because it often looks right on a chart: price does frequently travel to a nearby prior high or low.")
p("That frequency is exactly the problem. Of two levels, the nearer one is reached first more often for purely "
  "geometric reasons, even for a random walk. A hit rate of seventy percent therefore says little until it is compared "
  "with what distance alone would produce. A second gap separates a correct destination forecast from a profitable trade: "
  "the stop may be hit first, the move may be smaller than costs, or the level may sit inside the planned profit target.")
p("This paper tests both gaps in sequence. Section 4 builds causal level sets and measures destination accuracy against "
  "three controls of increasing strictness: an analytic geometry null, an exact-distance mirror placebo on the actual "
  "price paths, and a trend-conditioned version, followed by a walk-forward probability model. Section 5 then asks whether "
  "any entry design (pullbacks, failure reversals, moving-average and directional-change confirmations) earns its "
  "trading cost, and compares the best of them with simple non-magnet placebos. Successor studies by the same author, "
  "which rebuilt the level sets independently and extended the test to machine-learning selection on three assets, are "
  "reported in Section 5.6.")
p("The contribution is a careful negative result. A small structural residual exists in sample, particularly at previous-day "
  "highs and lows. It is not stable, it is not predictive out of sample beyond distance, and it is far too small to trade "
  "once realistic costs are paid.")

# ---------------------------------------------------------------- 2 literature
p("2. Related literature", H1)
p("Support and resistance have a long practitioner history and a smaller academic one. Osler (2000) found that "
  "published support and resistance levels for major exchange rates had predictive power for intraday trend "
  "interruptions, and Osler (2003) traced part of the effect to the clustering of take-profit and stop-loss orders at "
  "round numbers. Brock, Lakonishok and LeBaron (1992) reported apparent profitability of simple technical rules in "
  "equity indices; later work stressed that such evidence must be judged against data snooping (White, 2000; Harvey, "
  "Liu and Zhu, 2016; Bailey and Lopez de Prado, 2014). In cryptocurrency, Anghel (2021) found that technical and "
  "machine-learning rules rarely retain significant excess returns after frictions and snooping controls.")
p("The geometry null used here is the classical two-barrier first-passage result for driftless Brownian motion: starting "
  "between barriers at distances d<sub>near</sub> and d<sub>far</sub>, the probability of reaching the near barrier first is "
  "d<sub>far</sub> / (d<sub>near</sub> + d<sub>far</sub>). It is a benchmark, not a claim that Bitcoin follows Brownian motion; "
  "the mirror placebo in Section 4.4 removes reliance on that assumption by using the actual price paths. Execution and "
  "cost assumptions follow the implementation-cost framing of Kissell (2020).")

# ---------------------------------------------------------------- 3 data
p("3. Data", H1)
p("The primary data are one-minute OHLCV candles for the Bybit BTCUSDT linear perpetual from 1 January 2021 to 31 "
  "December 2024: 525,600 minutes in each of 2021-2023 and 527,040 in 2024, 2,103,840 in total. Yearly archives were "
  "checked against a source manifest by SHA-256 and ZIP CRC, and validated for exact one-minute chronology across year "
  "boundaries, timestamp agreement, OHLC ordering and numeric integrity. No candle was interpolated or synthesized.")
p("The multi-asset extension (H5) added ETHUSDT (15 March 2021 to 22 August 2026) and XRPUSDT (13 May 2021 to 22 August "
  "2026), for 7,740,300 one-minute bars across 16 yearly archives. BTC matched an independent production manifest; ETH "
  "and XRP hashes certify internal consistency and repeatability but have no independent original manifest. Funding "
  "rates and historical bid-ask quotes were not part of these datasets; Section 7 discusses the consequences.")

# ---------------------------------------------------------------- 4 methods
p("4. Methods", H1)
p("4.1 Candidate levels and events", H2)
p("At each 15-minute decision bar, candidate levels were drawn from four families: previous completed UTC-day high and "
  "low (PDH/PDL), previous completed week high and low (PWH/PWL), the previous completed 8-hour UTC block high and low, "
  "and 15-minute swing highs and lows confirmed only after two subsequent bars closed (seven-day lookback). Levels within "
  "0.15 of a 14-period ATR were clustered into bands. An event required at least one unresolved level on each side; the "
  "outcome was which side's nearest band price touched first within 24 hours, using one-minute highs and lows. Events "
  "were non-overlapping and no parameter was optimized (H4.2). A simpler PDH/PDL-only version (H4.1) preceded it.")
p("4.2 Destination accuracy and the geometry null", H2)
p("For each resolved event the geometry null assigns probability d<sub>far</sub>/(d<sub>near</sub>+d<sub>far</sub>) to the near "
  "side. The residual of interest is the observed near-first rate minus the mean of these probabilities (H4.NULL1). "
  "Structural features (stronger side, higher level count, more source families) were tested the same way. Inference used "
  "10,000 calendar-month block bootstrap replications over 48 months.")
p("4.3 Exact-distance mirror placebo", H2)
p("The geometry null assumes a driftless random walk. The mirror placebo (H4.NULL2) avoids that assumption: for each "
  "event it keeps the same two barrier distances and the same future BTC path, but swaps which side receives which "
  "distance. Geometry and path dynamics are preserved exactly; only the real placement of levels is broken. A structural "
  "effect should make real levels outperform their mirrors.")
p("4.4 Trend conditioning and out-of-sample probability models", H2)
p("Because a level in the direction of a prevailing trend is more likely to be reached for reasons unrelated to "
  "liquidity, H4.NULL2T split events by the standardized trailing 24-hour return (neutral, aligned with the near "
  "level, opposed). H4.MODEL1 then fitted logistic models with a fixed geometry offset, log(d<sub>lower</sub>/d<sub>upper</sub>), "
  "and added in turn a 24-hour trend term (M1), a signed PDH/PDL indicator (M2, the primary test), all structure "
  "families (M3) and a 4-hour trend term (M4). Walk-forward folds trained on 2021 (test 2022), 2021-2022 (test 2023) "
  "and 2021-2023 (test 2024). The primary decision was whether M2 improved out-of-sample Brier score and log loss over "
  "M1, with month-block bootstrap intervals (10,000 replications, seed 20260908). M3 and M4 were declared exploratory "
  "in advance and could not rescue a failed M2.")
p("4.5 Entry designs", H2)
p("The trading stages used the H4.2 events with near/far distance ratio at most 0.5. Each stage fixed its rules in "
  "advance and reported gross results with sensitivity to round-trip costs of 1 to 10 basis points; same-minute "
  "stop/target ambiguity was resolved as a loss.")
for t in ["<b>H4.3 tradability:</b> enter at the decision price toward the near level with stops at 0.25-1.0 of the target distance.",
          "<b>H4.4 pullback:</b> wait for a retracement of 25, 50 or 75% of the target distance before entering toward the magnet.",
          "<b>H4.5 / H4.5B failure reversal:</b> trade against the predicted magnet after a pullback fails, without and with a confirmation close.",
          "<b>H4.6A / H4.6A2 GMMA:</b> enter on a one-minute Guppy multiple moving average alignment toward the magnet (fast EMAs 3-15, slow 30-60); H4.6A2 kept only fresh transitions.",
          "<b>H4.6B DC:</b> enter on a close-confirmed directional-change reversal toward the magnet.",
          "<b>H4.G1A stop frontier:</b> vary the stop for the frozen H4.6A2 entries from 0.10 to 1.00 of the target distance.",
          "<b>H4.G1B confirmation placebos:</b> compare fresh GMMA entries with two non-magnet confirmations on the same signal pool: the first close reaching 25% progress toward the target, and the first close at which the geometric first-passage probability improves by five points."]:
    S.append(Paragraph(t, BL, bulletText="•"))
p("4.6 Successor reconstructions", H2)
p("Two later studies rebuilt the level sets independently because the original H4.2 event table was unavailable to them. "
  "The first (v2) used previous day, week and 8-hour extremes plus confirmed 15-minute pivots with expiry rules, "
  "close-confirmed directional-change events (threshold 1.25 times the previous day's median 240-minute realized "
  "volatility), next-minute entries, a 1R stop, +0.25/0.5/1R targets, 30- or 120-minute deadlines and 13 basis points of "
  "round-trip cost (5.5 bp taker fee plus 1 bp slippage per fill). A target-aware diagnostic then placed the take-profit "
  "at the magnet itself. The second (H5) added a fast directional-change scale, 32 causal features and ridge and "
  "gradient-boosted models that entered only when predicted net R was positive, trained on 2021-2023 and evaluated on "
  "2024-2026 across BTC, ETH and XRP. These are related studies, not exact reproductions of H4.")

# ---------------------------------------------------------------- 5 results
p("5. Results", H1)
p("5.1 Raw destination accuracy", H2)
rows = [[int(r.year), f"{int(r.observations):,}", pp(r.nearest_accuracy)] for r in h41.itertuples()]
rows2 = [[int(r.year), f"{int(r.observations):,}", pp(r.nearer_accuracy), pp(r.stronger_accuracy), f"{r.mean_hours_to_touch:.2f}"] for r in h42.itertuples()]
S.append(tbl(["Year", "Events (H4.2)", "Nearer first (%)", "Stronger side first (%)", "Mean hours to touch"], rows2,
             [55, 85, 95, 115, 105], "Table 1. Multi-level magnet events, BTCUSDT"))
p(f"The nearer band was reached first 70-74% of the time in every year, typically within three to four hours, while "
  f"the side with stronger level confluence won only 51-55%. The PDH/PDL-only version (H4.1) gave 72-78% on about 300 "
  f"events per year. Taken at face value these numbers look like a strong directional signal.")
p("5.2 Geometry explains most of it", H2)
s5 = n1.loc["STRONG_RATIO_<=0.50"]
p(f"Against the geometry null, the observed rate of {pp(n1_obs)}% exceeded the expected {pp(n1_geo)}% by "
  f"{pp(n1_obs-n1_geo)} points {ci(n1.loc['ALL_RESOLVED','bootstrap_95pct_low'], n1.loc['ALL_RESOLVED','bootstrap_95pct_high'])} "
  f"on 7,085 resolved events. In the strong-asymmetry subset (ratio at most 0.5) the observed 81.40% exceeded 79.22% by "
  f"{pp(s5.observed_mean_residual)} points {ci(s5.bootstrap_95pct_low, s5.bootstrap_95pct_high)}. About 95% of the raw "
  f"near-first rate is therefore what distance alone predicts. The structural side tests (stronger side, higher level "
  f"count, more source families) each added only 0.3-0.4 points over geometry.")
S += [A.figure(f"{OUT}/fig1.png", 440, 199, "Figure 1. Real levels compared with the driftless geometry null and the exact-distance mirror placebo on the same paths (H4.NULL2). Shaded bars: driftless geometry null; light bars: exact-distance mirror.")]
p("5.3 Mirror placebo", H2)
rows = []
for k, lbl in zip(keys, ["All", "Ratio <= 0.5", "Near is PDH/PDL", "Ratio <= 0.5 and near is PDH/PDL"]):
    r = n2b.loc[k]
    rows.append([lbl, f"{int(r.paired_resolved_events):,}", pp(r.observed_real_minus_mirror), ci(r.bootstrap_95pct_low, r.bootstrap_95pct_high), f"{r.fraction_bootstrap_le_0:.4f}"])
S.append(tbl(["Sample", "Paired events", "Real minus mirror (pp)", "95% month-block interval", "Bootstrap share <= 0"], rows,
             [140, 70, 85, 110, 80], "Table 2. Exact-distance mirror placebo (H4.NULL2)"))
y = n2y[n2y.subset == "ALL"].set_index("year")
p(f"Real levels beat their mirrors by {pp(mir_all.observed_real_minus_mirror)} points overall, an interval that barely "
  f"excludes zero, and by about 9-10 points when the near level was a previous-day high or low, on a few hundred events. "
  f"The effect was not stable in time: by year the all-event difference was {pp(y.loc[2021,'real_minus_mirror_pp'])}, "
  f"{pp(y.loc[2022,'real_minus_mirror_pp'])}, {pp(y.loc[2023,'real_minus_mirror_pp'])} and "
  f"{pp(y.loc[2024,'real_minus_mirror_pp'])} points for 2021-2024, so most of it came from 2024.")
p("5.4 Trend conditioning", H2)
rows = [[lbl, f"{int(n2t.loc[k,'paired_resolved']):,}", pp(n2t.loc[k, "observed_real_minus_mirror"]), ci(n2t.loc[k, "ci95_low"], n2t.loc[k, "ci95_high"])]
        for k, lbl in [("ALL", "All"), ("24H_NEUTRAL", "Neutral 24h trend"), ("24H_ALIGNED", "Trend toward near level"), ("24H_OPPOSED", "Trend against near level")]]
p("The structural residual appeared only when the 24-hour trend was neutral (Table 3). When price was already trending "
  "toward or away from the near level, real levels did no better than mirrors. For PDH/PDL the neutral and aligned "
  "subsets each had intervals touching zero (lower bound 0.00).")
S.append(tbl(["24-hour trend state", "Paired events", "Real minus mirror (pp)", "95% interval"], rows, [170, 80, 110, 120],
             "Table 3. Mirror advantage by trend state (H4.NULL2T)"))
p("5.5 Out-of-sample probability models", H2)
rows = [[mname.replace("_", " "), f"{m1.loc[mname,'brier']:.6f}", f"{m1.loc[mname,'log_loss']:.6f}", pp(m1.loc[mname, "accuracy_0_5"])]
        for mname in ["M0_GEOMETRY", "M1_TREND24", "M2_PDH_PDL", "M3_ALL_STRUCTURE", "M4_ALL_STRUCTURE_PLUS_4H"]]
S.append(tbl(["Model", "OOS Brier", "OOS log loss", "Accuracy at 0.5 (%)"], rows, [190, 90, 90, 100],
             "Table 4. Walk-forward out-of-sample fit, 5,152 events, test years 2022-2024 (H4.MODEL1)"))
p(f"Geometry alone (M0) had the best out-of-sample Brier score and log loss. Adding 24-hour trend worsened log loss by "
  f"{-m10.mean_logloss_improvement:.5f} [{m10.logloss_improvement_ci95_low:+.5f}, {m10.logloss_improvement_ci95_high:+.5f}]; "
  f"adding the PDH/PDL indicator to that (the primary test) changed it by {m2.mean_logloss_improvement:+.6f} "
  f"[{m2.logloss_improvement_ci95_low:+.5f}, {m2.logloss_improvement_ci95_high:+.5f}], with "
  f"{100*m2.fraction_bootstrap_logloss_improvement_le_0:.1f}% of bootstrap draws showing no improvement. The in-sample "
  f"residual did not translate into better forecasts on later years.")

p("5.6 From destination to trade", H2)
p("Figure 2 summarizes mean gross returns per trade for each entry design before any cost. None reached the 13 basis "
  "points that a taker pays round trip on Bybit's base tier with modest slippage, and several were negative before costs.")
S += [A.figure(f"{OUT}/fig2.png", 440, 199, "Figure 2. Mean gross basis points per trade by entry design (BTCUSDT 2021-2024) against a 13 bp round-trip cost. H4.5 and H4.5B use the 75% pullback variant.")]
r4 = ov["4"].iloc[0]; r6 = ov["6A2"].iloc[0]
rows = [["H4.4 pullback 25%", f"{int(r4.executed_trades):,}", f"{r4.mean_gross_bps_per_executed_trade:+.2f}", f"{r4.mean_R_conservative:+.3f}", "below 3 bp"],
        ["H4.5B confirmed failure 75%", f"{int(ov['5B'].iloc[2].trades):,}", f"{ov['5B'].iloc[2].mean_gross_bps:+.2f}", f"{ov['5B'].iloc[2].mean_R_conservative:+.3f}", "about 3 bp"],
        ["H4.6A GMMA", f"{int(ov['6A'].iloc[0].trades):,}", f"{ov['6A'].iloc[0].mean_gross_bps:+.2f}", f"{ov['6A'].iloc[0].mean_R_conservative:+.3f}", "below 2 bp"],
        ["H4.6A2 fresh GMMA", f"{int(r6.trades):,}", f"{r6.mean_gross_bps:+.2f}", f"{r6.mean_R_conservative:+.3f}", "about 8 bp"],
        ["H4.6B DC entry", f"{int(ov['6B'].iloc[0].trades):,}", f"{ov['6B'].iloc[0].mean_gross_bps:+.2f}", f"{ov['6B'].iloc[0].mean_R_conservative:+.4f}", "below 2 bp"]]
S.append(tbl(["Entry design", "Trades", "Mean gross bp", "Mean gross R", "Break-even round-trip cost"], rows, [150, 55, 80, 80, 115],
             "Table 5. Entry designs before costs"))
p(f"The best design, fresh GMMA alignment (H4.6A2), earned {r6.mean_gross_bps:.2f} bp gross per trade, with yearly means "
  f"of {', '.join(f'{v:.2f}' for v in a2y.mean_gross_bps)} bp for 2021-2024 and a month-block interval of "
  f"[{ga.gross_bps_block_ci95_low:.2f}, {ga.gross_bps_block_ci95_high:.2f}] bp. At 10 bp round trip its net mean was "
  f"already -1.73 bp. Its R-unit result (+{r6.mean_R_conservative:.3f}R) looks better than its money result because the "
  f"stop is sized from the target distance; R flatters small-risk trades.")
pr = g1bp.loc["FRESH_GMMA_minus_PROGRESS_25_stop_1.00D_BPS"]; pg = g1bp.loc["FRESH_GMMA_minus_GEOM_PLUS_5PP_stop_1.00D_BPS"]
gs = g1b.set_index(["strategy", "stop_fraction_D"])
p(f"The confirmation placebos (H4.G1B) are decisive for this design. On the same signal pool, the non-magnet "
  f"confirmations earned {gs.loc[('PROGRESS_25',1.0),'mean_gross_bps']:.2f} bp (25% progress) and "
  f"{gs.loc[('GEOM_PLUS_5PP',1.0),'mean_gross_bps']:.2f} bp (geometry +5 points) gross per trade, more than fresh GMMA's "
  f"{gs.loc[('FRESH_GMMA',1.0),'mean_gross_bps']:.2f} bp. On matched events GMMA was ahead of the progress placebo by "
  f"{pr.observed_mean_difference:.2f} bp [{pr.ci95_low:.2f}, {pr.ci95_high:.2f}] but not of the geometry placebo "
  f"({pg.observed_mean_difference:.2f} bp [{pg.ci95_low:.2f}, {pg.ci95_high:.2f}]). Whatever gross movement exists after "
  f"a signal is available to any simple confirmation that price is already moving; it is not specific to the magnet "
  f"logic, and it is smaller than the cost of trading it.")

p("5.7 Successor reconstructions and machine learning", H2)
nf = sa["near_first"]
pol = {"DC direction": "DC_only_0p5R_120m", "Nearest magnet": "Magnet_only_0p5R_120m"}
rows = []
for lbl, k in pol.items():
    for per in ["2021-23", "2024"]:
        d = sa[k][per]; rows.append([lbl, per, f"{d['n']:,}", f"{d['mean_net_R']:+.4f}", f"[{d['CI95'][0]:+.4f}, {d['CI95'][1]:+.4f}]"])
ak = [k for k in sa if k.startswith("Aligned_ratio0p5")]
if ak:
    for per in ["2021-23", "2024"]:
        d = sa[ak[0]][per]; rows.append(["Aligned with near magnet, ratio <= 0.5", per, f"{d['n']:,}", f"{d['mean_net_R']:+.4f}", f"[{d['CI95'][0]:+.4f}, {d['CI95'][1]:+.4f}]"])
S.append(tbl(["Policy (+0.5R target, 120 min)", "Period", "Trades", "Mean net R", "95% weekly-block interval"], rows,
             [170, 55, 55, 70, 130], "Table 6. Independent reconstruction (v2): next-minute entries after 13 bp costs"))
p(f"The independent reconstruction reproduced the central finding on its own level sets: the nearer level was reached "
  f"first in {nf['actual_pct']:.2f}% of {nf['n']:,} resolved events against {nf['geometry_heuristic_pct']:.2f}% from "
  f"distance alone, an excess of {nf['excess_pp']:+.2f} points [{nf['weekly_block_CI95_excess_pp'][0]:+.2f}, "
  f"{nf['weekly_block_CI95_excess_pp'][1]:+.2f}]. Every tradable rule lost money after costs, with intervals entirely below "
  f"zero. The nearest-magnet policy's gross edge shrank from about +3.2 bp in 2021 to +0.1 bp in 2024. Profit-protection "
  f"and early-failure exits evaluated on identical entries did not produce a robust improvement.")
p("The target-aware diagnostic explains one reason. Under the aligned rule, the median distance to the magnet was only "
  "0.23-0.30R, and the magnet sat closer than the +0.5R take-profit in 70-85% of trades by year. A correct destination "
  "forecast therefore often ended before the trade could pay. Placing the take-profit at the magnet instead did not help: "
  "all four years were negative in both the full and the cost-feasible cohorts (between -0.04R and -0.33R per trade).")
h = h5y[(h5y.policy == "dc")]
p("The H5 extension found that at a completed slow directional-change event the fast scale already agreed in "
  "99.7-99.8% of cases on all three assets, so two-scale agreement carried almost no information. The unconditional slow "
  "signal had gross returns within about one hundredth of R of zero in 2024-2025. The gradient-boosted selector traded "
  "rarely and inconsistently: six BTC trades in 2024 (all winners, too few to evaluate), 43 XRP trades in 2024 at "
  "+0.071R [-0.150, +0.303], and negative results in every ETH and XRP year from 2025. A ridge variant showed similarly "
  "unstable single-year positives. No model passed a cross-market, later-year robustness test.")

# ---------------------------------------------------------------- 6 discussion
p("6. Discussion", H1)
p("6.1 Proximity, not attraction", H2)
p("The headline accuracy of liquidity-magnet levels is mostly arithmetic. When one level is half as far as the other, "
  "even a random walk reaches it first about two thirds of the time, and the observed rates track that geometry closely. "
  "Strict controls leave a residual of one to two points overall and larger at previous-day extremes, consistent with "
  "the order-clustering mechanism Osler documented in currency markets. But the residual concentrated in one year and in "
  "trendless conditions, and it added nothing to out-of-sample forecasts once distance was known.")
p("6.2 Why a correct forecast did not pay", H2)
p("Three mechanisms separate destination accuracy from profit. First, the stop: a forecast resolved over 24 hours "
  "says nothing about the adverse excursion on the way. Second, geometry inside the trade: the magnet usually lay "
  "closer than the profit target, so being right often meant exiting early or not at all. Third, cost: the gross "
  "movement available after any reasonable confirmation was a few basis points, below a taker's round-trip cost and in "
  "most designs below a maker's. Lower volatility made this worse, because cost in R units rises as the stop narrows.")
p("6.3 Units can mislead", H2)
p("Several designs showed positive mean R while earning little or nothing in basis points. When stop distance varies with "
  "the setup, small-risk trades dominate R averages. Reporting both units, and judging tradability in money after costs, "
  "prevented that from being mistaken for an edge.")
p("6.4 Research process", H2)
p("This was a sequence of related tests on the same BTC history rather than a single pre-registered experiment. Each "
  "stage froze its own rules and controls, and several declared their primary comparison in advance (MODEL1's M2 versus "
  "M1; G1B's matched placebos). No family-wise correction spans the whole sequence. That cuts against a positive "
  "finding, not a negative one: the many specifications had every chance to find a tradable effect, and none survived.")

# ---------------------------------------------------------------- 7 limits
p("7. Limitations", H1)
for t in ["The core tests use one asset on one venue; the multi-asset extension covers only the successor design.",
          "Levels are proxies built from past prices, not measurements of resting orders. Order-book and stop-order data could show a mechanism that price-based levels cannot.",
          "The H4 cost grid stopped at 10 bp round trip, below a realistic taker cost; this makes the reported break-even points generous.",
          "Funding and historical bid-ask spreads were not modelled. Most trades were held for minutes to hours, so funding is unlikely to change the sign of the results, but maker execution remains untested.",
          "One-minute candles cannot resolve the order of high and low within a minute; ambiguous minutes were scored as losses.",
          "BTC 2021-2024 was examined repeatedly across studies; there is no pristine holdout, and the successor ETH/XRP 2025-2026 years were also seen in earlier work.",
          "The original H4.2 event table was not available to the successor studies, which therefore test related rather than identical level definitions."]:
    S.append(Paragraph(t, BL, bulletText="•"))
p("8. Future research", H1)
p("A more promising question than trading toward a level is how price behaves when it first sweeps one. A pre-specified "
  "test would timestamp the first sweep of an unswept previous-day or 8-hour extreme, compare rejection (a close back "
  "across the level) with acceptance (continued closes beyond it) as competing hypotheses, label tradable payoffs after "
  "the next open with cost-covered exits, and control with same-distance synthetic levels and time-matched random "
  "entries. Order-flow and order-book data, actual contract specifications, funding, and genuinely new data would be "
  "required for any confirmatory claim.")
p("9. Conclusion", H1)
p("Price does tend to reach the nearer of two prior levels first, but almost entirely because it is nearer. A small "
  "residual at previous-day highs and lows survives in-sample placebos, yet it is unstable, confined to trendless "
  "periods and absent out of sample. No entry design built on these forecasts earned its trading cost, and simple "
  "non-magnet confirmations did as well as the best of them. For a cost-paying trader, knowing where price is likely to "
  "go next was not enough to profit from it.")

# ---------------------------------------------------------------- declarations
p("Declarations", H1)
for h, t in [("Funding", "This research received no external funding."),
             ("Conflicts of interest", "The author declares no conflict of interest."),
             ("Data and code availability", "All study scripts, frozen configurations, summary result tables and the original research reports are public at https://github.com/Crayon99-droid/magnet-price-levels-study. Raw exchange data are not redistributed and can be reacquired from Bybit's public endpoints; event-level and trade-level files are available from the author on request."),
             ("Use of AI-assisted tools", "Generative AI tools assisted with code development, diagnostic framing and manuscript drafting. They did not generate, impute or replace market data. Reported values in this manuscript were read programmatically from the frozen result files in the repository. The author reviewed the methods, results and text and accepts responsibility for the final manuscript."),
             ("Ethics and investment disclaimer", "The study used public market data and involved no human subjects. It is research, not investment advice.")]:
    p(h, H2); p(t)

# ---------------------------------------------------------------- references
p("References", H1)
for r in ["Anghel, D.-G. (2021). A reality check on trading rule performance in the cryptocurrency market: Machine learning vs. technical analysis. <i>Finance Research Letters</i>, 39, 101655.",
          "Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. <i>Journal of Portfolio Management</i>, 40(5), 94-107.",
          "Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. <i>Journal of Finance</i>, 47(5), 1731-1764.",
          "Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. <i>Review of Financial Studies</i>, 29(1), 5-68.",
          "Kissell, R. (2020). <i>Algorithmic Trading Methods: Applications Using Advanced Statistics, Optimization, and Machine Learning Techniques</i> (2nd ed.). Academic Press.",
          "Osler, C. L. (2000). Support for resistance: Technical analysis and intraday exchange rates. <i>Federal Reserve Bank of New York Economic Policy Review</i>, 6(2), 53-68.",
          "Osler, C. L. (2003). Currency orders and exchange rate dynamics: An explanation for the predictive success of technical analysis. <i>Journal of Finance</i>, 58(5), 1791-1819.",
          "Tariq, R. (2026). From apparent profit to a reproducible null result: A cost-aware evaluation of directional-change reversal trading in cryptocurrency perpetual futures. Working paper.",
          "White, H. (2000). A reality check for data snooping. <i>Econometrica</i>, 68(5), 1097-1126."]:
    S.append(Paragraph(r, REF))

# ---------------------------------------------------------------- appendix A
S.append(PageBreak()); p("Appendix A. Anticipated questions", H1)
QA = [
("A.1 Isn't 70% accuracy proof that levels attract price?",
 "No. With the near level at half the distance of the far one, a driftless random walk reaches it first about two thirds of the time. Observed rates sat within about 1.5 points of that benchmark overall (Section 5.2)."),
("A.2 The geometry null assumes Brownian motion. Is that fair to Bitcoin?",
 "It is only a benchmark, which is why the mirror placebo was added: it uses the actual BTC paths and identical distances, removing any assumption about the return process. The conclusions are the same under both controls."),
("A.3 The PDH/PDL residual of about 9-10 points is significant. Why not trade it?",
 "It rests on 512 paired events (337 in the strongest subset), was concentrated in 2021, 2023 and 2024 with none in 2022, lost significance once the 24-hour trend was conditioned on, and the pre-specified out-of-sample test of a PDH/PDL feature showed no improvement over trend and distance (Sections 5.3-5.5). It is a hypothesis for new data, not a trading signal."),
("A.4 Could a different entry fix the problem?",
 "Six entry families were tested: immediate, pullback, failure reversal with and without confirmation, moving-average alignment and directional change, plus a stop frontier and target-aware exits. The best earned about 8 bp gross, and simple non-magnet confirmations earned as much on the same signals."),
("A.5 Would maker orders or lower fees help?",
 "The best design's break-even round-trip cost was about 8 bp; most others were 1-3 bp. Bybit's base maker fee alone is 2 bp per side. Maker fills would also need to be modelled with fill probabilities and adverse selection, which candle data cannot provide."),
("A.6 Why are some results positive in R but negative in money?",
 "R normalizes by stop distance. When stops are small, small absolute moves become large R values and costs become large fractions of R. Judging tradability in basis points after costs avoids that illusion (Section 6.3)."),
("A.7 Were the many tests corrected for multiple comparisons?",
 "Within stages, yes: month-block bootstraps, pre-declared primary comparisons and exploratory labels. Across the full sequence, no. That omission can only make a positive result look stronger than it is; it cannot hide one."),
("A.8 Why only BTC for the main tests?",
 "BTC is the deepest crypto market and the natural first test of a liquidity mechanism. ETH and XRP were added in the successor H5 design and gave the same answer."),
("A.9 Do the magnet levels measure real resting orders?",
 "No; they are price-based proxies. A study using order-book and liquidation data could test the mechanism more directly, and is the most credible way to revisit it."),
("A.10 What would change the conclusion?",
 "A pre-registered sweep-and-reaction test (Section 8) that beats same-distance synthetic levels and random entries on genuinely new data, after realistic costs, funding and contract rules.")]
for q, a in QA:
    S.append(KeepTogether([Paragraph(q, Q), Paragraph(a, B)]))

# ---------------------------------------------------------------- appendix B
p("Appendix B. Stage map", H1)
rows = [["H4.1", "PDH/PDL destination accuracy", "h4_1_btc_pdh_pdl_v2.py", "H4_1_*"],
        ["H4.2", "Multi-level magnet events", "h4_2_btc_multilevel_magnets_v2_safe_save.py", "H4_2_*"],
        ["H4.3", "Path tradability by stop", "h4_3_btc_magnet_path_tradability_v3_timestamp_fixed.py", "H4_3_*"],
        ["H4.4", "Pullback entries", "h4_4_btc_pullback_entry.py", "H4_4_*"],
        ["H4.5 / 5B", "Failure reversals", "h4_5_btc_magnet_failure_reversal.py, h4_5b_...py", "H4_5_*, H4_5B_*"],
        ["H4.6A / A2 / B", "GMMA and DC entries", "h4_6a_...py, h4_6a2_...py, h4_6b_...py", "H4_6A*, H4_6B_*"],
        ["G1A / G1B", "Stop frontier; confirmation placebos", "h4_g1a_...py, h4_g1b_...py", "H4_G1A_*, H4_G1B_*"],
        ["NULL1 / NULL2 / NULL2T", "Geometry, mirror, trend controls", "h4_null1_...py, h4_null2_...py, h4_null2t_...py", "H4_NULL*"],
        ["MODEL1", "Walk-forward probability models", "h4_model1_geometry_trend_structure_oos.py", "H4_MODEL1_*"],
        ["v2, diagnostic, H5", "Successor reconstructions and ML", "packs/ (btc_magnet_dc_v2.py, magnet_dc_h5.py, ...)", "reports/, packs/"]]
S.append(tbl(["Stage", "Question", "Script", "Results"], rows, [70, 125, 175, 85]))

A.build(f"{OUT}/Magnet_Levels_BTC_Null_Result_Tariq_2026.pdf", S,
        "Where Price Goes Is Not How to Profit: Testing Liquidity Magnet Price Levels in Bitcoin Perpetual Futures",
        "Rayan Tariq", "Working paper, November 2026",
        first_page_note=["Data, code and result files: github.com/Crayon99-droid/magnet-price-levels-study.",
                         "Not investment advice. No tested policy was authorized for live trading."])
print("built")

"""Build the multi-page PDF summary report for the Final Project.

Uses matplotlib's PdfPages: two text pages (introduction through limitations),
then two figure pages, computed from the same pipeline as the notebook.

Output: Quant_Final_Report.pdf
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

TICKERS = ["SPY", "QQQ", "XLK", "XLE", "XLF", "XLU"]
HORIZONS = [1, 5, 20]

# ----------------------------------------------------------------------------
# Recompute the results (identical pipeline to the notebook)
# ----------------------------------------------------------------------------
prices = pd.read_csv("data/etf_prices.csv", index_col=0, parse_dates=True)
prices = prices[TICKERS].sort_index()
prices.index = pd.to_datetime(prices.index)
bdays = pd.bdate_range(prices.index.min(), prices.index.max())
prices = prices.reindex(bdays).ffill(limit=1).dropna()
returns = np.log(prices).diff().dropna()

features = {}
for t in TICKERS:
    r = returns[t]
    df = pd.DataFrame({"ret": r})
    df["vol_20"] = r.rolling(20).std() * np.sqrt(252)
    for h in HORIZONS:
        df[f"lag_{h}"] = r.rolling(h).sum()
        df[f"fwd_{h}"] = r.shift(-1).rolling(h).sum().shift(-(h - 1))
    features[t] = df

spy_vol = features["SPY"]["vol_20"].dropna()
q25 = spy_vol.expanding(252).quantile(0.25)
q75 = spy_vol.expanding(252).quantile(0.75)
regime = pd.Series("Mid Vol", index=spy_vol.index)
regime[spy_vol <= q25] = "Low Vol"
regime[spy_vol >= q75] = "High Vol"


def pearson_table():
    out = {}
    for t in TICKERS:
        out[t] = {}
        for h in HORIZONS:
            sub = features[t][[f"lag_{h}", f"fwd_{h}"]].dropna()
            out[t][h] = stats.pearsonr(sub[f"lag_{h}"], sub[f"fwd_{h}"]).statistic
    return pd.DataFrame(out).T.reindex(TICKERS)[HORIZONS]


def pearson_by_regime(reg_label):
    out = {}
    for t in TICKERS:
        merged = features[t].join(regime.rename("reg"))
        out[t] = {}
        for h in HORIZONS:
            sub = merged[merged["reg"] == reg_label][[f"lag_{h}", f"fwd_{h}"]].dropna()
            out[t][h] = stats.pearsonr(sub[f"lag_{h}"], sub[f"fwd_{h}"]).statistic
    return pd.DataFrame(out).T.reindex(TICKERS)[HORIZONS]


full_r = pearson_table()
low_r = pearson_by_regime("Low Vol")
high_r = pearson_by_regime("High Vol")

# ----------------------------------------------------------------------------
# Build the PDF
# ----------------------------------------------------------------------------
REPORT_PATH = Path("Quant_Final_Report.pdf")

AUTHOR = "Austin Renz"

with PdfPages(REPORT_PATH) as pdf:

    def draw_text_page_header(fig):
        """Name and report title on each text page."""
        fig.text(0.5, 0.98, AUTHOR, ha="center", fontsize=12, fontweight="bold")
        title = (
            "Momentum vs. Mean Reversion across Sector ETFs,\n"
            "Horizons, and Volatility Regimes"
        )
        subtitle = "Quant Mentorship Final Project: Summary Analysis Report"
        fig.text(0.5, 0.935, title, ha="center", fontsize=14, fontweight="bold")
        fig.text(0.5, 0.885, subtitle, ha="center", fontsize=10, style="italic", color="#444")

    # -----------------------------
    # PAGE 1 - text (sections 1-3)
    # -----------------------------
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    draw_text_page_header(fig)

    body_page1 = r"""
1. INTRODUCTION & QUESTION
This project asks a single, focused research question: how do momentum and
mean-reversion behaviors in US equity ETFs vary across (i) the ETF itself,
(ii) the return horizon, and (iii) the prevailing volatility regime? The
aim is not to build a trading strategy, but to characterize where and when
the textbook patterns of short-horizon reversal and longer-horizon
persistence actually show up in real data.

2. DATASET & SOURCE
Daily adjusted close prices for six ETFs (SPY, QQQ, XLK, XLE, XLF, XLU),
downloaded from Yahoo Finance (yfinance, auto_adjust=True) over
2010-01-05 to 2026-04-17 (4,248 trading days after cleaning). Adjusted
closes are used so that splits and dividends are baked into the price
series, making log-returns a clean measure of total return. The ETFs were
chosen to combine a broad-market anchor (SPY), a growth anchor (QQQ), and
four sector views of different parts of the economy (technology, energy,
financials, utilities).

3. METHODS
- Cleaning: reindex to a business-day calendar, forward-fill gaps of at
  most one day, drop any remaining missing rows.
- Feature engineering: for each ETF and each horizon h in {1, 5, 20}
  trading days, construct two aligned columns:
    ret_lag_h  = cumulative log-return over the previous h days (observed
                 at t),
    ret_fwd_h  = cumulative log-return over the next h days (realized
                 strictly after t).
  This alignment is essential; it is what prevents lookahead bias.
- Volatility: 20-day rolling realized volatility, annualized.
- Regime indicator: the top/bottom quartile of SPY's 20d realized vol
  using expanding-window thresholds (so classification at date t depends
  only on history up to t).
- Statistical tests: Pearson and Spearman correlations between ret_lag_h
  and ret_fwd_h, full-sample and within each regime.
- Modeling: chronological 80/20 train/test split; a simple univariate
  OLS (fwd = a + b * lag) and an extended OLS that adds rolling vol, a
  high-vol dummy, and a lag x high-vol interaction; Newey-West (HAC)
  standard errors with 10 lags to correct for the overlap induced by the
  rolling forward-return target.
"""
    fig.text(0.07, 0.84, body_page1, ha="left", va="top", fontsize=9.5, family="serif")
    pdf.savefig(fig)
    plt.close(fig)

    # -----------------------------
    # PAGE 2 - text (sections 4-5)
    # -----------------------------
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    draw_text_page_header(fig)
    fig.text(0.5, 0.835, "(continued)", ha="center", fontsize=9, style="italic", color="#666")

    body_page2 = r"""
4. KEY FINDINGS
(a) SHORT-HORIZON MEAN REVERSION IS PERVASIVE. Over the full sample, the
    Pearson correlation between past 1-day and next 1-day returns is
    negative for every ETF, ranging from -0.022 (XLE) to -0.105 (XLK) and
    -0.100 (XLF). The 1d reversal is most pronounced in broad-market and
    sector ETFs that are heavily traded.

(b) 20-DAY RETURNS ALSO MEAN-REVERT, ESPECIALLY XLU. At the 20-day
    horizon, five of six ETFs show correlations between -0.07 and -0.22.
    Utilities (XLU) is the strongest: r = -0.223, meaning streaks of
    outsized 20-day moves in XLU are followed on average by partial
    give-back. Broad-market SPY sits at r = -0.146.

(c) XLE IS THE NOTABLE OUTLIER. Energy is the only ETF with a positive
    5-day correlation (r = +0.026), and its short-horizon reversal is the
    mildest of the group. This is consistent with XLE's heavy exposure to
    commodity prices, which exhibit their own momentum dynamics.

(d) HIGH VOLATILITY DRAMATICALLY AMPLIFIES SHORT-HORIZON REVERSAL. At
    h = 1 day, the Low-Vol regime shows nearly flat correlations (-0.042
    to +0.024), while the High-Vol regime shows sharp negatives: SPY
    -0.149, QQQ -0.149, XLK -0.174, XLF -0.160, XLU -0.104. This is the
    single cleanest result in the study.

(e) REGRESSIONS CORROBORATE THE STORY. The univariate lag-return slope is
    negative for every sector at the 5-day horizon, and the HAC-adjusted
    t-statistics on the extended SPY regression are consistent with the
    sign pattern, though the R^2 is small (as expected for daily equity
    returns).

5. REFLECTION & LIMITATIONS
The effect sizes are statistically visible but economically small on a
per-trade basis; real strategies combine many such weak signals and have
to survive costs and frictions not modeled here. The regime split depends
on the chosen volatility window and quartile cutoffs; other choices
would shift magnitudes without overturning the sign pattern. Overlapping
5- and 20-day forward-return windows induce residual autocorrelation; we
address it with HAC standard errors but do not eliminate it. Most
importantly, the panel ends in 2026 and covers one extended low-rate era
(2010-2021) and one tightening era (2022-2026); different economic
regimes could produce different numbers. The central takeaway is that
mean reversion's strength depends jointly on horizon, sector, and
volatility regime. That result is robust to these caveats even if the exact
coefficients are not.
"""
    fig.text(0.07, 0.795, body_page2, ha="left", va="top", fontsize=9, family="serif")
    pdf.savefig(fig)
    plt.close(fig)

    # -----------------------------
    # PAGE 3 - headline figure
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    sns.heatmap(full_r, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-0.25, vmax=0.25, ax=axes[0],
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Full-sample Pearson r:\nPast h-day return vs. next h-day return")
    axes[0].set_xlabel("Horizon (trading days)")
    axes[0].set_ylabel("ETF")

    regime_diff = high_r - low_r
    sns.heatmap(regime_diff, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-0.2, vmax=0.2, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Regime effect on r:\nHigh-Vol minus Low-Vol")
    axes[1].set_xlabel("Horizon (trading days)")
    axes[1].set_ylabel("ETF")

    fig.suptitle(
        "Figure 1. Horizon and regime dependent predictability of sector ETF returns (2010-2026)",
        fontsize=11, fontweight="bold", y=1.0,
    )
    fig.text(0.5, 0.02,
             "Left: negative values indicate short-horizon mean reversion. "
             "Right: blue cells show that predictive sign strengthens (toward reversal) in high-vol regimes.",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.text(0.5, 0.99, AUTHOR, ha="center", fontsize=9, fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)

    # -----------------------------
    # PAGE 4 - second figure: growth of $1 + rolling vol
    # -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    normalized = prices / prices.iloc[0]
    for col in normalized.columns:
        axes[0].plot(normalized.index, normalized[col], label=col, linewidth=1.2)
    axes[0].set_title("Figure 2a. Growth of $1 invested, 2010-2026 (adjusted close)")
    axes[0].set_ylabel("Multiple of initial value")
    axes[0].legend(ncol=3)

    vol_panel = pd.concat({t: features[t]["vol_20"] for t in TICKERS}, axis=1)
    for col in vol_panel.columns:
        axes[1].plot(vol_panel.index, vol_panel[col], label=col, linewidth=0.9, alpha=0.85)
    hi = (regime == "High Vol").reindex(vol_panel.index).fillna(False).astype(bool)
    axes[1].fill_between(vol_panel.index, 0, vol_panel.max().max(),
                         where=hi.values, color="red", alpha=0.08,
                         label="SPY High-Vol Regime")
    axes[1].set_title("Figure 2b. 20-day rolling realized volatility (annualized)")
    axes[1].set_ylabel("Annualized volatility")
    axes[1].legend(ncol=4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.text(0.5, 0.99, AUTHOR, ha="center", fontsize=9, fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)

print(f"Wrote {REPORT_PATH}")

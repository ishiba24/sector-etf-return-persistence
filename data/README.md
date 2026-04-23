# Data files (pre-downloaded for submission)

## `etf_prices.csv`

- **Source:** Yahoo Finance, retrieved with `yfinance` (`auto_adjust=True`).
- **Content:** One row per trading day, columns `SPY`, `QQQ`, `XLK`, `XLE`, `XLF`, `XLU` (adjusted close suitable for return calculations).
- **Purpose:** Graders can run the notebook **without** calling Yahoo or any other API.

## `synthesis_table.csv`

- Small summary table produced by the analysis (optional).

## `slide_headline.png`

- Asset used when regenerating `Quant_Final_Slide.pptx` via `build_slide.py`.

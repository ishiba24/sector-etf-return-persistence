# QM Final Project: Sector ETF Momentum vs. Mean Reversion

**Author:** Austin Renz  
**Course:** Quant Mentorship (Spring 2026)  

## What this repository contains

| Item | Description |
|------|-------------|
| `quant_final_project.ipynb` | Main analysis notebook (cleaning, EDA, horizons, regimes, modeling, conclusions). |
| `Quant_Final_Report.pdf` | Written summary with methods, findings, and figures. |
| `Quant_Final_Slide.pptx` | Single presentation slide (headline result). |
| `data/etf_prices.csv` | **Pre-downloaded** daily adjusted close prices (required for grading without live API calls). |
| `data/synthesis_table.csv` | Exported summary correlations (optional). |
| `build_notebook.py` | Regenerates the notebook from source (optional). |
| `build_report.py` / `build_slide.py` | Regenerate PDF and PPTX (optional). |
| `data/README.md` | Notes on bundled CSV data. |
| `REFERENCES.md` | Research and methodology citations. |

## Data (no API required to reproduce)

All Yahoo Finance data used in the project is **committed** as:

- **`data/etf_prices.csv`**  
  - **Tickers:** SPY, QQQ, XLK, XLE, XLF, XLU  
  - **Frequency:** daily (business days)  
  - **Field:** adjusted close (via `yfinance` `auto_adjust=True` at download time)

The notebook’s `load_prices()` function **loads this file first**. If the CSV is present, **no network download runs**, which matches the instruction: *we will not be running APIs or DB queries to download your data*.

If you ever delete the CSV and re-run only the download cell, you would need internet and `yfinance`; for submission, **keep `data/etf_prices.csv` in the repo**.

## How to run

1. Python 3.10+ recommended.  
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn yfinance statsmodels scikit-learn scipy jupyter nbformat nbclient
   ```

3. Open and run **`quant_final_project.ipynb`** from the repo root (or **Run All**). With `data/etf_prices.csv` present, analysis should run offline.

## Regenerating deliverables (optional)

From the repository root:

```bash
python3 build_report.py    # Quant_Final_Report.pdf
python3 build_slide.py     # Quant_Final_Slide.pptx (uses notebook pipeline data)
python3 build_notebook.py  # quant_final_project.ipynb (requires network if CSV missing)
```

## Research citations

See **`REFERENCES.md`** for papers and sources cited for momentum, mean reversion, volatility regimes, and methodology.

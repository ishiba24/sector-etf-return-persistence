# References and citations

The project design (lagged vs. forward returns across horizons, volatility regimes, and simple predictive regressions) is standard in empirical asset pricing and market microstructure. The following sources informed framing, interpretation, and methodology. **No single paper was replicated end-to-end**; the work is an original panel study on sector ETFs using public data.

## Academic and practitioner literature

1. **Jegadeesh, N., & Titman, S. (1993).** Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.  
   - Classic reference for **return continuation (momentum)** at medium horizons in equities.

2. **Lo, A. W., & MacKinlay, A. C. (1988).** Stock market prices do not follow random walks: Evidence from a simple specification test. *Review of Financial Studies*, 1(1), 41-66.  
   - Motivates **short-horizon predictability** and departures from the random walk; relevant to interpreting very short lag/future return relationships.

3. **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012).** Time series momentum. *Journal of Financial Economics*, 104(2), 228-250.  
   - Discusses **time-series momentum** and horizon dependence in trend-following style evidence.

4. **Cont, R. (2001).** Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223-236.  
   - **Stylized facts** (volatility clustering, fat tails) supporting rolling volatility and regime-style splits in exploratory analysis.

5. **Hamilton, J. D. (1989).** A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.  
   - Foundational **regime-switching** reference; our volatility quartile labels are a simple descriptive regime split, not a full Hamilton filter.

## Methodology and software

6. **Newey, W. K., & West, K. D. (1987).** A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.  
   - Supports **HAC (Newey–West)** standard errors where overlapping multi-day return windows induce serial correlation in regression residuals.

7. **pandas development team.** pandas. Zenodo / https://pandas.pydata.org/  
8. **McKinney, W.** Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference (2010).  
9. **yfinance:** Ran Aroussi et al., https://github.com/ranaroussi/yfinance (Yahoo Finance market data access; used only for the **initial** download that produced `data/etf_prices.csv`).

## Course materials

10. **Georgia Tech Investments Committee, Quant Mentorship (Spring 2026).** Final project design document (*S26 Quant Mentorship Final Project*) and mentorship slides: project scope, deliverables, and rubric.

---

If you add new models or literature during revisions, append numbered entries here and mention the key idea in the notebook or PDF.

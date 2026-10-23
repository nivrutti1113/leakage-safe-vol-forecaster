# Changelog

All notable changes to the `vol-forecaster` project are documented in this file.

## [2.1.0] - 2026-10-23

### Fixed
- **Volatility Targeting Realized Volatility Overshoot**: Resolved the discrepancy where uncalibrated volatility targeting realized 20%–22% volatility against a 15% target (exceeding Buy & Hold at 17.3%). Because the models forecast intraday Garman-Klass variance which omits overnight jumps, sizing by Garman-Klass volatility resulted in an over-leveraged position (by a factor of $\sqrt{1.87} \approx 1.37$). Forecasts are now calibrated to close-to-close variance inside each expanding training window using strictly past data. Realized volatility now closely matches the 15.0% target (14.4%–15.8% across models) while substantially reducing max drawdown (HAR MDD 22.70% vs Buy & Hold 33.92% on S&P 500).
- **Position Timing Strictness**: Position on day $t$ is explicitly sized using forecasts and prices available at the close of day $t-1$. Added unit test `test_vol_targeting_positions_use_only_past_close` verifying that perturbing prices on and after day $t$ has zero impact on $\text{position}[t]$.
- **Overclaiming Removal**: Removed all claims that purging "removes" or "eliminates" leakage. Documented that random shuffling is the primary driver of leakage under standard K-fold cross-validation, while contiguous blocks prevent feature interpolation. Purging and embargoing provide a modest boundary safeguard ($\approx 0.001$ QLIKE difference at $h=5$). Removed all usages of the word "alpha".

### Added
- **Block Bootstrap Calibration & Comparison**: Enforced block length $B \ge \max(h, 20)$ for moving block bootstrap confidence intervals to account for serial correlation. Reported the block size and analyzed the 3 borderline cases (out of 42 comparisons) where DM tests and bootstrap CIs slightly diverge due to asymptotic normality vs empirical resampling assumptions.
- **Master Tables in `results/`**:
  - `results/pairwise_dm.csv`: Pairwise DM statistics and Holm-Bonferroni step-down adjusted p-values.
  - `results/regimes.csv`: Walk-forward scores across High-Vol, Low-Vol (partitioned using trailing 66-day median volatility vs expanding median on past data only), and the Covid-19 (2020) crisis.
  - `results/mincer_zarnowitz.csv`: Standalone Mincer-Zarnowitz diagnostic table (intercept, slope, $R^2$) across all models, horizons, and tickers.
  - `results/vol_targeting_metrics.csv`: Master backtest scorecard with transaction cost sensitivities (0, 2, 5, 10 bps).

## [2.0.0] - 2026-10-23

### Fixed
- **GARCH Target Misalignment & Negative $R^2$ Diagnosis**: Diagnosed why raw GARCH(1,1) produced negative out-of-sample $R^2$ ($-0.32$ on `^GSPC`, $-0.62$ on `^NSEI`). Raw GARCH models close-to-close return variance (which includes overnight jump risk), whereas the evaluation target is intraday Garman-Klass variance (which omits overnight jumps). On S&P 500, close-to-close return variance averages $\approx 1.87\times$ Garman-Klass variance ($\approx +0.99$ offset in log space). This constant level bias destroyed out-of-sample MSE and produced negative $R^2$ despite tracking volatility dynamics accurately ($b \approx 0.96$).
- **Within-Training Calibration (Mincer-Zarnowitz)**: Implemented training-window calibration where $y_{\text{train}} = a + b \cdot \log(\text{GARCH}_{\text{in-sample}})$ is fit on the training block and applied out-of-sample. Out-of-sample $R^2$ shifted from negative to positive ($+0.3686$ on `^GSPC`, $+0.2448$ on `^NSEI` for $h=1$).
- **GJR-GARCH(1,1,1) Added**: Added Glosten-Jagannathan-Runkle asymmetric volatility to model the leverage effect ($o=1$ in `arch`), reducing QLIKE loss from $0.5677$ to $0.5303$ on `^GSPC` ($h=1$).

### Added
- **Multi-Horizon & Multi-Market Benchmark**: Automated walk-forward evaluations across $h \in \{1, 5, 22\}$ days on S&P 500 (`^GSPC`) and Nifty 50 (`^NSEI`), aggregating results into tidy `results/summary.csv`.
- **Leakage Demo on Aligned Test Samples**: Evaluated Shuffled K-Fold, Contiguous K-Fold, Purged K-Fold + Embargo, and Walk-Forward on the exact same out-of-sample date index across $h \in \{1, 5, 22\}$.
- **Automated CI & Configuration Engine**: Single CLI runner `python -m volfc.run`, YAML configs, `Makefile`, and GitHub Actions workflow `.github/workflows/ci.yml`.

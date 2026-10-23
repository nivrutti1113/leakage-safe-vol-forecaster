"""Data loading: CSV, Yahoo Finance (optional), and a synthetic OHLC generator for offline testing."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip().title()).sort_index()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Need Date index and {REQUIRED}.")
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=REQUIRED)
    df = df[(df.High >= df.Low) & (df.Low > 0)]
    keep = REQUIRED + (["Volume"] if "Volume" in df.columns else [])
    return df[keep].astype(float)


def load_csv(path: str) -> pd.DataFrame:
    """CSV with columns Date, Open, High, Low, Close and optionally Volume."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return _clean(df)


def download_yfinance(ticker: str, start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance (needs internet and `pip install yfinance`)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install yfinance (pip install yfinance) or use --source csv.") from e
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _clean(df)


def make_synthetic(n_days: int = 3500, seed: int = 0, steps: int = 78) -> pd.DataFrame:
    """Synthetic daily OHLCV with persistent stochastic volatility and a leverage effect.

    This is NOT market data. It exists so the pipeline and the leakage tests run offline.
    Daily log-vol follows an AR(1) with a negative response to yesterday's return shock.
    """
    rng = np.random.default_rng(seed)
    phi, eta, lev = 0.975, 0.12, 0.10
    mu = np.log(0.01)  # ~1% daily vol
    log_s = mu
    prev_close = 100.0
    rows = []
    z_prev = 0.0
    for _ in range(n_days):
        log_s = mu + phi * (log_s - mu) - lev * z_prev + eta * rng.standard_normal()
        sigma = float(np.exp(log_s))
        gap = 0.3 * sigma * rng.standard_normal()
        o = prev_close * np.exp(gap)
        incr = rng.standard_normal(steps) * sigma / np.sqrt(steps)
        path = o * np.exp(np.cumsum(incr))
        hi, lo, c = max(o, path.max()), min(o, path.min()), path[-1]
        z_prev = float(incr.sum() / sigma)
        vol = float(np.exp(15 + 0.5 * (log_s - mu) + 0.2 * rng.standard_normal()))
        rows.append((o, hi, lo, c, vol))
        prev_close = c
    idx = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])

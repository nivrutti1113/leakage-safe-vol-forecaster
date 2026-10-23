"""Features and target. Every feature at row t uses data up to the close of day t only.

Target at row t: log of the mean Garman-Klass variance over days t+1 ... t+h (h=1 is next-day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_WINDOW = 66  # longest rolling window used by any feature (days)
HAR_COLS = ["lv1", "lv5", "lv22"]
VAR_FLOOR = 1e-8


def gk_variance(df: pd.DataFrame) -> pd.Series:
    """Garman-Klass daily variance estimator from OHLC (floored to stay positive)."""
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    return gk.clip(lower=VAR_FLOOR)


def build_dataset(df: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    gk = gk_variance(df)
    lv = np.log(gk)
    ret = np.log(df["Close"] / df["Close"].shift(1))
    gap = np.log(df["Open"] / df["Close"].shift(1))

    X = pd.DataFrame(index=df.index)
    X["lv1"] = lv
    X["lv5"] = np.log(gk.rolling(5).mean())
    X["lv22"] = np.log(gk.rolling(22).mean())
    X["lv66"] = np.log(gk.rolling(MAX_WINDOW).mean())
    X["absret"] = ret.abs()
    X["neg_ret"] = ret.clip(upper=0.0)
    X["absgap"] = gap.abs()
    X["volvol22"] = lv.rolling(22).std()
    if "Volume" in df.columns and (df["Volume"] > 0).all():
        lvol = np.log(df["Volume"])
        X["volz"] = (lvol - lvol.rolling(MAX_WINDOW).mean()) / lvol.rolling(MAX_WINDOW).std()

    # mean variance over days t+1 .. t+h
    y = np.log(gk.rolling(horizon).mean().shift(-horizon))

    data = X.assign(y=y).replace([np.inf, -np.inf], np.nan).dropna()
    first, last = df.index.get_loc(data.index[0]), df.index.get_loc(data.index[-1])
    if len(data) != last - first + 1:
        raise ValueError("Rows were dropped in the middle of the sample; rows must be consecutive days.")
    return data.drop(columns="y"), data["y"]

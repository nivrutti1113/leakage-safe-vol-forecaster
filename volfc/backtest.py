"""Volatility targeting strategy backtest on index returns.

Position sizing: position[t] = clip(target_vol_daily / forecast_vol[t-1], 0, max_leverage).
Uses only forecasts and prices available at the close of day t-1 to size the position for day t.
Models targeting Garman-Klass variance are calibrated to close-to-close variance inside each
refit's training window using strictly past data, correcting for the ~1.87 variance ratio.
Evaluates realized volatility vs target (15%), Sharpe ratio, max drawdown, and turnover across
transaction cost assumptions (0, 2, 5, 10 bps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_forecasts_to_c2c(
    prices: pd.Series,
    pred: pd.DataFrame,
    step: int = 21,
) -> pd.DataFrame:
    """Calibrate model variance forecasts to close-to-close return variance using strictly past data.

    Inside each refit block, regresses past squared close-to-close returns on past model variance
    to estimate the scale factor, then applies that scale to the subsequent test block.
    """
    common_idx = pred.index
    p = prices.reindex(common_idx)
    r_c2c = p.pct_change()
    r_sq = r_c2c**2

    cal_pred = pd.DataFrame(index=common_idx)
    models = [c for c in pred.columns if c != "y"]
    n = len(pred)

    for m in models:
        raw_v = np.exp(pred[m].to_numpy(dtype=float))
        cal_v = np.empty(n)

        start = 0
        while start < n:
            stop = min(start + step, n)
            if start < 50:
                scale = 1.87 if m != "garch_raw" else 1.0
            else:
                past_r_sq = r_sq.iloc[:start].to_numpy(dtype=float)
                past_v = raw_v[:start]
                mask = np.isfinite(past_r_sq) & np.isfinite(past_v) & (past_v > 0)
                if mask.sum() > 30:
                    scale = float(past_r_sq[mask].mean() / past_v[mask].mean())
                else:
                    scale = 1.87 if m != "garch_raw" else 1.0
            cal_v[start:stop] = raw_v[start:stop] * max(scale, 0.1)
            start = stop
        cal_pred[m] = np.log(cal_v)
    return cal_pred


def single_vol_target_backtest(
    prices: pd.Series,
    forecast_logvar: pd.Series,
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
    cost_bps: float = 5.0,
) -> dict:
    """Run a single volatility-targeting backtest.

    Parameters
    ----------
    prices : pd.Series
        Close prices of the underlying index.
    forecast_logvar : pd.Series
        Log-variance forecast for day t, available at the close of day t-1.
    target_vol : float
        Annualized target volatility (0.15 = 15%).
    max_leverage : float
        Maximum leverage cap (e.g. 2.0).
    cost_bps : float
        One-way transaction cost in basis points (1 bps = 0.0001).
    """
    daily_target = target_vol / np.sqrt(252)

    # Forecast daily close-to-close volatility
    daily_vol = np.sqrt(np.exp(forecast_logvar))

    # Position on day t uses forecast from close t-1
    # shift(1) ensures that position[t] depends strictly on forecast[t-1] and prices up to t-1
    target_weights = np.clip(daily_target / daily_vol.shift(1), 0.0, max_leverage)

    # Asset return on day t: (Close[t] - Close[t-1]) / Close[t-1]
    asset_ret = prices.pct_change()

    # Align common dates
    idx = target_weights.dropna().index.intersection(asset_ret.dropna().index)
    w = target_weights.loc[idx]
    r = asset_ret.loc[idx]

    # Turnover is the day-to-day position adjustment
    turnover = w.diff().abs().fillna(w.iloc[0] if len(w) > 0 else 0.0)
    cost = turnover * (cost_bps * 1e-4)

    # Net strategy return on day t
    net_ret = w * r - cost

    ann_vol = float(net_ret.std() * np.sqrt(252)) if len(net_ret) > 1 else 0.0
    ann_ret = float(net_ret.mean() * 252) if len(net_ret) > 0 else 0.0
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-8 else 0.0

    cum = (1.0 + net_ret).cumprod()
    peak = cum.cummax()
    dd = (peak - cum) / peak
    mdd = float(dd.max()) if len(dd) > 0 else 0.0
    ann_turnover = float(turnover.mean() * 252) if len(turnover) > 0 else 0.0

    return {
        "target_vol": target_vol,
        "realized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "annual_turnover": ann_turnover,
        "net_ret": net_ret,
        "equity_curve": cum,
        "weights": w,
    }


def compare_vol_targeting(
    prices: pd.Series,
    pred: pd.DataFrame,
    target_vol: float = 0.15,
    max_leverage: float = 2.0,
    cost_bps_list: list[float] | None = None,
    step: int = 21,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare multiple models and Buy-and-Hold under calibrated volatility targeting."""
    if cost_bps_list is None:
        cost_bps_list = [0.0, 2.0, 5.0, 10.0]

    # Calibrate variance forecasts to close-to-close returns using past data
    cal_pred = calibrate_forecasts_to_c2c(prices, pred, step=step)

    models = [c for c in cal_pred.columns if c != "y"]
    rows = []
    equity_curves = {}

    # Buy and Hold benchmark
    asset_ret = prices.reindex(cal_pred.index).pct_change().dropna()
    bh_vol = float(asset_ret.std() * np.sqrt(252))
    bh_ann_ret = float(asset_ret.mean() * 252)
    bh_sharpe = float(bh_ann_ret / bh_vol) if bh_vol > 1e-8 else 0.0
    bh_cum = (1.0 + asset_ret).cumprod()
    bh_peak = bh_cum.cummax()
    bh_mdd = float(((bh_peak - bh_cum) / bh_peak).max())
    equity_curves["buy_and_hold"] = bh_cum

    for cost_bps in cost_bps_list:
        rows.append({
            "model": "buy_and_hold",
            "cost_bps": cost_bps,
            "target_vol": target_vol,
            "realized_vol": bh_vol,
            "sharpe": bh_sharpe,
            "max_drawdown": bh_mdd,
            "annual_turnover": 0.0,
        })

        for m in models:
            res = single_vol_target_backtest(
                prices=prices.reindex(cal_pred.index),
                forecast_logvar=cal_pred[m],
                target_vol=target_vol,
                max_leverage=max_leverage,
                cost_bps=cost_bps,
            )
            rows.append({
                "model": m,
                "cost_bps": cost_bps,
                "target_vol": target_vol,
                "realized_vol": res["realized_vol"],
                "sharpe": res["sharpe"],
                "max_drawdown": res["max_drawdown"],
                "annual_turnover": res["annual_turnover"],
            })
            if cost_bps == 5.0:
                equity_curves[m] = res["equity_curve"]

    summary_df = pd.DataFrame(rows)
    equity_df = pd.DataFrame(equity_curves)
    return summary_df, equity_df

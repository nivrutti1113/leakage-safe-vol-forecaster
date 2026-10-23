"""Single CLI entry point for vol-forecaster.

Usage:
    python -m volfc.run --config configs/sp500.yaml
    python -m volfc.run --ticker ^NSEI --horizons 1 5 22
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .backtest import compare_vol_targeting
from .data import download_yfinance, load_csv, make_synthetic
from .evaluate import (
    pairwise_dm_table,
    qlike,
    regime_breakdown,
    run_walk_forward,
    score_table,
)
from .features import build_dataset
from .models import make_models


def parse_args():
    parser = argparse.ArgumentParser(description="Leakage-safe volatility forecasting pipeline.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file.")
    parser.add_argument("--source", choices=["synthetic", "csv", "yfinance"], default=None)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--n-days", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=None)
    parser.add_argument("--initial-train", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--target-vol", type=float, default=None)
    parser.add_argument("--max-leverage", type=float, default=None)
    return parser.parse_args()


def load_config(args) -> dict:
    cfg = {
        "source": "synthetic",
        "ticker": "synthetic",
        "path": None,
        "start": "2010-01-01",
        "n_days": 1500,
        "seed": 0,
        "horizons": [1, 5, 22],
        "initial_train": 1000,
        "step": 21,
        "out": "results",
        "target_vol": 0.15,
        "max_leverage": 2.0,
        "transaction_costs_bps": [0.0, 2.0, 5.0, 10.0],
    }
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    cfg.update(loaded)

    # CLI arguments override config file
    for key in [
        "source", "ticker", "path", "start", "n_days", "seed",
        "initial_train", "step", "out", "target_vol", "max_leverage"
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.horizons is not None:
        cfg["horizons"] = args.horizons
    return cfg


def get_data(cfg: dict) -> pd.DataFrame:
    src = cfg["source"]
    if src == "synthetic":
        return make_synthetic(n_days=cfg.get("n_days", 3500), seed=cfg.get("seed", 0))
    if src == "csv":
        return load_csv(cfg["path"])
    return download_yfinance(cfg["ticker"], start=cfg.get("start", "2010-01-01"))


def plot_forecasts(pred: pd.DataFrame, out_path: Path):
    """Plot out-of-sample forecasts vs realised for the last 500 days."""
    ann = lambda lv: np.sqrt(np.exp(lv) * 252) * 100
    last = pred.tail(500)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(last.index, ann(last["y"]), color="0.65", lw=1.0, label="Realised (GK)")
    if "garch" in last.columns:
        ax.plot(last.index, ann(last["garch"]), color="tab:green", lw=1.1, label="GARCH(1,1) Calibrated")
    if "har" in last.columns:
        ax.plot(last.index, ann(last["har"]), color="tab:blue", lw=1.2, label="HAR")
    if "ridge" in last.columns:
        ax.plot(last.index, ann(last["ridge"]), color="tab:purple", lw=1.2, label="Ridge")
    if "gbm" in last.columns:
        ax.plot(last.index, ann(last["gbm"]), color="tab:red", lw=1.2, label="GBM")
    ax.set_ylabel("Annualised Volatility (%)")
    ax.set_title("Out-of-Sample Volatility Forecasts (Last 500 Days)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_qlike_cis(summary_df: pd.DataFrame, out_path: Path):
    """Plot QLIKE difference vs HAR with 95% bootstrap confidence intervals."""
    h1 = summary_df[summary_df["horizon"] == 1].copy()
    sub = h1[h1["model"] != "har"].copy()
    if sub.empty or "ci_lower_vs_har" not in sub.columns:
        return

    # diff = loss_m - loss_har. Negative means better than HAR.
    sub["diff"] = sub["qlike"] - float(h1[h1["model"] == "har"]["qlike"].iloc[0])
    sub["err_low"] = sub["diff"] - sub["ci_lower_vs_har"]
    sub["err_high"] = sub["ci_upper_vs_har"] - sub["diff"]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(sub))
    ax.errorbar(
        sub["diff"],
        y_pos,
        xerr=[sub["err_low"].clip(lower=0), sub["err_high"].clip(lower=0)],
        fmt="o",
        color="tab:blue",
        ecolor="0.4",
        capsize=4,
        lw=1.5,
    )
    ax.axvline(0, color="red", linestyle="--", lw=1, alpha=0.7, label="HAR Baseline")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["model"])
    ax.set_xlabel("QLIKE Difference vs HAR (Negative = Better than HAR)")
    ax.set_title("Out-of-Sample QLIKE vs HAR with 95% Block-Bootstrap CI (h=1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_vol_targeting(equity_df: pd.DataFrame, prices: pd.Series, target_vol: float, out_eq: Path, out_vol: Path):
    """Plot cumulative wealth and rolling realized volatility of vol-targeting portfolios."""
    # Equity curve plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in equity_df.columns:
        lw = 1.6 if col in ["har", "gbm", "buy_and_hold"] else 1.0
        ax.plot(equity_df.index, equity_df[col], label=col, lw=lw)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative Wealth (Log Scale, Base=1.0)")
    ax.set_title(f"Volatility Targeting Strategy Equity Curves (Target Vol = {int(target_vol*100)}%, Cost = 5 bps)")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_eq, dpi=130)
    plt.close(fig)

    # Rolling realized volatility plot (66 days)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ann_factor = np.sqrt(252) * 100
    for col in ["har", "gbm", "buy_and_hold"]:
        if col in equity_df.columns:
            ret = equity_df[col].pct_change()
            roll_vol = ret.rolling(66).std() * ann_factor
            ax.plot(roll_vol.index, roll_vol, label=f"{col} Realised", lw=1.2)
    ax.axhline(target_vol * 100, color="black", linestyle="--", lw=1.5, label=f"Target ({int(target_vol*100)}%)")
    ax.set_ylabel("66-Day Rolling Realised Vol (%)")
    ax.set_title("Realised Volatility vs Target Under Volatility Targeting")
    ax.legend(ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_vol, dpi=130)
    plt.close(fig)


def run_pipeline(cfg: dict) -> pd.DataFrame:
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker = cfg["ticker"]
    safe_ticker = ticker.replace("^", "")

    print(f"=== Running vol-forecaster for {ticker} ===")
    df = get_data(cfg)
    horizons = cfg["horizons"]
    all_scores = []
    h1_pred = None

    for h in horizons:
        print(f"\n--- Processing Horizon h = {h} ---")
        X, y = build_dataset(df, horizon=h)
        print(f"Data: {len(X)} usable days, {X.shape[1]} features, horizon={h}")

        models = make_models(horizon=h)
        pred = run_walk_forward(X, y, models, cfg["initial_train"], cfg["step"], h)
        pred_path = out_dir / f"predictions_{safe_ticker}_h{h}.csv"
        pred.to_csv(pred_path)
        print(f"Saved {pred_path}")

        if h == 1:
            h1_pred = pred
            # Default predictions.csv compatibility
            pred.to_csv(out_dir / "predictions.csv")

        # Score table
        st = score_table(pred, horizon=h, ref="har", n_boot=1000, seed=cfg.get("seed", 0))
        st.insert(0, "horizon", h)
        st.insert(0, "ticker", ticker)
        all_scores.append(st)

        # Pairwise DM with Holm correction
        var_true = np.exp(pred["y"].to_numpy())
        losses = {m: qlike(var_true, np.exp(pred[m].to_numpy())) for m in pred.columns if m != "y"}
        pdm = pairwise_dm_table(losses, horizon=h)
        pdm.to_csv(out_dir / f"pairwise_dm_{safe_ticker}_h{h}.csv", index=False)

        # Regime breakdown
        reg_df = regime_breakdown(pred, X, horizon=h, ref="har")
        reg_df.to_csv(out_dir / f"regimes_{safe_ticker}_h{h}.csv", index=False)

    summary_df = pd.concat(all_scores, ignore_index=True)

    # Save summary.csv
    # Existing summary.csv may exist from prior runs (e.g. other ticker); append or update tidy summary
    summary_path = out_dir / "summary.csv"
    if summary_path.exists():
        old_sum = pd.read_csv(summary_path)
        # remove rows with current ticker
        old_sum = old_sum[old_sum["ticker"] != ticker]
        merged = pd.concat([old_sum, summary_df], ignore_index=True)
        merged.to_csv(summary_path, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)
    print(f"\nUpdated {summary_path}")

    # Standard metrics.csv compatibility for h=1
    st_h1 = summary_df[summary_df["horizon"] == 1].drop(columns=["ticker", "horizon"])
    st_h1.to_csv(out_dir / "metrics.csv", index=False)

    # Generate Figures
    if h1_pred is not None:
        plot_forecasts(h1_pred, out_dir / "forecast.png")
        plot_qlike_cis(summary_df, out_dir / "qlike_ci.png")

        # Volatility Targeting Backtest
        print("\n--- Running Volatility Targeting Backtest (h=1) ---")
        prices = df["Close"].reindex(h1_pred.index)
        vt_summary, vt_equity = compare_vol_targeting(
            prices=prices,
            pred=h1_pred,
            target_vol=cfg["target_vol"],
            max_leverage=cfg["max_leverage"],
            cost_bps_list=cfg["transaction_costs_bps"],
        )
        vt_summary.to_csv(out_dir / f"vol_targeting_metrics_{safe_ticker}.csv", index=False)
        vt_equity.to_csv(out_dir / f"vol_targeting_equity_{safe_ticker}.csv")
        plot_vol_targeting(vt_equity, prices, cfg["target_vol"], out_dir / "vol_targeting_equity.png", out_dir / "vol_targeting_realized.png")
        print(f"Saved vol-targeting backtests and plots to {out_dir}")

    # Print summary table preview
    cols_show = ["ticker", "horizon", "model", "qlike", "mse_logvar", "r2_logvar", "dm_stat_vs_har", "dm_p_vs_har", "n_oos"]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print("\n=== Walk-Forward Summary Results ===")
        print(summary_df[cols_show].to_string(index=False))

    return summary_df


def main():
    args = parse_args()
    cfg = load_config(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()

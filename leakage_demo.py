"""Show how the validation scheme changes the score of the SAME model on the SAME data.

Schemes:
  - shuffled k-fold (leaky)
  - contiguous k-fold, no purge
  - purged k-fold + embargo
  - walk-forward

Evaluates all schemes on the exact SAME out-of-sample dates (initial_train .. n).
Reports purged vs non-purged contiguous k-fold honestly across horizons h = 1, 5, 22.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from volfc.cv import PurgedKFold
from volfc.data import download_yfinance, load_csv, make_synthetic
from volfc.evaluate import qlike, run_walk_forward
from volfc.features import MAX_WINDOW, build_dataset
from volfc.models import make_high_capacity_gbm, make_models


def get_data(args) -> pd.DataFrame:
    if args.source == "synthetic":
        return make_synthetic(n_days=args.n_days, seed=args.seed)
    if args.source == "csv":
        return load_csv(args.path)
    return download_yfinance(args.ticker, start=args.start)


def oof(X: pd.DataFrame, y: pd.Series, splits, make):
    pred = np.full(len(y), np.nan)
    for train, test in splits:
        pred[test] = make().fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
    return pred


def score(y, pred):
    mask = np.isfinite(y) & np.isfinite(pred)
    return float(qlike(np.exp(np.asarray(y)[mask]), np.exp(np.asarray(pred)[mask])).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "csv", "yfinance"], default="synthetic")
    ap.add_argument("--path")
    ap.add_argument("--ticker", default="^GSPC")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--n-days", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 22])
    ap.add_argument("--initial-train", type=int, default=1000)
    ap.add_argument("--step", type=int, default=63)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.horizon is not None:
        horizons = [args.horizon]
    else:
        horizons = args.horizons

    print(f"Loading data from {args.source}...")
    df = get_data(args)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    all_tables = []
    plot_tables = {}

    for h in horizons:
        print(f"\n==========================================")
        print(f"Running Leakage Demo for Horizon h = {h}")
        print(f"==========================================")
        X, y = build_dataset(df, horizon=h)
        n = len(X)
        eval_idx = np.arange(args.initial_train, n)
        y_eval = y.iloc[eval_idx].to_numpy()

        models = {"har": make_models(horizon=h)["har"], "gbm_high_capacity": make_high_capacity_gbm}

        schemes = {
            "shuffled k-fold (leaky)": list(KFold(args.folds, shuffle=True, random_state=args.seed).split(np.arange(n))),
            "contiguous k-fold, no purge": list(KFold(args.folds, shuffle=False).split(np.arange(n))),
            "purged k-fold + embargo": list(PurgedKFold(args.folds, horizon=h, embargo=MAX_WINDOW + h).split(n)),
        }

        rows = []
        for sname, splits in schemes.items():
            print(f"  Evaluating {sname}...")
            for mname, make in models.items():
                oof_full = oof(X, y, splits, make)
                # Evaluate on the SAME common out-of-sample sample
                oof_eval = oof_full[eval_idx]
                s = score(y_eval, oof_eval)
                rows.append({"horizon": h, "scheme": sname, "model": mname, "qlike": s})

        print("  Evaluating walk-forward...")
        wf = run_walk_forward(X, y, models, args.initial_train, args.step, h)
        for mname in models:
            s = score(wf["y"], wf[mname])
            rows.append({"horizon": h, "scheme": "walk-forward", "model": mname, "qlike": s})

        res_h = pd.DataFrame(rows)
        tbl_h = res_h.pivot(index="scheme", columns="model", values="qlike").loc[
            ["shuffled k-fold (leaky)", "contiguous k-fold, no purge", "purged k-fold + embargo", "walk-forward"]
        ]
        tbl_h.insert(0, "horizon", h)
        all_tables.append(tbl_h.reset_index())
        plot_tables[h] = tbl_h.drop(columns="horizon")

        print(f"\nQLIKE by Validation Scheme on Common Sample [N={len(eval_idx)}] (horizon={h}; lower=better):")
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
            print(tbl_h.drop(columns="horizon"))

    final_csv = pd.concat(all_tables, ignore_index=True)
    final_csv.to_csv(out / "leakage_demo.csv", index=False)
    print(f"\nSaved {out}/leakage_demo.csv")

    # Plot figure across horizons
    n_plots = len(plot_tables)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (h, tbl) in zip(axes, plot_tables.items()):
        tbl[["har", "gbm_high_capacity"]].plot(
            kind="bar", ax=ax, rot=20, color=["tab:blue", "tab:red"], legend=(h == horizons[0])
        )
        ax.set_title(f"Horizon h = {h}")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--")

    axes[0].set_ylabel("QLIKE (Lower is Better)")
    fig.suptitle("Validation Scheme Overoptimism: HAR vs High-Capacity GBM (Identical Test Sample)", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "leakage_demo.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}/leakage_demo.png")


if __name__ == "__main__":
    main()

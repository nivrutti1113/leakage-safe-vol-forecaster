"""Walk-forward volatility forecasting experiment.

Examples:
  python run_experiment.py --config configs/sp500.yaml
  python run_experiment.py --source synthetic
  python run_experiment.py --source yfinance --ticker ^GSPC --horizons 1 5 22
  python run_experiment.py --source yfinance --ticker ^NSEI --start 2010-01-01
"""
from __future__ import annotations

import argparse
from volfc.run import load_config, run_pipeline


def main():
    ap = argparse.ArgumentParser(description="Walk-forward volatility forecasting.")
    ap.add_argument("--config", help="Path to YAML config file.")
    ap.add_argument("--source", choices=["synthetic", "csv", "yfinance"], default=None)
    ap.add_argument("--path")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--n-days", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=None)
    ap.add_argument("--initial-train", type=int, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # If --horizon is specified, convert to horizons list
    if args.horizon is not None and args.horizons is None:
        args.horizons = [args.horizon]

    cfg = load_config(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()

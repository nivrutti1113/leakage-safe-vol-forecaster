"""Loss functions, Diebold-Mariano tests, Mincer-Zarnowitz diagnostics,
bootstrap confidence intervals, and walk-forward evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .cv import walk_forward_splits


def qlike(var_true, var_pred) -> np.ndarray:
    """QLIKE loss on variance levels (robust loss for volatility forecasts; lower is better)."""
    vt = np.asarray(var_true, dtype=float)
    vp = np.asarray(var_pred, dtype=float)
    vp = np.clip(vp, a_min=1e-12, a_max=None)
    r = vt / vp
    return r - np.log(r) - 1.0


def dm_test(loss_a, loss_b, horizon: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Returns (statistic, two-sided p-value). Negative statistic means model A has the lower loss.
    Uses a Newey-West (Bartlett) variance with bandwidth max(h-1, n^(1/3)) and the
    Harvey-Leybourne-Newbold small-sample correction.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    dc = d - d.mean()
    bw = max(horizon - 1, int(round(n ** (1 / 3))))
    var = dc @ dc / n
    for k in range(1, bw + 1):
        var += 2 * (1 - k / (bw + 1)) * (dc[k:] @ dc[:-k] / n)
    var = max(var, 1e-18)
    stat = d.mean() / np.sqrt(var / n)
    h = horizon
    stat *= np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    p = 2 * (1 - stats.t.cdf(abs(stat), df=max(n - 1, 1)))
    return float(stat), float(p)


def mincer_zarnowitz(y_true, y_pred) -> dict[str, float]:
    """Mincer-Zarnowitz diagnostic regression: y = a + b * y_pred.

    An ideal forecast has a = 0, b = 1, and high R^2.
    """
    y = np.asarray(y_true, dtype=float)
    x = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = len(y)
    if n < 5 or np.all(x == x[0]):
        return {"mz_intercept": np.nan, "mz_slope": np.nan, "mz_r2": np.nan, "mz_p": np.nan}
    res = stats.linregress(x, y)
    return {
        "mz_intercept": float(res.intercept),
        "mz_slope": float(res.slope),
        "mz_r2": float(res.rvalue**2),
        "mz_p": float(res.pvalue),
    }


def holm_bonferroni(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    if n <= 1:
        return p_arr
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    adj_p = np.empty(n)
    running_max = 0.0
    for i in range(n):
        val = (n - i) * sorted_p[i]
        running_max = max(running_max, val)
        adj_p[i] = min(1.0, running_max)
    rev_order = np.empty(n, dtype=int)
    rev_order[order] = np.arange(n)
    return adj_p[rev_order]


def block_bootstrap_ci(
    loss_a, loss_b, horizon: int = 1, n_boot: int = 1000, seed: int = 0
) -> dict[str, float | int]:
    """Moving block bootstrap 95% confidence interval for mean(loss_a - loss_b).
    Block size is set to at least max(horizon, 20) to capture serial correlation.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    b_size = max(horizon, 20, int(round(n ** (1 / 3))))
    if n < 10:
        return {"diff_mean": float(d.mean()), "ci_lower": np.nan, "ci_upper": np.nan, "boot_block_size": b_size}
    n_blocks = int(np.ceil(n / b_size))
    max_start = max(n - b_size + 1, 1)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([d[s : s + b_size] for s in starts])[:n]
        boot_means[b] = sample.mean()
    return {
        "diff_mean": float(d.mean()),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "boot_block_size": b_size,
    }


def pairwise_dm_table(losses: dict[str, np.ndarray], horizon: int = 1) -> pd.DataFrame:
    """Table of all pairwise Diebold-Mariano tests with Holm-Bonferroni corrected p-values."""
    models = list(losses.keys())
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m_a, m_b = models[i], models[j]
            stat, p = dm_test(losses[m_a], losses[m_b], horizon=horizon)
            rows.append({"model_a": m_a, "model_b": m_b, "dm_stat": stat, "raw_p": p})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["holm_p"] = holm_bonferroni(df["raw_p"].to_numpy())
    return df


def run_walk_forward(
    X: pd.DataFrame, y: pd.Series, factories: dict, initial_train: int, step: int, horizon: int
) -> pd.DataFrame:
    """Refit every `step` days on an expanding window; return out-of-sample log-variance forecasts."""
    n = len(X)
    preds = {k: [] for k in factories}
    idx = []
    for train, test in walk_forward_splits(n, initial_train, step, horizon):
        for name, make in factories.items():
            model = make().fit(X.iloc[train], y.iloc[train])
            preds[name].append(model.predict(X.iloc[test]))
        idx.append(test)
    idx = np.concatenate(idx)
    out = pd.DataFrame({k: np.concatenate(v) for k, v in preds.items()}, index=X.index[idx])
    out["y"] = y.iloc[idx].to_numpy()
    return out


def score_table(
    pred: pd.DataFrame, horizon: int, ref: str = "har", n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """QLIKE, log-variance MSE, R^2, Mincer-Zarnowitz diagnostics, and DM test vs ref."""
    models = [c for c in pred.columns if c != "y"]
    var_true = np.exp(pred["y"].to_numpy())
    losses = {m: qlike(var_true, np.exp(pred[m].to_numpy())) for m in models}
    sst = ((pred["y"] - pred["y"].mean()) ** 2).sum()
    rows = []
    for m in models:
        sse = ((pred[m] - pred["y"]) ** 2).sum()
        mz = mincer_zarnowitz(pred["y"], pred[m])
        row = {
            "model": m,
            "qlike": float(losses[m].mean()),
            "mse_logvar": float(sse / len(pred)),
            "r2_logvar": float(1 - sse / sst),
            "mz_intercept": mz["mz_intercept"],
            "mz_slope": mz["mz_slope"],
            "mz_r2": mz["mz_r2"],
            "n_oos": len(pred),
        }
        if m != ref and ref in losses:
            stat, p = dm_test(losses[m], losses[ref], horizon)
            boot = block_bootstrap_ci(losses[m], losses[ref], horizon=horizon, n_boot=n_boot, seed=seed)
            row.update({
                f"dm_stat_vs_{ref}": stat,
                f"dm_p_vs_{ref}": p,
                f"ci_lower_vs_{ref}": boot["ci_lower"],
                f"ci_upper_vs_{ref}": boot["ci_upper"],
                "boot_block_size": boot["boot_block_size"],
            })
        else:
            row.update({
                f"dm_stat_vs_{ref}": np.nan,
                f"dm_p_vs_{ref}": np.nan,
                f"ci_lower_vs_{ref}": np.nan,
                f"ci_upper_vs_{ref}": np.nan,
                "boot_block_size": max(horizon, 20, int(round(len(pred) ** (1 / 3)))),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True)


def regime_breakdown(
    pred: pd.DataFrame, X: pd.DataFrame, horizon: int, ref: str = "har"
) -> pd.DataFrame:
    """Score each model across market regimes using strictly past information.

    Regimes:
      - All: full out-of-sample period.
      - High Vol: day t trailing 66-day median vol > expanding median trailing 66-day vol.
      - Low Vol: day t trailing 66-day median vol <= expanding median trailing 66-day vol.
      - Covid 2020: year 2020.
    """
    # Align X with pred
    common_idx = pred.index.intersection(X.index)
    pred_sub = pred.loc[common_idx]
    X_sub = X.loc[common_idx]

    # Trailing 66d vol from X: 'lv66' is rolling 66d mean log variance
    if "lv66" in X.columns:
        trailing_vol = X["lv66"]
    else:
        trailing_vol = pred["y"]

    # Expanding median of trailing vol up to day t (strictly backward-looking)
    exp_med = trailing_vol.expanding(min_periods=66).median().loc[common_idx]
    trailing_sub = trailing_vol.loc[common_idx]

    high_vol_mask = trailing_sub > exp_med
    low_vol_mask = trailing_sub <= exp_med
    covid_mask = pd.Series(common_idx.year == 2020, index=common_idx)

    regimes = {
        "all": pred_sub,
        "high_vol": pred_sub[high_vol_mask],
        "low_vol": pred_sub[low_vol_mask],
        "covid_2020": pred_sub[covid_mask],
    }

    dfs = []
    for rname, rpred in regimes.items():
        if len(rpred) >= 20:
            tbl = score_table(rpred, horizon=horizon, ref=ref, n_boot=500)
            tbl.insert(0, "regime", rname)
            dfs.append(tbl)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
